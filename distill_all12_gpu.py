"""十二关大合并 · H20 GPU 版：875k 帧全量 obs（~25GB）常驻单卡显存，训练零 host→device 拷贝。

Mac 上这一步是内存墙：36GB 装不下 25GB obs，压缩内存把随机访问拖慢 25 倍（~29min/epoch），
磁盘 memmap 也救不了（25GB 远超 page cache，每 epoch 全 miss）。H20 单张 96GB 卡直接把
整份数据当成一个 uint8 显存张量放着，索引一个 batch 就是一次显存内 gather。

用法: python distill_all12_gpu.py [epochs] [out] [batch] [resume]   默认 32 / mario_all12_wide / 512
      resume 传字面量 "resume" → 复用已有 <out>.zip 权重续训。
"""
import warnings; warnings.filterwarnings("ignore")
import sys, glob, os, time
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stub_env import make_stub_env
from wide_cnn import WideNatureCNN

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
OUT = sys.argv[2] if len(sys.argv) > 2 else "mario_all12_wide"
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 512     # 沿用八关合并的配方(512 / lr 2.5e-4)便于对比
RESUME = len(sys.argv) > 4 and sys.argv[4] == "resume" and os.path.exists(f"{OUT}.zip")
DEVICE = "cuda"
# 消融开关：MARIO_DOUBLE_NORM=1 故意复现"双重 /255"（sb3 归一化 + WideNatureCNN 自己再 /255），
# 用来单变量隔离八关合并当年那 61% 里有多少是这个 bug 吃掉的（数据/epoch 全不变，只动这一个）。
DOUBLE_NORM = os.environ.get("MARIO_DOUBLE_NORM") == "1"
# 数据范围开关：MARIO_DATA_DIRS="distill_data,distill_data_w2" → 只喂八关那份(730k)，
# 用来把"数据多样性"从"epoch 数"里摘出来单独称重。默认三个世界全喂。
DATA_DIRS = os.environ.get("MARIO_DATA_DIRS", "distill_data,distill_data_w2,distill_data_w3").split(",")
SAVE_EVERY = 8                                             # 中途也存一份，长跑被抢卡不至于全丢

assert th.cuda.is_available(), "没有可用 CUDA —— 先确认这台机器的卡是真空的（nvidia-smi 显存 ≠ CUDA 可用）"
# 存出来的 zip 里 pickle 了 numpy 的内部布局：numpy 2.x 存的模型在 numpy 1.x 里 load 会报
# ModuleNotFoundError: numpy._core.numeric。评测/录像那条链路被 nes-py 钉在 numpy<2，所以蒸馏也必须在 numpy<2 里做。
assert np.__version__ < "2", f"当前 numpy {np.__version__} —— 换 numpy<2 的环境跑，否则存出的模型模拟器那侧加载不了"
print(f">>> GPU: {th.cuda.get_device_name(0)} | 显存 {th.cuda.get_device_properties(0).total_memory/2**30:.0f}GB", flush=True)

files = [f for d in DATA_DIRS for f in sorted(glob.glob(f"{d.strip()}/*.npz"))]
assert files, f"这些目录里没找到 npz：{DATA_DIRS}"
metas = []
for f in files:
    n = len(np.load(f)["probs"]); metas.append((f, n)); print(f"清点 {f}: {n} 条", flush=True)
N = sum(n for _, n in metas)
gb = N * 4 * 84 * 84 / 2**30
print(f">>> 数据 {'+'.join(d.strip() for d in DATA_DIRS)} 共 {N} 条 | obs {gb:.1f}GB 常驻显存 "
      f"| WideNatureCNN(686万参) | {EPOCHS} epochs | batch {BATCH}", flush=True)

# 全量搬进显存：逐文件读 npz（RAM 只过一份，峰值 ~2GB）→ 拷到显存对应区间
obs_gpu = th.empty((N, 4, 84, 84), dtype=th.uint8, device=DEVICE)
prob_list, off = [], 0
for f, n in metas:
    d = np.load(f)
    obs_gpu[off:off+n] = th.from_numpy(d["obs"]).to(DEVICE, non_blocking=True)
    prob_list.append(d["probs"]); off += n
    del d
    print(f"入显存 {f}: {n} 条 ({off}/{N})", flush=True)
probs_gpu = th.as_tensor(np.concatenate(prob_list), dtype=th.float32, device=DEVICE)
del prob_list
print(f">>> 数据就位，显存占用 {th.cuda.memory_allocated()/2**30:.1f}GB", flush=True)

dummy = make_stub_env()                                    # 只借形状，不跑模拟器
if RESUME:
    student = PPO.load(f"{OUT}.zip", device=DEVICE); print(f">>> 从 {OUT}.zip 续训", flush=True)
else:
    student = PPO("CnnPolicy", dummy, device=DEVICE, n_steps=64, verbose=0,
                  policy_kwargs=dict(features_extractor_class=WideNatureCNN,
                                     features_extractor_kwargs=dict(features_dim=1024),
                                     normalize_images=DOUBLE_NORM))  # 正常=False：WideNatureCNN 自己 /255，别让 sb3 再除一次
if DOUBLE_NORM:
    print(">>> 消融模式：normalize_images=True，输入会被 /255 两次（复现八关合并当年的 handicap）", flush=True)
opt = th.optim.Adam(student.policy.parameters(), lr=2.5e-4)

for ep in range(EPOCHS):
    t0 = time.time(); idx = th.randperm(N, device=DEVICE); tot = 0.0
    for i in range(0, N, BATCH):
        b = idx[i:i+BATCH]
        # obs 已是显存里的 uint8；sb3 preprocess(normalize_images=False) 只做 .float()，/255 交给 WideNatureCNN
        log_q = student.policy.get_distribution(obs_gpu[b]).distribution.logits
        loss = -(probs_gpu[b] * log_q).sum(1).mean()        # soft policy distillation: -Σ p_老师·log q_学生
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    dt = time.time() - t0
    print(f"epoch {ep+1}/{EPOCHS}  loss {tot/N:.4f}  {dt:.1f}s  ({N/dt/1000:.0f}k 帧/s)", flush=True)
    if (ep + 1) % SAVE_EVERY == 0 and ep + 1 < EPOCHS:
        student.save(OUT); print(f"    …中途存档 {OUT}.zip", flush=True)

student.save(OUT)
print(f">>> 十二关合并蒸馏完成，存为 {OUT}.zip", flush=True)
