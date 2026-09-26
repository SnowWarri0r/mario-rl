"""十二关大合并 · GPU 版：875k 帧全量 obs（~25GB）常驻单卡显存，训练零 host→device 拷贝。

Mac 上这一步是内存墙：36GB 装不下 25GB obs，压缩内存把随机访问拖慢 25 倍（~29min/epoch），
磁盘 memmap 也救不了（25GB 远超 page cache，每 epoch 全 miss）。单张 96GB 显存的卡直接把
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
from impala_cnn import ImpalaCNN
from big_cnn import BigCNN
# MARIO_BACKBONE=impala → 换更大的残差骨干。依据：四代学生数据量涨 3.3 倍而训练 loss
# 只从 0.8859 挪到 0.8570，连训练集都拟合不动＝容量限制；且 WideNatureCNN 的参数
# 95% 压在最后一个 Linear 上、卷积只占 4.5%，对要空间精度的任务分配是反的。
BACKBONE = os.environ.get("MARIO_BACKBONE", "wide")
if BACKBONE == "impala":
    _EXTRACTOR, _EXTRA = ImpalaCNN, dict(scale=int(os.environ.get("MARIO_SCALE", "64")))
elif BACKBONE == "big":
    # BigCNN：第四层带 stride 把 flatten 从 6272 压到 4608，省下的预算给卷积通道。
    # 卷积占比 4.4% → 25.6%(v1) / 36.8%(v2) / 38.3%(v3)，速度仍是 Wide 量级。
    _EXTRACTOR, _EXTRA = BigCNN, dict(variant=os.environ.get("MARIO_VARIANT", "v1"))
else:
    _EXTRACTOR, _EXTRA = WideNatureCNN, {}
SEED = int(os.environ.get("MARIO_SEED", "0"))

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
# ⚠️ banner 必须打**实际构造出来的**骨干，不能硬编码。原来这儿写死 "WideNatureCNN(686万参)"，
# 换 backbone 后照印不误；而第一版改成读 student 又放在了 student 建出来之前，直接 NameError。
# 这里用一次性实例只为称参数量，不参与训练。
from gymnasium import spaces as _sp
_probe_fe = _EXTRACTOR(_sp.Box(0, 255, (4, 84, 84), np.uint8), features_dim=1024, **_EXTRA)
BACKBONE_DESC = f"{type(_probe_fe).__name__}({sum(p.numel() for p in _probe_fe.parameters())/1e6:.2f}M参)"
del _probe_fe
print(f">>> 数据 {'+'.join(d.strip() for d in DATA_DIRS)} 共 {N} 条 | obs {gb:.1f}GB "
      f"| {BACKBONE_DESC} | {EPOCHS} epochs | batch {BATCH}", flush=True)

# MARIO_HOST_OBS=1：obs 留在**主机内存**（pinned），每个 batch 现传。
# 为什么要这条退路：共享服务器的卡经常被别人占满（装不下 28GB 的 obs），
# 而机器内存充足。代价是每 batch 一次 H2D 拷贝，epoch 慢几倍，但不用排队等卡。
# pin_memory 让拷贝走 DMA；probs 很小（N×7 float），照旧放显存。
HOST_OBS = os.environ.get("MARIO_HOST_OBS") == "1"
if HOST_OBS:
    print(">>> obs 留主机内存（pinned），每 batch 现传 —— 显存不足时的退路", flush=True)
    obs_gpu = th.empty((N, 4, 84, 84), dtype=th.uint8).pin_memory()
else:
    obs_gpu = th.empty((N, 4, 84, 84), dtype=th.uint8, device=DEVICE)
prob_list, off = [], 0
for f, n in metas:
    d = np.load(f)
    src = th.from_numpy(d["obs"])
    obs_gpu[off:off+n] = src if HOST_OBS else src.to(DEVICE, non_blocking=True)
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
    student = PPO("CnnPolicy", dummy, device=DEVICE, n_steps=64, verbose=0, seed=SEED,
                  policy_kwargs=dict(features_extractor_class=_EXTRACTOR,
                                     features_extractor_kwargs=dict(features_dim=1024, **_EXTRA),
                                     normalize_images=DOUBLE_NORM))  # 正常=False：WideNatureCNN 自己 /255，别让 sb3 再除一次
if DOUBLE_NORM:
    print(">>> 消融模式：normalize_images=True，输入会被 /255 两次（复现八关合并当年的 handicap）", flush=True)
# —— 开训前体检：骨干是不是活的 ——
# ⚠️ 这一步是拿 5 小时换来的。ImpalaCNN 第一版训满 32 epoch，loss 从 epoch 2 起钉死在
# 1.8557 —— 那正好是动作边缘分布的熵，网络退化成了**常数函数**（八个不同输入输出差异为 0）。
# 塌陷在第一分钟就已成定局，却要等整轮跑完才看得见。这里花 60 秒先判一次生死。
# 判据是"离常数解多远"，不是"loss 有没有降"：大数据上 loss 本来就降得慢，分不出死活。
# 用**另建的一次性模型**跑，避免热身权重污染受控对比。
def _preflight(sd):
    import copy
    n = min(2048, N)
    ob = obs_gpu[:n].to(DEVICE) if HOST_OBS else obs_gpu[:n]
    pb = probs_gpu[:n]
    pm = pb.mean(0)
    const = float(-(pm * th.log(pm.clamp_min(1e-9))).sum())
    probe = PPO("CnnPolicy", dummy, device=DEVICE, n_steps=64, verbose=0, seed=sd,
                policy_kwargs=dict(features_extractor_class=_EXTRACTOR,
                                   features_extractor_kwargs=dict(features_dim=1024, **_EXTRA),
                                   normalize_images=DOUBLE_NORM))
    o2 = th.optim.Adam(probe.policy.parameters(), lr=LR)
    for _ in range(300):
        lg = probe.policy.get_distribution(ob).distribution.logits
        l = -(pb * lg).sum(1).mean()
        o2.zero_grad(); l.backward(); o2.step()
    fin = float(l)
    del probe, o2
    th.cuda.empty_cache()
    print(f">>> 体检(seed {sd})：{n} 帧 300 步后 loss {fin:.4f}，常数解 {const:.4f}", flush=True)
    return fin < const * 0.8

LR = float(os.environ.get("MARIO_LR", "2.5e-4"))
# 塌陷是**初始化抽签**不是容量上限：同一个 seed 2 能同时放倒 BigCNN v2 和 v3，
# 而 v1 四个 seed 全活。所以不过就换 seed 重抽，而不是把网络改小。
if os.environ.get("MARIO_PREFLIGHT") == "1" and not RESUME:   # 默认关：实测它放过了 v35/v37 两次
    for _try in range(6):
        if _preflight(SEED):
            print(f">>> 体检通过，用 seed {SEED} 开训", flush=True)
            break
        SEED += 1
        print(f">>> 塌了，换 seed {SEED} 重抽", flush=True)
    else:
        raise SystemExit("连抽 6 个 seed 都塌 —— 这不是运气问题，别开整轮训练")
    student = PPO("CnnPolicy", dummy, device=DEVICE, n_steps=64, verbose=0, seed=SEED,
                  policy_kwargs=dict(features_extractor_class=_EXTRACTOR,
                                     features_extractor_kwargs=dict(features_dim=1024, **_EXTRA),
                                     normalize_images=DOUBLE_NORM))
opt = th.optim.Adam(student.policy.parameters(), lr=LR)

# —— 塌陷判据：全量数据上的常数解 ——
# 网络退化成常数函数时，loss 精确等于**边缘分布的熵**，且之后一动不动
# （v34 从 epoch 2 起 32 轮全是 1.8557，v35、v37 同样）。这个数是可以先算出来的，
# 不用等它"看起来不降了"再猜。
_pm = probs_gpu.mean(0)
CONST = float(-(_pm * th.log(_pm.clamp_min(1e-9))).sum())
print(f">>> 全量常数解 = {CONST:.4f}（epoch loss 贴到这个数＝塌了）", flush=True)

def _rebuild(sd):
    """换一个 init 抽签重来。塌陷是初始化彩票，不是容量上限：
    同一个 seed 2 能同时放倒 BigCNN v2 和 v3，而 v1 四个 seed 全活。"""
    st = PPO("CnnPolicy", dummy, device=DEVICE, n_steps=64, verbose=0, seed=sd,
             policy_kwargs=dict(features_extractor_class=_EXTRACTOR,
                                features_extractor_kwargs=dict(features_dim=1024, **_EXTRA),
                                normalize_images=DOUBLE_NORM))
    return st, th.optim.Adam(st.policy.parameters(), lr=LR)

MAX_RESEED = int(os.environ.get("MARIO_MAX_RESEED", "5"))
_reseeds = 0
ep = 0
while ep < EPOCHS:
    t0 = time.time()
    # HOST_OBS 时索引要在 CPU 上（用它切主机张量），否则在显存上
    idx = th.randperm(N, device="cpu" if HOST_OBS else DEVICE); tot = 0.0
    for i in range(0, N, BATCH):
        b = idx[i:i+BATCH]
        # obs 已是显存里的 uint8；sb3 preprocess(normalize_images=False) 只做 .float()，/255 交给骨干
        ob = obs_gpu[b].to(DEVICE, non_blocking=True) if HOST_OBS else obs_gpu[b]
        pb = probs_gpu[b.to(DEVICE)] if HOST_OBS else probs_gpu[b]
        log_q = student.policy.get_distribution(ob).distribution.logits
        loss = -(pb * log_q).sum(1).mean()        # soft policy distillation: -Σ p_老师·log q_学生
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    dt = time.time() - t0
    ep_loss = tot / N
    print(f"epoch {ep+1}/{EPOCHS}  loss {ep_loss:.4f}  {dt:.1f}s  ({N/dt/1000:.0f}k 帧/s)", flush=True)

    # ⚠️ 判据放在真实训练里，不用代理探针。
    # 2048 帧全批量的那道体检连着放过了 v35 和 v37 两次 —— 代理跟真实训练的
    # 梯度噪声和步数尺度都不一样，测不准。真实训练自己跑一个 epoch 只要 35-70s，
    # 是最便宜也最准的探测器。
    if ep_loss > CONST * 0.97:
        if _reseeds >= MAX_RESEED:
            raise SystemExit(f"连换 {MAX_RESEED} 个 seed 都塌在常数解 —— 不是运气问题，停。")
        _reseeds += 1
        SEED += 1
        print(f">>> epoch {ep+1} loss {ep_loss:.4f} 贴住常数解 {CONST:.4f} ⇒ 塌了，"
              f"换 seed {SEED} 从头重来（第 {_reseeds}/{MAX_RESEED} 次）", flush=True)
        del student, opt; th.cuda.empty_cache()
        student, opt = _rebuild(SEED)
        ep = 0
        continue

    ep += 1
    if ep % SAVE_EVERY == 0 and ep < EPOCHS:
        student.save(OUT); print(f"    …中途存档 {OUT}.zip", flush=True)

student.save(OUT)
print(f">>> 十二关合并蒸馏完成，存为 {OUT}.zip", flush=True)
