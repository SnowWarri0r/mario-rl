"""十二关大合并 · 磁盘 memmap 版：obs 存磁盘(~25GB)，训练按 batch 分页读，RAM 不爆。
36GB Mac 装不下 25GB obs in-RAM(被压缩内存拖慢25倍)→ 用磁盘换内存，也是扩到 32 关的正道。
全量 875k 数据，不抽样。用法: ./venv/bin/python -u distill_all12_mmap.py [epochs] [device] [out] [resume]
"""
import warnings; warnings.filterwarnings("ignore")
import sys, glob, os
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env
from wide_cnn import WideNatureCNN

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cpu"
OUT = sys.argv[3] if len(sys.argv) > 3 else "mario_all12_wide"
BATCH = 512
MMAP = "obs_all12.dat"

files = (sorted(glob.glob("distill_data/*.npz")) + sorted(glob.glob("distill_data_w2/*.npz"))
         + sorted(glob.glob("distill_data_w3/*.npz")))
prob_list, counts = [], []
for f in files:
    p = np.load(f)["probs"]; prob_list.append(p); counts.append((f, len(p)))
    print(f"清点 {f}: {len(p)} 条", flush=True)
N = sum(c for _, c in counts)
prob_all = np.concatenate(prob_list); del prob_list
print(f">>> 十二关总数据 {N} 条 → 磁盘 memmap {MMAP} | WideNatureCNN | {EPOCHS} epochs | {DEVICE}", flush=True)

# 建/复用磁盘 obs 缓冲：逐文件写入，RAM 只过一份(~2GB transient)
if not os.path.exists(MMAP) or os.path.getsize(MMAP) != N * 4 * 84 * 84:
    obs_mm = np.memmap(MMAP, dtype=np.uint8, mode="w+", shape=(N, 4, 84, 84))
    off = 0
    for f, c in counts:
        d = np.load(f); obs_mm[off:off+c] = d["obs"]; off += c
        del d; obs_mm.flush(); print(f"写盘 {f}: {c} 条 ({off}/{N})", flush=True)
    del obs_mm
obs = np.memmap(MMAP, dtype=np.uint8, mode="r", shape=(N, 4, 84, 84))   # 只读分页

dummy = make_env()
RESUME = len(sys.argv) > 4 and sys.argv[4] == "resume" and os.path.exists(f"{OUT}.zip")
if RESUME:
    student = PPO.load(f"{OUT}.zip", device=DEVICE); print(f">>> 从 {OUT}.zip 续训", flush=True)
else:
    student = PPO("CnnPolicy", dummy, device=DEVICE, n_steps=64, verbose=0,
                  policy_kwargs=dict(features_extractor_class=WideNatureCNN,
                                     features_extractor_kwargs=dict(features_dim=1024)))
opt = th.optim.Adam(student.policy.parameters(), lr=2.5e-4)
probs_t = th.as_tensor(prob_all, device=DEVICE)
for ep in range(EPOCHS):
    idx = np.random.permutation(N); tot = 0.0
    for i in range(0, N, BATCH):
        b = np.sort(idx[i:i+BATCH])                       # 排序索引→分页读更连续
        ob = np.asarray(obs[b])                            # 从磁盘取这一 batch 到 RAM
        ob_t, _ = student.policy.obs_to_tensor(ob)
        log_q = student.policy.get_distribution(ob_t).distribution.logits
        loss = -(probs_t[th.as_tensor(b)] * log_q).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    print(f"epoch {ep+1}/{EPOCHS}  loss {tot/N:.4f}", flush=True)
student.save(OUT)
print(f">>> 十二关合并蒸馏完成，存为 {OUT}.zip", flush=True)
