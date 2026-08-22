"""十二关大合并：World1+World2+World3 全数据蒸一个 WideNatureCNN。验证单网络扛不扛得住跨三世界 12 关。
用法: ./venv/bin/python distill_all12_wide.py [epochs] [device] [out] [resume]  默认 32 / cpu / mario_all12_wide
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

files = (sorted(glob.glob("distill_data/*.npz")) + sorted(glob.glob("distill_data_w2/*.npz"))
         + sorted(glob.glob("distill_data_w3/*.npz")))
# 内存安全：36GB 机器装不下 12 关全量(~25GB obs)→ 随机访问压缩内存慢25倍。
# 按比例抽样到 TARGET(回到 8 关那个 ~20GB 快速区)，预分配大缓冲逐文件拷入(峰值不翻倍)。
TARGET = 700_000
rng = np.random.default_rng(0)
prob_list, counts, raw_total = [], [], 0
metas = []
for f in files:
    n = len(np.load(f)["probs"]); raw_total += n; metas.append((f, n))
frac = min(1.0, TARGET / raw_total)
for f, n in metas:
    k = max(1, int(round(n * frac)))
    sel = np.sort(rng.choice(n, size=k, replace=False))
    prob_list.append(np.load(f)["probs"][sel]); counts.append((f, sel))
    print(f"清点 {f}: {n} → 抽 {k}", flush=True)
N = sum(len(s) for _, s in counts)
obs_all = np.empty((N, 4, 84, 84), dtype=np.uint8)
off = 0
for f, sel in counts:
    d = np.load(f); obs_all[off:off+len(sel)] = d["obs"][sel]; off += len(sel)
    del d; print(f"拷入 {f}: {len(sel)} 条 ({off}/{N})", flush=True)
prob_all = np.concatenate(prob_list); del prob_list
print(f">>> 十二关总数据 {N} 条 | WideNatureCNN(686万参) | {EPOCHS} epochs | {DEVICE}")

dummy = make_env()
RESUME = len(sys.argv) > 4 and sys.argv[4] == "resume" and os.path.exists(f"{OUT}.zip")
if RESUME:
    student = PPO.load(f"{OUT}.zip", device=DEVICE); print(f">>> 从 {OUT}.zip 续训")
else:
    student = PPO("CnnPolicy", dummy, device=DEVICE, n_steps=64, verbose=0,
                  policy_kwargs=dict(features_extractor_class=WideNatureCNN,
                                     features_extractor_kwargs=dict(features_dim=1024),
                                     normalize_images=False))  # WideNatureCNN 自己 /255，关掉 sb3 的二次归一化
opt = th.optim.Adam(student.policy.parameters(), lr=2.5e-4)
probs_t = th.as_tensor(prob_all, device=DEVICE)
for ep in range(EPOCHS):
    idx = th.randperm(N); tot = 0.0
    for i in range(0, N, BATCH):
        b = idx[i:i+BATCH]
        ob_t, _ = student.policy.obs_to_tensor(obs_all[b.numpy()])
        log_q = student.policy.get_distribution(ob_t).distribution.logits
        loss = -(probs_t[b] * log_q).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    print(f"epoch {ep+1}/{EPOCHS}  loss {tot/N:.4f}", flush=True)
student.save(OUT)
print(f">>> 十二关合并蒸馏完成，存为 {OUT}.zip", flush=True)
