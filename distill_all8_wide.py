"""八关大合并：World1(distill_data/) + World2(distill_data_w2/) 全部数据蒸进一个 WideNatureCNN。
验证单网络扛不扛得住跨两世界 8 关。监督蒸馏(软KL)。
用法: ./venv/bin/python distill_all8_wide.py [epochs] [device] [out] [resume]   默认 30 / cpu / mario_all8_wide
"""
import warnings; warnings.filterwarnings("ignore")
import sys, glob, os
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env
from wide_cnn import WideNatureCNN

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cpu"
OUT = sys.argv[3] if len(sys.argv) > 3 else "mario_all8_wide"
BATCH = 512

obs_all, prob_all = [], []
for f in sorted(glob.glob("distill_data/*.npz")) + sorted(glob.glob("distill_data_w2/*.npz")):
    d = np.load(f)
    obs_all.append(d["obs"]); prob_all.append(d["probs"])
    print(f"载入 {f}: {len(d['obs'])} 条")
obs_all = np.concatenate(obs_all); prob_all = np.concatenate(prob_all)
N = len(obs_all)
print(f">>> 八关总数据 {N} 条 | WideNatureCNN(686万参) | {EPOCHS} epochs | device {DEVICE}")

dummy = make_env()
RESUME = len(sys.argv) > 4 and sys.argv[4] == "resume" and os.path.exists(f"{OUT}.zip")
if RESUME:
    student = PPO.load(f"{OUT}.zip", device=DEVICE); print(f">>> 从 {OUT}.zip 续训")
else:
    student = PPO("CnnPolicy", dummy, device=DEVICE, n_steps=64, verbose=0,
                  policy_kwargs=dict(features_extractor_class=WideNatureCNN,
                                     features_extractor_kwargs=dict(features_dim=1024)))
opt = th.optim.Adam(student.policy.parameters(), lr=2.5e-4)

probs_t = th.as_tensor(prob_all, device=DEVICE)
for ep in range(EPOCHS):
    idx = th.randperm(N)
    tot = 0.0
    for i in range(0, N, BATCH):
        b = idx[i:i+BATCH]
        ob_t, _ = student.policy.obs_to_tensor(obs_all[b.numpy()])
        dist = student.policy.get_distribution(ob_t)
        log_q = dist.distribution.logits
        loss = -(probs_t[b] * log_q).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    print(f"epoch {ep+1}/{EPOCHS}  loss {tot/N:.4f}")

student.save(OUT)
print(f">>> 八关合并蒸馏完成，存为 {OUT}.zip")
