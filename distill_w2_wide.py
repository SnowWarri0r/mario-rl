"""World 2 蒸馏 · WideCNN 学生：用现有 305k 数据(原始+DAgger)蒸进 WideNatureCNN(686万参)。
验证容量假设：小 NatureCNN 扛不住 4 关(2-1/2-2 被挤压)，大网络能不能接住。
监督蒸馏是 WideCNN 的主场(RL 里它失败，蒸馏里它管用)。
用法: ./venv/bin/python distill_w2_wide.py [epochs] [device] [out]   默认 10 / cpu / mario_w2_wide
"""
import warnings; warnings.filterwarnings("ignore")
import sys, glob
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env
from wide_cnn import WideNatureCNN

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cpu"
OUT = sys.argv[3] if len(sys.argv) > 3 else "mario_w2_wide"
BATCH = 512

obs_all, prob_all = [], []
for f in sorted(glob.glob("distill_data_w2/*.npz")):
    d = np.load(f)
    obs_all.append(d["obs"]); prob_all.append(d["probs"])
    print(f"载入 {f}: {len(d['obs'])} 条")
obs_all = np.concatenate(obs_all); prob_all = np.concatenate(prob_all)
N = len(obs_all)
print(f">>> 总数据 {N} 条 | WideNatureCNN(686万参) | {EPOCHS} epochs | device {DEVICE}")

import os
dummy = make_env()
RESUME = len(sys.argv) > 4 and sys.argv[4] == "resume" and os.path.exists(f"{OUT}.zip")
if RESUME:
    student = PPO.load(f"{OUT}.zip", device=DEVICE)        # 续训：不浪费已训的 epoch
    print(f">>> 从 {OUT}.zip 续训")
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
print(f">>> World 2 WideCNN 蒸馏完成，存为 {OUT}.zip")
