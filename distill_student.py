"""蒸馏第3步：一个学生网络监督模仿 4 关老师的动作分布(soft policy distillation)。
loss = 软交叉熵 -Σ p_老师 · log q_学生。纯监督，不走 RL。学生默认小 NatureCNN。
用法: ./venv/bin/python distill_student.py [epochs] [device]   默认 8 / cpu
"""
import warnings; warnings.filterwarnings("ignore")
import sys, glob
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cpu"
BATCH = 512

# 载入 4 关数据
obs_all, prob_all = [], []
for f in sorted(glob.glob("distill_data/*.npz")):
    d = np.load(f)
    obs_all.append(d["obs"]); prob_all.append(d["probs"])
    print(f"载入 {f}: {len(d['obs'])} 条")
obs_all = np.concatenate(obs_all); prob_all = np.concatenate(prob_all)
N = len(obs_all)
print(f">>> 总数据 {N} 条 | 学生默认小 NatureCNN | {EPOCHS} epochs | device {DEVICE}")

# 用一个 env 构造学生策略(只为拿 obs/action space)
dummy = make_env()
student = PPO("CnnPolicy", dummy, device=DEVICE, n_steps=64, verbose=0)
opt = th.optim.Adam(student.policy.parameters(), lr=2.5e-4)

probs_t = th.as_tensor(prob_all, device=DEVICE)         # (N,7) 老师目标
for ep in range(EPOCHS):
    idx = th.randperm(N)
    tot = 0.0
    for i in range(0, N, BATCH):
        b = idx[i:i+BATCH]
        ob_t, _ = student.policy.obs_to_tensor(obs_all[b.numpy()])  # numpy(B,4,84,84)→tensor+预处理
        dist = student.policy.get_distribution(ob_t)
        log_q = dist.distribution.logits                            # torch Categorical: 已是归一化 log-prob
        loss = -(probs_t[b] * log_q).sum(1).mean()                  # 软交叉熵
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    print(f"epoch {ep+1}/{EPOCHS}  loss {tot/N:.4f}")

student.save("mario_distilled")
print(">>> 蒸馏完成，存为 mario_distilled.zip")
