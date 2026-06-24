"""World 3 student：WideCNN 监督蒸馏 4 关老师(3-1/3-2/3-3/3-4)。145k 数据，~4GB，36GB Mac 轻松跑。
镜像 W1/W2 student。用法: ./venv/bin/python -u distill_w3_student.py [epochs] [device] [out]  默认 30 / cpu / mario_w3_wide
"""
import warnings; warnings.filterwarnings("ignore")
import sys, glob
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env
from wide_cnn import WideNatureCNN

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cpu"
OUT = sys.argv[3] if len(sys.argv) > 3 else "mario_w3_wide"
BATCH = 512

obs_all, prob_all = [], []
for f in sorted(glob.glob("distill_data_w3/*.npz")):
    d = np.load(f); obs_all.append(d["obs"]); prob_all.append(d["probs"])
    print(f"载入 {f}: {len(d['obs'])} 条", flush=True)
obs_all = np.concatenate(obs_all); prob_all = np.concatenate(prob_all)
N = len(obs_all)
print(f">>> World 3 数据 {N} 条 | WideNatureCNN | {EPOCHS} epochs | {DEVICE}", flush=True)

dummy = make_env()
student = PPO("CnnPolicy", dummy, device=DEVICE, n_steps=64, verbose=0,
              policy_kwargs=dict(features_extractor_class=WideNatureCNN,
                                 features_extractor_kwargs=dict(features_dim=1024),
                                 normalize_images=False))  # WideNatureCNN 自己 /255，关掉 sb3 的二次归一化(否则 /255²→输入塌成~0→特征死)
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
print(f">>> World 3 student 蒸馏完成，存为 {OUT}.zip", flush=True)
