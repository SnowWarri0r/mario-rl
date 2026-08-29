"""把 2-2 的通关技能蒸进指定叠帧数的学生（4 或 8），作为后续抖动微调的起点。

学生用 sb3 自带的小 NatureCNN（跟梯子专家同架构，不带自己的 /255，所以不用碰 normalize_images）。
只蒸策略头：价值头蒸不了（老师的 V 是在 4 帧观测 + 另一套奖励归一化下学的），
后面 PPO 微调时会自己重新估 V —— 这是"先蒸后 RL"的固有代价，第一批更新会浪费在补 V 上。

用法: MARIO_STACK=8 python distill_stack.py [epochs] [out]   默认 30 / mario_22stack<k>
"""
import warnings; warnings.filterwarnings("ignore")
import sys, glob
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env, STACK_FRAMES

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = sys.argv[2] if len(sys.argv) > 2 else f"mario_22stack{STACK_FRAMES}"
BATCH = 512
KEY = f"obs{STACK_FRAMES}"
assert th.cuda.is_available()
assert np.__version__ < "2", "存出的模型要能被 numpy<2 的评测链路加载"

obs_l, prob_l = [], []
for f in sorted(glob.glob("distill_data_stack22/*.npz")):
    d = np.load(f)
    obs_l.append(d[KEY]); prob_l.append(d["probs"])
    print(f"载入 {f}: {len(d['probs'])} 条 ({KEY})", flush=True)
obs = np.concatenate(obs_l); probs = np.concatenate(prob_l)
del obs_l, prob_l
N = len(obs)
print(f">>> {N} 条 | 叠 {STACK_FRAMES} 帧 | {EPOCHS} epochs → {OUT}", flush=True)

obs_gpu = th.from_numpy(obs).cuda()
probs_gpu = th.as_tensor(probs, dtype=th.float32, device="cuda")
del obs, probs

dummy = make_env(stages=["2-2"], noop=0)        # 只借形状（叠帧数跟随 MARIO_STACK）
student = PPO("CnnPolicy", dummy, device="cuda", n_steps=64, verbose=0)
opt = th.optim.Adam(student.policy.parameters(), lr=2.5e-4)
for ep in range(EPOCHS):
    idx = th.randperm(N, device="cuda"); tot = 0.0
    for i in range(0, N, BATCH):
        b = idx[i:i+BATCH]
        log_q = student.policy.get_distribution(obs_gpu[b]).distribution.logits
        loss = -(probs_gpu[b] * log_q).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    print(f"epoch {ep+1}/{EPOCHS}  loss {tot/N:.4f}", flush=True)
dummy.close()
student.save(OUT)
print(f">>> 存为 {OUT}.zip", flush=True)
