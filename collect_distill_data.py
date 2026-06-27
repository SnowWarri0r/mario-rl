"""蒸馏第2步：收数据。用各关老师自己跑(stochastic)，记下 (画面 → 老师的7动作概率)。
1-3 用专家、1-1/1-2/1-4 用 mario_w1c_final。1-3 多收(难关需更多覆盖)。
存到 distill_data/<stage>.npz (obs uint8, probs float32)。
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env

os.makedirs("distill_data", exist_ok=True)

# (stage, 老师模型, 要收多少 transition)
JOBS = [
    ("1-3", "mario_13expert_final.zip", 60000),   # 难关多收
    ("1-1", "mario_w1c_final.zip",      30000),
    ("1-2", "mario_w1c_final.zip",      30000),
    ("1-4", "mario_w1c_final.zip",      30000),
]


def collect(stage, model_path, n):
    model = PPO.load(model_path, device="cpu")
    env = make_env(stages=[stage])
    obs_buf, prob_buf = [], []
    o, _ = env.reset()
    while len(obs_buf) < n:
        obs_t, _ = model.policy.obs_to_tensor(o)
        with th.no_grad():
            dist = model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.cpu().numpy()[0]      # (7,) 老师分布
            action = int(dist.sample().cpu().numpy()[0])          # 按老师策略走(stochastic)
        obs_buf.append(o.astype(np.uint8)); prob_buf.append(probs.astype(np.float32))
        o, r, term, trunc, info = env.step(action)
        if term or trunc:
            o, _ = env.reset()
        if len(obs_buf) % 10000 == 0:
            print(f"  {stage}: {len(obs_buf)}/{n}")
    env.close()
    np.savez_compressed(f"distill_data/{stage}.npz",
                        obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    print(f">>> {stage}: 存了 {n} 条 (老师 {model_path})")


if __name__ == "__main__":
    for stage, mp, n in JOBS:
        print(f"=== 收 {stage} ===")
        collect(stage, mp, n)
    print(">>> 全部收完 → distill_data/")
