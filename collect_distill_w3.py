"""World 3 蒸馏数据：收四关老师 (画面 → 老师7动作概率)。
3-1 用专家1.6M档(80%)、3-2/3-3/3-4 用混训 mario_w3_final(90/80/55%)。
3-4(城堡 55%)和 3-1 多收。存到 distill_data_w3/<stage>.npz。
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env

os.makedirs("distill_data_w3", exist_ok=True)

JOBS = [
    ("3-1", "checkpoints_31expert/mario_31exp_1600000_steps.zip", 40000),
    ("3-4", "mario_w3_final.zip", 40000),   # 城堡 55% 偏弱, 多收
    ("3-2", "mario_w3_final.zip", 30000),
    ("3-3", "mario_w3_final.zip", 35000),
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
            probs = dist.distribution.probs.cpu().numpy()[0]
            action = int(dist.sample().cpu().numpy()[0])
        obs_buf.append(o.astype(np.uint8)); prob_buf.append(probs.astype(np.float32))
        o, r, term, trunc, info = env.step(action)
        if term or trunc:
            o, _ = env.reset()
        if len(obs_buf) % 10000 == 0:
            print(f"  {stage}: {len(obs_buf)}/{n}")
    env.close()
    np.savez_compressed(f"distill_data_w3/{stage}.npz",
                        obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    print(f">>> {stage}: 存了 {n} 条 (老师 {model_path})")


if __name__ == "__main__":
    for stage, mp, n in JOBS:
        print(f"=== 收 {stage} ===")
        collect(stage, mp, n)
    print(">>> World 3 四关全部收完 → distill_data_w3/")
