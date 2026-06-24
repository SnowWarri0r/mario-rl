"""World 2 蒸馏第2步：收四关老师数据 (画面 → 老师7动作概率)。
2-1 用专家v2(87%)、2-2 用梯子老师(72%)、2-3/2-4 用混训(80/75%)。
2-2 多收(最难,覆盖要厚)。存到 distill_data_w2/<stage>.npz。
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env

os.makedirs("distill_data_w2", exist_ok=True)

# (stage, 老师模型, 要收多少 transition)
JOBS = [
    ("2-2", "mario_22ladder_final.zip", 45000),   # 水关最难，多收
    ("2-1", "mario_21expert_v2.zip",    40000),
    ("2-3", "mario_w2land_final.zip",   35000),
    ("2-4", "mario_w2land_final.zip",   35000),
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
    np.savez_compressed(f"distill_data_w2/{stage}.npz",
                        obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    print(f">>> {stage}: 存了 {n} 条 (老师 {model_path})")


if __name__ == "__main__":
    for stage, mp, n in JOBS:
        print(f"=== 收 {stage} ===")
        collect(stage, mp, n)
    print(">>> World 2 四关全部收完 → distill_data_w2/")
