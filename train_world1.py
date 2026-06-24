"""路线A 第一阶段：World 1（1-1~1-4）混合关训练一个网络。
8 并行 RandomStages 环境，每个 reset 随机选 1-1/1-2/1-3/1-4 之一。
用法: ./venv/bin/python train_world1.py [步数]   默认 300 万
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_world1

N_ENVS = 8
DEVICE = "cpu"


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000

    venv = make_vec_env(make_env_world1, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "CnnPolicy", venv,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,        # 多关任务更难探索，熵系数从 0.01 提到 0.05
        clip_range=0.2,
        device=DEVICE,
        verbose=1,
    )

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_w1",
                              name_prefix="mario_w1")

    print(f">>> World1 开训 {total:,} 步 | 8并行随机选关(1-1~1-4) + 奖励归一化 + ent_coef=0.05")
    model.learn(total_timesteps=total, callback=ckpt, progress_bar=False)
    model.save("mario_w1_final")
    venv.save("vecnormalize_w1.pkl")
    print(">>> World1 训练结束，存为 mario_w1_final.zip")


if __name__ == "__main__":
    main()
