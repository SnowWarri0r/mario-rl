"""World 3 混训：夜晚世界 4 关(3-1/3-2/3-3/3-4)随机采样单网络。无水关，全混。
跟 World 1 同套路：8×RandomStages + VecNormalize + ent_coef=0.02。
跑完 eval 看逐关，谁掉队(预判 3-3 athletic)再单独补专家，最后蒸进来。
用法: ./venv/bin/python train_world3.py [步数]  默认 400 万
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_world3

N_ENVS = 8


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
    venv = make_vec_env(make_env_world3, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "CnnPolicy", venv,
        learning_rate=2.5e-4, n_steps=512, batch_size=256, n_epochs=4,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.02, clip_range=0.2,
        device="cpu", verbose=1,
    )
    print(f">>> World 3 混训 {total:,} 步 | 3-1/3-2/3-3/3-4 RandomStages | ent_coef=0.02")

    ckpt = CheckpointCallback(save_freq=max(250_000 // N_ENVS, 1),
                              save_path="./checkpoints_w3", name_prefix="mario_w3")
    model.learn(total_timesteps=total, callback=ckpt)
    model.save("mario_w3_final")
    venv.save("vecnormalize_w3.pkl")
    print(">>> World 3 混训结束，存为 mario_w3_final.zip")


if __name__ == "__main__":
    main()
