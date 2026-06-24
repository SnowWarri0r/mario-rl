"""2-2 水下专家 · 精细控制版：skip=2(原4)，让 agent 能更快连点 A 划水。
其余跟原版一致(干净消融:到底是不是 frame-skip 卡住游泳)。
skip 减半→同样步数覆盖游戏帧少一半，所以默认 4M 步补偿。
用法: ./venv/bin/python train_2_2_expert_fine.py [步数]  默认 400 万
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_stage22_fine

N_ENVS = 8


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
    venv = make_vec_env(make_env_stage22_fine, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "CnnPolicy", venv,
        learning_rate=2.5e-4, n_steps=512, batch_size=256, n_epochs=4,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.02, clip_range=0.2,
        device="cpu", verbose=1,
    )
    print(f">>> 2-2 精细专家开训 {total:,} 步 | 单关 2-2 | skip=2 | ent_coef=0.02")

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_22fine", name_prefix="mario_22fine")
    model.learn(total_timesteps=total, callback=ckpt)
    model.save("mario_22fine_final")
    venv.save("vecnormalize_22fine.pkl")
    print(">>> 2-2 精细专家训练结束，存为 mario_22fine_final.zip")


if __name__ == "__main__":
    main()
