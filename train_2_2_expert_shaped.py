"""2-2 水下专家 · 奖励塑形版：死亡重罚(-50) + 通关大奖(+200)，治"全程死敌人手里"。
skip=4(标准)。用法: ./venv/bin/python train_2_2_expert_shaped.py [步数]  默认 300 万
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_stage22_shaped

N_ENVS = 8


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    venv = make_vec_env(make_env_stage22_shaped, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "CnnPolicy", venv,
        learning_rate=2.5e-4, n_steps=512, batch_size=256, n_epochs=4,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.02, clip_range=0.2,
        device="cpu", verbose=1,
    )
    print(f">>> 2-2 塑形专家开训 {total:,} 步 | 死罚-50 + 通关奖+200 | skip=4")

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_22shaped", name_prefix="mario_22shaped")
    model.learn(total_timesteps=total, callback=ckpt)
    model.save("mario_22shaped_final")
    venv.save("vecnormalize_22shaped.pkl")
    print(">>> 2-2 塑形专家训练结束，存为 mario_22shaped_final.zip")


if __name__ == "__main__":
    main()
