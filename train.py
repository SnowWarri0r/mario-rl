"""长训 v2：8 个并行环境 + 奖励归一化。
用法:
    ./venv/bin/python train.py            # 默认训 100 万步
    ./venv/bin/python train.py 3000000
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env

N_ENVS = 8
DEVICE = "cpu"


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000

    # 8 个环境各跑在独立进程里（SubprocVecEnv = 真并行，绕过 GIL）。make_vec_env 自动加 Monitor。
    venv = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    # 奖励归一化：把几百上千的奖励压成正常尺度。norm_obs=False —— 图像不归一化（CnnPolicy 自己 /255）。
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "CnnPolicy", venv,
        learning_rate=2.5e-4,
        n_steps=512,           # 每环境 512 步 → 每轮 rollout 收 8×512=4096 步（8 条独立来源）
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,         # 这次先不动，单独看"并行+归一化"的效果
        clip_range=0.2,
        device=DEVICE,
        verbose=1,
    )

    # save_freq 是"每环境步数"，要除以 N_ENVS 才是每 ~10 万总步存一次
    ckpt = CheckpointCallback(save_freq=max(100_000 // N_ENVS, 1),
                              save_path="./checkpoints",
                              name_prefix="mario_v2")

    print(f">>> v2 开训 {total:,} 步 | {N_ENVS} 并行 + 奖励归一化 | 设备 {model.policy.device}")
    model.learn(total_timesteps=total, callback=ckpt, progress_bar=False)
    model.save("mario_ppo_v2_final")
    venv.save("vecnormalize_v2.pkl")     # 存归一化统计量（续训用；纯看模型不需要）
    print(">>> v2 训练结束，存为 mario_ppo_v2_final.zip")


if __name__ == "__main__":   # macOS 多进程必须有这个保护
    main()
