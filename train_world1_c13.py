"""World1 课程加权续训：warm-start 现有网络，把 1-3 采样权重提到 ~40% 专攻钉子户。
ent_coef 保持 0.02。用法: ./venv/bin/python train_world1_c13.py [步数]  默认 300 万
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_world1_c13

N_ENVS = 8


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    base = "mario_w1c_final.zip"
    assert os.path.exists(base), f"缺 {base}"

    venv = make_vec_env(make_env_world1_c13, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    if os.path.exists("vecnormalize_w1c.pkl"):
        venv = VecNormalize.load("vecnormalize_w1c.pkl", venv); venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO.load(base, env=venv, device="cpu")
    model.ent_coef = 0.02
    print(f">>> 课程加权续训 {total:,} 步 | 1-3 采样~40% | warm-start from {base}")

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_w1_c13", name_prefix="mario_w1c13")
    model.learn(total_timesteps=total, callback=ckpt, reset_num_timesteps=True)
    model.save("mario_w1c13_final")
    venv.save("vecnormalize_w1c13.pkl")
    print(">>> 课程加权续训结束，存为 mario_w1c13_final.zip")


if __name__ == "__main__":
    main()
