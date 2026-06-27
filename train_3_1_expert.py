"""3-1 钉子户专家：混训卡 x≈2224(0%)，单关猛训补上。
warm-start 自 mario_w3_final(已会游 3-1 到 ~2989)，比从头快。
用法: ./venv/bin/python train_3_1_expert.py [步数]  默认 300 万
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_stage31

N_ENVS = 8
BASE_MODEL = "mario_w3_final.zip"
BASE_VECN = "vecnormalize_w3.pkl"


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    venv = make_vec_env(make_env_stage31, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    if os.path.exists(BASE_VECN):
        venv = VecNormalize.load(BASE_VECN, venv)
        venv.training = True; venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO.load(BASE_MODEL, env=venv, device="cpu")       # warm-start：已会游 3-1
    model.learning_rate = 2.5e-4
    print(f">>> 3-1 专家续训 {total:,} 步 | warm-start {BASE_MODEL} | 单关 3-1")

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_31expert", name_prefix="mario_31exp")
    model.learn(total_timesteps=total, callback=ckpt, reset_num_timesteps=True)
    model.save("mario_31expert_final")
    venv.save("vecnormalize_31expert.pkl")
    print(">>> 3-1 专家训练结束，存为 mario_31expert_final.zip")


if __name__ == "__main__":
    main()
