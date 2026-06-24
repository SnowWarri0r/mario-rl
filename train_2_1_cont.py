"""2-1 专家续训 v2：从 40% 的 mario_21expert_final 接着练，趁还在陡坡推上去。
用法: ./venv/bin/python train_2_1_cont.py [步数]  默认 300 万
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_stage21

N_ENVS = 8
BASE_MODEL = "mario_21expert_final.zip"
BASE_VECN = "vecnormalize_21expert.pkl"


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    venv = make_vec_env(make_env_stage21, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    if os.path.exists(BASE_VECN):
        venv = VecNormalize.load(BASE_VECN, venv)
        venv.training = True; venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO.load(BASE_MODEL, env=venv, device="cpu")
    model.learning_rate = 2.5e-4
    print(f">>> 2-1 专家续训 v2 {total:,} 步 | 续 {BASE_MODEL}(40%) | 单关 2-1")

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_21expert2", name_prefix="mario_21exp2")
    model.learn(total_timesteps=total, callback=ckpt, reset_num_timesteps=True)
    model.save("mario_21expert_v2")
    venv.save("vecnormalize_21expert_v2.pkl")
    print(">>> 2-1 专家续训 v2 结束，存为 mario_21expert_v2.zip")


if __name__ == "__main__":
    main()
