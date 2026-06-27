"""2-2 · checkpoint 塑形续训：在已会游到 x=2095 的塑形专家基础上，
给 x≈2100(Cheep-Cheep 鱼缝)前加一次性 +60，把通关胡萝卜挂到硬点前面。
赌点：穿缝本身有了即时梯度信号，策略能学会threading那道鱼缝。
续训而非从头——游泳技能已在策略网里，只补这一下。
用法: ./venv/bin/python train_2_2_ckpt.py [步数]  默认 300 万
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_stage22_ckpt

N_ENVS = 8
BASE_MODEL = "mario_22shaped_final.zip"
BASE_VECN = "vecnormalize_22shaped.pkl"


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    venv = make_vec_env(make_env_stage22_ckpt, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    # 继承旧的 reward 归一化统计(热启动)，新的 +60 会让它快速再适配
    if os.path.exists(BASE_VECN):
        venv = VecNormalize.load(BASE_VECN, venv)
        venv.training = True; venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO.load(BASE_MODEL, env=venv, device="cpu")       # 续训：游泳技能已在策略里
    model.learning_rate = 2.5e-4
    print(f">>> 2-2 checkpoint 续训 {total:,} 步 | 续 {BASE_MODEL} | 鱼缝前 x>2100 → +60")

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_22ckpt", name_prefix="mario_22ckpt")
    model.learn(total_timesteps=total, callback=ckpt, reset_num_timesteps=True)
    model.save("mario_22ckpt_final")
    venv.save("vecnormalize_22ckpt.pkl")
    print(">>> 2-2 checkpoint 续训结束，存为 mario_22ckpt_final.zip")


if __name__ == "__main__":
    main()
