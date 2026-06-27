"""2-2 · 梯子续训：把单个 checkpoint 换成一排(2100/2400/2700/2900 各 +50)，
让胡萝卜一路把策略拽到旗杆前，治单 checkpoint 的"推过一道墙就停"打地鼠。
续训自 mario_22ckpt_final(已把确定性前沿推到 ~2253)。
用法: ./venv/bin/python train_2_2_ladder.py [步数]  默认 300 万
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_stage22_ladder

N_ENVS = 8
BASE_MODEL = "mario_22ckpt_final.zip"
BASE_VECN = "vecnormalize_22ckpt.pkl"


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    venv = make_vec_env(make_env_stage22_ladder, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    if os.path.exists(BASE_VECN):
        venv = VecNormalize.load(BASE_VECN, venv)
        venv.training = True; venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO.load(BASE_MODEL, env=venv, device="cpu")       # 续训：已会游到 2253
    model.learning_rate = 2.5e-4
    print(f">>> 2-2 梯子续训 {total:,} 步 | 续 {BASE_MODEL} | 梯子 2100/2400/2700/2900 各 +50")

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_22ladder", name_prefix="mario_22ladder")
    model.learn(total_timesteps=total, callback=ckpt, reset_num_timesteps=True)
    model.save("mario_22ladder_final")
    venv.save("vecnormalize_22ladder.pkl")
    print(">>> 2-2 梯子续训结束，存为 mario_22ladder_final.zip")


if __name__ == "__main__":
    main()
