"""World1 复习续训：warm-start "1-3已点亮"的基座，切回均匀采样(4关各25%)，
让被饿着的 1-2/1-4 恢复(rehearsal 防遗忘)，同时保住已学到的 1-3。
用法: ./venv/bin/python train_world1_rehearsal.py [步数]  默认 300 万
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_world1   # 均匀采样 1-1~1-4

N_ENVS = 8


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    base = "mario_w1c13_base.zip"
    assert os.path.exists(base), f"缺 {base}"

    venv = make_vec_env(make_env_world1, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    vn = "vecnormalize_w1c.pkl"
    if os.path.exists(vn):
        venv = VecNormalize.load(vn, venv); venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO.load(base, env=venv, device="cpu")
    model.ent_coef = 0.02
    print(f">>> 复习续训 {total:,} 步 | 均匀采样(各25%) | warm-start {base}(1-3已58%)")

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_w1_reh", name_prefix="mario_w1reh")
    model.learn(total_timesteps=total, callback=ckpt, reset_num_timesteps=True)
    model.save("mario_w1reh_final")
    venv.save("vecnormalize_w1reh.pkl")
    print(">>> 复习续训结束，存为 mario_w1reh_final.zip")


if __name__ == "__main__":
    main()
