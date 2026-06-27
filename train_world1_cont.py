"""World1 续训：加载 3M 跑完的网络，把 ent_coef 压回 0.02 继续练。
目的：排除"熵系数太高(0.05)"这个干扰变量，看 1-1 通关率能不能爬回来。
用法: ./venv/bin/python train_world1_cont.py [续训步数]   默认再练 300 万
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_world1

N_ENVS = 8
NEW_ENT = 0.02   # 从 0.05 压回


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    assert os.path.exists("mario_w1_final.zip"), "等 3M 跑完生成 mario_w1_final.zip 再来"

    venv = make_vec_env(make_env_world1, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    # 接续奖励归一化的滚动统计量（别从零重估）
    if os.path.exists("vecnormalize_w1.pkl"):
        venv = VecNormalize.load("vecnormalize_w1.pkl", venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # warm-start：加载已训权重，只改 ent_coef
    model = PPO.load("mario_w1_final.zip", env=venv, device="cpu")
    old = model.ent_coef
    model.ent_coef = NEW_ENT
    print(f">>> 续训 {total:,} 步 | ent_coef {old} → {NEW_ENT} | warm-start from mario_w1_final")

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_w1_cont",
                              name_prefix="mario_w1c")
    # reset_num_timesteps=True：日志步数从 0 起，方便单独画这段续训曲线
    model.learn(total_timesteps=total, callback=ckpt, reset_num_timesteps=True)
    model.save("mario_w1c_final")
    venv.save("vecnormalize_w1c.pkl")
    print(">>> 续训结束，存为 mario_w1c_final.zip")


if __name__ == "__main__":
    main()
