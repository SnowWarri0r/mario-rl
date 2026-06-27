"""World1 加宽 NatureCNN 从零重训：uniform 4 关，解小网络容量瓶颈(用确定能训的浅网络)。
用法: ./venv/bin/python train_world1_wide.py [步数] [device]   默认 600 万 / cpu
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_world1
from wide_cnn import WideNatureCNN

N_ENVS = 8


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 6_000_000
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

    venv = make_vec_env(make_env_world1, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    policy_kwargs = dict(
        features_extractor_class=WideNatureCNN,
        features_extractor_kwargs=dict(features_dim=1024),
        normalize_images=False,
    )
    model = PPO(
        "CnnPolicy", venv, policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4, n_steps=512, batch_size=256, n_epochs=4,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.02, clip_range=0.2,
        device=device, verbose=1,
    )
    nparams = sum(p.numel() for p in model.policy.parameters())
    print(f">>> 加宽CNN从零训 {total:,} 步 | 参数 {nparams:,} | uniform 4关 | device {model.policy.device}")

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_w1_wide", name_prefix="mario_w1wide")
    model.learn(total_timesteps=total, callback=ckpt)
    model.save("mario_w1wide_final")
    venv.save("vecnormalize_w1wide.pkl")
    print(">>> 加宽CNN训练结束，存为 mario_w1wide_final.zip")


if __name__ == "__main__":
    main()
