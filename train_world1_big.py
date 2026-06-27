"""World1 大网络从零重训：IMPALA-CNN(残差) + uniform 采样 4 关，解小网络容量瓶颈。
用法: ./venv/bin/python train_world1_big.py [步数] [device]   默认 600 万 / cpu
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_world1
from impala_cnn import ImpalaCNN

N_ENVS = 8


def main():
    import os
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 6_000_000
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    warm = sys.argv[3] if len(sys.argv) > 3 else None   # 可选 warm-start 基座 zip

    venv = make_vec_env(make_env_world1, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    policy_kwargs = dict(
        features_extractor_class=ImpalaCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,   # 归一化交给 ImpalaCNN 自己做
    )
    if warm and os.path.exists(warm):
        model = PPO.load(warm, env=venv, device=device)
        model.ent_coef = 0.02
        mode = f"warm-start from {warm}"
    else:
        model = PPO(
            "CnnPolicy", venv, policy_kwargs=policy_kwargs,
            learning_rate=2.5e-4, n_steps=512, batch_size=256, n_epochs=4,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.02, clip_range=0.2,
            device=device, verbose=1,
        )
        mode = "从零"
    nparams = sum(p.numel() for p in model.policy.parameters())
    print(f">>> 大网络{mode} 训 {total:,} 步 | IMPALA-CNN 参数 {nparams:,} | uniform 4关 | device {model.policy.device}")

    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./checkpoints_w1_big", name_prefix="mario_w1big")
    model.learn(total_timesteps=total, callback=ckpt)
    model.save("mario_w1big_final")
    venv.save("vecnormalize_w1big.pkl")
    print(">>> 大网络训练结束，存为 mario_w1big_final.zip")


if __name__ == "__main__":
    main()
