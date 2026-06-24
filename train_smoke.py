"""短训冒烟：只验证 PPO 学习循环能在这台 Mac 上跑起来。"""
import warnings; warnings.filterwarnings("ignore")
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from make_env import make_env

# emulator 在 CPU 上跑、CNN 又小，MPS 帮助有限；冒烟先用 cpu 最稳。长训再试 mps。
DEVICE = "cpu"

env = Monitor(make_env())          # Monitor 负责统计每局总分（ep_rew_mean）

model = PPO(
    "CnnPolicy",                   # 用卷积网络看画面
    env,
    learning_rate=2.5e-4,
    n_steps=512,                   # 每玩 512 步复盘更新一次
    batch_size=64,
    n_epochs=4,
    gamma=0.99,                    # 未来奖励折扣：远的分打点折
    device=DEVICE,
    verbose=1,
)
print(">>> policy device:", model.policy.device)
print(">>> CNN 输入:", model.policy.observation_space.shape, " 动作数:", model.policy.action_space.n)

model.learn(total_timesteps=4096)  # 冒烟：约 8 次复盘更新
model.save("mario_ppo_smoke")
print(">>> SMOKE TRAIN DONE, saved mario_ppo_smoke.zip")
