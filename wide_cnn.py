"""WideNatureCNN：NatureCNN 的加宽版。同样 3 层浅卷积(确定能训)，通道翻倍 + 特征维度翻倍。
NatureCNN: 32→64→64, feat 512  |  Wide: 64→128→128, feat 1024。容量约 2-3×，无深网络信号衰减问题。
"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class WideNatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024):
        super().__init__(observation_space, features_dim)
        n_in = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 64, 8, stride=4), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs.float() / 255.0))
