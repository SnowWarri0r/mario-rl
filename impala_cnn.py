"""IMPALA-CNN：给卷积编码器加 ResNet 残差块，容量大、专治多关卡。
结构：3 组 ConvSequence(conv→maxpool→2残差块)，通道 32→64→64，最后 Linear 到 256。
"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c0 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        y = self.c0(th.relu(x))
        y = self.c1(th.relu(y))
        return x + y          # skip connection（ResNet 那一招）


class ConvSequence(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res0 = ResidualBlock(out_ch)
        self.res1 = ResidualBlock(out_ch)

    def forward(self, x):
        return self.res1(self.res0(self.pool(self.conv(x))))


class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, channels=(32, 64, 64)):
        super().__init__(observation_space, features_dim)
        n_in = observation_space.shape[0]   # 4（叠帧）
        seqs, c = [], n_in
        for oc in channels:
            seqs.append(ConvSequence(c, oc)); c = oc
        self.convs = nn.Sequential(*seqs)
        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flat = self.convs(sample).reshape(1, -1).shape[1]
        self.fc = nn.Linear(n_flat, features_dim)

    def forward(self, obs):
        x = obs.float() / 255.0          # 自己归一化（policy 设 normalize_images=False）
        x = th.relu(self.convs(x))
        x = x.reshape(x.shape[0], -1)
        return th.relu(self.fc(x))
