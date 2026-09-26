"""ImpalaCNN：给 28 关学生用的更大骨干。换它的依据是实测，不是"大一点总没坏处"。

**为什么换。** 四代学生的数据量从 90 万涨到 296 万帧（3.3×），
训练 loss 几乎不动：0.8859 → 0.8858 → 0.8657 → 0.8600 → 0.8570。
注意那是**训练** loss——数据翻三倍还拟合不动，说明连训练集都吃不下，
是欠拟合／容量限制，不是数据不够。逐关表现也印证：加一关的数据必然从别关扣走
（2-2 定点补课 +6.5pp，代价是 4-2 −23、5-3 −19），像在固定预算里挪钱。

**为什么不是简单加宽。** WideNatureCNN 的 686 万参数里 **640 万压在最后一个
Linear(6272→1024) 上，三层卷积一共才 46 万**。对一个要靠空间精度过窄缝的任务，
这个分配是反的——加宽只会让那个 Linear 更胖，卷积照样看不清。
Impala 式残差堆把参数放回卷积，同时用残差连接解决深网络信号衰减
（那正是当初选浅层 NatureCNN 的理由）。

结构：**4 个** stage，每个 = conv → maxpool → 2×残差块。
⚠️ 第一版只用 3 个 stage，84→42→21→11，flatten 还有 7744 维，
参数照样 95% 压在 Linear 上（卷积 0.39M / 线性 7.93M）——**等于没解决分配问题**。
加到 4 个 stage 把空间降到 6×6，再把通道加宽，卷积占比才真正上来。

⚠️ 跟 WideNatureCNN 一样自己做 /255，所以 sb3 侧必须 normalize_images=False，
否则会除两次（这个项目踩过，专门留了 MARIO_DOUBLE_NORM 消融开关）。
"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1),
                               nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.b(x)


class _Stage(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            _Residual(cout), _Residual(cout))

    def forward(self, x):
        return self.f(x)


class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024, scale=64):
        super().__init__(observation_space, features_dim)
        n_in = observation_space.shape[0]
        chans = (scale, scale * 2, scale * 4, scale * 4)
        stages, c = [], n_in
        for co in chans:
            stages.append(_Stage(c, co)); c = co
        self.cnn = nn.Sequential(*stages, nn.ReLU(), nn.Flatten())
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs.float() / 255.0))


if __name__ == "__main__":
    from gymnasium import spaces
    import numpy as np
    sp = spaces.Box(0, 255, (4, 84, 84), np.uint8)
    for sc in (32, 48, 64):
        m = ImpalaCNN(sp, scale=sc)
        conv = sum(p.numel() for p in m.cnn.parameters())
        lin = sum(p.numel() for p in m.linear.parameters())
        print(f"scale={sc:3d}  卷积 {conv/1e6:6.2f}M  线性 {lin/1e6:6.2f}M  "
              f"合计 {(conv+lin)/1e6:6.2f}M")
    from wide_cnn import WideNatureCNN
    w = WideNatureCNN(sp)
    print(f"\n对照 WideNatureCNN  卷积 {sum(p.numel() for p in w.cnn.parameters())/1e6:.2f}M  "
          f"线性 {sum(p.numel() for p in w.linear.parameters())/1e6:.2f}M  "
          f"合计 {sum(p.numel() for p in w.parameters())/1e6:.2f}M")
