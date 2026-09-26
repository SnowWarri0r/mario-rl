"""BigCNN：在"浅层朴素卷积"家族里加容量，而不是换成残差堆。

**为什么放弃 ImpalaCNN。** 实测（小批量探针，1500 步，batch 512，跨 28 关分层取样）：
    Wide 6.86M          0.989   9s
    Impala s32 @2.5e-4  1.762   塌（常数解 1.8428）
    Impala s32 +clip    1.840   塌
    Impala s32 @1e-4+clip+warmup  1.330   79s
    Impala s64          任何 lr（5e-5..3e-3）在 3000 步内塌
即：要低 lr + warmup + 裁剪伺候才活，活了也不如 Wide，每步还慢 9-19 倍
（Wide 28.4s/epoch，Impala s64 552s/epoch）。整轮 v34/v35 各烧掉数小时全废。

**真正要修的是参数分配，不是深度。** WideNatureCNN 686 万参数里 640 万在
Linear(6272→1024)，卷积只有 30 万（4.5%）。flatten 有 7×7×128=6272 维是根因。
这里加**第四层带 stride 的卷积**把空间压到 3×3，flatten 降到 4608 以下，
省下来的预算全给卷积通道——拓扑仍是 4 层朴素卷积，没有残差相加，
不会出现 Impala 那种连乘放大和晚发性崩溃。

⚠️ 自己做 /255，sb3 侧必须 normalize_images=False。
⚠️ 加层就要重新量激活尺度：sb3 对每层套 gain=√2 正交初始化，层数越多连乘越狠
   （Impala 20 层 + 8 次残差相加放大了 118 倍，直接把网络变成常数函数）。
"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 变体表：(通道序列, 最后一级 stride)。前三层沿用 Nature 的 8/s4 → 4/s2 → 3/s1。
VARIANTS = {
    "v1": (64, 128, 256, 512),
    "v2": (96, 192, 384, 512),
    "v3": (128, 256, 384, 640),
}


class BigCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024, variant="v1"):
        super().__init__(observation_space, features_dim)
        c1, c2, c3, c4 = VARIANTS[variant]
        n_in = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, c1, 8, stride=4), nn.ReLU(),   # 84 → 20
            nn.Conv2d(c1, c2, 4, stride=2), nn.ReLU(),     # 20 → 9
            nn.Conv2d(c2, c3, 3, stride=1), nn.ReLU(),     #  9 → 7
            nn.Conv2d(c3, c4, 3, stride=2), nn.ReLU(),     #  7 → 3  ← 这一级把 flatten 砍到 1/3
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs.float() / 255.0))


if __name__ == "__main__":
    import gymnasium as gym, numpy as np
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from wide_cnn import WideNatureCNN
    sp = spaces.Box(0, 255, (4, 84, 84), np.uint8)

    def report(m, name):
        conv = sum(p.numel() for p in m.cnn.parameters())
        lin = sum(p.numel() for p in m.linear.parameters())
        with th.no_grad():
            nf = m.cnn(th.zeros(1, 4, 84, 84)).shape[1]
        print(f"{name:12s} flatten {nf:6d}  卷积 {conv/1e6:5.2f}M  线性 {lin/1e6:5.2f}M  "
              f"合计 {(conv+lin)/1e6:5.2f}M  卷积占比 {conv/(conv+lin)*100:4.1f}%")

    report(WideNatureCNN(sp), "Wide(现役)")
    for v in VARIANTS:
        report(BigCNN(sp, variant=v), f"BigCNN {v}")

    class _Stub(gym.Env):
        observation_space, action_space = sp, spaces.Discrete(7)
        def reset(self, **k): return sp.sample(), {}
        def step(self, a): return sp.sample(), 0.0, False, False, {}

    print("\n=== 激活尺度（经 sb3 ortho_init，要走真实构造路径）===")
    for v in VARIANTS:
        pol = PPO("CnnPolicy", _Stub(), device="cpu", verbose=0,
                  policy_kwargs=dict(normalize_images=False, features_extractor_class=BigCNN,
                                     features_extractor_kwargs=dict(features_dim=1024, variant=v))).policy
        x = th.from_numpy(np.random.randint(0, 255, (8, 4, 84, 84), dtype=np.uint8))
        with th.no_grad():
            h = x.float() / 255.0; scales = [float(h.abs().mean())]
            for lyr in pol.features_extractor.cnn:
                h = lyr(h)
                if isinstance(lyr, nn.ReLU): scales.append(float(h.abs().mean()))
        print(f"  {v}: " + " → ".join(f"{s:.2f}" for s in scales) +
              f"   总放大 {scales[-1]/scales[0]:.1f}×")
