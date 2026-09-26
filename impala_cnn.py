"""ImpalaCNN：给 28 关学生用的更大骨干。换它的依据是实测，不是"大一点总没坏处"。

**为什么换。** 四代学生的数据量从 90 万涨到 296 万帧（3.3×），
训练 loss 几乎不动：0.8859 → 0.8858 → 0.8657 → 0.8600 → 0.8570。
注意那是**训练** loss——数据翻三倍还拟合不动，说明连训练集都吃不下，
是欠拟合／容量限制，不是数据不够。逐关表现也印证：加一关的数据必然从别关扣走
（2-2 定点补课 +6.5pp，代价是 4-2 −23、5-3 −19），像在固定预算里挪钱。

**为什么不是简单加宽。** WideNatureCNN 的 686 万参数里 **640 万压在最后一个
Linear(6272→1024) 上，三层卷积一共才 46 万**。对一个要靠空间精度过窄缝的任务，
这个分配是反的——加宽只会让那个 Linear 更胖，卷积照样看不清。

结构：**4 个** stage，每个 = conv → maxpool → 2×残差块。
⚠️ 第一版只用 3 个 stage，flatten 还有 7744 维，参数照样 95% 压在 Linear 上——
等于没解决分配问题。加到 4 个 stage 把空间降到 6×6，卷积占比才真正上来。

⚠️⚠️ **必须有 GroupNorm，别当它是可选的装饰。** 不带归一化的第一版训了 32 个
epoch（5 小时）完全白费：loss 从 epoch 2 起死死钉在 1.8557，而 1.8557 正好是
动作边缘分布的熵——网络退化成了**常数函数**，八个不同输入输出一模一样，
差异到小数点后 8 位都是 0。原因是 sb3 对 features_extractor 统一套
**gain=√2 的正交初始化**（那是给 3 层朴素 CNN 调的），在 20 层卷积 + 8 次残差
相加上会连乘：实测每个 stage 放大 3.2 倍，0.50→1.87→6.51→19.8→59.0，
训练后滚到 15932，ReLU 死掉 58%，于是 Adam 第一个 epoch 就把动作头权重归零、
只留 bias 拟合边缘分布，剩下 31 个 epoch 纯属空转。
选 GroupNorm 而不是"调初始化"，是因为**归一化对初始化免疫**——
sb3 在本文件之外动手脚也压不坏它；而调 init 会被那一道 ortho_init 覆盖掉。

⚠️ 跟 WideNatureCNN 一样自己做 /255，所以 sb3 侧必须 normalize_images=False。
"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

GROUPS = 8          # 64/128/256 都能整除


class _Residual(nn.Module):
    """pre-activation 残差块。GN 放在每个 conv 前面，压住权重尺度的连乘。"""

    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(
            nn.GroupNorm(GROUPS, c), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(GROUPS, c), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.b(x)


class _Stage(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.GroupNorm(GROUPS, cout),          # 主干上也要一道，否则 stage 间照样连乘
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


def activation_scale(policy, n=8):
    """逐 stage 量一遍前向的量级。**要传 sb3 建好的 policy，不是裸 ImpalaCNN**——
    把网络炸掉的那道 ortho_init 是 sb3 在本文件之外加的，只测裸模块看不见它。"""
    import numpy as np
    fe = policy.features_extractor
    dev = next(fe.parameters()).device
    x = th.from_numpy(np.random.randint(0, 255, (n, 4, 84, 84), dtype=np.uint8)).to(dev)
    out = []
    with th.no_grad():
        h = x.float() / 255.0
        out.append(("输入", float(h.abs().mean())))
        for i, st in enumerate(fe.cnn[:-2]):
            h = st(h)
            out.append((f"stage{i+1}", float(h.abs().mean())))
        f = fe.cnn(x.float() / 255.0)
        o = fe.linear(f)
        out.append(("flatten", float(f.abs().mean())))
        out.append(("linear", float(o.abs().mean())))
    return out


if __name__ == "__main__":
    from gymnasium import spaces
    import gymnasium as gym, numpy as np
    from stable_baselines3 import PPO

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

    # —— 自检：走 sb3 的真实构造路径量激活尺度 ——
    # 判据不是"看着还行"，是**逐级放大倍数**。无归一化那版每级 ×3.2、四级 118 倍，
    # 就是它退化成常数函数的直接原因。
    class _Stub(gym.Env):
        observation_space, action_space = sp, spaces.Discrete(7)
        def reset(self, **k): return sp.sample(), {}
        def step(self, a): return sp.sample(), 0.0, False, False, {}

    pol = PPO("CnnPolicy", _Stub(), device="cpu", verbose=0,
              policy_kwargs=dict(normalize_images=False,
                                 features_extractor_class=ImpalaCNN,
                                 features_extractor_kwargs=dict(features_dim=1024, scale=64))).policy
    print("\n=== 激活尺度自检（经 sb3 ortho_init 之后）===")
    rows = activation_scale(pol)
    prev = None
    for name, v in rows:
        r = f"  ×{v/prev:5.2f}" if prev else ""
        print(f"  {name:8s} |x| mean {v:9.3f}{r}")
        prev = v
    stages = [v for n, v in rows if n.startswith("stage")]
    grow = stages[-1] / rows[0][1]
    print(f"\n  输入→stage4 总放大 {grow:.1f} 倍")
    assert grow < 20, f"激活仍在连乘放大（{grow:.1f}×），归一化没起作用，别开训"
    print("  ✓ 未见连乘放大")
