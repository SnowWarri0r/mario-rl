"""把同一次训练后期若干档的权重平均成一个网络（SWA）。

动机：v40 的 25 个档在同一批 420 局上通关 347..374，档间标准差约 1.75pp，
而它们的 loss 只差千分之一量级 —— 训练停在一片平坦的低谷里来回晃，
停在哪一步决定了实战分数的一大部分。按实测挑档已验证无效（选择集第一的 ep37
在留出集上 394，输给不挑的 ep40 397，胜者诅咒）。
平均权重是压这种晃动的标准做法，而且结果仍是**一个**网络，不改口径。

用法: python swa.py <输出.zip> <档1.zip> <档2.zip> ...
"""
import sys, torch as th
from stable_baselines3 import PPO
import wide_cnn  # noqa: F401
out, srcs = sys.argv[1], sys.argv[2:]
base = PPO.load(srcs[0], device="cpu")
sd = {k: v.clone().float() for k, v in base.policy.state_dict().items()}
for p in srcs[1:]:
    for k, v in PPO.load(p, device="cpu").policy.state_dict().items():
        sd[k] += v.float()
for k in sd:
    sd[k] /= len(srcs)
base.policy.load_state_dict(sd)
base.save(out)
print(f"平均了 {len(srcs)} 个档 → {out}")
