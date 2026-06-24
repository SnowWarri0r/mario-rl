"""把 train.log 里的训练进展画成曲线。随时重跑刷新。
用法: ./venv/bin/python plot_progress.py [train.log]
"""
import re, sys
import matplotlib
matplotlib.use("Agg")
# macOS 自带的中文字体，否则中文标签会变方块
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

log = sys.argv[1] if len(sys.argv) > 1 else "train.log"
text = open(log).read()

def grab(block, key):
    m = re.search(r"\|\s*" + re.escape(key) + r"\s*\|\s*([-\d.eE+]+)\s*\|", block)
    return float(m.group(1)) if m else None

ts, rew, ev = [], [], []
for b in re.split(r"-{5,}", text):           # 每个 iteration 一块，用横线分隔
    t, r = grab(b, "total_timesteps"), grab(b, "ep_rew_mean")
    if t is not None and r is not None:
        ts.append(t); rew.append(r); ev.append(grab(b, "explained_variance"))

if not ts:
    print("还没有可画的数据点（第一块日志要等约 30 秒）"); sys.exit()

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(ts, rew, "o-", color="tab:blue", label="ep_rew_mean (走多远/学得好)")
ax1.set_xlabel("训练步数 total_timesteps")
ax1.set_ylabel("ep_rew_mean", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()                            # 第二根 y 轴放 explained_variance
ax2.plot(ts, ev, "s--", color="tab:orange", alpha=0.7, label="explained_variance (0→1 越懂)")
ax2.set_ylabel("explained_variance", color="tab:orange")
ax2.set_ylim(-0.05, 1.05)
ax2.tick_params(axis="y", labelcolor="tab:orange")

plt.title(f"马里奥训练进展（最新 {int(ts[-1]):,} 步，reward={rew[-1]:.0f}）")
fig.tight_layout()
fig.savefig("progress.png", dpi=110)
print(f">>> progress.png  数据点 {len(ts)} 个 | 最新步数 {int(ts[-1]):,} | ep_rew_mean {rew[-1]:.0f}")
