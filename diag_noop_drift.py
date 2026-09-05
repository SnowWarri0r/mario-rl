"""空按 k 帧之后，马里奥到底在哪：验证 no-op starts 是"只改相位"还是"顺带改了状态"。

Atari 把 no-op starts 定在 0-30 帧，是为了推开敌人相位而尽量不动别的。这条在陆地关成立，
**在 2-2 这种水下关不完全成立**：不划水就会一直下沉。这个脚本把每个 k 对应的落点打出来。

实测 2-2（见下）：y 从 184 一路沉到 79，**20 帧就到底了**，之后 y 恒定、计时器 200 帧才少 10。
所以 0-30 这个窗口内起点其实分成两段（水中 / 贴地），而 30 帧以上是纯相位变化。
⇒ 由此排除了一个我一度给出的错误解释：240 相位扫描里通关率从 65% 掉到 15%，
**不是**因为"状态一直在漂"（y 和 time 都不再变），更可能是因为模型只在 MARIO_NOOP=30
的窗口里训过，40 以上全是分布外的。

用法: python diag_noop_drift.py [关卡]   默认 2-2
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ["MARIO_NOOP_EXACT"] = "1"          # 必须在 import make_env 之前设
from make_env import make_env

STAGE = sys.argv[1] if len(sys.argv) > 1 else "2-2"

print(f"=== {STAGE}：空按 k 帧之后的落点 ===")
print(f"{'空按帧数':>8} {'x':>6} {'y':>6} {'剩余时间':>8}")
for k in (0, 5, 10, 20, 30, 60, 120, 200):
    env = make_env(stages=[STAGE], noop=k)
    env.reset()
    _, _, _, _, info = env.step(0)
    print(f"{k:8d} {info.get('x_pos'):6d} {info.get('y_pos'):6d} {info.get('time'):8d}", flush=True)
    env.close()
print(">>> y 随 k 单调下降＝在下沉（水下关）；y 和 time 都稳定之后，"
      "再加 k 就是纯相位变化了")
