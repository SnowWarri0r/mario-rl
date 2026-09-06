"""把岔路口前后的画面拼成一张 PNG，直接用眼睛看它在那儿干了什么。

到这一步是因为**猜不动了**：4-4 第二个岔路口已经吃掉三次机制推断——
max-x 势能（治住刷圈但没过口）、回卷即终局（过了第一个口，第二个不动）、
(x,y) 新鲜度（y 的探索范围确实从 27 拉到 84，还是 0/31）。
每次推断都被数据部分证实、却都不够，说明我对"那儿到底是什么"的想象是错的。
录像比再猜一轮便宜。

拼图而不是存 GIF：GIF 得挨帧点开，拼成网格能一眼看完整段，
也方便直接贴进对话里看。

用法: MARIO_DET=1 python diag_fork_frames.py <模型> [关卡] [起始相位]
      MARIO_XMIN=1800 只从这个 x 之后开始录（跳过前面顺利的部分）
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, collections
import numpy as np
import cv2

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_w4.zip"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "4-4"
PHASE = int(sys.argv[3]) if len(sys.argv) > 3 else 10
XMIN = int(os.environ.get("MARIO_XMIN", "1800"))
COLS = int(os.environ.get("MARIO_COLS", "6"))
EVERY = int(os.environ.get("MARIO_EVERY", "3"))     # 每 N 步取一帧
MAXF = int(os.environ.get("MARIO_MAXF", "24"))
DET = os.environ.get("MARIO_DET") == "1"


def main():
    import torch as th
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    model = PPO.load(MODEL, device="cpu")
    env = make_env(stages=[STAGE], noop=PHASE, exact=True)
    raw = env
    while hasattr(raw, "env"):
        raw = raw.env

    o, _ = env.reset()
    frames, labels = collections.deque(maxlen=MAXF), collections.deque(maxlen=MAXF)
    prevx, k = None, 0
    for t in range(4000):
        ot, _ = model.policy.obs_to_tensor(o)
        with th.no_grad():
            d = model.policy.get_distribution(ot).distribution
            a = int(d.probs.argmax().cpu()) if DET else int(d.sample().cpu()[0])
        o, r, term, trunc, info = env.step(a)
        x, y = int(info.get("x_pos", 0)), int(info.get("y_pos", 0))
        if x >= XMIN and t % EVERY == 0:
            frames.append(np.array(raw.render()))
            labels.append(f"t{t} x{x} y{y}")
        # 回卷即停：这一刻之前的画面才是我们要看的
        if prevx is not None and prevx - x >= 300 and prevx > XMIN:
            print(f">>> 在 t={t} 回卷：x {prevx} → {x}")
            break
        prevx = x
        if term or trunc:
            print(f">>> 在 t={t} 结束（死亡/超时），x={x}")
            break
    env.close()

    if not frames:
        raise SystemExit(f"没录到帧：这一局最远只到 x={prevx}，没到 XMIN={XMIN}")
    h, w = frames[0].shape[:2]
    rows = (len(frames) + COLS - 1) // COLS
    grid = np.zeros((rows * h, COLS * w, 3), np.uint8)
    for i, (f, lab) in enumerate(zip(frames, labels)):
        f = f.copy()
        cv2.putText(f, lab, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        r_, c_ = divmod(i, COLS)
        grid[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = f
    out = f"fork_{STAGE}_p{PHASE}.png"
    cv2.imwrite(out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f">>> {len(frames)} 帧 → {out}（{rows}×{COLS}，每 {EVERY} 步一帧）")


if __name__ == "__main__":
    main()
