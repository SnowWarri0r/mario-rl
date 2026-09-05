"""把「卡点」录成 GIF：只留死前那几秒，直接看它在障碍前做了什么。

`diag_progress.py` 只能告诉你卡在 x=884，说不出 884 那里是什么、它试过什么。
录整局没用（前面 800 x 都是顺利的），所以只保留死前 TAIL 帧 —— 看的就是
「面对那个障碍，它选择了什么动作」。

顺带打印死前的 x/y 轨迹：y 在不在动能区分两种病——
y 一直贴地＝**它根本没起跳**（策略里没有这个动作）；y 冲高又掉下去＝跳了但够不着/跳早了。
这两种病处方不同，光看 GIF 容易看错。

⚠️ nes-py 的屏幕是原地覆盖的内存，`raw.render()` 每次返回同一个数组对象，
必须 `np.array(...)` 拷贝，否则 GIF 全是最后一帧（这个项目最早踩过）。

用法: MARIO_DET=1 python diag_wall_gif.py <模型> <关卡> [空按帧数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, collections
import numpy as np
import imageio

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_w4.zip"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "4-3"
NOOP = int(sys.argv[3]) if len(sys.argv) > 3 else 0
DET = os.environ.get("MARIO_DET") == "1"
TAIL = int(os.environ.get("MARIO_TAIL", "90"))
MAXSTEP = int(os.environ.get("MARIO_MAXSTEP", "3000"))


def main():
    import torch as th
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

    os.environ["MARIO_NOOP_EXACT"] = "1"
    env = make_env(stages=[STAGE], noop=NOOP)
    # 找到底层的 nes-py env 拿原始彩色帧：包装链上每层都有 .env
    raw = env
    while hasattr(raw, "env"):
        raw = raw.env
    model = PPO.load(MODEL, device="cpu")

    o, _ = env.reset()
    frames, trace = collections.deque(maxlen=TAIL), collections.deque(maxlen=TAIL)
    w0, s0 = (int(x) for x in STAGE.split("-"))
    for t in range(MAXSTEP):
        ot, _ = model.policy.obs_to_tensor(o)
        with th.no_grad():
            d = model.policy.get_distribution(ot).distribution
            a = int(d.probs.argmax().cpu()) if DET else int(d.sample().cpu()[0])
        o, r, term, trunc, info = env.step(a)
        frames.append(np.array(raw.render()))        # ⚠️必须拷贝
        trace.append((t, info.get("x_pos"), info.get("y_pos"), SIMPLE_MOVEMENT[a]))
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (w0, s0):
            print("这一局通关了，换个相位再录"); break
        if term or trunc:
            break
    env.close()

    out = f"wall_{STAGE}_noop{NOOP}.gif"
    imageio.mimsave(out, list(frames), fps=20)
    print(f">>> 死前 {len(frames)} 帧 → {out}（止于 x={trace[-1][1]} y={trace[-1][2]}）")
    print("\n死前 30 步的 x / y / 动作：")
    for t, x, y, act in list(trace)[-30:]:
        print(f"  步{t:5d}  x={x:5}  y={y:4}  {'+'.join(act)}")
    ys = [y for _, _, y, _ in trace if y is not None]
    print(f"\n>>> 这段里 y 的范围 {min(ys)}-{max(ys)}："
          f"{'基本没离地，它没在这儿起跳' if max(ys) - min(ys) < 20 else '跳过，是跳的时机/距离不对'}")


if __name__ == "__main__":
    main()
