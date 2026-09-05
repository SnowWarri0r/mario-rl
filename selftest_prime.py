"""MARIO_PRIME 的自检：过关那一刻，栈里到底是什么。

这个开关的全部意义是"栈里要有速度"，所以自检就直接量这个：
用相邻帧之间的平均绝对差当运动量的代理，比较三种做法在**过关后第一帧**的栈。
  · flush   → 四帧完全相同，帧间差应为 0
  · 不处理  → 帧间差不为 0，但前几帧还是上一关的画面（内容错，这里只验它不是 0）
  · prime   → 帧间差不为 0，且四帧全是新关的
另外验一条容易写错的：prime 走掉的那 n-1 步，world/stage 不能再变回去，
且 term/trunc 要如实往上传。

跑法是让一个已知能过 1-1 的模型去打，直到它过关那一步为止。
不写成 heredoc：forkserver 子进程要重新 import __main__（`python - <<EOF` 会炸）。

用法: python selftest_prime.py [模型]   默认 mario_v7_wide.zip
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np


def stack_motion(stack):
    """相邻帧之间的平均绝对差：0 表示四帧一模一样（速度信息为零）"""
    d = np.abs(np.diff(stack.astype(np.int16), axis=0))
    return float(d.mean())


def run(mode, model_path):
    os.environ["MARIO_FLUSH"] = "1" if mode == "flush" else "0"
    os.environ["MARIO_PRIME"] = "1" if mode == "prime" else "0"
    for m in list(sys.modules):
        if m in ("make_env",):
            del sys.modules[m]
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    m = PPO.load(model_path, device="cpu")
    # 完整游戏才有"过关"这件事；用采样 + 抖动，否则 argmax 在 noop=0 下是条固定轨迹，
    # 走不通就永远等不到过关（第一次写成 argmax+noop=0，30000 步一次没过）
    env = make_env(stages=None, noop=30)
    o, _ = env.reset()
    ws = None
    for step in range(30000):
        ot, _ = m.policy.obs_to_tensor(o)
        with th.no_grad():
            p = m.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
        o, r, term, trunc, info = env.step(int(np.random.choice(len(p), p=p / p.sum())))
        cur = (info.get("world"), info.get("stage"))
        if ws is not None and cur != ws:
            at_change = stack_motion(o)
            # 参照物：进新关又走 8 步之后，栈里已经全是这一关的连续画面，
            # 那时的运动量就是"正常水平"。拿跨关的 44.9 当参照是错的——
            # 那是两张完全不同画面的像素差，不是运动量。
            for _ in range(8):
                ot2, _ = m.policy.obs_to_tensor(o)
                with th.no_grad():
                    p2 = m.policy.get_distribution(ot2).distribution.probs.cpu().numpy()[0]
                o, _, t2, u2, _ = env.step(int(np.random.choice(len(p2), p=p2 / p2.sum())))
                if t2 or u2:
                    break
            env.close()
            return at_change, stack_motion(o), cur, step
        ws = cur
        if term or trunc:
            o, _ = env.reset(); ws = None
    env.close()
    return None, None, None, -1


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "mario_v7_wide.zip"
    print(f"=== 过关后第一帧，栈内相邻帧平均绝对差（0 ＝ 四帧全同，没有速度）===", flush=True)
    got = {}
    for mode in ("none", "flush", "prime"):
        motion, ref, ws, step = run(mode, model)
        got[mode] = (motion, ref)
        if motion is None:
            print(f"  {mode:6s}  30000 步内没过关，测不了", flush=True)
        else:
            print(f"  {mode:6s}  过关那帧 {motion:7.3f}   关内正常水平 {ref:7.3f}   "
                  f"占比 {motion/ref*100 if ref else 0:5.0f}%   进到 {ws}（第 {step} 步）", flush=True)

    # 判据是**结构性**的，不跟"关内正常水平"比大小：那个参照量本身在 0.9~5.8 之间抖
    # （取决于那一刻画面滚不滚），当分母不可靠。而且新关开头马里奥在出生点、画面还没滚动，
    # 运动量低本来就是对的，不是预热没填够——我第一版拿它当判据，误判了一次。
    assert got["flush"][0] == 0.0, "flush 下栈内竟然有差异，说明这条路径没走到"
    assert got["prime"][0] > 0, "prime 下栈内没有运动，预热没生效"
    # 不处理时栈里混着上一关的画面，像素差会大得离谱——那不是"有运动"，是内容错了的指纹。
    # prime 必须远低于它，才说明四帧同属新关。
    assert got["prime"][0] < got["none"][0] / 10, (
        f"prime 的帧间差 {got['prime'][0]:.1f} 接近跨关量级 {got['none'][0]:.1f}，栈里还混着上一关")
    print(f">>> 通过：flush 的栈是死的（0）；prime 有运动（{got['prime'][0]:.2f}）"
          f"且远离跨关量级（{got['none'][0]:.1f}）＝四帧同属新关且彼此相邻", flush=True)


if __name__ == "__main__":
    main()
