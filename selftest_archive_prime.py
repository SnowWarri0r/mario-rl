"""存档开局 + prime 到底救没救回来：拿老师在同一批存档点上，比 prime 前后的通关率。

背景：Go-Explore 那条路上次是死在这儿的。存档开局的**位置**是真的（靠重放动作前缀走到的），
坏的是**栈**——reset 把一帧复制四份，速度信息为零，实测老师从 86% 掉到 argmax 21%、32 帧就死。
水关尤其致命：策略靠 4 帧差分判断鱼往哪游、多快，看到"静止的鱼"就直接撞上去。

所以在把 Go-Explore 用到迷宫关之前，先在**当初出问题的那一关（2-2）**上验证修复：
同一个老师、同一批存档点，只切 prime 开关。
判据：不 prime 应当明显低（复现旧的失败），prime 应当接近它在正常开局下的水平。
⚠️ 只测 prime 一侧会得到一个"看起来还行"的数字却不知道它是不是本来就行——必须两侧都测。

用法: ./venv/bin/python selftest_archive_prime.py [模型] [局数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODEL = os.environ.get("MARIO_MODEL", "mario_22robust.zip")
STAGE = os.environ.get("MARIO_STAGE", "2-2")
ARCHIVE = os.environ.get("MARIO_ARCHIVE", "states22_prefixes.npz")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
WORKERS = int(os.environ.get("MARIO_WORKERS", "24"))


def run(job):
    mode, seed = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import (MarioBase, NoopReset, ArchiveStart, SkipFrame,
                          GrayResize, FrameStack, SKIP_FRAMES)
    import wide_cnn  # noqa: F401

    model = PPO.load(MODEL, device="cpu")
    e = MarioBase(stages=[STAGE])
    # ⚠️ 存档开局时抖动放在哪儿是有讲究的：NoopReset 在 ArchiveStart **下面**，
    # 意味着每次恢复快照之后还要空按 0-30 帧——那是在关卡中段、水里、鱼群中间站着不动半秒。
    # 开局 x≈40 抖动无害，存档点抖动是送死。archive_noop0 这一档就是拿掉它做对照。
    nj = 0 if mode == "archive_noop0" else 30
    if nj:
        e = NoopReset(e, max_noop=nj, seed=seed)
    if mode != "normal":
        # allow_pickle：前缀是本项目自己录的动作序列（长度不齐所以是 object 数组），
        # 不是外部输入，跟 make_env 里既有的读法一致。
        pre = list(np.load(ARCHIVE, allow_pickle=True)["prefixes"])
        e = ArchiveStart(e, pre, seed=seed)
    e = SkipFrame(e, k=SKIP_FRAMES)
    e = GrayResize(e, 84)
    e = FrameStack(e, prime_on_reset=(mode == "prime"))

    w0, s0 = (int(x) for x in STAGE.split("-"))
    np.random.seed(seed)
    o, _ = e.reset()
    cleared, steps = False, 0
    for t in range(3000):
        ot, _ = model.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(model.policy.get_distribution(ot).distribution.probs.argmax().cpu())
        o, r, term, trunc, info = e.step(a)
        steps = t + 1
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (w0, s0):
            cleared = True; break
        if term or trunc:
            break
    e.close()
    return mode, cleared, steps


def main():
    if not os.path.exists(ARCHIVE):
        raise SystemExit(f"没有存档前缀文件 {ARCHIVE}")
    modes = ["normal", "archive_raw", "prime", "archive_noop0"]
    jobs = [(m, k) for m in modes for k in range(N)]
    print(f"=== 存档开局 prime 自检 | {MODEL} | {STAGE} | 每档 {N} 局 argmax ===", flush=True)
    agg = {m: [0, 0, []] for m in modes}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for m, c, st in pool.map(run, jobs, chunksize=1):
            agg[m][0] += c; agg[m][1] += 1; agg[m][2].append(st)
    label = {"normal": "正常开局（参照）", "archive_raw": "存档开局 · 不 prime",
             "prime": "存档开局 · prime", "archive_noop0": "存档开局 · 去掉抖动"}
    for m in modes:
        c, n, steps = agg[m]
        print(f"  {label[m]:22s} {c:3d}/{n}  = {c/n*100:5.1f}%   平均存活 {np.mean(steps):.0f} 步")
    raw, pr = agg["archive_raw"][0] / N, agg["prime"][0] / N
    print(f"\n>>> prime 相对不 prime：{raw*100:.0f}% → {pr*100:.0f}%")
    print(">>> 判据：不 prime 那一档若并不低，说明这台机器上根本没复现出当初的失败，"
          "这次比较就没有意义（别拿它当 prime 有效的证据）")


if __name__ == "__main__":
    main()
