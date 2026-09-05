"""迷宫关到底是不是「x 会倒退」：直接量 x 轨迹里有没有回卷。

笔记里一直写着「4-4/7-4/8-4 是迷宫关，走错会传回起点，x 不单调 → 梯子塑形失效」，
但这条从没量过，是从游戏常识推的。x 塑形是这个项目全部老师的奖励基础，
如果推错了，会为三关白造一套 (x,y) 闸门 / RND；如果推对了，那三关不改必然训不出来。
所以在动手之前先量：**跑若干局，看 x 有没有从高位掉回低位**。

判据：一局里若出现 x 从 >1500 掉到 <400，那就是被传回起点了（正常死亡会 reset 整局，
不会在同一局内回卷）。同时报 x 的最大回撤，回撤小＝其实是普通单向关。

⚠️ 用采样不用 argmax：argmax 策略在陌生关上往往几百步就卡死不动，
根本走不到分岔口，量不到回卷。

**⚠️ 这个诊断有前置条件，第一次跑就撞上了：喂它一个打不动那关的模型，结论无效。**
2026-09-04 用 W1 底模跑 4-4/7-4/8-4，回卷判定 0/10 —— 但最远 x 中位只有 135/311/167，
它根本没走到分岔口，「没测到回卷」是它太菜的假象，不是 x 单调的证据。
同批带的对照 4-2（最远 2064、回撤 2、0/10）说明**仪器本身是好的**，问题在被测对象。
⇒ 要等有了能走到分岔口的模型再来量。带一个已知单向的关当对照是必须的，
否则「全 0」既可能是仪器坏了也可能是真单调，分不开。

用法: python diag_maze.py <模型> <关卡逗号分隔> [每关局数]
      务必在关卡表里带一个已知单向的关（如 4-2）当对照
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_w4.zip"
STAGES = (sys.argv[2] if len(sys.argv) > 2 else "4-4,7-4,8-4").split(",")
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MAXSTEP = int(os.environ.get("MARIO_MAXSTEP", "2000"))


def run(job):
    stage, seed = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    np.random.seed(seed)
    model = PPO.load(MODEL, device="cpu")
    env = make_env(stages=[stage], noop=30)
    o, _ = env.reset()
    xs = []
    for _ in range(MAXSTEP):
        ot, _ = model.policy.obs_to_tensor(o)
        with th.no_grad():
            p = model.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
        o, r, term, trunc, info = env.step(int(np.random.choice(len(p), p=p / p.sum())))
        xs.append(int(info.get("x_pos", 0)))
        if term or trunc:
            break
    env.close()
    xs = np.array(xs)
    run_max = np.maximum.accumulate(xs)
    drawdown = int((run_max - xs).max()) if len(xs) else 0
    # 回卷 = 曾经到过 >1500，之后又跌回 <400
    warped = bool(len(xs) and (xs[np.argmax(run_max > 1500):] < 400).any() and (run_max > 1500).any())
    return stage, int(xs.max() if len(xs) else 0), drawdown, warped


def main():
    jobs = [(st, k) for st in STAGES for k in range(N)]
    print(f"=== x 是否单调（采样策略，每关 {N} 局，上限 {MAXSTEP} 步）| 模型 {MODEL} ===", flush=True)
    agg = {st: [] for st in STAGES}
    with ProcessPoolExecutor(max_workers=min(len(jobs), 32)) as pool:
        for stage, mx, dd, w in pool.map(run, jobs, chunksize=1):
            agg[stage].append((mx, dd, w))
    print(f"\n{'关卡':6s} {'最远x中位':>9s} {'最大回撤中位':>12s} {'回撤峰值':>8s} {'判定为回卷的局数':>14s}")
    for st in STAGES:
        v = agg[st]
        mxs = [a for a, _, _ in v]; dds = [b for _, b, _ in v]; ws = sum(c for _, _, c in v)
        print(f"{st:6s} {int(np.median(mxs)):9d} {int(np.median(dds)):12d} "
              f"{max(dds):8d} {ws:10d}/{len(v)}")
    print("\n>>> 读法：回撤只有几十 x＝掉坑/被打退，属正常，梯子塑形照用；"
          "回撤上千且有局判为回卷＝真被传回起点，x 塑形会奖励它在环里绕圈", flush=True)


if __name__ == "__main__":
    main()
