"""直接把迷宫岔路口的正确走法**搜出来**，而不是再设计一种奖励去指望策略自己撞上。

前四种办法（max-x 势能 / 回卷即终局 / 回合内 (x,y) novelty / 跨回合访问计数）
本质上都是同一件事：改奖励，然后希望策略在探索时碰巧走对。它们全败，
而画面告诉我们原因——那一段上下两条走廊 x 区间相同，正确路线在下层，
**任何以 x 为主的信号对两条路都是对称的**，给不出方向。

所以换个思路：先把"正确路线长什么样"当成一个**未知事实去测量**，而不是当成
策略应该自己学会的东西。做法：拿一个能稳定走到岔路口的档，在 x≥XSWITCH 之后
改成随机动作乱走，看哪些局能越过回卷点。成功局的 (x,y) 轨迹就是答案。

拿到答案之后能干两件事，都不是猜：
  ① 在正确位置放一个 (x,y) 定点奖励——这跟 2-2 当年手工放 checkpoints=[(2100,60)]
     是同一种做法，本项目的既有实践，不是作弊；
  ② 或者确认"随机走几千次也过不去"，那就说明它需要一段特定操作序列，
     属于探索问题而非奖励问题，再去考虑 Go-Explore 那条路值不值得修。

⚠️ x_pos 的 65535 脏读必须先挡掉，否则一次脏读就伪造一个"突破"（已中过一次）。
   自检：真突破必然在 XSWITCH..XGOAL 之间留下连续的格子；只在起点附近有格子就是假的。

⚠️ 判据要先看**有没有成功局**。一次都没成功时，"成功局的 y 分布"是空的，
任何从空集读出来的结论都是假的（这一程已经吃过一次空过的自检）。

用法: MARIO_XSWITCH=1800 python diag_fork_search.py <模型> [关卡] [尝试局数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, collections
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_mario_44exp/mario_44exp_7999488_steps.zip"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "4-4"
TRIES = int(sys.argv[3]) if len(sys.argv) > 3 else 400
XSWITCH = int(os.environ.get("MARIO_XSWITCH", "1800"))   # 过了这个 x 就交给随机
XGOAL = int(os.environ.get("MARIO_XGOAL", "2200"))       # 越过它就算突破了回卷点
WORKERS = int(os.environ.get("MARIO_WORKERS", "40"))
MAXSTEP = int(os.environ.get("MARIO_MAXSTEP", "1200"))


def run(seed):
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    rng = np.random.default_rng(seed)
    model = PPO.load(MODEL, device="cpu")
    env = make_env(stages=[STAGE], noop=int(seed % 31), exact=True)
    o, _ = env.reset()
    path, best, warped = [], 0, False
    for _ in range(MAXSTEP):
        ot, _ = model.policy.obs_to_tensor(o)
        x = path[-1][0] if path else 0
        if x < XSWITCH:
            with th.no_grad():                       # 前半程用策略，稳定走到岔路口
                a = int(model.policy.get_distribution(ot).distribution.probs.argmax().cpu())
        else:
            a = int(rng.integers(0, env.action_space.n))   # 岔路口之后纯随机
        o, r, term, trunc, info = env.step(a)
        nx, ny = int(info.get("x_pos", 0)), int(info.get("y_pos", 0))
        # ⚠️ x_pos 会偶发读出 65535（16 位下溢）。不挡住它，一次脏读就把 best 顶过 XGOAL，
        # 直接伪造出一个"突破"。第一版就中了：XSWITCH=900 报 46/400 突破，
        # 但成功局走过的格子只到 x≈1000、后面一片空白——真突破到 2200 必然一路走过 1100..2200。
        # 同一个脏读在 MaxXReward 里已经用 max_gain 挡过一次，这里漏了。
        if path and nx - path[-1][0] > 100:
            continue                              # 脏读：丢掉这一步，不进 path 也不更新 best
        if path and path[-1][0] - nx >= 300:
            warped = True; break
        path.append((nx, ny))
        best = max(best, nx)
        if term or trunc or best > XGOAL:
            break
    env.close()
    tail = [p for p in path if p[0] >= XSWITCH - 100]
    return best, warped, tail if best > XGOAL else []


def main():
    print(f"=== 岔路口搜索 | {STAGE} | {MODEL}", flush=True)
    print(f"    x<{XSWITCH} 用策略，之后纯随机；越过 x={XGOAL} 算突破 | {TRIES} 局 ===", flush=True)
    wins, warps, bests = [], 0, []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for best, warped, tail in pool.map(run, range(TRIES), chunksize=1):
            bests.append(best); warps += warped
            if tail:
                wins.append(tail)

    print(f"\n最远 x：中位 {int(np.median(bests))} / 最大 {max(bests)}；"
          f"回卷 {warps}/{TRIES} 局；**突破 {len(wins)}/{TRIES} 局**")
    if not wins:
        print("\n>>> 一次都没突破。⇒ 正确岔路不是'随机走就能碰到'的，需要一段特定操作序列，"
              "这是**探索问题**不是奖励问题。再加奖励项没有用，该考虑的是 Go-Explore "
              "或者把正确路线当成课程直接教。")
        return
    ys = collections.Counter()
    for tail in wins:
        for x, y in tail:
            if XSWITCH <= x <= XGOAL:
                ys[(x // 100 * 100, y // 32 * 32)] += 1
    print("\n>>> 突破局在岔路段走过的 (x,y) 格子（x 按 100、y 按 32 归档）：")
    for (x, y), c in sorted(ys.items()):
        print(f"     x≈{x:5d}  y≈{y:4d}   {c} 次")
    print("\n>>> 这些格子就是正确走廊。把它们做成 (x,y) 定点奖励喂进训练，"
          "跟 2-2 当年手工放 checkpoint 是同一种做法。", flush=True)


if __name__ == "__main__":
    main()
