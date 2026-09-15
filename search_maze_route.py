"""束搜索找出迷宫关的正确动作序列，找到后从干净开局重放验证。

**为什么这样绕。** 4-4 前后试过：四种奖励塑形、纯随机探索、672+144 组受控扰动，全部止步 x≈2046。
Go-Explore 本来是对症的，但它的地基没验通（存档开局比正常开局稳定低 20-30pp，
三个假设全证伪）。这里绕开那个问题：**找路线的时候根本不需要存档**——
每个候选都从 reset 重放一遍动作前缀（模拟器重放很快），找到完整序列后再从干净开局验证。
存档保真度的问题只在"拿存档当训练起点"时才致命，找路线用不着。

**评分踩过的坑（第一版就是这么废的）。** 第一版让前缀之后交给策略跑到底、按最终 x 打分。
结果搜了 17 轮死死卡在 2055-2077——因为**策略总会爬回上层走廊**，
前缀就算把马里奥放到下层，后面的策略也会把它带回去，分数完全看不出前缀的好坏。

这一版：① 分数只看**前缀自己**走到多远（不接策略），② **回卷=硬失败**（直接淘汰），
③ 束按**前沿的 y 分层**保留。②③ 是配套的：上下两条走廊 x 无差别，光靠 max-x 一定全挤到
策略已经在走的那条；回卷虽然能淘汰上层，但它发生在 x≈2046、一百七十多个 chunk 之后，
那时束里早就没有下层的候选可回溯了。按 y 分层就是为了把"另一条走廊"一直留在束里，
撑到回卷把答案揭晓。

⚠️ x_pos 的 65535 脏读要挡（这一程已经用它伪造过一次"突破"）。

用法: python search_maze_route.py [关卡] [轮数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

STAGE = sys.argv[1] if len(sys.argv) > 1 else "4-4"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
MODEL = os.environ.get("MARIO_MODEL", "checkpoints_mario_44p/mario_44p_7999488_steps.zip")
BEAM = int(os.environ.get("MARIO_BEAM", "8"))
FANOUT = int(os.environ.get("MARIO_FANOUT", "16"))
CHUNK = int(os.environ.get("MARIO_CHUNK", "12"))       # 每轮往前探几步
PHASE = int(os.environ.get("MARIO_PHASE", "10"))
XGOAL = int(os.environ.get("MARIO_XGOAL", "2300"))
MAXSTEP = int(os.environ.get("MARIO_MAXSTEP", "1600"))
WORKERS = int(os.environ.get("MARIO_WORKERS", "40"))
OUT = os.environ.get("MARIO_OUT", f"route_{STAGE}.json")


def evaluate(job):
    """只重放动作序列本身，不接策略。返回 (最远x, 终点y, 是否回卷/死亡, extra)."""
    prefix, extra = job
    from make_env import make_env
    env = make_env(stages=[STAGE], noop=PHASE, exact=True)
    o, _ = env.reset()
    acts = list(prefix) + list(extra)
    px, best, ylast, bad = None, 0, 0, False
    for a in acts:
        o, r, term, trunc, info = env.step(a)
        nx, ny = int(info.get("x_pos", 0)), int(info.get("y_pos", 0))
        if px is not None and nx - px > 100:
            continue                                   # 65535 脏读
        if px is not None and px - nx >= 300:
            bad = True; break                          # 回卷＝硬失败
        px = nx; best = max(best, nx); ylast = ny
        if term or trunc:
            bad = True; break                          # 死了也算失败
    env.close()
    return best, ylast, bad, extra


def main():
    from make_env import make_env
    nact = make_env(stages=[STAGE], noop=0, exact=True).action_space.n
    rng = np.random.default_rng(0)
    beam = [([], 0)]
    print(f"=== 束搜索找 {STAGE} 的通路 | beam={BEAM} fanout={FANOUT} chunk={CHUNK} "
          f"| 打分=前缀自己走到的最远x，回卷/死亡=淘汰 ===", flush=True)

    for rd in range(ROUNDS):
        jobs = []
        for prefix, _ in beam:
            for _ in range(FANOUT):
                jobs.append((prefix, [int(a) for a in rng.integers(0, nact, CHUNK)]))
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            out = list(pool.map(evaluate, jobs, chunksize=1))

        cand = []
        for (prefix, _), (best, ylast, bad, extra) in zip(jobs, out):
            if bad:
                continue                               # 回卷/死亡的候选直接丢
            cand.append((best, ylast, prefix + extra))
        if not cand:
            print("  本轮所有候选都回卷或死亡，回退上一轮的束继续试", flush=True)
            continue
        # 按 y 分层再各取最好：不这么做，束会在几轮内全被同一条走廊占满
        by_band = {}
        for sc, y, pre in cand:
            by_band.setdefault(y // 32, []).append((sc, pre))
        newbeam = []
        per = max(1, BEAM // max(1, len(by_band)))
        for band in sorted(by_band, reverse=True):
            for sc, pre in sorted(by_band[band], key=lambda t: -t[0])[:per]:
                newbeam.append((pre, sc))
        newbeam.sort(key=lambda t: -t[1])
        beam = newbeam[:BEAM]
        bands = sorted(by_band)
        print(f"     y 分层 {bands}（束里还剩 {len(by_band)} 个高度带）", flush=True)
        top = beam[0][1]
        print(f"  第 {rd+1:2d} 轮：前缀长 {len(beam[0][0]):4d}  最好分 {top}", flush=True)
        if top > XGOAL:
            print(f"\n>>> 突破！x 越过 {XGOAL}", flush=True)
            json.dump({"stage": STAGE, "phase": PHASE, "actions": beam[0][0]}, open(OUT, "w"))
            print(f">>> 动作序列已存 {OUT}（{len(beam[0][0])} 步）")
            return
    print(f"\n>>> {ROUNDS} 轮未突破，最好 {beam[0][1]}。前缀存下来供接着搜。")
    json.dump({"stage": STAGE, "phase": PHASE, "actions": beam[0][0],
               "best_x": beam[0][1]}, open(OUT, "w"))


if __name__ == "__main__":
    main()
