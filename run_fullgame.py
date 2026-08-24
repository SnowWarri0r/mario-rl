"""跑完整游戏：从 1-1 开始一路往下接，直到三条命打完（is_game_over）或步数上限。

逐关评测每关都是从头开始（小马里奥、满时间），真实连打时进入下一关的状态不一样
（可能是大马里奥/火力、剩余时间不同、命数在减少），所以"逐关都能过"不等于"连起来能过"。
这个脚本测的是后者。

三种打法：
  det    全程 argmax。确定性策略在确定性模拟器里每次轨迹相同——一旦死一次，重生后会死在同一个地方，
         三条命瞬间打完。所以它只适合"从不出错"的关。
  sto    全程采样。抗错但每关都有翻车概率。
  hybrid 默认 argmax；某条命死掉后，本次重试改成采样（换条路），过关后切回 argmax。
  auto   按 DET_STAGES 表逐关选模式（实测哪种好用哪种），死了同样退回采样重试。
         DET/STO 的优劣是逐关分裂的，全局二选一会在一半关卡上吃亏。

用法: python run_fullgame.py <model.zip> [det|sto|hybrid] [并行局数] [步数上限]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_all12_wide.zip"
MODE = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 24
MAX_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 30000

# mode=auto 用的逐关推理模式表：DET/STO 的优劣是逐关分裂的（实测 mario_honest12 @noop=30：
# 1-4 DET 100% vs STO 55%，而 1-2 DET 9% vs STO 55%）。哪种好按实测挑，别全局二选一。
DET_STAGES = {"1-4", "2-1", "2-2", "2-4", "3-1", "3-2", "3-3", "3-4"}


def play(run_id):
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    model = PPO.load(MODEL, device="cpu")
    env = make_env()                                    # stages=None → 完整游戏，过关自动接下一关
    o, _ = env.reset()
    cleared, deaths, log = [], 0, []
    # visited[关卡] = [进过几次, 清掉几次, 死几次]。"进了 1-2 之后清掉它的概率"才是连打里的真实成功率，
    # 逐关从头评测的 86% 是另一个口径（每次都是满血新开局）。
    visited = {"1-1": [1, 0, 0]}
    # 按"这次尝试开始时马里奥的形态"分桶：by_status[(关卡, 形态)] = [尝试次数, 清掉次数]。
    # 连打到 2-2 时他可能是大只/火力状态，而 2-2 专家是从小马里奥开局训的——大马里奥两格高，
    # 2-2 那道一格宽的鱼缝物理上钻不过去。形态是不是真凶，靠这个分桶判。
    by_status = {}
    cur = ("1-1", "small")                                           # 当前这次尝试：(关卡, 进关形态)
    by_status[cur] = [1, 0]
    def want_sto(w, s):
        if MODE == "auto":
            return f"{w}-{s}" not in DET_STAGES      # 表里的关用 argmax，其余采样
        return MODE == "sto"

    ws, life, sto_now = (1, 1), None, want_sto(1, 1)
    steps, prev_x, prev_time = 0, 0, None
    while steps < MAX_STEPS:
        a, _ = model.predict(o, deterministic=not sto_now)
        o, r, term, trunc, info = env.step(int(a))
        steps += 1
        w, s = info.get("world", ws[0]), info.get("stage", ws[1])
        lf = info.get("life")
        if life is not None and lf is not None and lf < life:        # 掉了一条命
            deaths += 1
            # 掉命这一步 info 已经是复活后的状态(x=出生点 40、time=400)，真实死亡位置要看上一步
            cause = "超时" if prev_time is not None and prev_time <= 1 else "死"
            log.append(f"{cause}在 {ws[0]}-{ws[1]} x={prev_x}（剩 {lf} 命）")
            visited.setdefault(f"{ws[0]}-{ws[1]}", [1, 0, 0])[2] += 1
            cur = (f"{ws[0]}-{ws[1]}", "small")                      # 复活一律小马里奥，算新的一次尝试
            by_status.setdefault(cur, [0, 0])[0] += 1
            if MODE in ("hybrid", "auto"):
                sto_now = True                                       # 这条命改采样，别再撞同一面墙
        life = lf
        if (w, s) != ws:                                             # 过关，接下一关
            cleared.append(f"{ws[0]}-{ws[1]}")
            log.append(f"过关 {ws[0]}-{ws[1]} → {w}-{s}（形态 {info.get('status')}）")
            visited.setdefault(f"{ws[0]}-{ws[1]}", [1, 0, 0])[1] += 1
            visited.setdefault(f"{w}-{s}", [0, 0, 0])[0] += 1
            by_status.setdefault(cur, [1, 0])[1] += 1                # 上一关这次尝试成功
            cur = (f"{w}-{s}", info.get("status", "small"))          # 新关卡的这次尝试
            by_status.setdefault(cur, [0, 0])[0] += 1
            ws = (w, s)
            if MODE in ("hybrid", "auto"):
                sto_now = want_sto(w, s)                             # 新关卡按表选模式
        prev_x, prev_time = info.get("x_pos", prev_x), info.get("time")
        if term or trunc:                                            # is_game_over
            break
    env.close()
    return run_id, cleared, deaths, steps, (ws, info.get("x_pos")), log, visited, by_status


def main():
    n = 1 if MODE == "det" else RUNS                    # det 每次一样，跑一局就够
    print(f"=== 完整游戏 {MODEL} · 打法 {MODE} · {n} 局 ===", flush=True)
    with ProcessPoolExecutor(max_workers=min(n, 48)) as pool:
        results = list(pool.map(play, range(n)))
    counts = [len(c) for _, c, _, _, _, _, _, _ in results]
    best = max(results, key=lambda r: len(r[1]))
    print(f"\n通关关数：平均 {np.mean(counts):.1f} / 最好 {max(counts)} / 最差 {min(counts)}", flush=True)

    agg = {}
    for r in results:
        for st, (ent, clr, dth) in r[6].items():
            a = agg.setdefault(st, [0, 0, 0])
            a[0] += ent; a[1] += clr; a[2] += dth
    print("\n=== 连打中的逐关真实成功率（进过 N 次 / 清掉 M 次）===")
    for st in sorted(agg):
        ent, clr, dth = agg[st]
        rate = f"{clr/ent*100:3.0f}%" if ent else "  -"
        print(f"  {st}: 进 {ent:3d} 次  清 {clr:3d} 次 = {rate}   死 {dth:3d} 次")

    st_agg = {}
    for r in results:
        for k, (att, clr) in r[7].items():
            a = st_agg.setdefault(k, [0, 0]); a[0] += att; a[1] += clr
    print("\n=== 按进关形态拆开（小马里奥 vs 大只/火力）===")
    for (stage, status) in sorted(st_agg):
        att, clr = st_agg[(stage, status)]
        if att >= 3:
            print(f"  {stage} 以 {status:8s} 开始：尝试 {att:3d} 次，清掉 {clr:3d} 次 = {clr/att*100:3.0f}%")
    print()
    for _, c, d, st, (endws, endx), _, _, _ in sorted(results, key=lambda r: -len(r[1]))[:5]:
        print(f"  过了 {len(c):2d} 关 {'→'.join(c) if c else '(一关没过)'} | 死 {d} 次 | "
              f"终止在 {endws[0]}-{endws[1]} x={endx}")
    print(f"\n=== 最好那局的全过程 ===")
    for line in best[5]:
        print("  " + line)


if __name__ == "__main__":
    main()
