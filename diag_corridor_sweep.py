"""按**目标高度**扫走廊：直接找出迷宫闸门要求的那条路，而不是让搜索去碰。

**这是对前面全部失败的纠正。** 之前四种奖励、随机探索、816 组扰动、两种束搜索，
共同点是都在**生产候选路线**——用探索去猜正确走法。但这个项目手里一直有廉价的
**验证**工具，缺的只是候选。而候选不需要搜：迷宫判定的是"通过闸门时马里奥在哪条走廊"，
也就是 y。那就直接把 y 当参数扫一遍。

控制器很土但够用：低于目标带就跳（right+A），高于就别跳让它落（right），
在带内就跑（right+B）。不需要它打得好看，只需要它**稳定停在指定高度**穿过闸门。

判据：哪个目标带能不回卷地越过 x≈2046，那条就是正确走廊。
拿到之后把这条带做成 (x,y) 路标奖励喂训练——跟 2-2 当年手放 checkpoint 同一种做法。

⚠️ x_pos 的 65535 脏读要挡（这一程用它伪造过一次"突破"）。

用法: python diag_corridor_sweep.py [关卡]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor

STAGE = sys.argv[1] if len(sys.argv) > 1 else "4-4"
# ⚠️ 只在**闸门窗口内**接管高度，窗口外一律交回策略。
# 第一版从 x=700 一路接管到底，结果九个高度带全停在 x≈1440——地形图显示 1342-1629 是岩浆段，
# 土控制器过不去，等于这一轮根本没测到 2046 那个闸门。策略能过岩浆，让它开车。
XLO = int(os.environ.get("MARIO_XLO", "1850"))
XHI = int(os.environ.get("MARIO_XHI", "2060"))
XGOAL = int(os.environ.get("MARIO_XGOAL", "2300"))
MAXSTEP = int(os.environ.get("MARIO_MAXSTEP", "1500"))
MODEL = os.environ.get("MARIO_MODEL", "checkpoints_mario_44p/mario_44p_7999488_steps.zip")
PHASES = [int(p) for p in os.environ.get("MARIO_PHASES", "3,10,17,24").split(",")]
BANDS = [int(b) for b in os.environ.get(
    "MARIO_BANDS", "40,64,88,112,136,160,184,208,232").split(",")]
HALF = int(os.environ.get("MARIO_HALF", "16"))         # 目标带半宽
WORKERS = int(os.environ.get("MARIO_WORKERS", "40"))


def run(job):
    target, phase = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    model = PPO.load(MODEL, device="cpu")
    env = make_env(stages=[STAGE], noop=phase, exact=True)
    o, _ = env.reset()
    px, py, best, warped, plife = None, None, 0, False, None
    crossed_y = None
    for _ in range(MAXSTEP):
        if px is None or not (XLO <= px <= XHI):
            ot, _ = model.policy.obs_to_tensor(o)      # 前半程交给策略，稳定走到闸门前
            with th.no_grad():
                a = int(model.policy.get_distribution(ot).distribution.probs.argmax().cpu())
        else:
            # 高度保持：低了就跳，高了就别跳，在带内就跑
            if py is None or py < target - HALF:
                a = 2          # right + A
            elif py > target + HALF:
                a = 1          # right（不跳，靠重力下落）
            else:
                a = 3          # right + B（带内全速跑）
        o, r, term, trunc, info = env.step(a)
        nx, ny = int(info.get("x_pos", 0)), int(info.get("y_pos", 0))
        if px is not None and nx - px > 100:
            continue                                    # 65535 脏读
        # ⚠️ 死亡重生 x 也会掉回关卡起点，跌幅同样 >300。不看 life 就会把"死了"记成"回卷"，
        # 前面几个脚本都有这个混淆。用命数变化区分。
        nlife = info.get("life")
        if px is not None and px - nx >= 300:
            if nlife is not None and plife is not None and nlife < plife:
                break                                   # 死亡，不是回卷
            warped = True; crossed_y = py; break
        plife = nlife
        px, py = nx, ny
        best = max(best, nx)
        if term or trunc or best > XGOAL:
            break
    env.close()
    return target, phase, best, warped, crossed_y


def main():
    jobs = [(b, p) for b in BANDS for p in PHASES]
    print(f"=== 走廊高度扫描 | {STAGE} | 闸门窗口 x∈[{XLO},{XHI}] | "
          f"目标带 {BANDS}（±{HALF}）× 相位 {PHASES} ===", flush=True)
    agg = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for target, phase, best, warped, cy in pool.map(run, jobs, chunksize=1):
            agg.setdefault(target, []).append((best, warped))
    print(f"\n{'目标y':>6s} {'最远x中位':>10s} {'最大':>6s} {'回卷':>6s} {'突破':>6s}")
    win = []
    for b in BANDS:
        v = agg.get(b, [])
        bests = sorted(x for x, _ in v)
        warps = sum(w for _, w in v)
        brk = sum(1 for x, _ in v if x > XGOAL)
        if brk:
            win.append(b)
        mid = bests[len(bests) // 2] if bests else 0
        print(f"{b:6d} {mid:10d} {max(bests) if bests else 0:6d} "
              f"{warps:4d}/{len(v):<2d} {brk:4d}/{len(v):<2d}")
    if win:
        print(f"\n>>> **正确走廊的高度带：{win}** —— 把它做成 (x,y) 路标奖励喂训练。")
    else:
        print("\n>>> 没有任何高度带能过。下一步：挪 MARIO_XLO/XHI 换个窗口再扫；"
              "若所有窗口都不行，说明判定的不是「通过闸门时的高度」。")


if __name__ == "__main__":
    main()
