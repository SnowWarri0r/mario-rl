"""一关卡在哪儿：报 max x 的分布，而不只是通关率。

**为什么值得单独写。** 扩关阶段最常见的读数是「0%」，而 0% 至少盖着三种完全不同的病：
① 开局就死（不会玩这关的基本动作）② 卡在某个固定 x（有一个过不去的障碍）
③ 到 x=95% 才死（就差一口气，加步数或多训一会儿就行）。三者的处方完全不同——
②要改奖励或加课程，③只要接着训。只看通关率区分不了，会把③误判成①去大改配方。

顺带报「卡点」：如果 max_x 的分布有一个尖峰，那个 x 就是障碍的位置，
可以直接去游戏里看那儿是什么（4-2 的隐藏管道、4-3 的移动平台之类）。

⚠️ 报的是 max x 不是终局 x：马里奥掉坑时 x 会回退，终局 x 会低估它到过多远。

模型位可以给多个（逗号分隔）或一个目录（扫该目录下所有 checkpoint）。扩关时这是主力用法：
**取最终档是错的**——实测 W4 最终档把 4-2 从 0/31 拉到 23/31，同时把 4-1 从 21/31 摔到 9/31。
混训里各关的峰值不在同一步，只能密存档 + 按实测通关率逐关挑。

⚠️ 别拿 `eval_noop_audit.py` 代替这个：那边的 noop 是 0..30 **随机抽**，31 局里相位重复，
有效样本量小于 31（同一模型 4-1 它报 45%、这里逐相位枚举报 68%）。argmax + 确定性模拟器下
相位才是样本，见 [[feedback_rl_eval_effective_sample_size]]。

用法: MARIO_DET=1 python diag_progress.py <模型逗号分隔|目录> <关卡逗号分隔> [每关局数] [并发]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

import glob, re as _re
_spec = sys.argv[1] if len(sys.argv) > 1 else "mario_w4.zip"
if os.path.isdir(_spec):
    MODELS = sorted(glob.glob(f"{_spec}/*.zip"),
                    key=lambda p: int(_re.search(r"(\d+)_steps", p).group(1)))
else:
    MODELS = [x for x in _spec.split(",") if x]
STAGES = (sys.argv[2] if len(sys.argv) > 2 else "4-1,4-2,4-3").split(",")
N = int(sys.argv[3]) if len(sys.argv) > 3 else 31
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 40
NOOP = int(os.environ.get("MARIO_NOOP", "30"))
DET = os.environ.get("MARIO_DET") == "1"
MAXSTEP = int(os.environ.get("MARIO_MAXSTEP", "3000"))
# MARIO_ONE_LIFE=1 → 第一次丢命就结束这一局。
# 为什么要有这个口径：默认的 done 是 game over，等于**默许三次尝试**。
# 连打里走到中后段时命常常已经在前面耗光，只剩一次机会，两边口径根本不同。
# 实测 v32 的 2-4 单关 87%、连打只有 45%：若三次 87%，单次约 1-(1-p)^3=0.87 → p≈49%，
# 跟 45% 几乎吻合 —— 所以"单关分"可能系统性高估了连打能力。
ONE_LIFE = os.environ.get("MARIO_ONE_LIFE") == "1"


def run(job):
    model_path, stage, seed = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    model = PPO.load(model_path, device="cpu")
    # 相位就是有效样本量：argmax + 确定性模拟器下，同一相位每次都是同一条轨迹。
    # 所以一个 seed 对应一个**确切**的空按帧数，而不是随机抽 0..seed。
    # ⚠️ 走 exact= 参数，不能设 MARIO_NOOP_EXACT 环境变量：那个在 make_env import 时就读死了，
    # 这里 import 在后，设了等于没设（第一版就是这么写的，扫出来的表整张作废）。
    env = make_env(stages=[stage], noop=seed, exact=True)
    w0, s0 = (int(x) for x in stage.split("-"))
    o, _ = env.reset()
    mx, cleared, life0 = 0, False, None
    for _ in range(MAXSTEP):
        ot, _ = model.policy.obs_to_tensor(o)
        with th.no_grad():
            d = model.policy.get_distribution(ot).distribution
            a = int(d.probs.argmax().cpu()) if DET else int(d.sample().cpu()[0])
        o, r, term, trunc, info = env.step(a)
        mx = max(mx, int(info.get("x_pos", 0)))
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (w0, s0):
            cleared = True; break
        life = info.get("life")
        if ONE_LIFE and life is not None and life0 is not None and life < life0:
            break                       # 丢了第一条命就收场
        if life0 is None:
            life0 = life
        if term or trunc:
            break
    env.close()
    return model_path, stage, mx, cleared


def main():
    import wide_cnn  # noqa: F401
    jobs = [(m, st, k) for m in MODELS for st in STAGES for k in range(N)]
    print(f"=== 推进深度 | {len(MODELS)} 个档 × {len(STAGES)} 关 | "
          f"{'argmax' if DET else '采样'} | 每格 {N} 个确切相位 | 上限 {MAXSTEP} 步"
          f"{' | 单命口径' if ONE_LIFE else ' | 三命(game over)口径'} ===", flush=True)
    res = {(m, st): [] for m in MODELS for st in STAGES}
    clr = {(m, st): 0 for m in MODELS for st in STAGES}
    with ProcessPoolExecutor(max_workers=min(len(jobs), WORKERS)) as pool:
        for m, stage, mx, c in pool.map(run, jobs, chunksize=1):
            res[(m, stage)].append(mx); clr[(m, stage)] += c

    name = lambda m: os.path.basename(m).replace(".zip", "")[-26:]
    if len(MODELS) > 1:
        print(f"\n{'档':28s}" + "".join(f"{st:>10s}" for st in STAGES) + f"{'合计':>8s}")
        for m in MODELS:
            tot = sum(clr[(m, st)] for st in STAGES)
            print(f"{name(m):28s}" + "".join(f"{clr[(m,st)]:>7d}/{N:<3d}" for st in STAGES)
                  + f"{tot:>6d}/{N*len(STAGES)}")
        print("\n>>> 逐关取最优档（混训里各关峰值不同步，这才是要拿去当老师的东西）")
        for st in STAGES:
            best = max(MODELS, key=lambda m: clr[(m, st)])
            print(f"  {st}  {clr[(best,st)]}/{N} = {clr[(best,st)]/N*100:.0f}%   {name(best)}")
        return

    print(f"\n{'关卡':6s} {'通关':>7s} {'max x 中位':>10s} {'最远':>6s} {'最近':>6s}  卡点分布（每 200 x 一档）")
    for st in STAGES:
        m = MODELS[0]
        v = np.array(res[(m, st)])
        lo, hi = v.min(), v.max()
        bins = np.arange(0, max(hi, 1) + 200, 200)
        h, _ = np.histogram(v, bins=bins)
        spark = " ".join(f"{int(b)}:{c}" for b, c in zip(bins[:-1], h) if c)
        print(f"{st:6s} {clr[(m,st)]:3d}/{N:<3d} {int(np.median(v)):10d} {int(hi):6d} {int(lo):6d}  {spark}")
    print("\n>>> 读法：中位数接近最远＝稳定地卡在同一处（去看那个 x 是什么障碍）；"
          "中位数很低而最远很高＝多数局早死，是基本操作不会；"
          "卡点只有一档且靠近终点＝就差一口气，接着训就行", flush=True)


if __name__ == "__main__":
    main()
