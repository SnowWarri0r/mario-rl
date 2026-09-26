"""连打 DAgger：学生按部署方式真打整局，每一帧问当前关的老师打标签。

**为什么不沿用 collect_dagger_29.py。** 那个是逐关、从起点跑学生（make_env(stages=[stage])），
按单关分数配比。它收的是"单关评测"的状态分布，而连打的状态分布不一样：
  - 中点复活：2-2 死了在水下半程复活，单关评测和训练数据几乎没覆盖这种起点；
  - 跨关入场相位（由上一关走了多久决定）；
  - 共用 3 条命下的第二、第三次尝试。
实测差距：2-2 单关从起点 argmax 58%，连打里每次尝试只有约 22%。
而优化"28 关平均"已证明对连打无效（SWA 单关 85.9%→90.6%，连打 5.6→5.7 关）。

这里收来的数据**天然按连打分布加权**：每局都要过 1-1..1-4，但一遍过的关只贡献一遍的帧；
卡在 2-2 反复重试的局会在 2-2 贡献大量帧；世界 5-8 几乎没人到，就几乎没有新数据。

⚠️ 新数据**叠加**在现有数据之上，不替换。collect_dagger_29 的教训：3-3 只分到 2000 帧，
   重蒸后 100%→16%。强关的覆盖靠原有数据保住。
⚠️ 重抖开关是 make_env 模块级常量，import 时读死 —— 环境变量必须在 import make_env 之前设。
⚠️ 两种开车方式交替：全程 argmax（复活点重抖防复读），和 argmax→死后本关改采样（hybrid）。
   两种部署方式实测持平（5.8 vs 5.65，N=96 内），都收，状态更多样。

用法: python collect_dagger_playthrough.py <学生.zip> <输出目录> [每 worker 帧数] [worker 数]
"""
import os
os.environ.setdefault("MARIO_REJITTER", "30")          # 必须在 import make_env 之前
os.environ.setdefault("MARIO_REJITTER_ON_DEATH", "1")
os.environ.pop("MARIO_REJITTER_STAGES", None)            # 全开：名单是按某个模型挑的，会偏
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings; warnings.filterwarnings("ignore")
import sys, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import teachers as T

STUDENT = sys.argv[1]
OUTDIR = sys.argv[2]
PER = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 100
MINSCORE = 40                                            # 不够格当老师的关不打标签（同 collect_dagger_29）


def work(wid):
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    rng = np.random.default_rng(wid)
    st = PPO.load(STUDENT, device="cpu")
    teachers = {}
    def teacher(stage):
        if stage not in teachers:
            ok = stage in T.TEACHERS and T.TEACHERS[stage][1] >= MINSCORE
            teachers[stage] = PPO.load(T.path(stage), device="cpu") if ok else None
        return teachers[stage]

    env = make_env()                                     # stages=None → 完整游戏
    buf = collections.defaultdict(lambda: ([], []))
    n, games, stages_cleared = 0, 0, []
    while n < PER:
        o, _ = env.reset()
        hybrid = (wid + games) % 2 == 1
        ws, life, sto, cleared = (1, 1), None, False, 0
        for _ in range(30000):
            stage = f"{ws[0]}-{ws[1]}"
            t = teacher(stage)
            ot, _ = st.policy.obs_to_tensor(o)
            with th.no_grad():
                d = st.policy.get_distribution(ot).distribution
                a = int(d.sample()[0]) if sto else int(d.probs.argmax())
                if t is not None:
                    tp = t.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
            if t is not None:                            # 存学生走到的状态 + 老师在那儿的意见
                b = buf[stage]; b[0].append(o.astype(np.uint8)); b[1].append(tp.astype(np.float32))
                n += 1
            o, r, term, trunc, info = env.step(a)
            w, s = info.get("world", ws[0]), info.get("stage", ws[1])
            lf = info.get("life")
            if life is not None and lf is not None and lf < life and hybrid:
                sto = True                               # hybrid：本关剩下的命改采样
            life = lf
            if (w, s) != ws:
                cleared += 1; ws = (w, s); sto = False
            if term or trunc or n >= PER:
                break
        games += 1; stages_cleared.append(cleared)
    env.close()
    os.makedirs(OUTDIR, exist_ok=True)
    counts = {}
    for stage, (ob, pr) in buf.items():
        np.savez_compressed(f"{OUTDIR}/{stage}_fg_w{wid:03d}.npz",
                            obs=np.array(ob, np.uint8), probs=np.array(pr, np.float32))
        counts[stage] = len(ob)
    return counts, stages_cleared


def main():
    print(f"=== 连打 DAgger | 学生 {STUDENT} | {WORKERS} worker × {PER} 帧 → {OUTDIR} ===", flush=True)
    tot, games = collections.Counter(), []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for counts, sc in pool.map(work, range(WORKERS)):
            tot.update(counts); games += sc
    allf = sum(tot.values())
    print(f"\n完整局 {len(games)} 局（最后一局可能因帧数到顶截断），平均过 {np.mean(games):.1f} 关")
    print(f"共 {allf:,} 帧，逐关分布（这就是连打的状态分布）：")
    for stage in sorted(tot, key=lambda x: tuple(map(int, x.split("-")))):
        print(f"  {stage}  {tot[stage]:8,d}  {tot[stage]/allf*100:5.1f}%")


if __name__ == "__main__":
    main()
