"""29 关（全部非迷宫关）的蒸馏数据采集：老师开车，存 (观测, 老师动作分布)。

跟 collect_distill_v5 的区别只有两条，但都是这次扩关踩出来的：
① 老师名册从 `teachers.py` 读，不再在本文件抄一份。名册抄在 8 个脚本里的时候，
   十二关勉强对得上，29 关必然漂移——而漂移不报错，是**悄悄用旧老师收了一批数据**，
   要等蒸出来学生变差才发现。
② 采样配比由 `teachers.weights()` 按实测分数自动算（越弱配越多），不再手写表。
   手写表每换一次班底就要重排一次，很容易忘了跟着分数走。

**动作空间用 SIMPLE(7)，不用 COMPLEX(12)。** 理由：
12 动作只有 8-4 需要（下管道），而三个迷宫关目前全部无解、没有老师；
现在切 12 动作等于凭空多一条"7 动作模型跑进 12 动作环境"的静默错配路径，
这一轮我已经在这类配置错配上栽过好几次（lr_schedule、NOOP_EXACT、VecNormalize 文件名）。
⇒ 等 8-4 真有老师了再整体切，那时学生本来也要重蒸。

⚠️ 采集必须开抖动（MARIO_NOOP=30）。不开就是在固定相位上收数据，学生学到的是背轨迹。

用法: MARIO_NOOP=30 python collect_distill_29.py [总帧数]   默认 900000
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

import teachers as T

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 900_000
OUTDIR = os.environ.get("MARIO_OUTDIR", "distill_data_29")
NOOP = int(os.environ.get("MARIO_NOOP", "30"))
SHARDS = int(os.environ.get("MARIO_SHARDS", "2"))
MIN_SCORE = int(os.environ.get("MARIO_MIN_SCORE", "40"))   # 低于这个分的关不收（数据是错的）
WORKERS = int(os.environ.get("MARIO_WORKERS", "40"))

STAGES = [s for s in T.ALL_STAGES if s not in T.MAZE_STAGES]


def collect(job):
    stage, teacher_path, n, shard = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    teacher = PPO.load(teacher_path, device="cpu")
    env = make_env(stages=[stage], noop=NOOP)
    w0, s0 = (int(x) for x in stage.split("-"))
    obs_buf, prob_buf = [], []
    o, _ = env.reset()
    cleared = attempts = 0
    while len(obs_buf) < n:
        ot, _ = teacher.policy.obs_to_tensor(o)
        with th.no_grad():
            p = teacher.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
        obs_buf.append(o.astype(np.uint8)); prob_buf.append(p.astype(np.float32))
        # 老师开车用**采样**而不是 argmax：argmax 只会产出一条轨迹，
        # 学生看不到"稍微偏一点该怎么回来"。偏移状态由后续 DAgger 补，这里先要覆盖度。
        o, r, term, trunc, info = env.step(int(np.random.choice(len(p), p=p / p.sum())))
        if term or trunc:
            attempts += 1
            cleared += bool(info.get("flag_get")
                            or (info.get("world"), info.get("stage")) != (w0, s0))
            o, _ = env.reset()
    env.close()
    path = f"{OUTDIR}/{stage}_s{shard}.npz"
    np.savez_compressed(path, obs=np.array(obs_buf[:n], np.uint8),
                        probs=np.array(prob_buf[:n], np.float32))
    return stage, n, cleared, attempts


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    assert NOOP, "必须开抖动，否则收的是固定相位下的数据（学生会学成背轨迹）"

    skipped = [(s, T.TEACHERS[s][1]) for s in STAGES
               if s not in T.TEACHERS or T.TEACHERS[s][1] < MIN_SCORE]
    use = [s for s in STAGES if s not in dict(skipped)]
    if skipped:
        # 不静默跳过：一个 13% 的老师收出来的数据大半是"怎么死"，会把学生带坏
        print(f"⚠️ 跳过 {len(skipped)} 关（老师分数 < {MIN_SCORE}%）：{skipped}", flush=True)

    w = T.weights(use)
    wsum = sum(w.values())
    jobs = []
    for s in use:
        per = max(TOTAL * w[s] // wsum // SHARDS, 3000)
        jobs += [(s, T.path(s), per, k) for k in range(SHARDS)]
    print(f"=== 蒸馏采集 {len(use)} 关 | 目标 {TOTAL:,} 帧 | 抖动 0-{NOOP} | "
          f"{len(jobs)} 进程 | 配比按实测分数自动算 ===", flush=True)

    tot, done = {}, 0
    with ProcessPoolExecutor(max_workers=min(len(jobs), WORKERS)) as pool:
        for stage, n, c, a in pool.map(collect, jobs, chunksize=1):
            done += n
            cc, aa = tot.get(stage, (0, 0)); tot[stage] = (cc + c, aa + a)
            print(f">>> {stage} {n} 帧，老师采样通关 {c}/{a}（累计 {done:,}）", flush=True)

    print("\n=== 老师在采集口径（采样）下的通关率，对照名册里的 argmax 分数 ===", flush=True)
    for s in use:
        c, a = tot.get(s, (0, 0))
        rate = c / a * 100 if a else 0
        flag = "  ⚠️ 比名册低很多" if a and rate < T.TEACHERS[s][1] - 30 else ""
        print(f"  {s}  采样 {rate:5.1f}%  名册(argmax) {T.TEACHERS[s][1]:3d}%{flag}")
    print(f"\n>>> 共 {done:,} 帧 → {OUTDIR}/", flush=True)
    print(">>> 采样通关率普遍低于 argmax 是正常的（采样会偏离），"
          "但**低太多的关标签质量差**，那是 DAgger 该重点补的地方")


if __name__ == "__main__":
    main()
