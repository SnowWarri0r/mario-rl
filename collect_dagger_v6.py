"""v6 专项 DAgger：v5 学生自己开车，新班底老师在它走偏的状态上打标签。

**为什么这一轮的 DAgger 比上一轮重要得多。** 上一轮十二关 DAgger 只值 +2pp，那时天花板在老师身上
（老师均值 56%，学生贴着它）。现在老师均值 94%，短板换了位置——是**学生抄不像**：
2-2 老师 89%、学生只有 45%，41pp 的落差。而且已经证明这不是别的关挤占的：
只喂 2-2 数据单独蒸一个同架构学生也才 50%（vs 混训 45%，只差 5pp）。
⇒ 落差是**误差累积**：champ 在 2-2 上是一条精确的低熵轨迹，每帧 99% 的准确率摊到 2000 步就什么都不剩。
这正是 DAgger 对症的东西——让学生自己走偏，然后在偏掉的状态上问老师"这里该怎么办"。

配比按 v5 逐关实测的短板给（每关取它自己最优推理模式，N=150）：
  1-1 91 / 1-2 41 / 1-3 99 / 1-4 86 / 2-1 100 / 2-2 48 /
  2-3 72 / 2-4 90 / 3-1 100 / 3-2 98 / 3-3 100 / 3-4 85
2-2 和 1-2 是两个明显的坑（1-2 另外还换了老师，基础数据也要重收）。强关给地板值防重蒸时漂移掉。

⚠️跟上一轮 `collect_dagger_all12.py` 的一个口径差别：那个脚本用的是 `make_env(stages=[stage])`
**没开抖动**，收到的是固定相位下的漂移状态。这里统一 noop 0-30，跟训练/评测口径对齐。

用法: MARIO_NOOP=30 python collect_dagger_v6.py [总帧数]   默认 240000
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

STUDENT = os.environ.get("MARIO_STUDENT", "mario_v5_wide.zip")
OUTDIR = os.environ.get("MARIO_OUTDIR", "distill_data_dagger_v6")
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 240_000
NOOP = int(os.environ.get("MARIO_NOOP", "30"))

# (关卡, 老师, 权重)
JOBS = [
    ("2-2", "mario_22champ.zip",                                      34),  # 学生 48 vs 老师 89
    ("1-2", "checkpoints_mario_12ent0/mario_12ent0_249984_steps.zip", 22),  # 学生 41 vs 新老师 88
    ("2-3", "mario_w2land_final.zip",                                 10),  # 学生 72 vs 老师 84
    ("3-4", "checkpoints_w3ent0/w3ent0_2250000_steps.zip",             7),  # 85 vs 96
    ("1-4", "checkpoints_w1ent0/w1ent0_3500000_steps.zip",             7),  # 86 vs 94
    ("2-4", "checkpoints_w2ent0/w2ent0_750000_steps.zip",              5),  # 90 vs 98
    ("1-1", "checkpoints_w1ent0/w1ent0_3500000_steps.zip",             5),  # 91 vs 93
    ("3-2", "checkpoints_w3ent0/w3ent0_6250000_steps.zip",             3),  # 地板值
    ("1-3", "checkpoints_w1ent0/w1ent0_6250000_steps.zip",             3),
    ("2-1", "checkpoints_s21ent0/s21ent0_3500000_steps.zip",           2),
    ("3-1", "checkpoints_w3ent0/w3ent0_2250000_steps.zip",             1),
    ("3-3", "checkpoints_w3ent0/w3ent0_2250000_steps.zip",             1),
]
SHARDS = 2          # 每关切两片并行，2-2 那 8 万帧一片跑太久


def collect(job):
    stage, teacher_path, n, shard = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401  学生是 WideNatureCNN，反序列化要它

    student = PPO.load(STUDENT, device="cpu")
    teacher = PPO.load(teacher_path, device="cpu")
    env = make_env(stages=[stage], noop=NOOP)
    obs_buf, prob_buf = [], []
    o, _ = env.reset()
    cleared = attempts = 0
    while len(obs_buf) < n:
        ot, _ = teacher.policy.obs_to_tensor(o)
        st, _ = student.policy.obs_to_tensor(o)
        with th.no_grad():
            tprobs = teacher.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
            a = int(student.policy.get_distribution(st).distribution.sample().cpu().numpy()[0])
        obs_buf.append(o.astype(np.uint8)); prob_buf.append(tprobs.astype(np.float32))
        o, r, term, trunc, info = env.step(a)     # 学生开车：走到哪儿由学生决定
        if term or trunc:
            attempts += 1
            w, s = int(stage.split("-")[0]), int(stage.split("-")[1])
            cleared += bool(info.get("flag_get") or (info.get("world"), info.get("stage")) != (w, s))
            o, _ = env.reset()
    env.close()
    path = f"{OUTDIR}/{stage}_dag_s{shard}.npz"
    np.savez_compressed(path, obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    return stage, n, cleared, attempts, path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    assert NOOP, "DAgger 也要在抖动分布上收，否则收的是固定相位下的漂移"
    wsum = sum(w for _, _, w in JOBS)
    jobs = []
    for stage, teacher, w in JOBS:
        per = max(TOTAL * w // wsum // SHARDS, 2000)
        jobs += [(stage, teacher, per, k) for k in range(SHARDS)]
    print(f"=== v6 DAgger：{STUDENT} 开车，{TOTAL} 帧按短板配比分到 12 关，抖动 0-{NOOP}，"
          f"{len(jobs)} 进程 ===", flush=True)
    tot, done = {}, 0
    with ProcessPoolExecutor(max_workers=min(len(jobs), 26)) as pool:
        for stage, n, c, a, path in pool.map(collect, jobs):
            done += n
            cc, aa = tot.get(stage, (0, 0)); tot[stage] = (cc + c, aa + a)
            print(f">>> {stage} 收完 {n} 帧，学生开车通关 {c}/{a}（累计 {done}）", flush=True)
    print(f"\n=== 学生自己开车的通关率（这就是它当前的真实水平，低于老师的部分正是 DAgger 要补的）===",
          flush=True)
    for stage, _, _ in JOBS:
        c, a = tot.get(stage, (0, 0))
        print(f"  {stage}  {c}/{a} = {c/a*100 if a else 0:.0f}%", flush=True)
    print(f">>> 共 {done} 帧 → {OUTDIR}/", flush=True)


if __name__ == "__main__":
    main()
