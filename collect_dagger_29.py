"""28 关 DAgger：学生自己开车，老师在它走偏的状态上打标签。

为什么这一步是必需的而不是锦上添花：蒸馏只在**老师的轨迹分布**上给标签，
学生每帧 99% 的准确率摊到两千步就什么都不剩，误差累积会把它带到老师从没去过的状态，
那里它没有任何监督。实测 v29 学生 78.9%、老师 93.2%，差 14pp，
而且差得最狠的几关（8-1 差 55pp、5-1 差 45pp）恰恰是关卡长、累积步数多的。

**配比自动从学生的实测分数算**，不手写表：
手写配比每换一个学生就要重排一次，而且很容易忘了跟着分数走（上一轮就是这个坑）。
这里直接解析 `diag_progress` 的输出日志，越弱的关配越多帧。

⚠️ 老师名册从 teachers.py 读，别在这里抄第二份。
⚠️ 必须开抖动（MARIO_NOOP=30），否则收的是固定相位下的漂移状态。

用法: MARIO_NOOP=30 MARIO_EVAL_LOG=eval_v29.log python collect_dagger_29.py [总帧数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, re
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

import teachers as T

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
STUDENT = os.environ.get("MARIO_STUDENT", "mario_v29.zip")
EVAL_LOG = os.environ.get("MARIO_EVAL_LOG", "eval_v29.log")
OUTDIR = os.environ.get("MARIO_OUTDIR", "distill_data_dagger_29")
NOOP = int(os.environ.get("MARIO_NOOP", "30"))
SHARDS = int(os.environ.get("MARIO_SHARDS", "2"))
WORKERS = int(os.environ.get("MARIO_WORKERS", "40"))
# 强关的地板值。⚠️ 第一轮用 2 太低，出过事：3-3 本来 100%（31/31），DAgger 里按地板只分到
# 2000 帧（占合并数据的 0.65%），重蒸之后掉到 16%（5/31）——它仍然能走到 x=2409 那个终点
# 附近（跟满分时同一个位置），只是最后一下过不去了。也就是说**地板太低护不住已经满分的关**，
# 而且失败方式很局部（不是忘了怎么玩，是终点前那一下丢了）。
FLOOR = int(os.environ.get("MARIO_FLOOR", "10"))


def student_scores(path):
    """从 diag_progress 的输出里解析每关通关数。格式: `1-1     28/31   ...`"""
    out = {}
    for ln in open(path):
        m = re.match(r"^(\d-\d)\s+(\d+)/(\d+)\s", ln)
        if m:
            out[m.group(1)] = int(m.group(2)) / int(m.group(3)) * 100
    if not out:
        raise SystemExit(f"{path} 里没解析出任何逐关成绩——确认它是 diag_progress 的输出")
    return out


def collect(job):
    stage, teacher_path, n, shard = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    student = PPO.load(STUDENT, device="cpu")
    teacher = PPO.load(teacher_path, device="cpu")
    env = make_env(stages=[stage], noop=NOOP)
    w0, s0 = (int(x) for x in stage.split("-"))
    obs_buf, prob_buf = [], []
    o, _ = env.reset()
    cleared = attempts = 0
    while len(obs_buf) < n:
        ot, _ = teacher.policy.obs_to_tensor(o)
        st, _ = student.policy.obs_to_tensor(o)
        with th.no_grad():
            tp = teacher.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
            a = int(student.policy.get_distribution(st).distribution.sample().cpu().numpy()[0])
        # 存的是**学生走到的状态** + **老师在那儿的意见**，这正是 DAgger 的全部内容
        obs_buf.append(o.astype(np.uint8)); prob_buf.append(tp.astype(np.float32))
        o, r, term, trunc, info = env.step(a)          # 学生开车
        if term or trunc:
            attempts += 1
            cleared += bool(info.get("flag_get")
                            or (info.get("world"), info.get("stage")) != (w0, s0))
            o, _ = env.reset()
    env.close()
    path = f"{OUTDIR}/{stage}_dag_s{shard}.npz"
    np.savez_compressed(path, obs=np.array(obs_buf[:n], np.uint8),
                        probs=np.array(prob_buf[:n], np.float32))
    return stage, n, cleared, attempts


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    assert NOOP, "DAgger 也要在抖动分布上收，否则收的是固定相位下的漂移"
    sc = student_scores(EVAL_LOG)
    stages = [s for s in sc if s in T.TEACHERS and T.TEACHERS[s][1] >= 40]
    # 权重 = 学生离满分的距离，强关留地板值
    w = {s: max(FLOOR, round(100 - sc[s])) for s in stages}
    wsum = sum(w.values())

    print(f"=== DAgger：{STUDENT} 开车，{len(stages)} 关，目标 {TOTAL:,} 帧，抖动 0-{NOOP} ===")
    print("配比（学生分 → 权重 → 帧数）：")
    for s in sorted(stages, key=lambda x: sc[x]):
        print(f"  {s}  学生 {sc[s]:5.1f}%  权重 {w[s]:3d}  {TOTAL*w[s]//wsum:7,d} 帧")

    jobs = []
    for s in stages:
        per = max(TOTAL * w[s] // wsum // SHARDS, 2000)
        jobs += [(s, T.path(s), per, k) for k in range(SHARDS)]

    tot, done = {}, 0
    with ProcessPoolExecutor(max_workers=min(len(jobs), WORKERS)) as pool:
        for stage, n, c, a in pool.map(collect, jobs, chunksize=1):
            done += n
            cc, aa = tot.get(stage, (0, 0)); tot[stage] = (cc + c, aa + a)
            print(f">>> {stage} {n} 帧，学生开车通关 {c}/{a}（累计 {done:,}）", flush=True)

    print(f"\n=== 学生自己开车的通关率（低于老师的部分正是 DAgger 要补的）===")
    for s in sorted(stages, key=lambda x: sc[x]):
        c, a = tot.get(s, (0, 0))
        print(f"  {s}  {c}/{a} = {c/a*100 if a else 0:5.1f}%   （argmax 评测是 {sc[s]:.0f}%）")
    print(f">>> 共 {done:,} 帧 → {OUTDIR}/", flush=True)


if __name__ == "__main__":
    main()
