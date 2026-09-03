"""裁 HUD 版 DAgger：学生看**裁过的**图开车，老师看**未裁的**图打标签。

为什么必须双视角：学生 `mario_v13crop` 是在裁过的观测上蒸出来的，只能喂它裁过的栈；
而各关老师都是在含 HUD 的观测上训的 RL 专家，喂它们裁过的图＝OOD，标签会变差
（这也是 `collect_distill_crop.py` 当初的设计——老师只当标注 oracle，一个都不用重训）。
所以一帧算两遍：未裁的 (4,84,84) 给老师出概率，裁过的 (4,84,84) 给学生决策并存进数据集。

要回答的问题：裁 HUD 在 base-only 上把连打全通率从 1.0% 抬到 4.2%，
那在**完整配方（含 DAgger）**里值多少？对标的是 v10 的 10.4%。

⚠️手搭 env 链必须自己补 `NoopReset`（`collect_distill_crop.py` 原版漏了，
收出来全是相位 0 的数据 —— 正是本项目最早发现的记忆陷阱本身）。

用法: MARIO_NOOP=30 python collect_dagger_crop.py [总帧数]   默认 240000
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, collections
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 240_000
STUDENT = os.environ.get("MARIO_STUDENT", "mario_v13crop.zip")
OUTDIR = os.environ.get("MARIO_OUTDIR", "distill_data_dagger_v13crop")
NOOP = int(os.environ.get("MARIO_NOOP", "30"))
HUD_ROWS = 40
SHARDS = 2

# 配比按 v13crop 的短板给（逐关 argmax / 连打条件成功率）：
#   2-2 35/25 · 2-3 68/80 · 1-3 97/68 · 3-4 82/75 · 3-3 100/76 · 2-4 95/74 · 2-1 88/99
JOBS = [
    ("2-2", "mario_22robust.zip",                                     30),
    ("1-3", "checkpoints_w1ent0/w1ent0_6250000_steps.zip",            16),
    ("2-3", "mario_23robust.zip",                                     12),
    ("3-4", "mario_34robust.zip",                                     10),
    ("2-4", "checkpoints_mario_24fine/mario_24fine_99840_steps.zip",   8),
    ("3-3", "checkpoints_w3ent0/w3ent0_2250000_steps.zip",             8),
    ("2-1", "checkpoints_s21ent0/s21ent0_3500000_steps.zip",           6),
    ("3-1", "checkpoints_w3ent0/w3ent0_2250000_steps.zip",             4),
    ("1-4", "mario_14robust.zip",                                      3),
    ("1-1", "mario_11robust.zip",                                      1),
    ("1-2", "mario_12robust.zip",                                      1),
    ("3-2", "mario_32robust.zip",                                      1),
]


def _prep(frame, crop):
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if crop:
        g = g[HUD_ROWS:]
    return cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA).astype(np.uint8)


def collect(job):
    stage, teacher_path, n, shard = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import MarioBase, SkipFrame, NoopReset
    import wide_cnn  # noqa: F401  学生是 WideNatureCNN，反序列化要它

    student = PPO.load(STUDENT, device="cpu")
    teacher = PPO.load(teacher_path, device="cpu")
    env = SkipFrame(NoopReset(MarioBase(stages=[stage]), max_noop=NOOP), k=4)
    w0, s0 = (int(x) for x in stage.split("-"))

    full = collections.deque(maxlen=4)      # 未裁：老师看
    crop = collections.deque(maxlen=4)      # 裁过：学生看，也是存下来的

    def reset():
        frame, _ = env.reset()
        for dq, c in ((full, False), (crop, True)):
            f = _prep(frame, c)
            dq.clear()
            for _ in range(4):
                dq.append(f)

    reset()
    obs_buf, prob_buf = [], []
    cleared = attempts = 0
    while len(obs_buf) < n:
        ot, _ = teacher.policy.obs_to_tensor(np.stack(full, 0))
        st, _ = student.policy.obs_to_tensor(np.stack(crop, 0))
        with th.no_grad():
            tprobs = teacher.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
            a = int(student.policy.get_distribution(st).distribution.sample().cpu().numpy()[0])
        obs_buf.append(np.stack(crop, 0)); prob_buf.append(tprobs.astype(np.float32))
        frame, r, term, trunc, info = env.step(a)          # 学生开车
        full.append(_prep(frame, False)); crop.append(_prep(frame, True))
        if term or trunc:
            attempts += 1
            cleared += bool(info.get("flag_get")
                            or (info.get("world"), info.get("stage")) != (w0, s0))
            reset()
    env.close()
    path = f"{OUTDIR}/{stage}_dagcrop_s{shard}.npz"
    np.savez_compressed(path, obs=np.array(obs_buf[:n], np.uint8),
                        probs=np.array(prob_buf[:n], np.float32))
    return stage, n, cleared, attempts, path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    assert NOOP, "DAgger 也要在抖动分布上收，否则收的是固定相位下的漂移"
    wsum = sum(w for _, _, w in JOBS)
    jobs = []
    for stage, teacher, w in JOBS:
        per = max(TOTAL * w // wsum // SHARDS, 2000)
        jobs += [(stage, teacher, per, k) for k in range(SHARDS)]
    print(f"=== 裁 HUD 版 DAgger：{STUDENT} 看裁过的图开车 / 老师看未裁图打标签，"
          f"抖动 0-{NOOP}，{len(jobs)} 进程 ===", flush=True)
    tot, done = {}, 0
    with ProcessPoolExecutor(max_workers=min(len(jobs), 26)) as pool:
        for stage, n, c, a, path in pool.map(collect, jobs):
            done += n
            cc, aa = tot.get(stage, (0, 0)); tot[stage] = (cc + c, aa + a)
            print(f">>> {stage} 收完 {n} 帧，学生开车通关 {c}/{a}（累计 {done}）", flush=True)
    print("\n=== 学生自己开车的通关率（低于老师的部分正是 DAgger 要补的）===", flush=True)
    for stage, _, _ in JOBS:
        c, a = tot.get(stage, (0, 0))
        print(f"  {stage}  {c}/{a} = {c/a*100 if a else 0:.0f}%", flush=True)
    print(f">>> 共 {done} 帧 → {OUTDIR}/", flush=True)


if __name__ == "__main__":
    main()
