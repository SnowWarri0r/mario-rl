"""裁掉 HUD 的蒸馏数据：老师看未裁画面出概率，我们存裁过的画面给学生。

**这一版是为「连打落差只剩 HUD 没排除」这个结论重做的。** 逐关能力稳在 92-96%，
而连打全通率在 5.9%-21.5% 之间乱跳；相位、帧栈、步数上限、马里奥形态都已逐个排除
（1-3 与 3-3 逐相位扫过，121 个相位 96-100%）。剩下的唯一系统性差异就是 HUD：
单关评测时分数 0 / 命数 2，连打到 1-3 时分数几千 / 命数在变，而这些像素就在观测顶部 40 行。
⚠️ 当初「裁掉 HUD 没用」是在 2-2 只有 4% 的模型上量的——那正是让我在帧栈清空上判错的
同一种「闸门关着，闸门后面所有改动的测量值都是 0」的情形。

关键点：老师是在含状态栏的观测上训出来的，喂它裁过的图＝OOD，标签会变差。
所以这里一帧算两遍——未裁的 (4,84,84) 给老师做决策和打标签，裁过的 (4,84,84) 存进数据集。
老师在这条链路里只是个标注 oracle，**一个 RL 专家都不用重训**。

每关切 SHARDS 份并行收（12 关 × 3 = 36 进程），192 核的机器上十分钟量级。
用法: python collect_distill_crop.py [每关帧数] [每关分片数]   默认 70000 / 3
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, collections
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

PER_STAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 70_000
SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUTDIR = os.environ.get("MARIO_OUTDIR", "distill_data_crop")
NOOP = int(os.environ.get("MARIO_NOOP", "30"))
HUD_ROWS = 40

# robust 班底（主指标 no-op 0-30 / 鲁棒性 0-120），跟 collect_distill_v5.py 保持一致
TEACHERS = {
    "1-1": "mario_11robust.zip",                             # 100/100
    "1-2": "mario_12robust.zip",                             #  95/94
    "1-3": "checkpoints_w1ent0/w1ent0_6250000_steps.zip",    # 100/100
    "1-4": "mario_14robust.zip",                             # 100/100
    "2-1": "checkpoints_s21ent0/s21ent0_3500000_steps.zip",  # 100/98
    "2-2": "mario_22robust.zip",                             #  84/60
    "2-3": "mario_23robust.zip",                             #  91/91
    "2-4": "checkpoints_mario_24fine/mario_24fine_99840_steps.zip",   # 100/100
    "3-1": "checkpoints_w3ent0/w3ent0_2250000_steps.zip",    # 100/100
    "3-2": "mario_32robust.zip",                             # 100/100
    "3-3": "checkpoints_w3ent0/w3ent0_2250000_steps.zip",    # 100/100
    "3-4": "mario_34robust.zip",                             # 100/100
}


def _prep(frame, crop):
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if crop:
        g = g[HUD_ROWS:]
    return cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA).astype(np.uint8)


def collect(job):
    stage, shard, n = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import MarioBase, SkipFrame, NoopReset

    teacher = PPO.load(TEACHERS[stage], device="cpu")
    raw = MarioBase(stages=[stage])
    # ⚠️这条链是手搭的，原版漏了 NoopReset —— 不补上收出来全是相位 0 的数据，
    # 那正是这个项目最早发现的记忆陷阱本身。放在跳帧之前拿单帧粒度。
    env = SkipFrame(NoopReset(raw, max_noop=NOOP), k=4)      # 到这一层还是原始 240x256 帧
    full = collections.deque(maxlen=4)              # 未裁：给老师看
    crop = collections.deque(maxlen=4)              # 裁过：存给学生

    frame, _ = env.reset()
    for dq, c in ((full, False), (crop, True)):
        f = _prep(frame, c)
        for _ in range(4):
            dq.append(f)

    obs_buf, prob_buf = [], []
    w0, s0 = (int(x) for x in stage.split("-"))
    cleared = attempts = 0
    while len(obs_buf) < n:
        ot, _ = teacher.policy.obs_to_tensor(np.stack(full, 0))
        with th.no_grad():
            dist = teacher.policy.get_distribution(ot)
            probs = dist.distribution.probs.cpu().numpy()[0]
            a = int(dist.sample().cpu().numpy()[0])          # 老师开车(采样)，跟原来的收数据方式一致
        obs_buf.append(np.stack(crop, 0)); prob_buf.append(probs.astype(np.float32))
        frame, r, term, trunc, info = env.step(a)
        full.append(_prep(frame, False)); crop.append(_prep(frame, True))
        if term or trunc:
            attempts += 1
            cleared += bool(info.get("flag_get")
                            or (info.get("world"), info.get("stage")) != (w0, s0))
            frame, _ = env.reset()
            for dq, c in ((full, False), (crop, True)):
                f = _prep(frame, c)
                for _ in range(4):
                    dq.append(f)
    env.close()
    path = f"{OUTDIR}/{stage}_s{shard}.npz"
    np.savez_compressed(path, obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    return stage, shard, n, cleared, attempts, path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    per = PER_STAGE // SHARDS
    jobs = [(st, k, per) for st in TEACHERS for k in range(SHARDS)]
    print(f"=== 裁 HUD 收数据：12 关 × {PER_STAGE} 帧（切 {SHARDS} 份并行，共 {len(jobs)} 进程）"
          f"，老师看未裁图、存裁过的图，抖动 0-{NOOP} ===", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=min(len(jobs), 40)) as pool:
        tot = {}
        for stage, shard, n, c, a, path in pool.map(collect, jobs):
            done += n
            cc, aa = tot.get(stage, (0, 0)); tot[stage] = (cc + c, aa + a)
            print(f">>> {stage} 分片{shard} 收完 {n} 帧，其间通关 {c}/{a}（累计 {done}）", flush=True)
    print("\n=== 收集期间老师的实际通关率（数据质量得当场看见）===", flush=True)
    for st in TEACHERS:
        c, a = tot.get(st, (0, 0))
        print(f"  {st}  {c}/{a} = {c/a*100 if a else 0:.0f}%", flush=True)
    print(f">>> 共 {done} 帧 → {OUTDIR}/", flush=True)


if __name__ == "__main__":
    main()
