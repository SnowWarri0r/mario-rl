"""裁掉 HUD 的蒸馏数据：老师看未裁画面出概率，我们存裁过的画面给学生。

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
OUTDIR = "distill_data_crop"
HUD_ROWS = 40

TEACHERS = {
    "1-1": "mario_w1c_final.zip",
    "1-2": "mario_w1c_final.zip",
    "1-3": "mario_13expert_final.zip",
    "1-4": "mario_w1c_final.zip",
    "2-1": "mario_21expert_v2.zip",
    "2-2": "mario_22ladder_final.zip",
    "2-3": "mario_w2land_final.zip",
    "2-4": "mario_w2land_final.zip",
    "3-1": "checkpoints_31expert/mario_31exp_1600000_steps.zip",
    "3-2": "mario_w3_final.zip",
    "3-3": "mario_w3_final.zip",
    "3-4": "mario_w3_final.zip",
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
    from make_env import MarioBase, SkipFrame

    teacher = PPO.load(TEACHERS[stage], device="cpu")
    raw = MarioBase(stages=[stage])
    env = SkipFrame(raw, k=4)                       # 到这一层还是原始 240x256 帧
    full = collections.deque(maxlen=4)              # 未裁：给老师看
    crop = collections.deque(maxlen=4)              # 裁过：存给学生

    frame, _ = env.reset()
    for dq, c in ((full, False), (crop, True)):
        f = _prep(frame, c)
        for _ in range(4):
            dq.append(f)

    obs_buf, prob_buf = [], []
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
            frame, _ = env.reset()
            for dq, c in ((full, False), (crop, True)):
                f = _prep(frame, c)
                for _ in range(4):
                    dq.append(f)
    env.close()
    path = f"{OUTDIR}/{stage}_s{shard}.npz"
    np.savez_compressed(path, obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    return stage, shard, n, path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    per = PER_STAGE // SHARDS
    jobs = [(st, k, per) for st in TEACHERS for k in range(SHARDS)]
    print(f"=== 裁 HUD 收数据：12 关 × {PER_STAGE} 帧（切 {SHARDS} 份并行，共 {len(jobs)} 进程）"
          f"，老师看未裁图、存裁过的图 ===", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=min(len(jobs), 40)) as pool:
        for stage, shard, n, path in pool.map(collect, jobs):
            done += n
            print(f">>> {stage} 分片{shard} 收完 {n} 帧（累计 {done}）", flush=True)
    print(f">>> 共 {done} 帧 → {OUTDIR}/", flush=True)


if __name__ == "__main__":
    main()
