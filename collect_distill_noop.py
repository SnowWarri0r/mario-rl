"""带 no-op starts 重收蒸馏数据：老师在抖动过相位的开局上跑，覆盖各种相位而不是永远那一种。

为什么要重收：原来的数据全是从"游戏刚启动"那个固定相位采的，学生跟着老师背同一段舞步。
开了抖动之后，同一段路会以不同的敌人相位出现多次，学生才有机会学"看画面做决定"。
老师自己在抖动下也会掉分（平均 74%→56%），所以这份数据的标签质量确实更差——
但它是诚实的，而且覆盖的状态分布跟真实连打一致。

用法: python collect_distill_noop.py [每关帧数] [每关分片数] [抖动帧数]  默认 70000 / 2 / 30
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

PER_STAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 70_000
SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
NOOP = int(sys.argv[3]) if len(sys.argv) > 3 else 30
OUTDIR = "distill_data_noop"

TEACHERS = {
    "1-1": "mario_w1c_final.zip", "1-2": "mario_w1c_final.zip",
    "1-3": "mario_13expert_final.zip", "1-4": "mario_w1c_final.zip",
    "2-1": "mario_21expert_v2.zip", "2-2": "mario_22noop_gpu.zip",   # 2-2 换成抖动下重训过的那个
    "2-3": "mario_w2land_final.zip", "2-4": "mario_w2land_final.zip",
    "3-1": "checkpoints_31expert/mario_31exp_1600000_steps.zip",
    "3-2": "mario_w3_final.zip", "3-3": "mario_w3_final.zip", "3-4": "mario_w3_final.zip",
}


def collect(job):
    stage, shard, n = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env

    teacher = PPO.load(TEACHERS[stage], device="cpu")
    env = make_env(stages=[stage], noop=NOOP)
    obs_buf, prob_buf = [], []
    o, _ = env.reset()
    while len(obs_buf) < n:
        ot, _ = teacher.policy.obs_to_tensor(o)
        with th.no_grad():
            dist = teacher.policy.get_distribution(ot)
            probs = dist.distribution.probs.cpu().numpy()[0]
            a = int(dist.sample().cpu().numpy()[0])
        obs_buf.append(o.astype(np.uint8)); prob_buf.append(probs.astype(np.float32))
        o, r, term, trunc, info = env.step(a)
        if term or trunc:
            o, _ = env.reset()                        # 每次 reset 都重新抽一个相位
    env.close()
    path = f"{OUTDIR}/{stage}_s{shard}.npz"
    np.savez_compressed(path, obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    return stage, shard, n, path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    per = PER_STAGE // SHARDS
    jobs = [(st, k, per) for st in TEACHERS for k in range(SHARDS)]
    print(f"=== 抖动下重收蒸馏数据：12 关 × {PER_STAGE} 帧，开局随机空按 0-{NOOP} 帧，{len(jobs)} 进程 ===", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=min(len(jobs), 26)) as pool:
        for stage, shard, n, path in pool.map(collect, jobs):
            done += n
            print(f">>> {stage} 分片{shard} 收完 {n} 帧（累计 {done}）", flush=True)
    print(f">>> 共 {done} 帧 → {OUTDIR}/", flush=True)


if __name__ == "__main__":
    main()
