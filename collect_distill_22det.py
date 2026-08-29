"""专门给 2-2 重收数据：让老师用它最强的 argmax 模式开车。

为什么要单独收：现在学生在 2-2 上只有 10-13%，老师是 26%(采样)/36%(argmax)，落差 20pp+，
而其他关的常规落差只有 6pp。原因在收数据的方式上——收数据时老师一直在**采样**，
等于让它在自己不擅长的模式下示范，轨迹里大半是失败的；学生跟着学到的是"怎么失败"。
改成 argmax 开车，轨迹分布就落在它真正能通关的那条路上（标签仍用完整的动作概率分布）。

抖动照旧开 0-30：argmax + 抖动才是它拿到 36% 的那个设定，也是学生将来要面对的设定。

用法: python collect_distill_22det.py [帧数] [分片数]   默认 140000 / 4（比常规的 70k 加倍，给 2-2 加权）
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 140_000
SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TEACHER = "mario_22cur_30.zip"          # 课程撞出来的尖解：argmax 36% / 采样 22%
OUTDIR = "distill_data_22det"
NOOP = 30


def collect(job):
    shard, n = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env

    teacher = PPO.load(TEACHER, device="cpu")
    env = make_env(stages=["2-2"], noop=NOOP)
    obs_buf, prob_buf = [], []
    o, _ = env.reset()
    cleared = attempts = 0
    while len(obs_buf) < n:
        ot, _ = teacher.policy.obs_to_tensor(o)
        with th.no_grad():
            dist = teacher.policy.get_distribution(ot)
            probs = dist.distribution.probs.cpu().numpy()[0]
        a = int(np.argmax(probs))                      # argmax 开车（标签还是完整分布）
        obs_buf.append(o.astype(np.uint8)); prob_buf.append(probs.astype(np.float32))
        o, r, term, trunc, info = env.step(a)
        if term or trunc:
            attempts += 1
            cleared += bool(info.get("flag_get") or (info.get("world"), info.get("stage")) != (2, 2))
            o, _ = env.reset()
    env.close()
    path = f"{OUTDIR}/2-2_det_s{shard}.npz"
    np.savez_compressed(path, obs=np.array(obs_buf[:n], np.uint8),
                        probs=np.array(prob_buf[:n], np.float32))
    return shard, n, cleared, attempts, path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    per = TOTAL // SHARDS
    print(f"=== 2-2 专项重收：{TOTAL} 帧，老师 {TEACHER} 用 argmax 开车，抖动 0-{NOOP} ===", flush=True)
    tot_c = tot_a = 0
    with ProcessPoolExecutor(max_workers=SHARDS) as pool:
        for shard, n, c, a, path in pool.map(collect, [(k, per) for k in range(SHARDS)]):
            tot_c += c; tot_a += a
            print(f">>> 分片{shard} {n} 帧，其间通关 {c}/{a} 局 → {path}", flush=True)
    rate = tot_c / tot_a * 100 if tot_a else 0
    print(f">>> 共 {per*SHARDS} 帧；收集期间老师通关率 {tot_c}/{tot_a} = {rate:.0f}%"
          f"（应该接近它的 argmax 36%，明显偏低说明数据没落在正确的分布上）", flush=True)


if __name__ == "__main__":
    main()
