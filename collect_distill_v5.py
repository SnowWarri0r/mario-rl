"""按"熵为零手术"之后的新老师班底重收蒸馏数据（v5）。

跟 collect_distill_noop.py 的区别只有两个：老师换成手术后逐关实测最强的那一档，
以及每关顺带报一个收集期间的实际通关率——数据质量得能当场看见，不能等蒸完学生才发现喂错了。

选老师的原则还是"一关一关挑"，跟上一轮一样，只是这次的教训更明确：
熵为零（ent_coef=0 + 小 lr + 密集存档 + 按 DET 挑档）在需要**精确和坚决**的关上是大胜
（2-1 46→100、3-1 64→100、2-4 69→98、1-3 39→100、3-4 79→96、2-2 34→88），
但在本来就宽松的关上反而有害（2-3 84→83、1-2 45→9），那两关保留原版老师。
没有普适的超参，只有配得上这一关的超参。

开车一律用**采样**而不是 argmax，哪怕新老师的 DET 分数更高：argmax 轨迹是条极窄的管道，
学生一偏出去就没见过那个画面（上次实测 11%→0%）。模仿学习要的是状态覆盖，不是最优示范。
好在熵为零训出来的策略分布本身就接近 one-hot，采样≈argmax，覆盖靠的是残余的那点熵。

用法: python collect_distill_v5.py [每关帧数] [每关分片数] [抖动帧数]  默认 70000 / 2 / 30
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

PER_STAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 70_000
SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
NOOP = int(sys.argv[3]) if len(sys.argv) > 3 else 30
OUTDIR = "distill_data_v5"

# 括号里是该老师在这一关的实测通关率（N=150 @ noop=30，取它自己最优的推理模式）
TEACHERS = {
    "1-1": "checkpoints_w1ent0/w1ent0_3500000_steps.zip",    # 93% STO（原 68）
    # ⚠️这一档是后来才找到的：手术版在 1-2 上曾判"有害（9%）"，那是因为当时最早的存档是 750k，
    # 早滑过峰值了。按 2.5 万步重扫，10 万-40 万步是一整段 81-91% 的平台，N=400 复核 88%，
    # 而且 noop=0 与 noop=30 只差 1pp（原老师差 27pp，那 27 全是背相位的水分）。
    "1-2": "checkpoints_mario_12ent0/mario_12ent0_249984_steps.zip",  # 88% STO（原 45）
    "1-3": "checkpoints_w1ent0/w1ent0_6250000_steps.zip",    # 100% DET（原 39）
    "1-4": "checkpoints_w1ent0/w1ent0_3500000_steps.zip",    # 94% DET（原 82）
    "2-1": "checkpoints_s21ent0/s21ent0_3500000_steps.zip",  # 100% DET（原 46）
    "2-2": "mario_22champ.zip",                              # 88% DET（原 34）
    "2-3": "mario_w2land_final.zip",                         # 84%（手术版 83，打平保留原版）
    # ⚠️边界档不是峰值：w2ent0 那条臂最早的存档就是 750k，按 2.5 万步细扫，
    # 7.4 万-22.4 万步是一整段 100%。同样的毛病 3-2 也有（它取的是最后一档）。
    "2-4": "checkpoints_mario_24fine/mario_24fine_99840_steps.zip",   # 100% DET（原 69→98→100）
    "3-1": "checkpoints_w3ent0/w3ent0_2250000_steps.zip",    # 100% DET（原 64）
    "3-2": "checkpoints_mario_32fine/mario_32fine_99840_steps.zip",   # 100% DET（原 85→89→100）
    "3-3": "checkpoints_w3ent0/w3ent0_2250000_steps.zip",    # 100% DET（原 98）
    "3-4": "checkpoints_w3ent0/w3ent0_2250000_steps.zip",    # 96% DET（原 79）
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
    cleared = attempts = 0
    while len(obs_buf) < n:
        ot, _ = teacher.policy.obs_to_tensor(o)
        with th.no_grad():
            dist = teacher.policy.get_distribution(ot)
            probs = dist.distribution.probs.cpu().numpy()[0]
            a = int(dist.sample().cpu().numpy()[0])
        obs_buf.append(o.astype(np.uint8)); prob_buf.append(probs.astype(np.float32))
        o, r, term, trunc, info = env.step(a)
        if term or trunc:
            attempts += 1
            # 单关环境里过了旗子就会跳到下一关，world/stage 变了也算通关
            w, s = int(stage.split("-")[0]), int(stage.split("-")[1])
            cleared += bool(info.get("flag_get") or (info.get("world"), info.get("stage")) != (w, s))
            o, _ = env.reset()                        # 每次 reset 都重新抽一个相位
    env.close()
    path = f"{OUTDIR}/{stage}_s{shard}.npz"
    np.savez_compressed(path, obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    return stage, shard, n, cleared, attempts, path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    per = PER_STAGE // SHARDS
    # MARIO_ONLY="1-2,2-2" → 只重收这几关（换了某关老师时不必把 84 万帧全收一遍）
    only = [s for s in os.environ.get("MARIO_ONLY", "").split(",") if s]
    stages = only or list(TEACHERS)
    jobs = [(st, k, per) for st in stages for k in range(SHARDS)]
    print(f"=== v5 重收：12 关 × {PER_STAGE} 帧，采样开车，开局随机空按 0-{NOOP} 帧，{len(jobs)} 进程 ===", flush=True)
    tot = {}
    done = 0
    with ProcessPoolExecutor(max_workers=min(len(jobs), 26)) as pool:
        for stage, shard, n, c, a, path in pool.map(collect, jobs):
            done += n
            cc, aa = tot.get(stage, (0, 0))
            tot[stage] = (cc + c, aa + a)
            print(f">>> {stage} 分片{shard} 收完 {n} 帧，其间通关 {c}/{a}（累计 {done}）", flush=True)
    print(f"\n=== 收集期间老师的实际通关率（采样模式，跟上面表里的最优模式分数会有出入）===", flush=True)
    for st in stages:
        c, a = tot.get(st, (0, 0))
        print(f"  {st}  {c}/{a} = {c/a*100 if a else 0:.0f}%", flush=True)
    print(f">>> 共 {done} 帧 → {OUTDIR}/", flush=True)


if __name__ == "__main__":
    main()
