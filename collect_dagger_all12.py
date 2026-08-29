"""十二关 DAgger：学生（mario_all12_wide）自己开车，各关老师在它走偏的状态上打标签。

治的是蒸馏的分布漂移——学生一旦偏离老师示范过的成功轨迹就懵，而它偏到哪里只有让它自己跑才知道。
配比按学生当前的逐关实测（每关 100 局）来：弱关多收，强关也收一点防止重蒸时漂移掉。
存到 distill_data_dagger12/（单独目录，不污染之前六组消融用的那两个数据集）。

用法: python collect_dagger_all12.py [总帧数]   默认 200000，按下面权重分到 12 关
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

STUDENT = "mario_all12_wide.zip"
OUTDIR = "distill_data_dagger12"
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000

# (关卡, 老师, 权重)。权重≈按学生当前通关率的短板程度给，强关给地板值防遗忘。
# 学生现状(N=100)：1-1 60 / 1-2 86 / 1-3 95 / 1-4 57 / 2-1 76 / 2-2 70 /
#                  2-3 76 / 2-4 71 / 3-1 73 / 3-2 79 / 3-3 78 / 3-4 54
JOBS = [
    ("3-4", "mario_w3_final.zip",         30),   # 54% 最弱
    ("1-4", "mario_w1c_final.zip",        30),   # 57%
    ("1-1", "mario_w1c_final.zip",        30),   # 60%
    ("2-2", "mario_22ladder_final.zip",   20),   # 70% 反应式水关，历来最先掉血
    ("2-4", "mario_w2land_final.zip",     18),   # 71%
    # 3-1 用 1.6M 档 checkpoint（当年实测比 3M 的 final 好，也是 3-1.npz 的标签来源，换老师会打架）
    ("3-1", "checkpoints_31expert/mario_31exp_1600000_steps.zip", 15),   # 73%
    ("2-1", "mario_21expert_v2.zip",      12),   # 76%
    ("2-3", "mario_w2land_final.zip",     12),   # 76%
    ("3-3", "mario_w3_final.zip",         12),   # 78%
    ("3-2", "mario_w3_final.zip",         10),   # 79%
    ("1-2", "mario_w1c_final.zip",         6),   # 86% 地板值
    ("1-3", "mario_13expert_final.zip",    5),   # 95% 地板值
]


def collect(job):
    stage, teacher_path, n = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401  student 是 WideNatureCNN，反序列化要它

    student = PPO.load(STUDENT, device="cpu")
    teacher = PPO.load(teacher_path, device="cpu")
    env = make_env(stages=[stage])
    obs_buf, prob_buf = [], []
    o, _ = env.reset()
    while len(obs_buf) < n:
        ot, _ = teacher.policy.obs_to_tensor(o)
        st, _ = student.policy.obs_to_tensor(o)
        with th.no_grad():
            tprobs = teacher.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
            a = int(student.policy.get_distribution(st).distribution.sample().cpu().numpy()[0])
        obs_buf.append(o.astype(np.uint8)); prob_buf.append(tprobs.astype(np.float32))
        o, r, term, trunc, info = env.step(a)     # 学生开车：走到哪儿由学生决定
        if term or trunc:
            o, _ = env.reset()
    env.close()
    path = f"{OUTDIR}/{stage}_dagger12.npz"
    np.savez_compressed(path, obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    return stage, n, teacher_path, path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    wsum = sum(w for _, _, w in JOBS)
    jobs = [(s, t, max(1000, round(TOTAL * w / wsum))) for s, t, w in JOBS]
    print(f"=== 十二关 DAgger：学生 {STUDENT} 开车，共收 {sum(n for _, _, n in jobs)} 帧，12 关并行 ===", flush=True)
    for s, t, n in jobs:
        print(f"  {s}: {n:6d} 帧  老师 {t}")
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for stage, n, tp, path in pool.map(collect, jobs):
            print(f">>> {stage} 收完 {n} 帧 → {path}", flush=True)
    print(f">>> DAgger 数据已落 {OUTDIR}/，重蒸时把它加进 MARIO_DATA_DIRS", flush=True)


if __name__ == "__main__":
    main()
