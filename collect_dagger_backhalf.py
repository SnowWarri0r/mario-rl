"""定点 DAgger：只在 2-2 的后半段收数据，因为分歧就集中在那里。

**诊断依据**（`diag_agreement22.py`，老师 argmax 开车逐帧比对学生）：
    x    0-1400  argmax 不一致  5-6%   KL 0.01   ← 前半程几乎逐帧一致
    x 1400-2000               14-20%  KL 0.18
    x 2200-2800               29-30%  KL 0.38-0.47  ← 峰值
    x 3000-3200                0.4%   KL 0.01   ← 旗杆前的直路
学生自己开车的死亡中位数 x=2214，跟分歧峰值对上。
⇒ 是**局部精度问题**，不是表示/容量问题（把整张网专供给 2-2 也只买到 8pp）。
2-2 全长 3161 像素，均匀采的数据里真正难的那 800 像素分到不足四分之一，
其中还有一大半是失败轨迹（老师采样模式在 2-2 只有 62% 通关）。这里就是把标签堆到那 800 像素上。

**为什么不用存档开局（`ArchiveStart`）来省掉前半段。** 试过，是错的：把**老师**放到 x=1408 的
存档点，它从整关 86% 掉到 argmax 21%，局长中位只有 32 帧，往前走 80 像素就死
（`diag_archive_start.py`）。原因是 `FrameStack` 在复位时被填成"同一帧复制四份"，
**观测里的速度信息全没了**——2-2 是水下关，策略靠 4 帧差分判断鱼往哪游、多快，
看到一堆"静止的鱼"就会撞上去。那样收来的状态分布在真实游戏里不存在，连标签也是假的
（champ 在畸形观测上给出的动作概率，不代表它真会怎么做）。
⚠️当初记"存档机制跑通了（开局 x≈1880）"只验证了**位置**对，没验证**状态**可用——这是两回事。

所以这里就老老实实从头打，**只记录 x ≥ MARIO_XMIN 之后的帧**：状态分布天然正确，
代价只是前半段白跑（学生本来就有一半以上的局能走到 x≈2214，不算浪费）。

学生用**采样**开车而不是 argmax：按 DAgger 的道理该在部署分布上收，而 2-2 部署时走 argmax；
但 argmax 在确定性模拟器里每局轨迹完全相同，只有开局抖动那点差异，覆盖太窄
（这个项目在"argmax 开车收数据"上栽过一次，11%→0%）。采样覆盖 argmax 的邻域，是这里的折中。

用法: MARIO_NOOP=30 MARIO_XMIN=1300 python collect_dagger_backhalf.py [总帧数] [并发]  默认 160000 / 24
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 160_000
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
STUDENT = os.environ.get("MARIO_STUDENT", "mario_v7_wide.zip")
TEACHER = os.environ.get("MARIO_TEACHER", "mario_22champ.zip")
OUTDIR = os.environ.get("MARIO_OUTDIR", "distill_data_dagger_back")
XMIN = int(os.environ.get("MARIO_XMIN", "1300"))   # 只记这个 x 之后的帧（分歧从 1400 起飙）


def collect(job):
    wid, n = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    student = PPO.load(STUDENT, device="cpu")
    teacher = PPO.load(TEACHER, device="cpu")
    env = make_env(stages=["2-2"], noop=int(os.environ.get("MARIO_NOOP", "30")))
    obs_buf, prob_buf = [], []
    o, _ = env.reset()
    x = 0
    kept = seen = cleared = attempts = 0
    while len(obs_buf) < n:
        st, _ = student.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(student.policy.get_distribution(st).distribution.sample().cpu().numpy()[0])
        seen += 1
        if x >= XMIN:                       # 只在后半段问老师，前半段连前向都省了
            ot, _ = teacher.policy.obs_to_tensor(o)
            with th.no_grad():
                tprobs = teacher.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
            obs_buf.append(o.astype(np.uint8)); prob_buf.append(tprobs.astype(np.float32))
            kept += 1
        o, r, term, trunc, info = env.step(a)
        x = int(info.get("x_pos", x))
        if term or trunc:
            attempts += 1
            cleared += bool(info.get("flag_get") or (info.get("world"), info.get("stage")) != (2, 2))
            o, _ = env.reset(); x = 0
    env.close()
    start_x = [kept * 100 // max(seen, 1)]      # 复用返回位：记录"留存率%"
    path = f"{OUTDIR}/2-2_back_w{wid}.npz"
    np.savez_compressed(path, obs=np.array(obs_buf[:n], np.uint8),
                        probs=np.array(prob_buf[:n], np.float32))
    return wid, n, cleared, attempts, start_x[0], path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    per = max(TOTAL // WORKERS, 1000)
    print(f"=== 2-2 后半段定点 DAgger：{STUDENT} 开车 / {TEACHER} 打标签，"
          f"{per*WORKERS} 帧，{WORKERS} 进程，只记 x>={XMIN} 的帧 ===", flush=True)
    tot_c = tot_a = 0
    xs = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for wid, n, c, a, mx, path in pool.map(collect, [(k, per) for k in range(WORKERS)]):
            tot_c += c; tot_a += a; xs.append(mx)
            print(f">>> w{wid} {n} 帧（占跑过的 {mx}%），学生整关通关 {c}/{a}", flush=True)
    print(f"\n>>> 共 {per*WORKERS} 帧 → {OUTDIR}/", flush=True)
    print(f">>> 留存率中位 {int(np.median(xs))}%（跑过的帧里有多少落在 x>={XMIN}）", flush=True)
    print(f">>> 学生整关通关率 {tot_c}/{tot_a} = {tot_c/tot_a*100 if tot_a else 0:.0f}%"
          f"（采样模式，对照它的 argmax 56%）", flush=True)


if __name__ == "__main__":
    main()
