"""把"学生 85.9% vs 老师 93.2%"这 7.3pp 拆成两块：分布内 argmax 翻转 vs 分布漂移。

**为什么要拆。** 蒸馏拟合已经到顶（全量 FLOOR=0.8346，v32 实测 0.8600，
吃掉可压缩空间的 97.5%；换大网络、锐化目标都已证否）。所以剩下的差距只能
来自两处，而处方完全相反：

  ① **分布内 argmax 翻转**：在老师走过的状态上，学生就已经选了别的动作。
     软交叉熵小不代表 argmax 一致 —— 确定性模拟器里一帧翻转就可能送命。
     若主因在这儿，该改的是训练目标（让它对齐 argmax），不是再收数据。
  ② **分布漂移**：学生自己走出去，到了老师数据里没有的状态才乱。
     若主因在这儿，DAgger 才是对症的。

同一把尺子量两遍就能分开：
  A 分布内 = 老师数据里的每一帧，学生 argmax ≠ 老师 argmax 的比例。
    老师的输出概率已经存在 npz 里，不用重新加载老师，几乎免费。
  B 在策略 = 学生自己 argmax 走一局，每一步现问老师，统计同样的不一致率。
B 显著高于 A ⇒ 漂移为主；两者接近 ⇒ 翻转为主。

⚠️ B 必须走 exact= 参数指定相位，不能设 MARIO_NOOP_EXACT 环境变量
   （那个在 make_env import 时就读死了，设了等于没设，这一程栽过两次）。

用法: python diag_shift.py <学生.zip> [关卡逗号分隔] [每关相位数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, glob, re
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

STUDENT = sys.argv[1] if len(sys.argv) > 1 else "mario_v32.zip"
import teachers as T
STAGES = (sys.argv[2].split(",") if len(sys.argv) > 2 else
          sorted({re.match(r"(\d+-\d+)", os.path.basename(f)).group(1)
                  for f in glob.glob("distill_data_29/*.npz")},
                 key=lambda s: tuple(map(int, s.split("-")))))
NPHASE = int(sys.argv[3]) if len(sys.argv) > 3 else 8
WORKERS = int(os.environ.get("MARIO_WORKERS", "56"))
MAXSTEP = int(os.environ.get("MARIO_MAXSTEP", "3000"))
DIRS = "distill_data_29,distill_data_dagger_29,distill_data_dagger_30,distill_data_dagger_31".split(",")


def indist(stage):
    """A：老师数据上的 argmax 不一致率。老师的 probs 已存在 npz 里。"""
    import torch as th; th.set_num_threads(2)
    from stable_baselines3 import PPO
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    m = PPO.load(STUDENT, device="cpu")
    files = [f for d in DIRS for f in glob.glob(f"{d}/{stage}_*.npz")]
    bad = tot = 0
    for f in files:
        d = np.load(f)
        obs, pr = d["obs"], d["probs"]
        for i in range(0, len(obs), 1024):
            ob = th.from_numpy(obs[i:i+1024])
            with th.no_grad():
                sa = m.policy.get_distribution(ob).distribution.probs.argmax(1).numpy()
            ta = pr[i:i+1024].argmax(1)
            bad += int((sa != ta).sum()); tot += len(sa)
    return stage, bad, tot


def onpolicy(job):
    """B：学生自己 argmax 走，每步现问老师。"""
    stage, phase = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    st = PPO.load(STUDENT, device="cpu")
    tc = PPO.load(T.path(stage), device="cpu")
    env = make_env(stages=[stage], noop=phase, exact=True)     # ⚠️ 走参数，不走环境变量
    o, _ = env.reset()
    w0, s0 = (int(x) for x in stage.split("-"))
    bad = tot = 0
    for _ in range(MAXSTEP):
        ot, _ = st.policy.obs_to_tensor(o)
        with th.no_grad():
            sa = int(st.policy.get_distribution(ot).distribution.probs.argmax())
            ta = int(tc.policy.get_distribution(ot).distribution.probs.argmax())
        bad += (sa != ta); tot += 1
        o, r, term, trunc, info = env.step(sa)                 # 走学生自己的动作
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (w0, s0):
            break
        if term or trunc:
            break
    env.close()
    return stage, bad, tot


def main():
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    print(f"=== 分布内 vs 在策略 的 argmax 不一致率 | {STUDENT} | {len(STAGES)} 关 ===", flush=True)
    A, B = {}, {}
    with ProcessPoolExecutor(max_workers=min(len(STAGES), WORKERS)) as pool:
        for stage, bad, tot in pool.map(indist, STAGES):
            A[stage] = (bad, tot)
    print(">>> A(分布内) 跑完，开始 B(在策略)", flush=True)
    jobs = [(s, p) for s in STAGES for p in range(NPHASE)]
    with ProcessPoolExecutor(max_workers=min(len(jobs), WORKERS)) as pool:
        for stage, bad, tot in pool.map(onpolicy, jobs, chunksize=1):
            b = B.setdefault(stage, [0, 0]); b[0] += bad; b[1] += tot

    print(f"\n{'关卡':6s} {'A 分布内':>9s} {'B 在策略':>9s} {'B−A':>8s}   {'老师分':>5s}")
    rows = []
    for s in STAGES:
        a = A[s][0] / max(A[s][1], 1) * 100
        b = B[s][0] / max(B[s][1], 1) * 100
        rows.append((b - a, s, a, b))
    for d, s, a, b in sorted(rows, reverse=True):
        print(f"{s:6s} {a:8.2f}% {b:8.2f}% {d:+7.2f}pp   {T.TEACHERS[s][1]:4d}%")
    ta = sum(A[s][0] for s in STAGES) / sum(A[s][1] for s in STAGES) * 100
    tb = sum(B[s][0] for s in STAGES) / sum(B[s][1] for s in STAGES) * 100
    print(f"\n合计   {ta:8.2f}% {tb:8.2f}% {tb-ta:+7.2f}pp")
    print("\n>>> 判读：B ≫ A ⇒ 分布漂移为主，DAgger 对症；"
          "\n    B ≈ A ⇒ 分布内就在翻 argmax，再收数据没用，该改训练目标。", flush=True)


if __name__ == "__main__":
    main()
