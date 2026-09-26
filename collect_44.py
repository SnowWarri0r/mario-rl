"""给学生收 4-4 蒸馏数据，标签来自 expert_44.Expert44（状态专家，119/121）。

两种开车方式：
  teacher：专家自己 argmax 开车——学生看到完整正确路线（含"往左走进 1 格缺口"这一步）。
  student：学生开车、专家打标签——DAgger，覆盖学生自己会走到的错误状态。
相位随机 0..NOOP（默认 120，和鲁棒性口径一致）。

⚠️ 专家有状态（底层单向标志），每个回合开始必须 reset()，之后每步按顺序调 probs()。

用法: python collect_44.py <teacher|student> <输出目录> [总帧数] [worker 数]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings; warnings.filterwarnings("ignore")
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODE = sys.argv[1]
OUTDIR = sys.argv[2]
TOTAL = int(sys.argv[3]) if len(sys.argv) > 3 else 150_000
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 60
STUDENT = os.environ.get("MARIO_STUDENT", "mario_v42_swa.zip")
NOOP = int(os.environ.get("MARIO_NOOP44", "120"))


def work(wid):
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env, _nes_of
    from expert_44 import Expert44
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    rng = np.random.default_rng(wid)
    ex = Expert44()
    st = PPO.load(STUDENT, device="cpu") if MODE == "student" else None
    per = TOTAL // WORKERS
    obs_b, pr_b, eps, clears = [], [], 0, 0
    while len(obs_b) < per:
        env = make_env(stages=["4-4"], noop=int(rng.integers(0, NOOP + 1)), exact=True)
        o, _ = env.reset(); n = _nes_of(env); ex.reset()
        for _ in range(3000):
            p = ex.probs(o, n.ram)
            obs_b.append(o.astype(np.uint8)); pr_b.append(p)
            if MODE == "teacher":
                a = int(np.argmax(p))
            else:
                ot, _ = st.policy.obs_to_tensor(o)
                with th.no_grad():
                    a = int(st.policy.get_distribution(ot).distribution.sample()[0])
            o, r, term, trunc, info = env.step(a)
            if info.get("flag_get") or (info.get("world"), info.get("stage")) != (4, 4):
                clears += 1; break
            if term or trunc or len(obs_b) >= per:
                break
        eps += 1
        env.close()
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez_compressed(f"{OUTDIR}/4-4_{MODE}_w{wid:03d}.npz",
                        obs=np.array(obs_b[:per], np.uint8), probs=np.array(pr_b[:per], np.float32))
    return len(obs_b[:per]), eps, clears


def main():
    print(f"=== 4-4 数据 | {MODE} 开车 | {WORKERS} worker → {OUTDIR}，共 {TOTAL:,} 帧 ===", flush=True)
    tot = eps = cl = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for n, e, c in pool.map(work, range(WORKERS)):
            tot += n; eps += e; cl += c
    print(f">>> {tot:,} 帧，{eps} 回合，开车者通关 {cl} 回合", flush=True)


if __name__ == "__main__":
    main()
