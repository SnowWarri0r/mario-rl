"""逐个相位扫 2-2：那个"平均 74%"底下是均匀的，还是几个相位全过不去、几个全过得去？

**为什么问这个。** v9 在 2-2 上单关 74%（noop 0-30，argmax），连打里只有 60%，差 14pp；
而 v7 单关 56%、连打 54%，几乎不差。**落差是跟着单关能力一起长出来的**，这不合直觉。
一条线索：v9 在 `noop=0` 时 2-2 是 **0%**——它在某个特定相位上完全过不去，
那个 74% 是对 0-30 各相位求的平均。

而连打时进 2-2 的相位**不是均匀抽的**：它由 agent 穿过 2-1 的用时决定，
分布可能很集中。如果集中在几个坏相位上，连打成绩就会系统性低于"对相位取平均"的单关成绩，
而且策略越尖（相位间差异越大），这个偏差越大——正好解释"落差随单关能力增长"。

这个脚本把 `MARIO_NOOP_EXACT=1` 打开，逐个固定相位量通关率，看曲线有多尖。
读法：
  · 各相位都在均值附近 → 相位不是原因，去查别的进入条件（HUD 分数/命数等）
  · 有若干相位是 0%、另一些是 100% → 尖，连打成绩就取决于到达相位落在哪儿

用法: python diag_phase_scan.py [每相位局数] [最大相位] [并发]   默认 60 / 30 / 40
      MARIO_MODELS="a.zip,b.zip" 指定模型；MARIO_EVAL_STAGE 指定关卡
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["MARIO_NOOP_EXACT"] = "1"          # 必须在 import make_env 之前设
from concurrent.futures import ProcessPoolExecutor
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MAXP = int(sys.argv[2]) if len(sys.argv) > 2 else 30
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
MODELS = os.environ.get("MARIO_MODELS", "mario_v9_wide.zip,mario_v7_wide.zip").split(",")
STAGE = os.environ.get("MARIO_EVAL_STAGE", "2-2")
DET = os.environ.get("MARIO_DET", "1") == "1"


def run(job):
    model_path, phase, n = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    m = PPO.load(model_path, device="cpu")
    env = make_env(stages=[STAGE], noop=phase)     # NOOP_EXACT 已开 → 恰好空按 phase 帧
    w0, s0 = (int(x) for x in STAGE.split("-"))
    cleared = 0
    for _ in range(n):
        o, _ = env.reset()
        done = False
        while not done:
            ot, _ = m.policy.obs_to_tensor(o)
            with th.no_grad():
                p = m.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
            a = int(np.argmax(p)) if DET else int(np.random.choice(len(p), p=p / p.sum()))
            o, r, term, trunc, info = env.step(a)
            done = term or trunc
        cleared += bool(info.get("flag_get") or (info.get("world"), info.get("stage")) != (w0, s0))
    env.close()
    return model_path, phase, cleared, n


def main():
    print(f"=== {STAGE} 逐相位扫描（{'argmax' if DET else '采样'}，每相位 {N} 局）===", flush=True)
    jobs = [(mp, ph, N) for mp in MODELS for ph in range(MAXP + 1)]
    res = {mp: {} for mp in MODELS}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for mp, ph, c, n in pool.map(run, jobs):
            res[mp][ph] = c / n * 100
    for mp in MODELS:
        r = res[mp]
        vals = [r[p] for p in range(MAXP + 1)]
        print(f"\n--- {os.path.basename(mp)} ---", flush=True)
        for p in range(MAXP + 1):
            bar = "#" * int(vals[p] / 4)
            print(f"  空按 {p:2d} 帧  {vals[p]:5.0f}%  {bar}", flush=True)
        arr = np.array(vals)
        print(f"  均值 {arr.mean():.1f}%   标准差 {arr.std():.1f}   "
              f"全灭的相位 {int((arr == 0).sum())}/{MAXP+1}   "
              f"满分的相位 {int((arr == 100).sum())}/{MAXP+1}", flush=True)
    print("\n>>> 读法：标准差大 / 有 0% 和 100% 并存 ＝ 策略是逐相位尖的，"
          "连打成绩就取决于到达相位落在哪几个上；各相位齐平 ＝ 相位不是原因", flush=True)


if __name__ == "__main__":
    main()
