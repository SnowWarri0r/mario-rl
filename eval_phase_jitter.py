"""相位抖动评测：单关从头开局，但在策略接手前先随机塞 k 帧 NOOP（k∈[0,JITTER]）。

为什么要这么测：单关评测每局都从"游戏刚启动"的同一状态开始，敌人和移动平台的相位每次一样，
策略只要背下这段舞步就能过；真实连打时进关的相位不一样，策略必须真的反应。
塞几帧 NOOP 只改相位，地形/形态/时间/模型全不变——单变量隔离"背舞步"和"会反应"。

用法: python eval_phase_jitter.py <model.zip> [每关局数] [抖动上限帧] [并发]
      抖动上限 0 = 退化成普通单关评测（做对照组用）
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_all12_wide.zip"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
JITTER = int(sys.argv[3]) if len(sys.argv) > 3 else 60
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 40
STAGES = ["1-1", "1-2", "1-3", "1-4", "2-1", "2-2", "2-3", "2-4", "3-1", "3-2", "3-3", "3-4"]
CHUNK = 5


def run_chunk(task):
    stage, n_eps, seed = task
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    model = PPO.load(MODEL, device="cpu")
    env = make_env(stages=[stage])
    rng = np.random.default_rng(seed)
    w0, s0 = int(stage[0]), int(stage[2])
    clears, xs = 0, []
    for _ in range(n_eps):
        o, _ = env.reset()
        k = int(rng.integers(0, JITTER + 1)) if JITTER else 0
        term = trunc = False
        for _ in range(k):                                  # 只按 NOOP 空转，把敌人相位推开
            o, r, term, trunc, info = env.step(0)
            if term or trunc:
                break
        maxx, flag, w, s = 0, False, w0, s0
        done = term or trunc
        while not done:
            a, _ = model.predict(o, deterministic=False)
            o, r, term, trunc, info = env.step(int(a))
            done = term or trunc
            maxx = max(maxx, info.get("x_pos", 0))
            flag = flag or info.get("flag_get", False)
            w, s = info.get("world", w), info.get("stage", s)
        clears += bool(flag or (w, s) != (w0, s0))
        xs.append(maxx)
    env.close()
    return stage, clears, n_eps, xs


def main():
    tasks, seed = [], 0
    for st in STAGES:
        left = N
        while left > 0:
            k = min(CHUNK, left); tasks.append((st, k, seed)); seed += 1; left -= k
    print(f"=== 相位抖动评测 {MODEL}｜每关 {N} 局｜开局随机 NOOP 0-{JITTER} 帧｜{WORKERS} 并发 ===", flush=True)
    agg = {st: [0, 0, []] for st in STAGES}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for stage, clears, n_eps, xs in pool.map(run_chunk, tasks):
            a = agg[stage]; a[0] += clears; a[1] += n_eps; a[2] += xs
    print("\n=== 逐关结果 ===")
    rates = []
    for st in STAGES:
        c, n, xs = agg[st]
        rates.append(c / n * 100)
        print(f"  {st}: 通关 {c:3d}/{n} = {c/n*100:3.0f}%   平均最远x {np.mean(xs):.0f}")
    print(f">>> 十二关平均: {np.mean(rates):.1f}%")


if __name__ == "__main__":
    main()
