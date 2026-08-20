"""十二关逐关评估 · 多进程版：W1+W2+W3 共 12 关，每关跑 N 局（stochastic）统计通关率。

Mac 上逐关串行评测是这套流水线最慢的一环（NES 模拟器是 CPU bound）。H20 一台机 192 核，
把 (关卡, 局数) 切成小任务撒进进程池，12×30=360 局能在几分钟内跑完。
每个 worker 自己 load 一份模型、torch 线程钉成 1（否则 48 个 worker 各开 192 线程互相打架）。

用法: python eval_all12.py <model.zip> [每关局数] [并发数]      默认 30 局 / min(48, 核数)
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_all12_wide.zip"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else min(48, mp.cpu_count())
STAGES = [f"{w}-{s}" for w in (1, 2, 3) for s in (1, 2, 3, 4)]
CHUNK = 5                                                  # 每个任务跑 5 局，摊掉建环境/载模型的开销

_model = None


def _get_model():
    global _model
    if _model is None:
        import torch as th
        th.set_num_threads(1)
        from stable_baselines3 import PPO
        import wide_cnn                                     # noqa: F401  反序列化 WideNatureCNN 需要
        _model = PPO.load(MODEL, device="cpu")
    return _model


def run_chunk(task):
    try:
        return _run_chunk(task)
    except BaseException as ex:                             # 把 worker 里的异常原样带回主进程，别让它变成静默死亡
        stage, n_eps = task
        return stage, 0, n_eps, [], f"{type(ex).__name__}: {ex}"


def _run_chunk(task):
    stage, n_eps = task
    from make_env import make_env
    model = _get_model()
    env = make_env(stages=[stage])
    clears, xs = 0, []
    w0, s0 = int(stage[0]), int(stage[2])
    for _ in range(n_eps):
        o, _ = env.reset()
        done, maxx, flag, w, s = False, 0, False, w0, s0
        while not done:
            a, _ = model.predict(o, deterministic=False)   # 未必收敛的策略看 stochastic，det 会低估
            o, r, term, trunc, info = env.step(int(a))
            done = term or trunc
            maxx = max(maxx, info.get("x_pos", 0))
            flag = flag or info.get("flag_get", False)
            w, s = info.get("world", w), info.get("stage", s)
        clears += bool(flag or (w, s) != (w0, s0))          # 摸旗杆 或 关卡号变了 = 通关
        xs.append(maxx)
    env.close()
    return stage, clears, n_eps, xs, None


def main():
    tasks = []
    for st in STAGES:
        left = N
        while left > 0:
            k = min(CHUNK, left); tasks.append((st, k)); left -= k
    print(f"=== 十二关评估 {MODEL}（每关 {N} 局 stochastic，{WORKERS} 并发，{len(tasks)} 个任务）===", flush=True)

    agg = {st: [0, 0, []] for st in STAGES}
    failed = 0
    # 用 ProcessPoolExecutor 而不是 mp.Pool：worker 被硬杀(SIGKILL/段错误)时它会抛 BrokenProcessPool，
    # mp.Pool 只会静静地等一个永远不来的结果(踩过：64 worker 全死，主进程挂了 10 分钟毫无输出)。
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for stage, clears, n_eps, xs, err in pool.map(run_chunk, tasks):
            if err:
                failed += n_eps
                print(f"  !! {stage} 这块 {n_eps} 局失败: {err}", flush=True)
                continue
            a = agg[stage]; a[0] += clears; a[1] += n_eps; a[2] += xs
            print(f"  … {stage} 累计 {a[0]}/{a[1]}", flush=True)
    if failed:
        print(f"!! 共 {failed} 局没跑成，下面的数字是残缺样本，先修错误再看结论", flush=True)

    print("\n=== 逐关结果 ===")
    rates = []
    for st in STAGES:
        c, n, xs = agg[st]
        if n == 0:
            print(f"  {st}: 无有效样本"); continue
        rate = c / n * 100; rates.append(rate)
        print(f"  {st}: 通关 {c:2d}/{n} = {rate:3.0f}%   平均最远x {np.mean(xs):.0f}")
    if not rates:
        print(">>> 全部失败，没有可用结果"); return
    print(f">>> 十二关平均通关率: {np.mean(rates):.0f}%   最差关: "
          f"{STAGES[int(np.argmin(rates))]} {min(rates):.0f}%")


if __name__ == "__main__":
    main()
