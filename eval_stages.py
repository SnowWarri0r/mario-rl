"""逐关评估一个模型：对 1-1/1-2/1-3/1-4 分别测通关率（stochastic）。
用法: ./venv/bin/python eval_stages.py <model.zip> [每关局数]
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
from stable_baselines3 import PPO
from make_env import make_env

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_w1_final.zip"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
STAGES = ["1-1", "1-2", "1-3", "1-4"]

model = PPO.load(MODEL, device="cpu")
print(f"=== 逐关评估 {MODEL}（每关 {N} 局，随机采样）===")
overall = []
for stage in STAGES:
    env = make_env(stages=[stage])
    clears, xs = 0, []
    for _ in range(N):
        o, _ = env.reset()
        done, maxx, flag, w, s = False, 0, False, int(stage[0]), int(stage[2])
        start = (w, s)
        while not done:
            a, _ = model.predict(o, deterministic=False)
            o, r, term, trunc, info = env.step(int(a))
            done = term or trunc
            maxx = max(maxx, info.get("x_pos", 0))
            flag = flag or info.get("flag_get", False)
            w, s = info.get("world", w), info.get("stage", s)
        cleared = flag or (w, s) != start
        clears += cleared
        xs.append(maxx)
    env.close()
    rate = clears / N * 100
    overall.append(rate)
    print(f"  {stage}: 通关 {clears:2d}/{N} = {rate:3.0f}%   平均最远x {np.mean(xs):.0f}")
print(f">>> World 1 平均通关率: {np.mean(overall):.0f}%")
