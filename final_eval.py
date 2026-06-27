"""最终评估 + 录最好成绩 GIF。测 deterministic 和 stochastic 的通关率，
录其中最好的一局（优先真·通关）成 mario_FINAL.gif。"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import imageio
from stable_baselines3 import PPO
from make_env import make_env, MarioBase, SkipFrame, GrayResize, FrameStack

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_ppo_v2_final.zip"
model = PPO.load(MODEL, device="cpu")

# 录像要原始彩色帧，agent 决策用预处理 obs
raw = MarioBase()
env = FrameStack(GrayResize(SkipFrame(raw, 4)), 4)

best = {"score": -1, "frames": None, "x": 0, "ws": "1-1", "cleared": False}

def run(mode, n):
    det = (mode == "deterministic")
    clears = 0
    for ep in range(n):
        o, _ = env.reset()
        frames, maxx, flag, w, s = [], 0, False, 1, 1
        for _ in range(4000):
            a, _ = model.predict(o, deterministic=det)
            o, r, term, trunc, info = env.step(int(a))
            maxx = max(maxx, info.get("x_pos", 0))
            flag = flag or info.get("flag_get", False)
            w, s = info.get("world", 1), info.get("stage", 1)
            frames.append(np.array(raw.render()))
            if term or trunc:
                break
        cleared = flag or (w, s) != (1, 1)
        if cleared:
            clears += 1
        # 打分：通关优先，其次冲得远（进了1-2的用 大基数+x 体现）
        score = (10000 if cleared else 0) + (w - 1) * 5000 + (s - 1) * 3000 + maxx
        if score > best["score"]:
            best.update(score=score, frames=frames, x=maxx, ws=f"{w}-{s}", cleared=cleared)
        print(f"  {mode[:4]} ep{ep:2d}: x={maxx:4d} 到达{w}-{s} 通关={cleared}")
    print(f">>> {mode} 通关率: {clears}/{n} = {clears/n*100:.0f}%\n")
    return clears, n

print(f"=== 评估 {MODEL} ===")
dc, dn = run("deterministic", 15)
sc, sn = run("stochastic", 15)

imageio.mimsave("mario_FINAL.gif", best["frames"][::2], fps=20)
print(f">>> mario_FINAL.gif 录的是最好一局: 通关={best['cleared']} 最远x={best['x']} 到达{best['ws']} ({len(best['frames'])}帧)")
print(f">>> 总通关率  deterministic {dc}/{dn} | stochastic {sc}/{sn}")
