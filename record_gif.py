"""加载一个存档，让马里奥打一局，录成 GIF 看它现在多厉害。
用法:
    ./venv/bin/python record_gif.py mario_ppo_smoke.zip
    ./venv/bin/python record_gif.py checkpoints/mario_ppo_500000_steps.zip
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import imageio
from stable_baselines3 import PPO
from make_env import make_env, MarioBase, SkipFrame, GrayResize, FrameStack

model_path = sys.argv[1] if len(sys.argv) > 1 else "mario_ppo_smoke.zip"
out = (model_path.split("/")[-1].replace(".zip", "")) + ".gif"

# 录像要原始彩色大画面，但 agent 决策仍用预处理后的 (4,84,84)。两路并行。
raw = MarioBase()                                  # 原始彩色帧，用来录像
env = FrameStack(GrayResize(SkipFrame(raw, 4)), 4)  # agent 实际看到的

model = PPO.load(model_path, device="cpu")

# 跑 N 局随机采样，录其中冲得最远的那局（没收敛的策略，随机比确定性强）
N_EPISODES = 8
best = {"x": -1, "frames": None, "r": 0.0, "flag": False}
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    frames, total_r, maxx, flag = [], 0.0, 0, False
    for _ in range(3000):
        action, _ = model.predict(obs, deterministic=False)   # 随机采样
        obs, r, term, trunc, info = env.step(int(action))
        total_r += r
        maxx = max(maxx, info.get("x_pos", 0))
        flag = flag or info.get("flag_get", False)
        frames.append(np.array(raw.render()))                 # copy：屏幕缓冲会被原地覆盖
        if term or trunc:
            break
    print(f"   ep{ep}: x={maxx} flag={flag} r={total_r:.0f}")
    if maxx > best["x"]:
        best = {"x": maxx, "frames": frames, "r": total_r, "flag": flag}

imageio.mimsave(out, best["frames"][::2], fps=20)
print(f">>> {out}  {len(best['frames'])}帧 | 最好那局 总分{best['r']:.0f} | 最远 x={best['x']} | flag={best['flag']}")
