"""World3 成果 montage：让 WideCNN student 依次打 3X，各录一个通关局拼成一条 GIF。"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import imageio
from stable_baselines3 import PPO
from make_env import MarioBase, SkipFrame, GrayResize, FrameStack
import wide_cnn  # noqa: 注册 WideNatureCNN 供反序列化

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_w3_wide.zip"
model = PPO.load(MODEL, device="cpu")

montage = []
for stage in ["3-1", "3-2", "3-3", "3-4"]:
    raw = MarioBase(stages=[stage])
    env = FrameStack(GrayResize(SkipFrame(raw, 4)), 4)
    start = (int(stage[0]), int(stage[2]))
    best_frames, best_x, got_clear = None, -1, False
    for ep in range(40):
        o, _ = env.reset()
        frames, maxx, flag, w, s = [], 0, False, start[0], start[1]
        done = False
        while not done:
            a, _ = model.predict(o, deterministic=False)
            o, r, term, trunc, info = env.step(int(a)); done = term or trunc
            frames.append(np.array(raw.render()))
            maxx = max(maxx, info.get("x_pos", 0)); flag = flag or info.get("flag_get", False)
            w, s = info.get("world", w), info.get("stage", s)
        cleared = flag or (w, s) != start
        if cleared:
            best_frames, got_clear = frames, True
            break
        if maxx > best_x:
            best_x, best_frames = maxx, frames
    env.close()
    print(f"{stage}: {'通关✓' if got_clear else f'最远x={best_x}'} ({len(best_frames)}帧)")
    montage += best_frames[::2]

imageio.mimsave("mario_world3.gif", montage, fps=20)
print(f">>> mario_world3.gif  共 {len(montage)} 帧")
