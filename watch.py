"""实时窗口看马里奥打游戏。**在你自己的终端里跑**，会弹出游戏窗口。
用法:
    ./venv/bin/python watch.py                       # 看随机瞎按
    ./venv/bin/python watch.py mario_ppo_smoke.zip    # 看某个模型打
    ./venv/bin/python watch.py checkpoints/mario_ppo_500000_steps.zip
按 Ctrl+C 退出。
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, collections
import numpy as np
import cv2
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

model_path = sys.argv[1] if len(sys.argv) > 1 else None

# render_mode='human' → nes-py 会开一个真实的游戏窗口
base = gym_super_mario_bros.make("SuperMarioBros-v0",
                                 apply_api_compatibility=True,
                                 render_mode="human")
base = JoypadSpace(base, SIMPLE_MOVEMENT)

model = None
if model_path:
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    print(f">>> 加载模型 {model_path}，按它学到的策略打")
else:
    print(">>> 没给模型，随机瞎按（训练前的样子）")


def proc(o):                       # 跟训练时一样：灰度 + 缩到 84x84
    g = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA).astype(np.uint8)


def reset():
    o = base.reset()
    o = o[0] if isinstance(o, tuple) else o
    return collections.deque([proc(o)] * 4, maxlen=4)   # 叠 4 帧


stack = reset()
try:
    while True:
        base.render()                                    # 刷新窗口
        if model is not None:
            action, _ = model.predict(np.stack(stack, 0), deterministic=True)
            action = int(action)
        else:
            action = base.action_space.sample()
        for _ in range(4):                               # 跳帧 4：一个动作维持 4 帧
            o, r, term, trunc, info = base.step(action)
            base.render()
            time.sleep(1 / 60)                            # 按真实游戏速度放慢
            if term or trunc:
                break
        stack.append(proc(o))
        if term or trunc:
            print(f"   一局结束 | 最远 x={info.get('x_pos')} | 过关={info.get('flag_get')}")
            stack = reset()
except KeyboardInterrupt:
    print("\n>>> 退出")
finally:
    base.close()
