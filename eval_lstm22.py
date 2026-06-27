"""2-2 LSTM 专家评测 · 带 LSTM 隐藏状态管理(RecurrentPPO 必须这样评，否则当无记忆模型评，偏低)。
标准 skip=4 干净环境(评通关率与训练用的塑形奖励无关)。
用法: ./venv/bin/python eval_lstm22.py [模型.zip] [回合数]
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
from sb3_contrib import RecurrentPPO
from make_env import make_env

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_22lstm_final.zip"
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def main():
    model = RecurrentPPO.load(MODEL, device="cpu")
    env = make_env(stages=["2-2"])
    clears = 0
    max_x = []
    for ep in range(N_EP):
        obs, _ = env.reset()
        lstm_states = None
        ep_start = np.ones((1,), dtype=bool)   # 告诉策略：新回合，清记忆
        done = False
        cleared = False
        last_x = 0
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=ep_start, deterministic=False)
            ep_start = np.zeros((1,), dtype=bool)
            obs, r, term, trunc, info = env.step(int(action))
            last_x = info.get("x_pos", last_x)
            w, s = info.get("world"), info.get("stage")
            if info.get("flag_get") or (w and (w, s) != (2, 2)):
                cleared = True
            done = term or trunc
        clears += int(cleared)
        max_x.append(last_x)
        print(f"  ep{ep+1:02d}: {'✓通关' if cleared else '✗'}  最远x={last_x}")
    env.close()
    print(f"\n2-2 LSTM 通关率: {clears}/{N_EP} = {100*clears/N_EP:.0f}%  | 平均最远x={np.mean(max_x):.0f}")


if __name__ == "__main__":
    main()
