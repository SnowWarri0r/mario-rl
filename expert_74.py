"""7-4 的状态专家：迷宫段和库巴段各用一个网络，按马里奥 x 切换（同 expert_44 的拼接思路）。

- 迷宫段 FIRST：LoopGuard 训练的 mario_74g_a 600 万档。六个迷宫检查全对，失败几乎都死在 x≈3230 的库巴。
- 库巴段 BOSS：从库巴前存档开局专练的 mario_74boss2（每个环境存档点不同 + 每局换随机数）。
  单独拿它打全关会在迷宫段 x≈425 / 1658 死掉（专练把前面忘了），但走到库巴的局全部通关。
x ≥ SWITCH 用 BOSS，否则用 FIRST。无状态，可直接给任意状态打标签。

用法: e = Expert74(); p = e.probs(obs, ram)；自测 python expert_74.py [相位数]
"""
import os
import numpy as np

FIRST = os.environ.get("MARIO_74_FIRST", "checkpoints_mario_74g_a/mario_74g_a_6000000_steps.zip")
BOSS = os.environ.get("MARIO_74_BOSS", "checkpoints_mario_74boss2/mario_74boss2_3499776_steps.zip")
SWITCH = int(os.environ.get("MARIO_74_SWITCH", "2900"))


class Expert74:
    def __init__(self, first=FIRST, boss=BOSS):
        from stable_baselines3 import PPO
        import wide_cnn  # noqa: F401
        self.m1 = PPO.load(first, device="cpu")
        self.m2 = PPO.load(boss, device="cpu")

    def reset(self):
        pass

    def probs(self, obs, ram):
        import torch as th
        x = int(ram[0x6D]) * 256 + int(ram[0x86])
        m = self.m2 if x >= SWITCH else self.m1
        ot, _ = m.policy.obs_to_tensor(obs)
        with th.no_grad():
            return m.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0].astype(np.float32)


def _selftest(phase):
    import torch as th; th.set_num_threads(1)
    from make_env import make_env, _nes_of
    e = Expert74()
    env = make_env(stages=["7-4"], noop=phase, exact=True)
    o, _ = env.reset(); n = _nes_of(env)
    for _ in range(4000):
        o, r, term, trunc, info = env.step(int(np.argmax(e.probs(o, n.ram))))
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (7, 4):
            return phase, True, 0
        if term or trunc:
            return phase, False, int(n.ram[0x6D]) * 256 + int(n.ram[0x86])
    return phase, False, -1


if __name__ == "__main__":
    import sys
    from concurrent.futures import ProcessPoolExecutor
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    with ProcessPoolExecutor(max_workers=min(N, 60)) as pool:
        res = list(pool.map(_selftest, range(N)))
    print("失败位置:", sorted(x for _, ok, x in res if not ok))
    print(f"7-4 状态专家：通关 {sum(ok for _, ok, _ in res)}/{N}")
