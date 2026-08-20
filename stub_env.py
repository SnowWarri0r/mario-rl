"""空壳环境：只提供 observation_space / action_space 的形状，不跑 NES 模拟器。

蒸馏是纯监督学习，env 唯一的用处是让 sb3 知道输入输出形状（(4,84,84) uint8 → 7 个动作），
一步都不 step。用空壳替真环境，蒸馏这条链路就跟 nes-py / gym-super-mario-bros 完全解耦
（H20 上先蒸馏、模拟器慢慢装也不阻塞；spaces 与 make_env() 逐字一致，存出来的模型通用）。
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StubMarioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_stack=4, size=84, n_actions=7):
        self.observation_space = spaces.Box(0, 255, (n_stack, size, size), np.uint8)
        self.action_space = spaces.Discrete(n_actions)
        self._zero = np.zeros((n_stack, size, size), np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._zero, {}

    def step(self, a):
        return self._zero, 0.0, False, False, {}


def make_stub_env():
    return StubMarioEnv()
