"""马里奥环境 + 预处理。每个 wrapper 是一个'积木'，决定 agent 看到什么。"""
import warnings; warnings.filterwarnings("ignore")
import collections
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces


# --- 积木 A：把老的 gym 马里奥，翻译成 sb3 要的 gymnasium 接口 ---
# 马里奥库是几年前的 gym 写的，sb3 只认新的 gymnasium。这层就是个翻译官。
class MarioBase(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, stages=None):
        # stages=None → 默认完整游戏（从 1-1 顺序打）；
        # 给 stages 列表（如 ['1-1','1-2','1-3','1-4']）→ 每次 reset 随机选一关（路线A 混合训练）
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        if stages:
            e = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0",
                                          stages=list(stages),
                                          apply_api_compatibility=True,
                                          render_mode="rgb_array")
        else:
            e = gym_super_mario_bros.make("SuperMarioBros-v0",
                                          apply_api_compatibility=True,
                                          render_mode="rgb_array")
        self._e = JoypadSpace(e, SIMPLE_MOVEMENT)      # 把 256 种按键组合，砍成 7 个常用动作
        self.observation_space = spaces.Box(0, 255, (240, 256, 3), np.uint8)
        self.action_space = spaces.Discrete(len(SIMPLE_MOVEMENT))
        self._last = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        o = self._e.reset()
        o = o[0] if isinstance(o, tuple) else o
        self._last = np.asarray(o, np.uint8)
        return self._last, {}

    def step(self, a):
        o, r, term, trunc, info = self._e.step(int(a))
        self._last = np.asarray(o, np.uint8)
        return self._last, float(r), bool(term), bool(trunc), info

    def render(self):
        return self._last

    def close(self):
        self._e.close()


# --- 积木 B：跳帧。一个动作连按 k 帧 ---
# 游戏每秒 60 帧，但你按一下方向键也不会只持续 1/60 秒。让 agent 每决策一次就维持 4 帧，
# 既贴近真人操作，又把要决策的次数砍到 1/4，学得快很多。这 4 帧的奖励加总。
class SkipFrame(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k

    def step(self, a):
        total = 0.0
        term = trunc = False
        for _ in range(self.k):
            o, r, term, trunc, info = self.env.step(a)
            total += r
            if term or trunc:
                break
        return o, total, term, trunc, info


# --- 积木 C：转灰度 + 缩小到 84x84 ---
# 彩色 240x256 对 CNN 太重，颜色对'往右冲'也没用。压成灰度小图，信息够用、算得快。
class GrayResize(gym.ObservationWrapper):
    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(0, 255, (size, size), np.uint8)

    def observation(self, obs):
        g = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.uint8)


# --- 积木 D：叠帧。把最近 4 张摞成 (4,84,84) ---
# 单张静止图看不出马里奥在往哪动、速度多快。摞 4 张连续帧，agent 就能'看出运动'。
class FrameStack(gym.Wrapper):
    def __init__(self, env, n=4):
        super().__init__(env)
        self.n = n
        self.frames = collections.deque(maxlen=n)
        self.observation_space = spaces.Box(0, 255, (n, 84, 84), np.uint8)

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        for _ in range(self.n):
            self.frames.append(o)
        return np.stack(self.frames, 0), info

    def step(self, a):
        o, r, term, trunc, info = self.env.step(a)
        self.frames.append(o)
        return np.stack(self.frames, 0), r, term, trunc, info


def make_env(stages=None, skip=4):
    """把 4 块积木叠起来：原始画面 -> 跳帧 -> 灰度缩小 -> 叠4帧。最终 agent 看到 (4,84,84)。
    stages=None → 单一/完整游戏；stages=['1-1',...] → 随机选关混合训练。
    skip=跳帧数（陆地 4；水下可调 2 拿更精细的连点控制）。"""
    env = MarioBase(stages=stages)
    env = SkipFrame(env, k=skip)
    env = GrayResize(env, size=84)
    env = FrameStack(env, n=4)
    return env


# World 1 混合训练用：4 关随机采样。SubprocVecEnv 需要可 pickle 的顶层函数，所以单独定义。
def make_env_world1():
    return make_env(stages=["1-1", "1-2", "1-3", "1-4"])


# 课程加权：1-3 在列表里重复 → 采样占 2/5≈40%, 集中火力攻钉子户; 其余各 20%。
# RandomStages 每次 reset 从列表里均匀挑, 重复 = 提高权重。
def make_env_world1_c13():
    return make_env(stages=["1-1", "1-2", "1-3", "1-3", "1-4"])


# 单关 1-3 专家训练用
def make_env_stage13():
    return make_env(stages=["1-3"])


# World 2 陆地关混训：2-1/2-3/2-4 随机采样(2-2 水关另有梯子专家)。跟 World 1 同套路。
def make_env_world2_land():
    return make_env(stages=["2-1", "2-3", "2-4"])


# 2-1 钉子户专家训练用(混训卡 x≈2066，单独补)
def make_env_stage21():
    return make_env(stages=["2-1"])


# World 3 混训：夜晚世界 4 关(3-1/3-2/3-3/3-4)随机采样，无水关，全混。预判 3-3 athletic 是钉子户。
def make_env_world3():
    return make_env(stages=["3-1", "3-2", "3-3", "3-4"])


# 3-3 夜晚 athletic 钉子户专家备用
def make_env_stage33():
    return make_env(stages=["3-3"])


# 3-1 钉子户专家训练用(混训卡 x≈2224，单独补)
def make_env_stage31():
    return make_env(stages=["3-1"])


# 单关 2-2 水下专家训练用
def make_env_stage22():
    return make_env(stages=["2-2"])


# 2-2 水下专家 · 精细控制版（skip=2，让它能更快连点 A 划水）
def make_env_stage22_fine():
    return make_env(stages=["2-2"], skip=2)


# --- 奖励塑形：让"安全通关"远比"冲一段就死"值钱（治水关"死一片"）---
class ShapeReward(gym.Wrapper):
    def __init__(self, env, start_ws=(2, 2), death_pen=50.0, clear_bonus=200.0, checkpoints=None):
        super().__init__(env)
        self.start_ws = start_ws; self.death_pen = death_pen; self.clear_bonus = clear_bonus
        # checkpoints=[(x, bonus), ...]：马里奥首次冲过某个 x，立刻发一次性奖励。
        # 治"通关奖太远够不着"：把胡萝卜挂到硬点(x≈2100 鱼缝)前面，给穿缝的即时梯度信号。
        self.checkpoints = sorted(checkpoints or [])

    def reset(self, **kw):
        out = self.env.reset(**kw); self._cleared = False
        self._hit = set()                                      # 本回合已领过的 checkpoint
        return out

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        ws = (info.get("world"), info.get("stage"))
        x = info.get("x_pos", 0) or 0
        for cx, cb in self.checkpoints:                        # 过线即奖，每个每回合只发一次
            if cx not in self._hit and x > cx:
                r += cb; self._hit.add(cx)
        if not self._cleared and (info.get("flag_get") or (ws[0] and ws != self.start_ws)):
            r += self.clear_bonus; self._cleared = True       # 真·通关 → 大奖(直接奖励目标)
        if (term or trunc) and not self._cleared:
            r -= self.death_pen                                # 半路送死 → 重罚
        return obs, r, term, trunc, info


def make_env_stage22_shaped():
    # MarioBase → ShapeReward(看原始 r+info) → SkipFrame → GrayResize → FrameStack
    e = MarioBase(stages=["2-2"])
    e = ShapeReward(e)
    e = SkipFrame(e, k=4)
    e = GrayResize(e, 84)
    e = FrameStack(e, 4)
    return e


# 2-2 · checkpoint 塑形版：在 x≈2100(那道 Cheep-Cheep 鱼缝)前发一次性 +60，
# 让"穿过硬点"本身有即时奖励，不必等到遥远的通关。续训已会游到 2095 的塑形专家用。
def make_env_stage22_ckpt():
    e = MarioBase(stages=["2-2"])
    e = ShapeReward(e, checkpoints=[(2100.0, 60.0)])
    e = SkipFrame(e, k=4)
    e = GrayResize(e, 84)
    e = FrameStack(e, 4)
    return e


# 2-2 · 梯子版：单个 checkpoint 只把它拽到 ~2250 就停(信号真空)，
# 改成一排胡萝卜 2100/2400/2700/2900 各 +50，一路拽到旗杆(3161)前，治"打地鼠"。
def make_env_stage22_ladder():
    e = MarioBase(stages=["2-2"])
    e = ShapeReward(e, checkpoints=[(2100.0, 50.0), (2400.0, 50.0),
                                    (2700.0, 50.0), (2900.0, 50.0)])
    e = SkipFrame(e, k=4)
    e = GrayResize(e, 84)
    e = FrameStack(e, 4)
    return e


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = make_env()
    check_env(env, warn=True)          # sb3 自检：接口合不合规
    o, _ = env.reset()
    print("reset obs shape:", o.shape, o.dtype)
    o, r, term, trunc, info = env.step(env.action_space.sample())
    print("step obs shape :", o.shape, "| reward:", r, "| mario x:", info.get("x_pos"))
    print("ENV OK")
