"""马里奥环境 + 预处理。每个 wrapper 是一个'积木'，决定 agent 看到什么。"""
import warnings; warnings.filterwarnings("ignore")
import collections
import os
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

# 顶部状态栏（MARIO / 分数 / 硬币 / WORLD / TIME）占画面前 40 行，实测裁掉它播放区完整保留。
# 为什么要裁：状态栏进了观测，"打到第 5 关时的 1-2"和"单独打 1-2"在像素上就是两张不同的图
# （分数不同），确定性策略的轨迹跨不过关卡——完整游戏里平均只连过 2.5 关，逐关却有 73%。
# MARIO_CROP=1 全局打开，评测/录像脚本不用改代码。
HUD_ROWS = 40
CROP_HUD = os.environ.get("MARIO_CROP") == "1"
# MARIO_FLUSH=1：过关瞬间把叠帧缓冲清空重填。策略只在"4 帧同属一关"的输入上训过，
# 关卡交界处那几步的输入是"3 帧上一关 + 1 帧新关"，属于训练里没有的分布。
FLUSH_ON_STAGE_CHANGE = os.environ.get("MARIO_FLUSH") == "1"
# MARIO_NOOP=k：每次 reset 后先随机空按 0~k 帧，把敌人/移动平台的相位推开（Atari 那套 no-op starts）。
# 为什么必须有：不加它，每局都从"游戏刚启动"的同一状态开始，相位固定，策略能靠背一段舞步拿分——
# 实测 2-2 无抖动 64%、抖 0-60 帧后 0/50，那 64% 全是记死的轨迹。训练和评测都要开，否则分数是假的。
NOOP_JITTER = int(os.environ.get("MARIO_NOOP", "0"))
# MARIO_SKIP=k：一个动作连按几帧（默认 4）。2-2 那道一格宽的鱼缝要帧级精度，
# 每个动作硬按 4 帧可能物理上就不够细——改 2 让它能更快连点划水。
# 注意：训练和评测必须用同一个值，动作粒度变了策略就不通用。
SKIP_FRAMES = int(os.environ.get("MARIO_SKIP", "4"))
# MARIO_STACK=k：叠几帧（默认 4）。叠更多＝能看到更长一段的敌人运动轨迹，
# 对 2-2 这种"来回游的鱼"也许有用。注意改了它模型就不通用（输入通道数变了，要从零训）。
STACK_FRAMES = int(os.environ.get("MARIO_STACK", "4"))
# MARIO_STICKY=p：每步有 p 的概率忽略新动作、重复上一个动作（Machado 2018 的 sticky actions）。
# no-op starts 只扰动开局，sticky 在整个回合里持续注入扰动——我们的病灶正是"策略跟敌人逐帧锁死"，
# 这是直接治它的那味药。训练时开了，学出来的策略就不可能再依赖逐帧对齐。
STICKY_P = float(os.environ.get("MARIO_STICKY", "0"))


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


# --- 积木 A2：no-op starts。reset 后随机空按几帧，只改相位不改别的 ---
class NoopReset(gym.Wrapper):
    def __init__(self, env, max_noop=None, seed=None):
        super().__init__(env)
        self.max_noop = NOOP_JITTER if max_noop is None else max_noop
        self.rng = np.random.default_rng(seed)

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        for _ in range(int(self.rng.integers(0, self.max_noop + 1)) if self.max_noop else 0):
            o, r, term, trunc, info = self.env.step(0)
            if term or trunc:                                # 极少见：空按到死，重开一次就好
                o, info = self.env.reset(**kw)
                break
        return o, info


# --- 积木 A3：sticky actions。以概率 p 重复上一个动作 ---
class StickyActions(gym.Wrapper):
    def __init__(self, env, p=None, seed=None):
        super().__init__(env)
        self.p = STICKY_P if p is None else p
        self.rng = np.random.default_rng(seed)
        self._last_a = 0

    def reset(self, **kw):
        self._last_a = 0
        return self.env.reset(**kw)

    def step(self, a):
        if self.rng.random() < self.p:
            a = self._last_a                 # 粘住上一个动作：agent 的指令这一步不生效
        self._last_a = a
        return self.env.step(a)


# --- 积木 B2：裁掉顶部状态栏（可选）---
# 分数/命数/时间这些数字对"怎么跳过这个坑"毫无用处，但它们会随游戏进度变化，
# 等于给同一个场景配了个会变的水印，让策略学到的轨迹依赖当前分数。裁掉＝去掉这个干扰源。
class CropHUD(gym.ObservationWrapper):
    def __init__(self, env, top=HUD_ROWS):
        super().__init__(env)
        self.top = top
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(0, 255, (h - top, w, c), np.uint8)

    def observation(self, obs):
        return obs[self.top:]


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
    def __init__(self, env, n=None, flush_on_stage_change=None):
        super().__init__(env)
        self.n = n = STACK_FRAMES if n is None else n
        self.frames = collections.deque(maxlen=n)
        self.observation_space = spaces.Box(0, 255, (n, 84, 84), np.uint8)
        self.flush = FLUSH_ON_STAGE_CHANGE if flush_on_stage_change is None else flush_on_stage_change
        self._ws = None

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        for _ in range(self.n):
            self.frames.append(o)
        self._ws = None
        return np.stack(self.frames, 0), info

    def step(self, a):
        o, r, term, trunc, info = self.env.step(a)
        self.frames.append(o)
        if self.flush:
            ws = (info.get("world"), info.get("stage"))
            if self._ws is not None and ws != self._ws:      # 刚进新关卡 → 4 帧全填新画面
                for _ in range(self.n):
                    self.frames.append(o)
            self._ws = ws
        return np.stack(self.frames, 0), r, term, trunc, info


def make_env(stages=None, skip=None, crop=None, noop=None):
    """把积木叠起来：原始画面 -> 跳帧 ->（裁状态栏）-> 灰度缩小 -> 叠4帧。agent 看到 (4,84,84)。
    stages=None → 单一/完整游戏；stages=['1-1',...] → 随机选关混合训练。
    skip=跳帧数（默认跟随 MARIO_SKIP，陆地 4；水下可调 2 拿更精细的连点控制）。
    crop=是否裁顶部状态栏（默认跟随环境变量 MARIO_CROP；裁与不裁的模型不通用，得配对使用）。
    noop=开局随机空按 0~noop 个模拟器帧（默认跟随 MARIO_NOOP）。抖相位用，防止策略背轨迹。"""
    env = MarioBase(stages=stages)
    k = NOOP_JITTER if noop is None else noop
    if k:
        env = NoopReset(env, max_noop=k)     # 单帧粒度地抖相位，要放在跳帧之前
    if STICKY_P:
        env = StickyActions(env)             # 放在跳帧之前，按模拟器帧粘
    env = SkipFrame(env, k=SKIP_FRAMES if skip is None else skip)
    if CROP_HUD if crop is None else crop:
        env = CropHUD(env)
    env = GrayResize(env, size=84)
    env = FrameStack(env)
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


# 任意单关工厂：`MARIO_STAGE=2-4 ... train_world_noop.py single`。
# 走环境变量而不是闭包/partial，因为 SubprocVecEnv 的 forkserver 子进程是**重新 import 模块**
# 拿到工厂的，闭包捕获的变量传不过去（同一个机制也让 `python - <<EOF` 探测 SubprocVecEnv 会炸）。
def make_env_single():
    return make_env(stages=[os.environ["MARIO_STAGE"]])


# 单关 2-3 专家训练用：2-3 是最后一个还停在 84% 的关，两次判"保留原版"用的都是 750k 起步的粗档，
# 而 1-2 已经证明真峰值常在 10 万-40 万步之间——那一段过去从来没看过。
def make_env_stage23():
    return make_env(stages=["2-3"])


# 单关 1-2 专家训练用：熵为零那套手术在 1-2 上反而把 45% 打成 9%（地下管道关靠随机扰动脱困，
# argmax 会锁死在墙上推到超时），所以这关要拆开验：手术其实是三件事，
# 「熵归零」只是其一，另两件（低 lr + 密存档 + 按实测挑档）不该跟着一起被否掉。
def make_env_stage12():
    return make_env(stages=["1-2"])


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
    e = FrameStack(e)
    return e


# 2-2 · checkpoint 塑形版：在 x≈2100(那道 Cheep-Cheep 鱼缝)前发一次性 +60，
# 让"穿过硬点"本身有即时奖励，不必等到遥远的通关。续训已会游到 2095 的塑形专家用。
def make_env_stage22_ckpt():
    e = MarioBase(stages=["2-2"])
    e = ShapeReward(e, checkpoints=[(2100.0, 60.0)])
    e = SkipFrame(e, k=4)
    e = GrayResize(e, 84)
    e = FrameStack(e)
    return e


# 2-2 · 梯子版：单个 checkpoint 只把它拽到 ~2250 就停(信号真空)，
# 改成一排胡萝卜 2100/2400/2700/2900 各 +50，一路拽到旗杆(3161)前，治"打地鼠"。
def make_env_stage22_ladder():
    e = MarioBase(stages=["2-2"])
    e = ShapeReward(e, checkpoints=[(2100.0, 50.0), (2400.0, 50.0),
                                    (2700.0, 50.0), (2900.0, 50.0)])
    e = SkipFrame(e, k=4)
    e = GrayResize(e, 84)
    e = FrameStack(e)
    return e


# 2-2 · 梯子 + no-op starts：每局开局随机空按 0-30 帧，把鱼的相位推开。
# 不加这个，2-2 的"70% 通关"是一段跟鱼帧级锁死的舞步——抖 2 帧就腰斩、抖 30 帧只剩 4%。
# 加了它，策略再也背不了固定序列，只能真的看着鱼做决定。NoopReset 放在 ShapeReward 之前，
# 空按的那几帧不该产生奖励也不该算进 checkpoint 判定。
def make_env_stage22_ladder_noop():
    # 抖动量取 MARIO_NOOP（默认 30）。课程式训练就是逐级把它从 4 抬到 30。
    e = MarioBase(stages=["2-2"])
    if STICKY_P:
        e = StickyActions(e)
    e = NoopReset(e, max_noop=NOOP_JITTER)   # 0 就是不抖(别写 `or 30`，会把 0 悄悄变成 30)
    e = ShapeReward(e, checkpoints=[(2100.0, 50.0), (2400.0, 50.0),
                                    (2700.0, 50.0), (2900.0, 50.0)])
    e = SkipFrame(e, k=SKIP_FRAMES)
    e = GrayResize(e, 84)
    e = FrameStack(e)
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


# --- Go-Explore / Backplay 那套：直接把 agent 放到硬点前面反复练 ---
# 梯子塑形是用奖励"拽"策略过去，这个是直接从存档点开局，省掉每回合先游 2000 像素的成本。
# 模拟器状态存取藏在 6 层 wrapper 底下（JoypadSpace → TimeLimit → OrderEnforcing →
# PassiveEnvChecker → EnvCompatibility → RandomStages → SuperMarioBrosEnv 才有 _backup）。
# 单槽存档：所以做法是"重放一段动作前缀 → _backup() → 之后每次 reset 都 _restore()"，
# 重放成本（~0.9s）摊到 ROTATE 个回合上。前缀本身带不同相位，档案自带多样性。
def _find_nes(env):
    """穿过所有 wrapper 找到带 _backup 的 SuperMarioBrosEnv。
    注意 MarioBase 是 gym.Env 不是 Wrapper，它把底层环境放在 `_e` 上，只顺着 `.env` 钻会一步都下不去。"""
    node = env
    for _ in range(12):
        if hasattr(node, "_backup"):
            return node
        nxt = getattr(node, "env", None)
        if nxt is None:
            nxt = getattr(node, "_e", None)
        if nxt is None or nxt is node:
            break
        node = nxt
    raise RuntimeError("找不到能存档的 NES 环境层")


class ArchiveStart(gym.Wrapper):
    """开局就站在硬点前：重放一段动作前缀到目标位置，然后 _backup()。

    关键机制：nes_py 的 reset() 是 `if self._has_backup: self._restore()`——所以覆盖掉备份之后，
    每次正常 reset 都会落到我们的快照上，整条 wrapper 链（TimeLimit / OrderEnforcing / ShapeReward /
    FrameStack）都照常重置。不要自己绕过 reset 去 _restore，那样内层 wrapper 的回合状态不会清，
    会把 worker 搞崩（踩过：EOFError）。
    单槽存档意味着一个 env 只能守一个起点，所以"完整关卡"的回合靠**另一批环境**提供，不在回合间横跳。
    """

    def __init__(self, env, prefixes, seed=None):
        super().__init__(env)
        self.prefixes = prefixes
        self.rng = np.random.default_rng(seed)
        self._snapped = False

    def reset(self, **kw):
        if self._snapped:
            return self.env.reset(**kw)          # 自动恢复到快照
        for _ in range(8):                       # 前缀可能因相位不同走死，多试几条
            o, info = self.env.reset(**kw)
            pre = self.prefixes[int(self.rng.integers(len(self.prefixes)))]
            ok = True
            # 前缀录于 skip=4 的环境（每动作维持 4 帧），这里在 MarioBase 层重放要重复 SKIP_FRAMES 次，
            # 否则只走到目标距离的 1/4（踩过：x=484 vs 1850）
            for a in pre:
                for _ in range(SKIP_FRAMES):
                    o, r, term, trunc, info = self.env.step(int(a))
                    if term or trunc:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                _find_nes(self.env)._backup()
                self._snapped = True
                return o, info
        return self.env.reset(**kw)              # 都失败就老老实实从头开始


def make_env_stage22_archive():
    """2-2 · 存档起点 + 梯子塑形：MARIO_ARCHIVE 指向前缀文件，MARIO_ARCHIVE_P 控制存档开局的比例。"""
    path = os.environ.get("MARIO_ARCHIVE", "states22_prefixes.npz")
    p = float(os.environ.get("MARIO_ARCHIVE_P", "0.75"))
    e = MarioBase(stages=["2-2"])
    # 按环境切分：这一份 env 以概率 p 成为"硬点开局"环境，其余保持完整关卡，
    # 让策略不会只会打后半段（单槽存档没法在回合间来回切）。
    if np.random.default_rng().random() < p:
        e = ArchiveStart(e, list(np.load(path, allow_pickle=True)["prefixes"]))
    e = ShapeReward(e, checkpoints=[(2100.0, 50.0), (2400.0, 50.0),
                                    (2700.0, 50.0), (2900.0, 50.0)])
    e = SkipFrame(e, k=SKIP_FRAMES)
    e = GrayResize(e, 84)
    e = FrameStack(e)
    return e
