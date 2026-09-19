"""马里奥环境 + 预处理。每个 wrapper 是一个'积木'，决定 agent 看到什么。"""
import warnings; warnings.filterwarnings("ignore")
import collections
import math
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
# MARIO_PRIME=1：过关瞬间**多走几帧把栈填满真实的连续画面**，再把控制权交回策略。
# 为什么需要第三个选项：flush 和不 flush 都不对，只是错法不同——
#   不 flush ＝ 3 帧上一关的旗杆 + 1 帧新关（内容错）
#   flush   ＝ 同一帧复制四份（内容对，但**速度信息为零**）
# 后者对 2-2 这种水下关是致命的：策略靠 4 帧差分判断鱼往哪游、多快，看到"静止的鱼"就撞上去。
# 实测把老师放到一个同样冻结的栈上（存档开局），它从 86% 掉到 argmax 21%、32 帧就死。
# prime 则两头都对：帧是新关的、且彼此相邻带着运动。代价是新关开头有 n-1 步不由策略决定
# （沿用它跨关那一步的动作），发生在 x≈40 的出生点附近，那里没有威胁。
# ⚠️**结论：三种做法在 N=288 下完全打平**（不处理 21.5% / flush 19.1% / prime 18.4% 全通率），
# 名义最优是"什么都不做"。之前"flush 值 +0.9 关""符号跟着模型翻"都是 N=64/96 的噪声，已撤回。
# 留着这两个开关只为复现旧数字。为什么存档开局是致命的、过关时却无所谓：
# 过关时马里奥出生在 x≈40 周围什么都没有，三帧退化的观测不要钱；x=1408 是在 2-2 水下、
# 身边全是鱼，同样三帧就是致命的——同一个缺陷，代价取决于它发生在哪里。
PRIME_ON_STAGE_CHANGE = os.environ.get("MARIO_PRIME") == "1"
# MARIO_NOOP=k：每次 reset 后先随机空按 0~k 帧，把敌人/移动平台的相位推开（Atari 那套 no-op starts）。
# 为什么必须有：不加它，每局都从"游戏刚启动"的同一状态开始，相位固定，策略能靠背一段舞步拿分——
# 实测 2-2 无抖动 64%、抖 0-60 帧后 0/50，那 64% 全是记死的轨迹。训练和评测都要开，否则分数是假的。
NOOP_JITTER = int(os.environ.get("MARIO_NOOP", "0"))
# MARIO_NOOP_EXACT=1：空按恰好 MARIO_NOOP 帧（而不是 0~k 随机），用来逐个相位扫描
NOOP_EXACT = os.environ.get("MARIO_NOOP_EXACT") == "1"
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
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
        if stages:
            e = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0",
                                          stages=list(stages),
                                          apply_api_compatibility=True,
                                          render_mode="rgb_array")
        else:
            e = gym_super_mario_bros.make("SuperMarioBros-v0",
                                          apply_api_compatibility=True,
                                          render_mode="rgb_array")
        # MARIO_COMPLEX=1 → 12 动作。**8-4 必须用它**：正确路线要下管道，而 SIMPLE_MOVEMENT
        # 里根本没有 `down`（实测 down 只在 COMPLEX 的第 10 号位），拿 SIMPLE 训 8-4 是无解的。
        # 已核实 `SIMPLE_MOVEMENT == COMPLEX_MOVEMENT[:7]` 逐字为真，所以：
        #   · 现有 7 动作老师全部不用重训，它们的 7 维概率补 5 个 0 就能喂 12 动作学生；
        #   · 但 7 动作模型和 12 动作 env 的 action_space 对不上，不能混用，得显式开关。
        moves = COMPLEX_MOVEMENT if os.environ.get("MARIO_COMPLEX") == "1" else SIMPLE_MOVEMENT
        self._e = JoypadSpace(e, moves)                # 把 256 种按键组合砍成 7/12 个常用动作
        self.observation_space = spaces.Box(0, 255, (240, 256, 3), np.uint8)
        self.action_space = spaces.Discrete(len(moves))
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
    def __init__(self, env, max_noop=None, seed=None, exact=None):
        super().__init__(env)
        self.max_noop = NOOP_JITTER if max_noop is None else max_noop
        # exact 做成实例属性（而不是只读全局），评测时可以逐局改成不同的固定相位，
        # 不必为每个相位重建一个环境（nes-py 初始化要 ~1s）
        self.exact = NOOP_EXACT if exact is None else exact
        self.rng = np.random.default_rng(seed)

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        # MARIO_NOOP_EXACT=1 → 空按**恰好** max_noop 帧，而不是 0~max 里随机取。
        # 用来逐个相位量通关率：0-30 的平均值会把"某几个相位完全过不去"这件事抹平，
        # 而连打时进新关的相位不是均匀抽的（由穿过上一关的用时决定），抹平了就看不出系统性偏差。
        k = self.max_noop if self.exact else (
            int(self.rng.integers(0, self.max_noop + 1)) if self.max_noop else 0)
        for _ in range(k):
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
    def __init__(self, env, n=None, flush_on_stage_change=None, prime_on_stage_change=None,
                 prime_on_reset=False):
        super().__init__(env)
        self.n = n = STACK_FRAMES if n is None else n
        # prime_on_reset：**重置之后**再走 n-1 步，让栈里是真实相邻的帧而不是同一帧复制 n 份。
        # 专为存档开局而加。存档开局本身位置是真的（靠重放动作前缀走到的），
        # 坏就坏在栈：reset 把一帧复制四份＝速度信息为零，实测老师从 86% 掉到 argmax 21%、32 帧就死。
        # 用哪个动作来 prime：ArchiveStart 会把快照那一刻正在按的动作放进 info["prime_action"]，
        # 沿用它才能保住动量方向（拿 NOOP 去 prime 等于先松手减速，那是另一种失真）。
        self.prime_on_reset = prime_on_reset
        self.frames = collections.deque(maxlen=n)
        self.observation_space = spaces.Box(0, 255, (n, 84, 84), np.uint8)
        self.flush = FLUSH_ON_STAGE_CHANGE if flush_on_stage_change is None else flush_on_stage_change
        self.prime = PRIME_ON_STAGE_CHANGE if prime_on_stage_change is None else prime_on_stage_change
        assert not (self.flush and self.prime), "MARIO_FLUSH 和 MARIO_PRIME 是同一处的两种做法，只能开一个"
        self._ws = None

    def reset(self, **kw):
        # ⚠️ prime 有可能把这一局走死（存档点就在危险位置时）。走死了**必须重开**，
        # 不能带着一个 done 的环境返回——外层紧接着 step 会直接炸
        # `ValueError: cannot step in a done environment`（踩过）。
        dead = False
        for _ in range(8):
            o, info = self.env.reset(**kw)
            self.frames.clear()
            self.frames.append(o)
            if not self.prime_on_reset:
                break
            a = int(info.get("prime_action", 0))
            dead = False
            while len(self.frames) < self.n:
                o, r, term, trunc, info = self.env.step(a)
                self.frames.append(o)
                if term or trunc:
                    dead = True
                    break
            if not dead:
                break
        if dead:
            # 重试还是死：存档点本身就在"再按这个动作就撞上去"的位置上。
            # 退回不 prime 的普通重置——栈会退化，但至少返回的是个活环境。
            # 绝不能带着 done 的环境返回，外层紧接着 step 会炸。
            o, info = self.env.reset(**kw)
            self.frames.clear()
        while len(self.frames) < self.n:
            self.frames.append(o)
        self._ws = None
        return np.stack(self.frames, 0), info

    def step(self, a):
        o, r, term, trunc, info = self.env.step(a)
        self.frames.append(o)
        if self.flush or self.prime:
            ws = (info.get("world"), info.get("stage"))
            changed = self._ws is not None and ws != self._ws
            if changed and self.flush:                       # 4 帧全填新画面（速度信息归零）
                for _ in range(self.n):
                    self.frames.append(o)
            elif changed and self.prime:
                # 多走 n-1 步把栈填满**相邻的**新关画面。沿用跨关这一步的动作 a：
                # 策略当时在往右跑，继续往右跑最接近它本来会做的事。奖励照常累加，
                # 中途真结束了就停（出生点附近几乎不会，但不能不防）。
                for _ in range(self.n - 1):
                    if term or trunc:
                        break
                    o, r2, term, trunc, info = self.env.step(a)
                    self.frames.append(o)
                    r += r2
                ws = (info.get("world"), info.get("stage"))
            self._ws = ws
        return np.stack(self.frames, 0), r, term, trunc, info


def make_env(stages=None, skip=None, crop=None, noop=None, exact=None):
    """把积木叠起来：原始画面 -> 跳帧 ->（裁状态栏）-> 灰度缩小 -> 叠4帧。agent 看到 (4,84,84)。
    stages=None → 单一/完整游戏；stages=['1-1',...] → 随机选关混合训练。
    skip=跳帧数（默认跟随 MARIO_SKIP，陆地 4；水下可调 2 拿更精细的连点控制）。
    crop=是否裁顶部状态栏（默认跟随环境变量 MARIO_CROP；裁与不裁的模型不通用，得配对使用）。
    noop=开局随机空按 0~noop 个模拟器帧（默认跟随 MARIO_NOOP）。抖相位用，防止策略背轨迹。
    exact=True → 空按**恰好** noop 帧（逐相位扫描用）。⚠️ 必须走这个参数，别靠设
    `os.environ["MARIO_NOOP_EXACT"]`：那个变量在本模块 import 时就读死了（第 38 行），
    在 import 之后再设完全无效。踩过——`diag_progress.py` 以为自己在枚举 31 个确切相位，
    实际每局是在 [0,k] 里随机抽，既不可复现、相位还重复，扫出来的整张表全废。"""
    env = MarioBase(stages=stages)
    k = NOOP_JITTER if noop is None else noop
    if k:
        env = NoopReset(env, max_noop=k, exact=exact)   # 单帧粒度地抖相位，要放在跳帧之前
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


# 任意多关混训工厂：`MARIO_STAGES=4-1,4-2,4-3 ... train_world_noop.py multi`。
# 扩到 World 4-8 时不必再为每个世界写一个函数（w1/w2land/w3 那三个是历史遗留）。
# 同样走环境变量而不是闭包——forkserver 子进程是重新 import 模块拿工厂的。
def make_env_multi():
    return make_env(stages=[x for x in os.environ["MARIO_STAGES"].split(",") if x])


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


# MARIO_MAZE_NOVELTY=2 → 迷宫关额外给 (x,y) 格子新鲜度奖励（默认 0 关闭）
MAZE_NOVELTY = float(os.environ.get("MARIO_MAZE_NOVELTY", "0"))
# MARIO_MAZE_PERSIST=1 → 访问计数跨回合累计，奖励 1/sqrt(n) 衰减
MAZE_PERSIST = os.environ.get("MARIO_MAZE_PERSIST") == "1"
# MARIO_MAZE_GATES="x:ylo:yhi:bonus,..." → 已知正解的 (x,y) 路标
MAZE_GATES = tuple(tuple(float(v) for v in g.split(":"))
                   for g in os.environ.get("MARIO_MAZE_GATES", "").split(",") if g.count(":") == 3)


# --- 迷宫关奖励：只对"刷新历史最远"发钱（治 4-4/7-4/8-4 的绕圈刷分）---
class MaxXReward(gym.Wrapper):
    """把原生的 delta-x 奖励换成**单调势能**：只有超过本回合历史最远 x 才有奖励。

    为什么必须换。原生 SMB 奖励是逐帧 delta-x。迷宫城堡走错路会把你传回环的起点，
    x 从 1023 一步掉回 11——实测**那一步的奖励是 +9**（不是 -1012，也不是截断后的 -15）。
    于是绕一圈净赚 ~1012，agent 学会的是刷圈不是通关：实测 569 步里绕了 4 圈，
    整局正奖励 5723、负奖励只有 -14，`ep_rew_mean` 7.6e3 量的其实是圈数。
    这也是为什么 4-4 的 ep_rew_mean 比别的关高 3 倍——那不是学得好，是刷得多。

    换成 max-x 势能后，第二次跑同一段走廊收益为 0，想拿分只能走出没走过的路，
    也就是找到正确岔路。回退不扣分（掉坑已有 death_pen 管），所以不会因为怕扣分而不敢探索。

    ⚠️ 这个 wrapper 要放在 SkipFrame **之前**（跟 ShapeReward 一样看原始 info）。
    """

    def __init__(self, env, start_ws=None, death_pen=50.0, clear_bonus=500.0, time_pen=0.1,
                 max_gain=40, warp_drop=300, cell=16, novelty=0.0, persist=False,
                 lane_gates=()):
        super().__init__(env)
        self.start_ws = start_ws; self.death_pen = death_pen
        self.clear_bonus = clear_bonus; self.time_pen = time_pen
        # ⚠️ max_gain 不是调参，是防脏读。`x_pos` 偶尔会读出 65535（16 位下溢），
        # 无截断的势能会为这一步发 +65535。实测中招后每局奖励是双峰的：
        # 正常局 100-900，中招局 65,000 上下，ep_rew_mean 被拉到 2.3e4。
        # 原生 delta-x 奖励把每帧增量截在 ±15，一直替我们盖住了这个脏读；换成势能就露出来了。
        # 马里奥一个模拟器帧最多前进 ~6 px，40 已经很宽松，超过就是脏读，丢弃不更新 maxx。
        self.max_gain = max_gain
        # ⚠️ warp_drop：x 一步倒退这么多＝走错岔路被传回环的起点，**直接结束这一局**。
        # 光把绕圈收益压成 0 是不够的：agent 照样在环里跑，只是不赚钱了。实测 4-4 每局 650 步
        # （4-3 只要 200），第一次回卷之后的三分之二时间全在零收益地重跑同一条走廊，
        # 岔路口的尝试次数被摊薄成三分之一。结束这一局＝把这些步数换成新的一次岔路采样，
        # 同时让"走错"真的有代价（吃 death_pen）。马里奥正常一帧退不了 300，不会误伤。
        self.warp_drop = warp_drop
        # novelty>0：对**首次踏入的 (x,y) 格子**发一次性小奖，格子边长 cell（一个 tile=16px）。
        # 为什么 max-x 势能不够：它只为**水平**推进付钱。迷宫的两条岔路往往 x 相同、只差 y
        # （在哪条走廊里），势能对二者完全无差别；而往上/往下试探要花时间却不涨 x，净亏。
        # 实测 4-4 卡在第二个岔路口 5.5M 步，回卷瞬间的 y 只在 191-218 之间（不到两个 tile），
        # 也就是**它从来没试过别的高度**。(x,y) novelty 让"换条走廊"本身有收益。
        self.cell = cell; self.novelty = novelty
        # persist=True → 访问计数**跨回合**累计，奖励按 1/sqrt(次数) 衰减（count-based exploration）。
        # 为什么非要跨回合：回合内 novelty 治不了 4-4，实测它把 y 的探索跨度从 27 拉到 84
        # （机制确实生效了）却仍是 0/31。看了画面才明白——那一段有上下两条走廊，
        # 正确路线在下层，而**两条走廊 x 区间相同**：max-x 势能对二者无差别，
        # 回合内 per-cell novelty 也对称（跑上层同样在开新格子），中间没有任何梯度区分两条路。
        # 跨回合计数才打破对称：上层被走过几千遍、奖励衰减到近 0，下层始终是新的。
        # 计数按 worker 各存各的（64 个子进程不做 IPC）——每个 worker 都在重复走上层，
        # 不共享也照样能压低上层的分，够用且零通信开销。
        self.persist = persist
        self._counts = collections.Counter() if persist else None
        # lane_gates = [(x, y_lo, y_hi, bonus), ...]：**首次**在 y∈[y_lo,y_hi] 的条件下越过 x，发一次性奖励。
        # 这是把**已知正解**直接编成路标，而不是再让探索去碰——迷宫城堡的走法是公开资料：
        #   4-4：两段迷宫，第一段走**上**、第二段走**下**（fandom / strategywiki）
        #   7-4：下 → 中 → 上
        #   8-4：要下管道，必须 MARIO_COMPLEX=1 才有 down
        # 之前四种奖励 / 随机 / 816 组扰动 / 两种束搜索全败，共同点是都在**生产候选路线**；
        # 但候选根本不用搜，查一下就有，手里缺的从来只是候选而不是验证手段。
        # 形式上这跟 2-2 当年手放 checkpoints=[(2100,60)] 是同一种做法，本项目既有实践。
        self.lane_gates = sorted(lane_gates)

    def reset(self, **kw):
        out = self.env.reset(**kw)
        # ⚠️ 起点必须用"第一次看到的 x"来播种，不能用 0：马里奥开局 x 就有 40 上下，
        # 若 maxx 从 0 起，第一步的 gain 直接超过 max_gain 被判成脏读，maxx 永远推不动，
        # 整局奖励恒为负——脏读保护会把正常关卡也一起锁死。
        self._maxx = None; self._cleared = False; self._ws0 = self.start_ws
        self._prevx = None
        self._seen = set()          # 本回合已发过奖的格子（同一格一回合只发一次）
        self._gates = set()         # 本回合已领过的路标
        self.glitches = 0; self.warped = 0
        return out

    def step(self, a):
        obs, _r, term, trunc, info = self.env.step(a)
        ws = (info.get("world"), info.get("stage"))
        if self._ws0 is None and ws[0]:
            self._ws0 = ws                      # 第一步才知道自己在哪关，不必外部传
        x = info.get("x_pos", 0) or 0
        r = -self.time_pen                      # 每步小额时间成本，压住原地磨蹭
        if self._maxx is None:
            self._maxx = x                      # 播种，不发钱
        gain = x - self._maxx
        if gain > self.max_gain:
            self.glitches += 1                  # 脏读：既不发钱也不推高 maxx
        elif gain > 0:
            r += gain                           # 只为"新地方"付钱；回退给 0，不倒扣
            self._maxx = x
        if self.novelty and gain <= self.max_gain:      # 脏读的 x 不能拿去记格子
            key = (x // self.cell, (info.get("y_pos", 0) or 0) // self.cell)
            if key not in self._seen:
                self._seen.add(key)
                if self.persist:
                    self._counts[key] += 1
                    r += self.novelty / math.sqrt(self._counts[key])
                else:
                    r += self.novelty
        y = info.get("y_pos", 0) or 0
        for gi, (gx, ylo, yhi, gb) in enumerate(self.lane_gates):
            if gi not in self._gates and x > gx and ylo <= y <= yhi:
                self._gates.add(gi); r += gb
        if not self._cleared and (info.get("flag_get") or (ws[0] and self._ws0 and ws != self._ws0)):
            r += self.clear_bonus; self._cleared = True
        # 走错岔路被传回起点 → 当作一次失败收场，别让它在环里空耗
        if (self._prevx is not None and not self._cleared
                and self._prevx - x >= self.warp_drop):
            self.warped += 1; term = True
        self._prevx = x
        if (term or trunc) and not self._cleared:
            r -= self.death_pen
        return obs, r, term, trunc, info


def build_maze_env(stage, noop=None, exact=None, novelty=None, persist=None, gates=None):
    """迷宫关的链路：与 make_env 一致，只是在 SkipFrame 前插 MaxXReward。
    单独拆出带参版本，是为了让自检能钉死相位——`make_env_maze` 必须无参（SubprocVecEnv 要 pickle），
    但两种奖励下要跑同一条轨迹做对比，就得能指定 exact 相位。"""
    e = MarioBase(stages=[stage])
    k = NOOP_JITTER if noop is None else noop
    if k:
        e = NoopReset(e, max_noop=k, exact=exact)
    if STICKY_P:
        e = StickyActions(e)
    e = MaxXReward(e, novelty=MAZE_NOVELTY if novelty is None else novelty,
                   persist=MAZE_PERSIST if persist is None else persist,
                   lane_gates=MAZE_GATES if gates is None else gates)
    e = SkipFrame(e, k=SKIP_FRAMES)
    if CROP_HUD:
        e = CropHUD(e)
    e = GrayResize(e, size=84)
    e = FrameStack(e)
    return e


def make_env_maze():
    """迷宫关工厂：`MARIO_STAGE=4-4 ... train_world_noop.py maze`"""
    return build_maze_env(os.environ["MARIO_STAGE"])





def make_env_shaped():
    """通用塑形工厂：`MARIO_STAGE=7-2 MARIO_CKPTS=800:40,1400:40,... train_world_noop.py shaped`

    为什么给 7-2 补这个：7-2 打了 ~36M 步卡在 13%，死点从 x=518 铺到 3161、**没有卡点**，
    也就是"到处死"而不是"卡在一处"。我先前把它归成"跟 2-2 同病、可能得改 skip"，
    但漏了一件事——**2-2 的 84% 是带塑形拿到的，7-2 至今一直是裸奖励**。
    ShapeReward 的注释写得很清楚，它当初就是为"治水关死一片"加的：
    死一次重罚 death_pen，安全通关给 clear_bonus，让"苟着走完"比"冲一段就死"值钱。
    ⇒ 在动 skip（会让模型不通用、破坏"一个网络打完全部关卡"）之前，先把这一招补上。
    """
    stage = os.environ["MARIO_STAGE"]
    cks = [(float(x.split(":")[0]), float(x.split(":")[1]))
           for x in os.environ.get("MARIO_CKPTS", "").split(",") if ":" in x]
    w, st = (int(v) for v in stage.split("-"))
    e = MarioBase(stages=[stage])
    if NOOP_JITTER:
        e = NoopReset(e, max_noop=NOOP_JITTER)
    if STICKY_P:
        e = StickyActions(e)
    e = ShapeReward(e, start_ws=(w, st),
                    death_pen=float(os.environ.get("MARIO_DEATH_PEN", "50")),
                    clear_bonus=float(os.environ.get("MARIO_CLEAR_BONUS", "200")),
                    checkpoints=cks)
    e = SkipFrame(e, k=SKIP_FRAMES)
    if CROP_HUD:
        e = CropHUD(e)
    e = GrayResize(e, size=84)
    e = FrameStack(e)
    return e


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
        self._prime_action = 0                   # 快照那一刻正在按的动作，交给 FrameStack 去 prime

    def reset(self, **kw):
        if self._snapped:
            o, info = self.env.reset(**kw)       # 自动恢复到快照
            info = dict(info); info["prime_action"] = self._prime_action
            return o, info
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
                self._prime_action = int(pre[-1]) if len(pre) else 0
                info = dict(info); info["prime_action"] = self._prime_action
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
