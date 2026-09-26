"""2-2 的状态专家：现有 2-2 老师 + 读 RAM 的"飞鱼避让护盾"。——⚠️ 已证否，留作记录。

**结论（2026-09-28）：所有变体都比老师自己差**（121 相位单命：老师 80；护盾 6-62）。
  - 按方向躲（鱼在下就上浮、在上就下沉）：31-62。水面附近会沉进下方的鱼里；水里惯性大，松键后还会上漂好几步。
  - 一律上浮到水面：6-10。马里奥在水面最高约 Y≈14，而 Y=40 那档飞鱼能摆到 Y=25，
    高度差 15 像素照样相撞（dbg 实录：Y=14 vs 鱼 Y=29、dx=12 → 死）。水面不是够得着的安全带。
  - 还有一层：老师本来会提前躲，护盾横插一脚打乱它的节奏。


**为什么是飞鱼。** diag_22_deaths.py 实测（121 相位、单命、argmax）：老师 41 次死亡里 31 次被飞鱼撞
（灰 20、红 11），墨鱼只有 4 次；学生 v44 75 次里 64 次是飞鱼。多半从前下方或正前方同高撞来。
**为什么能躲。** smbdis.asm 的 MoveSwimmingCheepCheep：飞鱼只会慢慢往左游（灰每帧 $40/256 像素、
红 $80/256），前两个槽位完全不上下动，其余槽位在出生高度 ±15 像素内摆动（CheepCheepOrigYPos）。
即每条飞鱼只占一条约 30 像素高的固定横带，交会时离开那条带就行。

规则：平时照老师；前方 [-8, LOOK] 像素内有飞鱼/墨鱼且 |dy| < BAND 时接管：
  鱼在同高或下方（dy ≥ 0，Y 越大越靠下）→ 右+A（往上游）；鱼在上方 → 只按右（往下沉）。
贴近水面（Y < TOP）时不能再往上，改为往下沉。

用法：e = Expert22(); p = e.probs(obs, ram)；自检 python expert_22.py [相位数]
"""
import os
import numpy as np

TEACHER = os.environ.get("MARIO_22_TEACHER", "mario_22robust.zip")
LOOK = int(os.environ.get("MARIO_22_LOOK", "56"))
BAND = int(os.environ.get("MARIO_22_BAND", "22"))
TOP = int(os.environ.get("MARIO_22_TOP", "40"))
# SIMPLE_MOVEMENT：0 无 / 1 右 / 2 右+A / 5 A / 6 左。
# ⚠️ 第一版躲避时一直按右（UP=2、SINK=1），121 相位从老师的 80 掉到 36：水里按右会加速，
#    等于迎着往左游的飞鱼冲过去。
UP = int(os.environ.get("MARIO_22_UP", "5"))
SINK = int(os.environ.get("MARIO_22_SINK", "0"))
FISH = (0x07, 0x0A, 0x0B)                        # 墨鱼、灰飞鱼、红飞鱼
SHIELD_MODE = os.environ.get("MARIO_22_MODE", "up")   # up：有威胁就上浮；dir：旧的按方向躲（已证否）


def _onehot(a, n=7):
    p = np.zeros(n, np.float32); p[a] = 1.0
    return p


def threat(ram):
    """返回最近的威胁 (dx, dy) 或 None。dx>0 表示在马里奥前方，dy>0 表示在下方。"""
    mx = int(ram[0x6D]) * 256 + int(ram[0x86]); my = int(ram[0xCE])
    best = None
    for i in range(5):
        if ram[0x0F + i] and int(ram[0x16 + i]) in FISH:
            dx = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) - mx
            dy = int(ram[0xCF + i]) - my
            if -8 <= dx <= LOOK and abs(dy) < BAND and (best is None or dx < best[0]):
                best = (dx, dy)
    return best, my


class Expert22:
    def __init__(self, teacher=TEACHER):
        from stable_baselines3 import PPO
        import wide_cnn  # noqa: F401
        self.m = PPO.load(teacher, device="cpu")

    def reset(self):
        pass

    def shield(self, ram):
        t, my = threat(ram)
        if t is None:
            return None
        # 飞鱼出生高度只有 Y=40..168 八档、摆动 ±15，121 局里 Y<24 一条都没出现过；墨鱼往上游
        # 不低于 Y=$20（BlooperSwim 的 cmp #$20）。马里奥 16 像素高，Y≤8 时两种敌人都碰不到。
        # ⇒ 有威胁就一律往上浮到水面，不判断鱼在哪边。
        # ⚠️ 前两版"鱼在上方就往下沉 / 贴近水面改成往下沉"是错的：在水面附近等于沉进下方的鱼里。
        return UP if SHIELD_MODE == "up" else (UP if (t[1] >= 0 and my >= TOP) else SINK)

    def probs(self, obs, ram):
        a = self.shield(ram)
        if a is not None:
            return _onehot(a)
        import torch as th
        ot, _ = self.m.policy.obs_to_tensor(obs)
        with th.no_grad():
            return self.m.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0].astype(np.float32)


def _selftest(phase):
    import torch as th; th.set_num_threads(1)
    from make_env import make_env, _nes_of
    e = Expert22()
    env = make_env(stages=["2-2"], noop=phase, exact=True)
    o, _ = env.reset(); n = _nes_of(env)
    over = steps = 0
    for _ in range(3000):
        p = e.probs(o, n.ram)
        over += e.shield(n.ram) is not None; steps += 1
        o, r, term, trunc, info = env.step(int(np.argmax(p)))
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (2, 2):
            return phase, True, over, steps, None
        if term or trunc:
            return phase, False, over, steps, int(n.ram[0x6D]) * 256 + int(n.ram[0x86])
    return phase, False, over, steps, "超时"


if __name__ == "__main__":
    import sys
    from concurrent.futures import ProcessPoolExecutor
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 121
    ok, ov, st, dx = 0, 0, 0, []
    with ProcessPoolExecutor(max_workers=min(N, 80)) as pool:
        for phase, cleared, over, steps, died in pool.map(_selftest, range(N)):
            ok += cleared; ov += over; st += steps
            if died is not None: dx.append(died)
    print(f"MODE={SHIELD_MODE} UP={UP} SINK={SINK} LOOK={LOOK} BAND={BAND} TOP={TOP}：通关 {ok}/{N}，护盾接管 {ov/st*100:.1f}% 的步数，"
          f"死亡位置 {sorted(d for d in dx if isinstance(d, int))[:12]}…")
