"""4-4 的状态专家：给任意 4-4 状态出动作分布，供蒸馏和 DAgger 打标签。

拼接专家（demo_44_route.py）是**时序**脚本："按住右直到站上中层，再按住左直到站上底层"。
DAgger 要的是学生开到**任意状态**都能打标签，所以入口区改写成只看当前 RAM 的规则：

  x < 1500                               → 44p 的输出（前半段，第 5 页检查它每个相位都过）
  已进入底层（见下）                       → 44b 的输出（第二段专项，31/31）
  其余 x ≥ 1500：
      站/走在中层右侧（Y∈[$78,$8F]，x ≥ 1523） → 左（走回 1 格缺口掉下去）
      其他（上层平台、空中）                  → 右（跳上平台，从 x≈1596 或 1755 的洞掉到中层）

⚠️ "在不在底层"是**路径状态**，不能用某一帧的 Y 判：第一版用 Y≥$90 判，马里奥在底层一跳、
   或者走到 x>2100 的库巴桥区域，Y 就落进"中层"范围，被判成"往左"，121 个相位全死在 x=2418。
   但底层走廊顶上罩着中层砖线，进去就出不来，所以用一个单向标志：
   第一次站上底层（x≥1515、Y≥$A0、站地）起置位，x 回到 1500 以下（回卷/重开）才清零。
   ⇒ 专家是有状态的：每回合开始调 reset()，之后每一步都按顺序调 probs()。

依据：smbdis.asm 的 ProcLoopCommand（第 9 页检查要求那一帧 Player_Y==$B0 且站地）；
楼层实测 $40/$80/$B0 三层；底层入口只能从右边进（缺口左侧柱顶悬着一块砖）。

用法（打标签）: e = Expert44(); p = e.probs(obs, ram)   # obs 是 make_env 的观测，ram 是 NES RAM
"""
import os
import numpy as np

FIRST = os.environ.get("MARIO_44_FIRST", "checkpoints_mario_44p/mario_44p_7999488_steps.zip")
SECOND = os.environ.get("MARIO_44_SECOND", "mario_44b_final.zip")
RIGHT, LEFT = 1, 6


def _onehot(a, n=7):
    p = np.zeros(n, np.float32); p[a] = 1.0
    return p


class Expert44:
    def __init__(self, first=FIRST, second=SECOND):
        from stable_baselines3 import PPO
        import wide_cnn  # noqa: F401
        self.m1 = PPO.load(first, device="cpu")
        self.m2 = PPO.load(second, device="cpu")
        self.bottom = False

    def reset(self):
        self.bottom = False

    def zone(self, ram):
        x = int(ram[0x6D]) * 256 + int(ram[0x86]); y = int(ram[0xCE])
        if x < 1500:
            self.bottom = False
            return "first"
        if not self.bottom and x >= 1515 and y >= 0xA0 and int(ram[0x1D]) == 0:
            self.bottom = True
        if self.bottom:
            return "second"
        if 0x78 <= y <= 0x8F and x >= 1523:
            return "left"
        return "right"

    def _policy_probs(self, m, obs):
        import torch as th
        ot, _ = m.policy.obs_to_tensor(obs)
        with th.no_grad():
            return m.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0].astype(np.float32)

    def probs(self, obs, ram):
        z = self.zone(ram)
        if z == "first":
            return self._policy_probs(self.m1, obs)
        if z == "second":
            return self._policy_probs(self.m2, obs)
        return _onehot(LEFT if z == "left" else RIGHT)


def _selftest(phase):
    import torch as th; th.set_num_threads(1)
    from make_env import make_env, _nes_of
    e = Expert44()
    env = make_env(stages=["4-4"], noop=phase, exact=True)
    o, _ = env.reset(); n = _nes_of(env)
    e.reset()
    zones = []
    for _ in range(3000):
        z = e.zone(n.ram)
        if not zones or zones[-1] != z:
            zones.append(z)
        a = int(np.argmax(e.probs(o, n.ram)))
        o, r, term, trunc, info = env.step(a)
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (4, 4):
            return phase, True, zones
        if term or trunc:
            return phase, False, zones + [f"死于x={int(n.ram[0x6D])*256+int(n.ram[0x86])}"]
    return phase, False, zones + ["超时"]


if __name__ == "__main__":
    import sys
    from concurrent.futures import ProcessPoolExecutor
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    ok = 0
    with ProcessPoolExecutor(max_workers=min(N, 60)) as pool:
        for phase, cleared, zones in pool.map(_selftest, range(N)):
            ok += cleared
            if not cleared:
                print(f"相位 {phase}: 失败  区段 {' → '.join(zones[:12])}{' …' if len(zones) > 12 else ''}")
    print(f"状态专家自己开车：通关 {ok}/{N}")
