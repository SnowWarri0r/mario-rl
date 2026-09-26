"""4-4 第二段底路的确定性示范：上平台 → 从 x≈1596 的洞掉到中层 → 往左走掉进 1 格缺口 → 底层直走过检查。

来由（见 diag_loop_ram.py / search_44_drop.py 的注释）：第 9 页检查要求那一帧站在底层（Player_Y==$B0）。
底层走廊从左边走不进去（岩浆 + 通到地面的柱子），中层在 x 1523-1539 有 1 格缺口，缺口左边的柱子
顶上悬着一块砖，所以**从左边翻过去不可能**——局部搜索 2 万次 × 28 格全是 0 就是这个原因。
入口要从右边：先到中层 x>1590，再往左走进缺口。

这里把示范拆成三段：策略前缀（到 x≥X_TAKE）→ 脚本段（按右到站上中层 → 按左到站上底层）→ 交回策略。
每段都用 RAM 判定是否达成，不看像素。

用法: python demo_44_route.py [相位逗号分隔]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor

MODEL = os.environ.get("MARIO_MODEL", "checkpoints_mario_44p/mario_44p_7999488_steps.zip")
# 脚本段之后交给谁。44p 从没走过底层走廊、也没见过 4-4 的库巴；学生打过 1-4/2-4/3-4 的火棍和库巴桥。
SUFFIX = os.environ.get("MARIO_SUFFIX", MODEL)
PHASES = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else ",".join(map(str, range(31)))).split(",")]
X_TAKE = int(os.environ.get("MARIO_XTAKE", "1500"))
RIGHT, LEFT, NOOP = 1, 6, 0


def nes_of(env):
    n = env.unwrapped._e
    while not hasattr(n, "ram"):
        n = n.env
    return n


def run(phase):
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401
    m = PPO.load(MODEL, device="cpu")
    ms = PPO.load(SUFFIX, device="cpu") if SUFFIX != MODEL else m
    env = make_env(stages=["4-4"], noop=phase, exact=True)
    o, _ = env.reset(); n = nes_of(env)
    X = lambda: int(n.ram[0x6D]) * 256 + int(n.ram[0x86])
    Y = lambda: int(n.ram[0xCE]); ST = lambda: int(n.ram[0x1D])
    acts, stage_log = [], []
    def pol(o, mm=None):
        mm = mm or m
        ot, _ = mm.policy.obs_to_tensor(o)
        with th.no_grad():
            return int(mm.policy.get_distribution(ot).distribution.probs.argmax())
    def step(a):
        nonlocal o
        acts.append(a)
        o, r, term, trunc, info = env.step(a)
        return term or trunc, info
    # 段 1：策略开到接管点
    while X() < X_TAKE:
        d, info = step(pol(o))
        if d: return phase, {"ok": False, "why": f"前缀死在 x={X()}"}, acts
        if len(acts) > 2500: return phase, {"ok": False, "why": "前缀超时"}, acts
    stage_log.append(("接管", X(), f"${Y():02X}"))
    # 段 2a：按右，直到站在中层（$80）且 x>1590
    for _ in range(200):
        d, info = step(RIGHT)
        if d: return phase, {"ok": False, "why": f"按右时死在 x={X()}", "log": stage_log}, acts
        if Y() == 0x80 and ST() == 0 and X() > 1590: break
    else:
        return phase, {"ok": False, "why": f"没落到中层 x={X()} Y=${Y():02X}", "log": stage_log}, acts
    stage_log.append(("站上中层", X(), f"${Y():02X}"))
    # 段 2b：按左，直到站在底层（$B0）
    for _ in range(200):
        d, info = step(LEFT)
        if d: return phase, {"ok": False, "why": f"按左时死在 x={X()}", "log": stage_log}, acts
        if Y() == 0xB0 and ST() == 0: break
    else:
        return phase, {"ok": False, "why": f"没掉进缺口 x={X()} Y=${Y():02X}", "log": stage_log}, acts
    stage_log.append(("站上底层", X(), f"${Y():02X}"))
    # 段 3：交回策略，看能否过第 9 页检查、能否通关
    mx, check2 = X(), None
    for _ in range(2500):
        prev = (X(), Y(), ST())
        d, info = step(pol(o, ms))
        x = X(); mx = max(mx, x)
        if prev[0] >= 1900 and x < prev[0] - 600:
            return phase, {"ok": False, "why": f"第9页检查回卷 @x={prev[0]} Y=${prev[1]:02X} state={prev[2]}", "log": stage_log}, acts
        if check2 is None and x > 2100:
            check2 = prev; stage_log.append(("过了第9页检查", prev[0], f"${prev[1]:02X}"))
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (4, 4):
            return phase, {"ok": True, "cleared": True, "log": stage_log, "steps": len(acts)}, acts
        if d:
            return phase, {"ok": check2 is not None, "cleared": False, "why": f"死在 x={x}", "log": stage_log}, acts
    return phase, {"ok": False, "why": f"超时 最远 {mx}", "log": stage_log}, acts


def main():
    print(f"    前缀 {MODEL} | 脚本段之后 {SUFFIX}")
    print(f"=== 4-4 底路示范验证 | 相位 {PHASES[0]}..{PHASES[-1]} ({len(PHASES)} 个) | 接管 x≥{X_TAKE} ===", flush=True)
    demos, n_check, n_clear = {}, 0, 0
    with ProcessPoolExecutor(max_workers=len(PHASES)) as pool:
        for phase, res, acts in pool.map(run, PHASES):
            print(f"相位 {phase:2d}: {res}", flush=True)
            n_check += any(s[0] == "过了第9页检查" for s in res.get("log", []))
            n_clear += bool(res.get("cleared"))
            if res.get("cleared"):
                demos[phase] = acts
    print(f"\n>>> 过了第 9 页检查 {n_check}/{len(PHASES)}，通关 {n_clear}/{len(PHASES)}")
    json.dump(demos, open(os.environ.get("MARIO_DEMO_OUT", "demo_44_actions.json"), "w"))
    print(f">>> 通关的动作序列存 demo_44_actions.json（{len(demos)} 条）", flush=True)


if __name__ == "__main__":
    main()
