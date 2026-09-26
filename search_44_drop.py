"""4-4 第二段的底路入口：从中层跳过小柱子、落进 1 格缺口、掉到底层。用存读档局部搜出来，再从头重放验证。

**为什么是这一个动作。** 判定逻辑（smbdis.asm ProcLoopCommand）：第 9 页检查那一帧要求
Player_Y==$B0（底层）且站地。逐帧 RAM 实测：策略每个相位都在 x≈2053 于上层跳着过检查 → 回卷。
拼接地图显示底层走廊从左边走不进去（x 1411-1522 两段岩浆 + x≈1443 一根通到地面的柱子），
唯一入口是中层砖线在 x≈1522-1539 的 1 格缺口，紧贴一根立在中层上的小柱子（x 1507-1522）。
进了底层头顶一直有中层罩着，爬不回去，直走到检查点，中间只有一根火棍 —— 与攻略
"第二段三条路走最下面，底路直走，只有一根火棍"一致。

这不是盲搜：成功条件是精确的 RAM 状态（Y==$B0、站地、x∈[1520,1600]），搜索范围只有几十步。

⚠️ nes-py 只有一个存档槽，reset 靠它回关卡起点。这里接管后 _backup() 会覆盖它，
   之后只能 _restore()，不能再 reset；验证另开干净环境从头重放。

用法: python search_44_drop.py [相位逗号分隔] [每个接管点的试验次数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODEL = "checkpoints_mario_44p/mario_44p_7999488_steps.zip"
PHASES = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,5,10,15,20,25,30").split(",")]
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
TAKES = [1380, 1400, 1420, 1440]
MAXLEN = 45
Y_B0 = 0xB0


def nes_of(env):
    n = env.unwrapped._e
    while not hasattr(n, "ram"):
        n = n.env
    return n


def policy_prefix(m, env, n, x_take):
    import torch as th
    o, _ = env.reset()
    n = nes_of(env)
    steps = 0
    X = lambda: int(n.ram[0x6D]) * 256 + int(n.ram[0x86])
    while X() < x_take and steps < 2500:
        ot, _ = m.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(m.policy.get_distribution(ot).distribution.probs.argmax())
        o, r, term, trunc, info = env.step(a)
        steps += 1
        if term or trunc:
            return None, None, n
    return o, steps, n


def verify(m, phase, x_take, seq):
    """干净环境从头重放：策略开到接管点 → 执行 seq → 交回策略。报检查结果与是否通关。"""
    import torch as th
    from make_env import make_env
    env = make_env(stages=["4-4"], noop=phase, exact=True)
    o, steps, n = policy_prefix(m, env, None, x_take)
    if o is None:
        return {"ok": False, "why": "前缀里死了"}
    X = lambda: int(n.ram[0x6D]) * 256 + int(n.ram[0x86])
    mx, loop_at, check2 = 0, None, None
    life0 = int(n.ram[0x075A])
    acts = list(seq)
    for t in range(2500):
        if t < len(acts):
            a = acts[t]
        else:
            ot, _ = m.policy.obs_to_tensor(o)
            with th.no_grad():
                a = int(m.policy.get_distribution(ot).distribution.probs.argmax())
        prev = (X(), int(n.ram[0xCE]), int(n.ram[0x1D]))
        o, r, term, trunc, info = env.step(a)
        x = X(); mx = max(mx, x)
        if check2 is None and prev[0] >= 2000 and x < prev[0] - 600:
            loop_at = prev; break                       # 第 9 页检查失败回卷
        if check2 is None and prev[0] >= 2000 and x > 2090:
            check2 = prev                               # 过了检查点还没回卷
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (4, 4):
            env.close(); return {"ok": True, "cleared": True, "check2": check2, "mx": mx}
        if int(n.ram[0x075A]) < life0 or term or trunc:
            env.close(); return {"ok": check2 is not None, "cleared": False, "died_at": x, "check2": check2, "mx": mx}
    env.close()
    return {"ok": False, "cleared": False, "loop_at": loop_at, "mx": mx}


def work(job):
    phase, x_take = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401
    m = PPO.load(MODEL, device="cpu")
    env = make_env(stages=["4-4"], noop=phase, exact=True)
    o, steps, n = policy_prefix(m, env, None, x_take)
    if o is None:
        return phase, x_take, 0, []
    n._backup()                                        # 覆盖掉 reset 的存档点：此后只用 _restore
    rng = np.random.default_rng(phase * 1000 + x_take)
    X = lambda: int(n.ram[0x6D]) * 256 + int(n.ram[0x86])
    life0 = int(n.ram[0x075A])
    found = []
    for trial in range(TRIALS):
        n._restore()
        n.done = False                                 # ⚠️ 读档不清 nes-py 的 done 标志；上一次试验死了它就还是 True
        seq = []
        while len(seq) < MAXLEN:                       # 游程编码的随机动作：选一个动作按住 1-8 步
            a = int(rng.choice(7, p=[.08, .30, .22, .10, .12, .10, .08]))
            seq += [a] * int(rng.integers(1, 9))
        seq = seq[:MAXLEN]
        for k, a in enumerate(seq):
            _, _, term, trunc, _ = env.step(a)
            if term or trunc or int(n.ram[0x075A]) < life0:   # 单关环境一进入死亡动画就 done
                break
            if int(n.ram[0xCE]) == Y_B0 and int(n.ram[0x1D]) == 0 and 1520 <= X() <= 1600:
                found.append(seq[:k + 1]); break
        if len(found) >= 5:
            break
    env.close()
    res = []
    for s in found[:3]:
        res.append((s, verify(m, phase, x_take, s)))
    return phase, x_take, trial + 1, res


def main():
    jobs = [(p, x) for p in PHASES for x in TAKES]
    print(f"=== 4-4 底路入口搜索 | 相位 {PHASES} × 接管点 {TAKES} | 每格最多 {TRIALS} 次 ===", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for phase, x_take, tried, res in pool.map(work, jobs):
            print(f"\n相位 {phase:2d} 接管 x={x_take}: 试了 {tried} 次，落到底层 {len(res)} 条（最多验证 3 条）")
            for s, v in res:
                print(f"   {len(s):2d} 步 {s}\n      → 重放：{v}")
                out.append({"phase": phase, "x_take": x_take, "seq": s, "verify": v})
    json.dump(out, open("search_44_drop.json", "w"), default=str)
    ok = [r for r in out if r["verify"].get("check2")]
    cl = [r for r in out if r["verify"].get("cleared")]
    print(f"\n>>> 过了第 9 页检查的：{len(ok)} 条；通关的：{len(cl)} 条", flush=True)


if __name__ == "__main__":
    main()
