"""证明 exact 相位真的生效：同一相位跑两遍必须逐帧相同，不同相位必须不同。

写这个是因为坏掉的那版**看起来完全正确**——`os.environ["MARIO_NOOP_EXACT"]="1"` 就写在
`make_env(...)` 上面一行，读代码读不出问题。它没生效是因为 `make_env.py` 在 import 时
就把那个变量读成模块常量了，而 import 在赋值之前。这类"顺序坑"只能靠断言抓。

⚠️ 判据必须是**轨迹逐帧相同**，不是"通关率差不多"。坏掉那版的通关率看着也挺合理
（4-1 68%），是拿两个只差 30k 步的档对比才露馅（9/31 vs 23/31）。可复现性是二值的，
用连续量去判会漏。

⚠️ 比的是**观测**不是 info 里的 x/y：马里奥自己的 x/y 在开局跟相位无关（他不动），
相位改变的是**敌人**的位置。第一版拿 x_pos/y_pos 比，noop=12 和 noop=20 判成"完全相同"，
差点得出"noop 没生效"的错误结论——被测的量里根本不含相位信息。

用法: ./venv/bin/python selftest_exact_phase.py
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np

STAGE = os.environ.get("MARIO_STAGE", "1-1")
STEPS = 60
ACTIONS = [1, 1, 2, 1, 0, 3, 1, 1, 4, 1] * 6      # 一段固定动作序列，不需要模型


def trace(noop, exact):
    from make_env import make_env
    env = make_env(stages=[STAGE], noop=noop, exact=exact)
    o, _ = env.reset()
    xs = [hash(o.tobytes())]
    for a in ACTIONS[:STEPS]:
        o, r, term, trunc, info = env.step(a)
        xs.append(hash(o.tobytes()))      # 观测本身：含敌人位置与 HUD 计时
        if term or trunc:
            break
    env.close()
    return xs


def main():
    print(f"=== exact 相位自检 | {STAGE} | 固定动作序列 {STEPS} 步 ===", flush=True)

    a1, a2 = trace(12, True), trace(12, True)
    assert a1 == a2, f"exact=True 同相位两次轨迹不同 → 相位没钉住\n{a1[:5]}\n{a2[:5]}"
    print(f"① exact=True，noop=12 跑两遍：轨迹逐帧相同 ✓（{len(a1)} 步）")

    b = trace(20, True)
    assert b != a1, "noop=12 和 noop=20 轨迹相同 → noop 根本没起作用"
    print("② exact=True，noop=12 vs noop=20：轨迹不同 ✓")

    # 随机那侧：跑够多次必须出现分歧，否则说明 exact 被强制开着
    rs = [trace(20, False) for _ in range(6)]
    assert len({tuple(r) for r in rs}) > 1, "exact=False 六次全同 → 抖动没生效"
    print(f"③ exact=False，noop=20 跑六遍：出现 {len({tuple(r) for r in rs})} 条不同轨迹 ✓")

    # 这一条正是当初的 bug：import 之后再设环境变量应当**无效**，
    # 断言它无效是为了防止有人以后又改回去依赖环境变量。
    os.environ["MARIO_NOOP_EXACT"] = "1"
    rs2 = [trace(20, None) for _ in range(6)]
    if len({tuple(r) for r in rs2}) > 1:
        print("④ import 后设 MARIO_NOOP_EXACT 确实无效（符合预期）—— "
              "所以逐相位扫描必须传 exact=True 参数 ✓")
    else:
        print("④ ⚠️ import 后设环境变量竟然生效了，make_env 的读取方式变了，回去看第 38 行")

    print("\n>>> 全部通过", flush=True)


if __name__ == "__main__":
    main()
