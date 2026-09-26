"""2-2 的死因诊断：每次死亡那一刻，马里奥在哪、离它最近的敌人是什么、在哪个方向。

**为什么要做这个，而不是继续调奖励。** 4-4 的教训是：先读出游戏自己怎么判，再设计办法。
2-2 上已经有八条路证否，而死亡位置是"到处都死"（121 个相位里 1200-2800 分散），
单看位置给不出处方。死因（被墨鱼追上 / 撞飞鱼 / 超时）和相对方位（敌人从上方压下来还是正面撞）
才决定该怎么躲。

墨鱼的运动（smbdis.asm 的 ProcSwimmingB / ChkNearPlayer）：下沉时一直跟到马里奥上方 16 像素
才转头往上游，代码里**没有**绝对的最低高度——"贴海底就安全"没有代码支持。

RAM（smbdis.asm 的 '=' 定义）：敌人 5 个槽位，Enemy_Flag $0F+i、Enemy_ID $16+i、
Enemy_X $87+i、Enemy_PageLoc $6E+i、Enemy_Y $CF+i。类型码：墨鱼 $07，灰/红飞鱼 $0A/$0B。

用法: python diag_22_deaths.py <模型逗号分隔> [相位数] [并发]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, collections
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODELS = sys.argv[1].split(",")
NPH = int(sys.argv[2]) if len(sys.argv) > 2 else 121
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 80
NAMES = {0x07: "墨鱼", 0x0A: "灰飞鱼", 0x0B: "红飞鱼"}


def run(job):
    model, phase = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env, _nes_of
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    m = PPO.load(model, device="cpu")
    env = make_env(stages=["2-2"], noop=phase, exact=True)
    o, _ = env.reset(); n = _nes_of(env)
    last = None
    for _ in range(3000):
        r = n.ram
        mx = int(r[0x6D]) * 256 + int(r[0x86]); my = int(r[0xCE])
        ens = []
        for i in range(5):
            if r[0x0F + i]:
                ex = int(r[0x6E + i]) * 256 + int(r[0x87 + i]); ey = int(r[0xCF + i])
                ens.append((abs(ex - mx) + abs(ey - my), int(r[0x16 + i]), ex - mx, ey - my))
        last = (mx, my, int(r[0x07F8]) * 100 + int(r[0x07F9]) * 10 + int(r[0x07FA]), sorted(ens)[:1])
        ot, _ = m.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(m.policy.get_distribution(ot).distribution.probs.argmax())
        o, rew, term, trunc, info = env.step(a)
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (2, 2):
            env.close(); return model, phase, "通关", last
        if term or trunc:
            env.close(); return model, phase, "死", last
    env.close()
    return model, phase, "步数上限", last


def main():
    jobs = [(m, p) for m in MODELS for p in range(NPH)]
    res = collections.defaultdict(list)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for model, phase, outcome, last in pool.map(run, jobs, chunksize=1):
            res[model].append((outcome, last))
    for model in MODELS:
        rs = res[model]
        clr = sum(o == "通关" for o, _ in rs)
        deaths = [l for o, l in rs if o == "死"]
        print(f"\n=== {os.path.basename(model)}：通关 {clr}/{len(rs)}，死 {len(deaths)} ===")
        cause = collections.Counter(); rel = collections.Counter(); ybin = collections.Counter()
        for mx, my, t, near in deaths:
            if t <= 1:
                cause["超时"] += 1; continue
            if not near or near[0][0] > 48:
                cause["身边 48px 内没有敌人"] += 1; continue
            d, eid, dx, dy = near[0]
            nm = NAMES.get(eid, f"其他${eid:02X}")
            cause[nm] += 1
            vert = "上方" if dy < -8 else ("下方" if dy > 8 else "同高")
            horz = "前方" if dx > 4 else ("后方" if dx < -4 else "正对")
            rel[f"{nm}·{vert}{horz}"] += 1
            ybin[my // 32 * 32] += 1
        print("  死因:", dict(cause.most_common()))
        print("  相对方位(敌人相对马里奥):", dict(rel.most_common(8)))
        print("  死亡时马里奥 Y(每32一档，越大越靠下):", dict(sorted(ybin.items())))


if __name__ == "__main__":
    main()
