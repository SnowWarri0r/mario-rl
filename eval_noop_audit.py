"""no-op starts 体检：同一个模型、同一关，只改开局随机空按的帧数，看通关率怎么塌。

塌得越狠，说明那个分数里"背敌人相位"的成分越多。0 帧＝历史上一直用的口径（每局从游戏刚启动的
同一状态开始，相位固定）；30 帧＝Atari benchmark 的 no-op starts 标准口径。

用法: python eval_noop_audit.py <spec> [每格局数] [并发]
  spec = teachers  各关老师在自己那关上体检（老师是数据源头，它背轨迹＝全部蒸馏数据都带这个毛病）
       = students  三个合并学生的十二关重测
       = curve22   2-2 的降级曲线（0/2/4/8/16/30/60 帧）
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

SPEC = sys.argv[1] if len(sys.argv) > 1 else "teachers"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 44
CHUNK = 5
# MARIO_DET=1 → argmax 推理。相位无关的策略用 DET 只是去掉采样噪声（2-2 是精度任务，噪声很贵）；
# 背轨迹的策略用 DET 才会虚高，那个陷阱现在不成立了。
DET = os.environ.get("MARIO_DET") == "1"
# models22/ckpt22 这两个 spec 默认测 2-2，改这个变量可以拿去测别的关
EVAL_STAGE = os.environ.get("MARIO_EVAL_STAGE", "2-2")

TEACHERS = {
    "1-1": "mario_w1c_final.zip", "1-2": "mario_w1c_final.zip",
    "1-3": "mario_13expert_final.zip", "1-4": "mario_w1c_final.zip",
    "2-1": "mario_21expert_v2.zip", "2-2": "mario_22ladder_final.zip",
    "2-3": "mario_w2land_final.zip", "2-4": "mario_w2land_final.zip",
    "3-1": "checkpoints_31expert/mario_31exp_1600000_steps.zip",
    "3-2": "mario_w3_final.zip", "3-3": "mario_w3_final.zip", "3-4": "mario_w3_final.zip",
}
STAGES = list(TEACHERS)
STUDENTS = ["mario_all12_wide.zip", "mario_dag12_warm.zip"]


def build_cells():
    """每个 cell = (标签, 模型, 关卡, noop帧数)"""
    if SPEC == "teachers":
        return [(f"老师 {st}", TEACHERS[st], st, k) for st in STAGES for k in (0, 30)]
    if SPEC == "students":
        return [(f'{os.path.basename(m).replace(".zip", "")} {st}', m, st, k)
                for m in STUDENTS for st in STAGES for k in (0, 30)]
    if SPEC == "curve22":
        return [(f"{os.path.basename(m).replace('.zip','')} @2-2", m, "2-2", k)
                for m in ("mario_22ladder_final.zip", "mario_all12_wide.zip")
                for k in (0, 2, 4, 8, 16, 30, 60)]
    if SPEC == "w1pilot":
        # World1 pilot：抖动重训的老师 vs 原老师，在 W1 四关上按 noop=30 对比
        return [(f"{'抖动重训' if 'noop' in m else '原老师  '} {st}", m, st, 30)
                for m in ("mario_w1c_final.zip", "mario_w1noop.zip")
                for st in ("1-1", "1-2", "1-3", "1-4")]
    if SPEC == "w23pilot":
        # W2/W3 抖动重训版 vs 原版，逐关比。挑老师要一关一关挑：W1 那轮重训版在 1-2 上反而更差，
        # 一刀切全换会把好老师换掉。
        pairs = {
            "2-1": ("mario_w2noop.zip", "mario_w2land_final.zip", "mario_21expert_v2.zip"),
            "2-3": ("mario_w2noop.zip", "mario_w2land_final.zip"),
            "2-4": ("mario_w2noop.zip", "mario_w2land_final.zip"),
            "3-1": ("mario_w3noop.zip", "mario_w3_final.zip",
                    "checkpoints_31expert/mario_31exp_1600000_steps.zip"),
            "3-2": ("mario_w3noop.zip", "mario_w3_final.zip"),
            "3-3": ("mario_w3noop.zip", "mario_w3_final.zip"),
            "3-4": ("mario_w3noop.zip", "mario_w3_final.zip"),
        }
        return [(f"{st} {os.path.basename(m).replace('.zip','')[:22]}", m, st, 30)
                for st, ms in pairs.items() for m in ms]
    if SPEC.startswith("robust:"):
        # robust:<a.zip,b.zip,...> → 双口径：0-30 是对外可比的主指标（Atari no-op starts 标准），
        # 0-120 是鲁棒性副指标。为什么要第二条：**训练窗口和评测窗口同为 0-30 时，
        # 「背下这 31 个相位」就是一种可行策略**——实测 champ 在 0-30 内 27/31，
        # 出了窗口（31-120）只剩 37%。主指标不动是为了可比，副指标是为了不被记忆策略刷高。
        return [(os.path.basename(p).replace(".zip", ""), p, EVAL_STAGE, k)
                for p in SPEC.split(":", 1)[1].split(",") for k in (30, 120)]
    if SPEC.startswith("models22:"):
        # models22:<a.zip,b.zip,...> → 指定几个模型在 2-2 上按 noop=0/30 各测一遍
        return [(os.path.basename(p).replace(".zip", ""), p, EVAL_STAGE, k)
                for p in SPEC.split(":", 1)[1].split(",") for k in (0, 30)]
    if SPEC.startswith("ckpt22"):
        # ckpt22:<目录> → 该目录下每个 checkpoint 在 2-2 上按 noop=30 测一遍，
        # 看"抖动下的真实通关率"在训练过程中有没有往上走（ep_rew_mean 含塑形奖励，看不出这个）
        import glob
        import re as _re
        d = SPEC.split(":", 1)[1] if ":" in SPEC else "checkpoints_22noop_gpu"
        def steps_of(path):
            return int(_re.search(r"(\d+)_steps", path).group(1))
        cells = []
        for p in sorted(glob.glob(f"{d}/*.zip"), key=steps_of):
            cells.append(("%5dk步" % (steps_of(p) // 1000), p, EVAL_STAGE, 30))
        cells.append(("起点参照", os.environ.get("MARIO_REF", "mario_22ladder_final.zip"),
                      EVAL_STAGE, 30))
        return cells
    raise SystemExit(f"未知 spec: {SPEC}")


def _find_noop(env):
    """在 wrapper 链里找 NoopReset，用来逐局指定精确相位"""
    while env is not None:
        if type(env).__name__ == "NoopReset":
            return env
        env = getattr(env, "env", None)
    return None


def run_chunk(task):
    label, model_path, stage, noop, n_eps, seed, off = task
    try:
        import torch as th; th.set_num_threads(1)
        from stable_baselines3 import PPO
        from make_env import make_env
        import wide_cnn  # noqa: F401
        model = PPO.load(model_path, device="cpu")
        env = make_env(stages=[stage], noop=noop)
        w0, s0 = int(stage[0]), int(stage[2])
        clears, xs = 0, []
        # ⚠️argmax 模式下"跑 N 局"是假的样本量：确定性模拟器 + argmax + 固定相位
        # ＝ 每局轨迹逐帧相同，结果非 0 即 100。实测 2-2 的 31 个相位每个都是纯 0% 或纯 100%。
        # 所以有效样本量是**相位数**（noop=30 时是 31），不是局数——标准误 ±9pp 而不是 ±3pp。
        # 这里改成逐个相位各跑一局：同样的算力，有效样本量大几十倍。
        # off 是这一 chunk 在本格内的起始相位编号（由 main 传入），
        # 不能用全局 seed 推——seed 跨格递增，会让每一格从不同相位开始
        phases = [(off + i) % (noop + 1) for i in range(n_eps)] if DET else None
        nr = _find_noop(env) if phases else None
        if nr is not None:
            nr.exact = True
        for i in range(n_eps):
            if nr is not None:
                nr.max_noop = phases[i]
            o, _ = env.reset()
            done, maxx, flag, w, s = False, 0, False, w0, s0
            while not done:
                a, _ = model.predict(o, deterministic=DET)
                o, r, term, trunc, info = env.step(int(a))
                done = term or trunc
                maxx = max(maxx, info.get("x_pos", 0))
                flag = flag or info.get("flag_get", False)
                w, s = info.get("world", w), info.get("stage", s)
            clears += bool(flag or (w, s) != (w0, s0))
            xs.append(maxx)
        env.close()
        return (label, stage, noop), clears, n_eps, float(np.mean(xs)), None
    except BaseException as ex:
        return (label, stage, noop), 0, n_eps, 0.0, f"{type(ex).__name__}: {ex}"


def main():
    cells = build_cells()
    tasks, seed = [], 0
    for label, m, st, k in cells:
        # argmax 模式下超过相位数的局数是纯浪费（每个相位的结果完全相同），砍到相位数
        # noop=0 也只有一个相位，argmax 下跑 300 局就是把同一局重复 300 次
        left = min(N, k + 1) if DET else N
        off = 0
        while left > 0:
            c = min(CHUNK, left)
            tasks.append((label, m, st, k, c, seed, off))
            seed += 1; off += c; left -= c
    print(f"=== no-op 体检 [{SPEC}]｜{len(cells)} 格 × {N} 局｜{WORKERS} 并发 ===", flush=True)

    agg = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for key, clears, n_eps, mx, err in pool.map(run_chunk, tasks):
            if err:
                print(f"  !! {key} 失败: {err}", flush=True); continue
            a = agg.setdefault(key, [0, 0, []]); a[0] += clears; a[1] += n_eps; a[2].append(mx)

    print("\n=== 结果 ===")
    labels = sorted({k[0] for k in agg})
    noops = sorted({k[2] for k in agg})
    header = "关卡/模型".ljust(22) + "".join(f"noop={k:<3d}".rjust(11) for k in noops)
    print(header)
    for lb in labels:
        row = lb.ljust(22)
        vals = {}
        for k in noops:
            hits = [v for kk, v in agg.items() if kk[0] == lb and kk[2] == k]
            if hits:
                c = sum(h[0] for h in hits); n = sum(h[1] for h in hits)
                vals[k] = c / n * 100
                row += f"{c/n*100:6.0f}% ({n:2d})".rjust(11)
            else:
                row += "".rjust(11)
        if 0 in vals and 30 in vals:
            row += f"   Δ{vals[30]-vals[0]:+.0f}"
        print(row)

    print("-" * len(header))
    for k in noops:                                   # 各 noop 档的总体均值
        tot = [(v[0], v[1]) for kk, v in agg.items() if kk[2] == k]
        c = sum(t[0] for t in tot); n = sum(t[1] for t in tot)
        if n:
            print(f"{('全部平均 noop=' + str(k)):22s}{c/n*100:6.1f}% ({n:4d} 局)")


if __name__ == "__main__":
    main()
