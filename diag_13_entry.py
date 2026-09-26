"""连打条件下 1-3 的第一次尝试：死在哪、进关时的状态是什么。

**为什么要有它。** 单关评测里 1-3 argmax 单命 0-120 相位四个模型都是 97-98%，
但连打里 v44/v45 在 1-3 死 83-85 次、v42 只有 13-24 次。共用 3 条命下一次失手最多连带 3 次死亡，
所以 v44 至少有 ~15% 的进关是第一次尝试就失败——单关评测测不到的条件下发生的。
帧栈残留这个假设以前在 v7 上 N=288 测过（不处理 / 清空 / 预热三者打平），不重做。

做法：从 1-1 开始按连打方式打（进关重抖 0-30、全程 argmax，等于连打的第一条命），
第一次死亡或离开 1-3 就停。记 1-3 第一次尝试的结局、死亡 x，和进关那一刻的 RAM：
形态 PlayerStatus($0756)、FrameCounter($09)、计时器、以及在前两关走了多少步。

用法: python diag_13_entry.py <模型逗号分隔> [每个模型局数]
"""
import os
os.environ["MARIO_REJITTER"] = "30"              # 必须在 import make_env 之前
os.environ.pop("MARIO_REJITTER_STAGES", None)
os.environ.pop("MARIO_REJITTER_ON_DEATH", None)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings; warnings.filterwarnings("ignore")
import sys, collections
from concurrent.futures import ProcessPoolExecutor

MODELS = sys.argv[1].split(",")
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
TAG = f"[FLUSH={os.environ.get('MARIO_FLUSH','0')} PRIME={os.environ.get('MARIO_PRIME','0')}]"


def run(job):
    model, seed = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env, _nes_of
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    m = PPO.load(model, device="cpu")
    env = make_env()
    env.unwrapped._e  # noqa
    o, _ = env.reset(seed=seed); n = _nes_of(env)
    # 进关重抖的随机源默认按进程时间，这里显式按 seed 区分每一局
    for w in [env] + [getattr(env, "env", None)]:
        pass
    steps, entry, prevx = 0, None, 0
    life0 = int(n.ram[0x075A])
    for _ in range(12000):
        ot, _ = m.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(m.policy.get_distribution(ot).distribution.probs.argmax())
        o, r, term, trunc, info = env.step(a); steps += 1
        ws = (info.get("world"), info.get("stage"))
        if entry is None and ws == (1, 3):
            r_ = n.ram
            entry = dict(steps=steps, status=int(r_[0x0756]), fc=int(r_[0x09]),
                         timer=int(r_[0x07F8]) * 100 + int(r_[0x07F9]) * 10 + int(r_[0x07FA]))
        if int(n.ram[0x075A]) < life0 or term or trunc:
            # ⚠️ 掉命那一步 RAM 已是复活后的状态（出生点 x=40），真实死亡位置取上一步（第一版栽过）
            x = prevx
            env.close()
            if entry is None:
                return model, "没到1-3", None, None
            return model, "1-3 第一次尝试死", entry, x
        prevx = int(n.ram[0x6D]) * 256 + int(n.ram[0x86])
        if entry is not None and ws != (1, 3):
            env.close(); return model, "1-3 一次过", entry, None
    env.close()
    return model, "步数上限", entry, None


def main():
    jobs = [(m, s) for m in MODELS for s in range(RUNS)]
    agg = collections.defaultdict(list)
    with ProcessPoolExecutor(max_workers=120) as pool:
        for model, out, entry, x in pool.map(run, jobs, chunksize=1):
            agg[model].append((out, entry, x))
    for model in MODELS:
        rs = agg[model]
        c = collections.Counter(o for o, _, _ in rs)
        reach = c["1-3 第一次尝试死"] + c["1-3 一次过"]
        print(f"\n=== {os.path.basename(model)} {TAG}：{dict(c)}")
        if reach:
            print(f"   到了 1-3 的 {reach} 局里，第一次尝试失败 {c['1-3 第一次尝试死']} 局"
                  f" = {c['1-3 第一次尝试死']/reach*100:.1f}%")
        xs = sorted(x for o, _, x in rs if o == "1-3 第一次尝试死")
        print(f"   死亡 x：{xs}")
        for label in ("1-3 一次过", "1-3 第一次尝试死"):
            es = [e for o, e, _ in rs if o == label and e]
            if es:
                st = collections.Counter(e["status"] for e in es)
                print(f"   [{label}] 进关形态 {dict(st)}  前两关步数 中位 {sorted(e['steps'] for e in es)[len(es)//2]}"
                      f"  计时器 {collections.Counter(e['timer'] for e in es).most_common(3)}")


if __name__ == "__main__":
    main()
