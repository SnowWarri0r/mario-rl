"""2-2：撞死马里奥的那条鱼，死前在画面里出现了多久？——检验"光看画面不够"。

背景：存档搜索补的硬标签学得进、却不迁移（search_22.py）。一种解释是活路依赖像素看不到的信息
（随机数、还没游进屏幕的鱼）。如果凶手大多在死前很短时间内才进画面，那就是信息不够，靠训练补不回来；
如果凶手早就在画面里待了一秒以上，信息是有的，问题在网络没学会。

做法：老师 argmax、单命、121 相位，每局换一套评测半边的随机数（同 diag_22_rng.py 的 both）。
每个智能体步（4 帧）记录 5 个敌人槽位（Enemy_Flag $0F+i、Enemy_ID $16+i、X 页 $6E+i/$87+i、Y $CF+i）
和屏幕左缘（ScreenLeft_PageLoc $071A、ScreenLeft_X_Pos $071C）。死亡时取离马里奥最近的敌人当凶手，
往回数它连续"同槽位同类型且在屏幕横向范围内"的步数。

结果（2026-10-09，mario_22rngB，121 相位单命，评测半边随机数）：通关 67，有凶手的死亡 47（灰飞鱼 25、红飞鱼 14、墨鱼 8）。
凶手死前连续在屏内 13-104 步，中位 18 步（约 1.2 秒），**没有一例 ≤ 0.53 秒**，≤1 秒的只有 8/47。
凶手几乎都从马里奥前方约 137 像素（屏幕右缘）进画面，马里奥死时多在屏幕 x≈112。
⇒ "致命的鱼在画面外/来不及看见"不成立：信息在画面里待了一秒以上，问题在策略没学会怎么躲，不在看不见。
  （没排除的：飞鱼上漂/下漂要看几帧运动才分得出，叠 4 帧 = 16 帧能看到速度，应该够。）

用法: python diag_22_visible.py <模型> [相位数] [并发]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, collections
os.environ.setdefault("OMP_NUM_THREADS", "1")
assert not os.environ.get("MARIO_RNG_POOL")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else None
NPH = int(sys.argv[2]) if len(sys.argv) > 2 else 121
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 11
NAMES = {0x07: "墨鱼", 0x0A: "灰飞鱼", 0x0B: "红飞鱼"}


def snap(r):
    sl = int(r[0x071A]) * 256 + int(r[0x071C])
    mx = int(r[0x6D]) * 256 + int(r[0x86]); my = int(r[0xCE])
    ens = []
    for i in range(5):
        if r[0x0F + i]:
            ex = int(r[0x6E + i]) * 256 + int(r[0x87 + i])
            ens.append((i, int(r[0x16 + i]), ex, int(r[0xCF + i])))
    return sl, mx, my, ens


def run(phase):
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env, _nes_of
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    m = PPO.load(MODEL, device="cpu")
    env = make_env(stages=["2-2"], noop=phase, exact=True)
    o, _ = env.reset(); n = _nes_of(env)
    pool = np.load("rng22_pool.npy")[1::2]
    row = pool[np.random.default_rng(1000 + phase).integers(len(pool))]
    n.ram[0x07A7:0x07AE] = row[:7]; n.ram[0x09] = row[7]
    hist = []
    for _ in range(3000):
        hist.append(snap(n.ram))
        ot, _ = m.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(m.policy.get_distribution(ot).distribution.probs.argmax())
        o, rew, term, trunc, info = env.step(a)
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (2, 2):
            env.close(); return phase, "通关", None
        if term or trunc:
            break
    env.close()
    sl, mx, my, ens = hist[-1]
    if not ens:
        return phase, "死", None
    d, i, eid, ex, ey = min((abs(ex - mx) + abs(ey - my), i, eid, ex, ey) for i, eid, ex, ey in ens)
    if d > 48:
        return phase, "死", ("远", eid, None, None, mx - sl)
    vis = 0
    first = None
    for sl_t, mx_t, my_t, ens_t in reversed(hist):
        e = [x for x in ens_t if x[0] == i and x[1] == eid]
        if not e or not (0 <= e[0][2] - sl_t < 256):
            break
        vis += 1; first = (e[0][2] - mx_t, e[0][3] - my_t)
    return phase, "死", (NAMES.get(eid, f"${eid:02X}"), eid, vis, first, mx - sl)


def main():
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        res = list(pool.map(run, range(NPH), chunksize=1))
    clr = sum(o == "通关" for _, o, _ in res)
    deaths = [k for _, o, k in res if o == "死" and k]
    print(f"{os.path.basename(MODEL)}：通关 {clr}/{NPH}，有凶手的死亡 {len(deaths)}")
    near = [k for k in deaths if k[0] != "远"]
    print("凶手:", dict(collections.Counter(k[0] for k in deaths)))
    vs = sorted(k[2] for k in near)
    if vs:
        print(f"凶手死前连续在屏内的步数（1 步 = 4 帧 ≈ 0.067 秒）：中位 {vs[len(vs)//2]}，分布 {vs}")
        for th_ in (4, 8, 15, 30):
            print(f"  ≤{th_} 步（≤{th_*4/60:.2f} 秒）: {sum(v <= th_ for v in vs)}/{len(vs)}")
        print("首次入屏时相对马里奥 (dx,dy)：", [k[3] for k in near][:30])
        print("死时马里奥在屏幕上的 x：", sorted(k[4] for k in near))


if __name__ == "__main__":
    main()
