"""在岔路段做**受控扰动搜索**：沿用策略保命，只在一个位置强插一小段动作，看能不能改走另一条路。

为什么不能用纯随机（上一版就是这么错的）：4-4 是城堡关，中段全是岩浆和火棍，
7 个动作均匀乱按活不过一百来像素——实测从 x=900 交给随机，最远 x 中位只有 1028，
**根本没走到岔路口**。所以那次"0/400 突破"证明的是"随机策略太菜"，
不是"正确路线找不到"。我当时把它读成了后者，是过度解读。

这一版的做法：全程用策略（它能稳定活到 x≈2046），只在 x0 处强行插入 k 步固定动作，
然后交回策略。这样既保命、又能系统地试"在这里换个高度会怎样"。
扫 x0 × 动作 × k，看哪一组能避开回卷。

⚠️ x_pos 的 65535 脏读要挡（否则一次脏读伪造一个突破，已中过一次）。
⚠️ 判据：真突破必然在 x0..XGOAL 之间留下连续轨迹。

用法: python diag_fork_probe.py <模型> [关卡]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_mario_44p/mario_44p_7999488_steps.zip"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "4-4"
XGOAL = int(os.environ.get("MARIO_XGOAL", "2200"))
MAXSTEP = int(os.environ.get("MARIO_MAXSTEP", "1500"))
WORKERS = int(os.environ.get("MARIO_WORKERS", "40"))
PHASES = [int(x) for x in os.environ.get("MARIO_PHASES", "5,10,17").split(",")]
# 扫哪些位置插入。地形图显示 x≈1150-1250 有一组不同高度的方块缝，是最可疑的"选高度"闸门
X0S = [int(x) for x in os.environ.get(
    "MARIO_X0S", "900,1000,1050,1100,1150,1200,1250,1300,1400,1500,1600,1700,1800,1900").split(",")]
# 插什么动作。SIMPLE_MOVEMENT: 0 NOOP / 1 right / 2 right+A / 3 right+B / 4 right+A+B / 5 A / 6 left
ACTS = [int(a) for a in os.environ.get("MARIO_ACTS", "0,6,5,1").split(",")]
KS = [int(k) for k in os.environ.get("MARIO_KS", "4,10,20,35").split(",")]


def run(job):
    x0, act, k, phase = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    model = PPO.load(MODEL, device="cpu")
    env = make_env(stages=[STAGE], noop=phase, exact=True)
    o, _ = env.reset()
    px, best, warped, forced = None, 0, False, 0
    ytrace = []
    for _ in range(MAXSTEP):
        if px is not None and px >= x0 and forced < k:
            a = act; forced += 1                      # 强插的那一小段
        else:
            ot, _ = model.policy.obs_to_tensor(o)
            with th.no_grad():
                a = int(model.policy.get_distribution(ot).distribution.probs.argmax().cpu())
        o, r, term, trunc, info = env.step(a)
        nx, ny = int(info.get("x_pos", 0)), int(info.get("y_pos", 0))
        if px is not None and nx - px > 100:
            continue                                   # 脏读，丢弃
        if px is not None and px - nx >= 300:
            warped = True; break
        px = nx; best = max(best, nx)
        if nx >= x0:
            ytrace.append((nx, ny))
        if term or trunc or best > XGOAL:
            break
    env.close()
    return (x0, act, k, phase), best, warped, len(ytrace)


def main():
    jobs = [(x0, a, k, p) for x0 in X0S for a in ACTS for k in KS for p in PHASES]
    print(f"=== 受控扰动搜索 | {STAGE} | {len(jobs)} 组"
          f"（{len(X0S)} 个位置 × {len(ACTS)} 个动作 × {len(KS)} 个长度 × {len(PHASES)} 相位）===",
          flush=True)
    hits, res = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for key, best, warped, n in pool.map(run, jobs, chunksize=1):
            res.append((best, key, warped))
            if best > XGOAL:
                hits.append((key, best))
    res.sort(reverse=True)
    print(f"\n最远 x 排行（前 12）：")
    for best, (x0, a, k, p) in [(b, kk) for b, kk, _ in res[:12]]:
        print(f"   x0={x0:5d} 动作={a} k={k:3d} 相位={p:3d}  →  最远 {best}")
    base = max(b for b, kk, _ in res if kk[2] == min(KS)) if res else 0
    print(f"\n**突破 x>{XGOAL} 的组合：{len(hits)}**")
    for key, best in hits[:20]:
        print(f"   x0={key[0]} 动作={key[1]} k={key[2]} 相位={key[3]}  最远 {best}")
    if not hits:
        print(">>> 一组都没突破。所有扰动都还是回到同一条上层走廊 ⇒ "
              "分岔可能不在这段 x 里，或者需要的是多步组合而不是单段强插。")


if __name__ == "__main__":
    main()
