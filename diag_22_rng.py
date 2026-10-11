"""2-2 老师连打掉分，是不是因为游戏随机数没见过？不训练，只换随机数状态再测一遍。

依据（smbdis.asm）：随机数是 RAM $07A7 起 7 字节的移位寄存器，冷启动设种子 $A5，之后每帧推进一次
（NMI 里紧跟 `inc FrameCounter`，暂停时也推进）。飞鱼出生读它决定上漂/下漂，墨鱼读它决定何时转向。
单关环境每次进 2-2 的随机数都一样，开局空按 0-120 帧只是在同一条序列上挪几格；
连打进 2-2 时它取决于开机以来的总帧数，老师基本没见过。

三组，都是 121 个相位、单命、argmax：
  base  不动
  rng   reset 后把随机数换成"真机跑出来的"另一个状态，帧计数器不动
  both  随机数和帧计数器 $09 一起换成同一帧的值（等于"晚到了 K 帧"，最接近连打）
随机数状态不是按公式推，而是让模拟器空跑 20 万帧沿途采样（POOL）：实测 nes-py 单步之间
寄存器并不总是整体转一位，手写公式对不上真机，采样才保证每个状态游戏自己真能走到。
判读：rng/both 掉到连打进关的 43.5% 附近 → 病根是随机数；和 base 差不多 → 不是。

结果（2026-10-09，22robust，随机数池 2062 个）：base 80/121=66.1%、rng 50/121=41.3%、both 57/121=47.1%，
后两者和连打进关的 43.5% 对上（base vs 换随机数差约 3σ）。base→both 逐相位：过→过 33、过→死 47、死→过 24、死→死 17，
同一个开局时机换了随机数就大面积翻转 → 老师背的是这一个随机数状态下的鱼群，病根坐实。

续训试点（2026-10-09，未过预设标准）：从 22robust 热启动 800 万步，每次 reset 从池的偶数行写随机数
（make_env.RandomizeRNG），A 只换随机数、B 再加粘滞 0.25，lr 1e-4 / ent 0.01 / 相位 0-120。
留出奇数行挑档（老师 50/121）：A 最好 65（350 万步，档间 15-65 乱跳）、B 最好 67（750 万步，后 6 档 52-67）。
连打第一次进 2-2（各 300 局）：老师 93/227=41.0%、A 106/244=43.4%、B 119/236=50.4%（B 比老师约 +2σ）。
预设标准 ≥55% 未达。换随机数后 base≈both，背鱼群的问题没了，但在没见过的鱼群下躲鱼本身只有五成左右。

用法: python diag_22_rng.py <模型逗号分隔> [相位数] [并发]   MARIO_ARMS=base,both 选组（默认三组）
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, collections
os.environ.setdefault("OMP_NUM_THREADS", "1")
assert not os.environ.get("MARIO_RNG_POOL"), "评测脚本自己写随机数，别再让 make_env 写一遍（base 组会被污染）"
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else None
NPH = int(sys.argv[2]) if len(sys.argv) > 2 else 121
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
RNG, FRAME = 0x07A7, 0x09


POOL = os.environ.get("MARIO_RNG_EVAL_POOL", "rng22_pool.npy")   # 偶数行给训练（make_env.RandomizeRNG），这里只用奇数行


def build_pool(frames=200_000, every=97):
    from make_env import MarioBase, _nes_of
    env = MarioBase(stages=["2-2"]); env.reset(); n = _nes_of(env)
    out = []
    for t in range(frames):
        n._frame_advance(0)   # 绕过 done 检查、不 reset：reset 会恢复同一个快照、随机数跟着倒回去；死了/游戏结束随机数照样每帧走
        if t % every == 0:
            out.append([int(x) for x in n.ram[RNG:RNG + 7]] + [int(n.ram[FRAME])])
    np.save(POOL, np.array(out, dtype=np.uint8))
    print(f"随机数池 {len(out)} 个，互不相同 {len(set(map(tuple, out)))} 个")


def run(job):
    model, arm, phase = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env, _nes_of
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    m = PPO.load(model, device="cpu")
    env = make_env(stages=["2-2"], noop=phase, exact=True)
    o, _ = env.reset(); n = _nes_of(env)
    if arm != "base":
        pool = np.load(POOL)[1::2]
        row = pool[np.random.default_rng(int(os.environ.get("MARIO_EVAL_SEED", "1000")) + phase).integers(len(pool))]
        n.ram[RNG:RNG + 7] = row[:7]
        assert list(n.ram[RNG:RNG + 7]) == list(row[:7]), "RAM 写不进去"
        if arm == "both":
            n.ram[FRAME] = row[7]
    for _ in range(3000):
        ot, _ = m.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(m.policy.get_distribution(ot).distribution.probs.argmax())
        o, rew, term, trunc, info = env.step(a)
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (2, 2):
            env.close(); return model, arm, phase, 1
        if term or trunc:
            env.close(); return model, arm, phase, 0
    env.close()
    return model, arm, phase, 0


def main():
    if not os.path.exists(POOL):
        build_pool()
    arms = os.environ.get("MARIO_ARMS", "base,rng,both").split(",")
    models = MODEL.split(",")
    jobs = [(m, a, p) for m in models for a in arms for p in range(NPH)]
    res = collections.defaultdict(dict)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for model, arm, phase, ok in pool.map(run, jobs, chunksize=1):
            res[(model, arm)][phase] = ok
    for model in models:
        line = "  ".join(f"{a} {sum(res[(model, a)].values())}/{NPH}" for a in arms)
        print(f"{os.path.basename(model):40s} {line}", flush=True)

if __name__ == "__main__":
    main()
