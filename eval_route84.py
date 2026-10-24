"""8-4 分房间评测：从录好的路线开局（make_env_route，MARIO_ROUTE），模型 argmax 打，记最远 x、有没有往前传送、怎么结束。

用法: MARIO_COMPLEX=1 MARIO_STAGE=8-4 MARIO_ROUTE=route84_room1.npz python eval_route84.py <模型逗号分隔> [局数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, collections
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["MARIO_COMPLEX"] = "1"
from concurrent.futures import ProcessPoolExecutor


def run(job):
    model, ep = job
    os.environ["MARIO_ROUTE_NOOP"] = str(ep * 3 % 61)
    import numpy as np
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    import wide_cnn, impala_cnn  # noqa: F401
    from make_env import make_env_route, _nes_of
    m = PPO.load(model, device="cpu")
    env = make_env_route()
    o, _ = env.reset(); n = _nes_of(env); R = n.ram
    X = lambda: int(R[0x6D]) * 256 + int(R[0x86])
    xmax, warps, px = X(), [], X()
    for t in range(3000):
        ot, _ = m.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(m.policy.get_distribution(ot).distribution.probs.argmax())
        o, r, term, trunc, info = env.step(a)
        x = X()
        if abs(x - px) > 150:
            warps.append((px, x, hex(int(R[0x750]))))
        px = x; xmax = max(xmax, x)
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (8, 4):
            return model, "通关", xmax, warps
        if term or trunc:
            return model, f"结束 x={x}", xmax, warps
    return model, "步数上限", xmax, warps


if __name__ == "__main__":
    models = sys.argv[1].split(",")
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with ProcessPoolExecutor(max_workers=min(len(models) * N, 64)) as pool:
        res = list(pool.map(run, [(m, e) for m in models for e in range(N)]))
    for m in models:
        print("==", os.path.basename(m))
        for _, end, xmax, w in [r for r in res if r[0] == m]:
            print(f"   {end}  最远 x={xmax}  传送 {w[:4]}")
