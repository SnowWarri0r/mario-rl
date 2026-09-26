"""4-4 后半段的楼层地图：从策略轨迹上不同 x 接管、只按"右"，记下落地后站在哪一层（Player_Y）。

目的：第 9 页检查要求检查那一帧 Player_Y==$B0 且站地（见 diag_loop_ram.py 和 smbdis.asm 的
ProcLoopCommand）。策略一直走最上层，在 x≈2053 跳着过检查 → 回卷。要造示范，先得知道
"$B0 是哪一层、从哪儿能下去"。截图看得到三层走廊，但像素估 Y 不可靠，这里直接测。

用法: python diag_44_floors.py [相位]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor

MODEL = "checkpoints_mario_44p/mario_44p_7999488_steps.zip"
PHASE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
X0S = list(range(1100, 2060, 20))
FORCE = 60                                            # 接管后按"右"多少步（×4 帧）


def nes_of(env):
    n = env.unwrapped._e
    while not hasattr(n, "ram"):
        n = n.env
    return n


def run(x0):
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401
    m = PPO.load(MODEL, device="cpu")
    env = make_env(stages=["4-4"], noop=PHASE, exact=True)
    o, _ = env.reset(); n = nes_of(env)
    X = lambda: int(n.ram[0x6D]) * 256 + int(n.ram[0x86])
    for _ in range(2500):                             # 先让策略开到 x0
        if X() >= x0: break
        ot, _ = m.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(m.policy.get_distribution(ot).distribution.probs.argmax())
        o, r, term, trunc, info = env.step(a)
        if term or trunc: return x0, None, []
    start = (X(), int(n.ram[0xCE]))
    stands, life0 = [], n.ram[0x075A]
    for _ in range(FORCE):
        o, r, term, trunc, info = env.step(1)         # 1 = right
        if n.ram[0x075A] < life0 or term or trunc:
            stands.append(("死", X())); break
        if n.ram[0x1D] == 0:
            y = int(n.ram[0xCE])
            if not stands or stands[-1][0] != y:
                stands.append((y, X()))
    env.close()
    return x0, start, stands


def main():
    print(f"=== 4-4 楼层地图 | 相位 {PHASE} | 从策略轨迹 x0 接管、只按右 {FORCE} 步 ===")
    print("    每行：接管点(x, Y) → 依次站过的 (Y, 从哪个 x 起)；Y 是 RAM Player_Y，越大越靠下")
    with ProcessPoolExecutor(max_workers=len(X0S)) as pool:
        for x0, start, stands in pool.map(run, X0S):
            if start is None:
                print(f"  x0={x0}: 策略没开到"); continue
            seq = "  ".join(f"{'死' if s[0]=='死' else f'${s[0]:02X}'}@{s[1]}" for s in stands)
            print(f"  x0={x0:4d} 起点(x={start[0]},Y=${start[1]:02X})  →  {seq}")


if __name__ == "__main__":
    main()
