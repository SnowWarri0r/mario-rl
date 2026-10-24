"""8-4 的管道地图：现成策略开车，沿途每一步都试"按下 10 帧"，看哪些位置会钻进管道、钻进去到了哪。

依据 smbdis.asm ProcLoopCommand：8-4 在页 6/11/16 检查 Player_Y==$F0（站地不可能）→ 走路到那里必定回卷，
只能靠钻管道绕过。哪根管道通哪里要实测：从存档点按 down（COMPLEX 动作 10）10 帧，
若 AreaPointer($0750) 变了或马里奥 x 跳了 >100，就记一条"x → 新区域 / 新 x"。

用法: MARIO_COMPLEX=1 python probe_84.py <开车模型> [相位]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ["MARIO_COMPLEX"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np


def main():
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    from make_env import MarioBase, _nes_of, SkipFrame, GrayResize, FrameStack
    m = PPO.load(sys.argv[1], device="cpu")
    phase = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    env = FrameStack(GrayResize(SkipFrame(MarioBase(stages=["8-4"]), k=4), 84))
    o, _ = env.reset(); n = _nes_of(env); R = n.ram
    for _ in range(phase):
        o, *_ = env.step(0)
    X = lambda: int(R[0x6D]) * 256 + int(R[0x86])
    seen = set(); found = []
    for t in range(1500):
        x, area = X(), int(R[0x750])
        if int(R[0x1D]) == 0 and x // 8 not in seen:          # 站地时，每 8 像素试一次
            seen.add(x // 8)
            n._backup()
            for _ in range(12):
                n._frame_advance(1 << 5)                       # 手柄 down 位（nes-py 按位：A B select start up down left right → down=1<<5）
            x2, a2 = X(), int(R[0x750])
            if a2 != area or abs(x2 - x) > 100:
                found.append((x, int(R[0xCE]), area, a2, x2))
                print(f"钻管道：x={x} Y={int(R[0xCE])} 区域 {area:#x} → {a2:#x}, x → {x2}", flush=True)
            n._restore(); n.done = False
        ot, _ = m.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(m.policy.get_distribution(ot).distribution.probs.argmax())
        o, r, term, trunc, info = env.step(a)
        if term or trunc:
            print(f"开车结束 t={t} x={X()} 区域 {int(R[0x750]):#x}"); break
    print("找到:", found)


if __name__ == "__main__":
    main()
