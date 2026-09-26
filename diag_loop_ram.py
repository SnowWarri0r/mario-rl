"""4-4 回卷判定的逐帧 RAM 诊断：在检查那一帧，马里奥站在哪、是什么状态。

**判定逻辑（原文见 SMB 反汇编 smbdis.asm 的 ProcLoopCommand）**：
  4-4（WorldNumber=$03）有两个检查点：第 5 页要求 Player_Y_Position==$40，第 9 页要求 ==$B0，
  并且 Player_State==0（站在地面上，不在跳/落）。不满足就 ExecGameLoopback：所有页位置减 4。
  检查发生在**一帧**：LoopCommand 已被关卡数据置位，且渲染器刚开始加载第 N 页第 0 列
  （CurrentPageLoc==N 且 CurrentColumnPos==0）。渲染器跑在马里奥前面大约一屏，
  所以检查时马里奥**还在第 N 页之前约一屏的位置**。

**为什么之前十二种办法都失败。** 我把它当成"闸门处的位置"或"走过哪条路"，
而真实判据是"某一帧的精确高度 + 是否站地"。在错的 x 窗口扫高度、不要求站地，永远对不上。

⚠️ 必须逐帧读：检查只在一帧内发生，LoopCommand 当帧就被清零，隔着 SkipFrame 的 4 帧会漏。
   这里把 NES 模拟器实例的 step 包一层，每个模拟器帧都记 RAM。

用法: python diag_loop_ram.py <模型.zip> [相位逗号分隔]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_mario_44p/mario_44p_7999488_steps.zip"
PHASES = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,5,10,15,20,25,30").split(",")]
STAGE = os.environ.get("MARIO_STAGE", "4-4")
MAXSTEP = int(os.environ.get("MARIO_MAXSTEP", "2500"))

# RAM 地址（smbdis.asm 的 '=' 定义）
LoopCommand, MLCorrect, MLPass = 0x0745, 0x06D9, 0x06DA
CurPage, CurCol = 0x0725, 0x0726
PlayerY, PlayerState, PlayerPage, PlayerX = 0xCE, 0x1D, 0x6D, 0x86
REQ = {5: 0x40, 9: 0xB0}                        # 4-4 的两个检查点：页 → 要求的 Player_Y


def run(phase):
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    m = PPO.load(MODEL, device="cpu")
    env = make_env(stages=[STAGE], noop=phase, exact=True)
    u = env.unwrapped                              # make_env 的 MarioBase（gym→gymnasium 转接层）
    frames = []
    ram = None
    orig = u.step
    def rec(a):                                   # 每个模拟器帧都记一次 RAM
        out = orig(a)
        r = ram
        if r is None:
            return out
        frames.append((int(r[LoopCommand]), int(r[CurPage]), int(r[CurCol]),
                       int(r[PlayerY]), int(r[PlayerState]), int(r[PlayerPage]) * 256 + int(r[PlayerX])))
        return out
    u.step = rec
    o, _ = env.reset()
    # ⚠️ RAM 在最底层 SuperMarioBrosEnv 上，要顺着 .env 链往下找；而且必须在 reset **之后**找：
    #    RandomStages 在 reset 时才挑出当前关的子环境，reset 之前拿到的那个 WorldNumber 读出来是 0。
    n = u._e
    while not hasattr(n, "ram"):
        n = n.env
    ram = n.ram
    assert (ram[0x075F] + 1, ram[0x075C] + 1) == tuple(int(x) for x in STAGE.split("-")), \
        f"RAM 里的关卡号 {ram[0x075F]+1}-{ram[0x075C]+1} 对不上 {STAGE}，读错了模拟器实例"
    frames.clear()
    w0, s0 = (int(x) for x in STAGE.split("-"))
    cleared = False
    for _ in range(MAXSTEP):
        ot, _ = m.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(m.policy.get_distribution(ot).distribution.probs.argmax())
        o, r, term, trunc, info = env.step(a)
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != (w0, s0):
            cleared = True; break
        if term or trunc:
            break
    env.close()
    # 找检查帧：上一帧 LoopCommand!=0，这一帧被清零，且 CurCol==0、页在检查表里
    events = []
    for i in range(1, len(frames)):
        lc0, pg, _, y0, st0, x0 = frames[i-1]         # 判定读的是本帧开始时的值：取上一帧末尾
        lc, _, _, _, _, x = frames[i]
        # ⚠️ 页号必须取检查**之前**的：回卷会把 CurrentPageLoc 减 4（9→5），
        #    第一版在回卷后读页号，把失败的第 9 页检查记成了"第 5 页检查"。
        if lc0 and not lc and pg in REQ:
            back = x < x0 - 600                            # 回卷＝同一帧里 x 被减掉 4 页
            events.append((pg, x0, y0, st0, REQ[pg], back))
    return phase, cleared, events, max(f[5] for f in frames) if frames else 0


def main():
    print(f"=== {STAGE} 回卷判定逐帧诊断 | {MODEL} | 相位 {PHASES} ===")
    with ProcessPoolExecutor(max_workers=len(PHASES)) as pool:
        for phase, cleared, events, mx in pool.map(run, PHASES):
            print(f"\n相位 {phase:2d}：{'通关' if cleared else '未通关'}，最远 x={mx}，检查 {len(events)} 次")
            for pg, x, y, st, req, back in events:
                ok = (y == req and st == 0)
                print(f"   第 {pg} 页检查 @马里奥 x={x:5d}  Y=${y:02X}({y:3d}) 要求 ${req:02X}({req:3d})  "
                      f"state={st}{'(站地)' if st == 0 else '(空中)'}  "
                      f"{'✓ 应通过' if ok else '✗ 应回卷'}  实际{'回卷' if back else '没回卷'}")


if __name__ == "__main__":
    main()
