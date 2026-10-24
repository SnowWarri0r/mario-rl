"""7-4 / 8-4 的迷宫检查：让一个现成策略开车，逐帧记下每次"检查"发生时马里奥在哪。

依据 smbdis.asm ProcLoopCommand：表 LoopCmdWorldNumber/PageNumber/YPosition
（世界号从 0 起：$06 = 7-4，$07 = 8-4）——
  7-4：页 4/5/6 要求 Y = $B0/$80/$40，页 8/9/10 要求 $40/$80/$40，三个一组；
       每组三次检查都对（Y 相等且 Player_State==0 站地）才放行，否则第三次时回卷 4 页。
  8-4：页 6/11/16 要求 Y = $F0（站地不可能是 $F0）→ 走着到这些页一定回卷，只能钻管道绕过。
检查发生在渲染器开始加载那一页第 0 列的那一帧（CurrentColumnPos==0），马里奥此时约在前一屏。

逐帧挂在 MarioBase.step 上（跳帧之下），检测 LoopCommand($0745) 从非零变零的那一帧，
记录 页号 CurrentPageLoc($0725)、马里奥 x/Y、Player_State($1D)、两个计数器（$06DA 次数 / $06D9 正确数），
以及有没有回卷（马里奥页号一帧内掉 ≥3）。

用法: python diag_74.py <模型> <关卡 7-4|8-4> [相位数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, collections
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor

MODEL = sys.argv[1] if len(sys.argv) > 1 else None
STAGE = sys.argv[2] if len(sys.argv) > 2 else "7-4"
NPH = int(sys.argv[3]) if len(sys.argv) > 3 else 8


def run(phase):
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    from make_env import make_env, _nes_of, MarioBase
    m = PPO.load(MODEL, device="cpu")
    env = make_env(stages=[STAGE], noop=phase, exact=True)
    o, _ = env.reset(); n = _nes_of(env); R = n.ram
    events = []
    base = env.unwrapped
    orig = MarioBase.step
    state = {"lc": 0, "pg": int(R[0x6D])}

    def hooked(self, a):
        out = orig(self, a)
        lc = int(R[0x745]); pg = int(R[0x6D])
        if state["lc"] and not lc:
            events.append(("检查", int(R[0x725]), int(R[0x6D]) * 256 + int(R[0x86]), int(R[0xCE]),
                           int(R[0x1D]), int(R[0x6DA]), int(R[0x6D9])))
        if pg <= state["pg"] - 3:
            events.append(("回卷", pg))
        state["lc"], state["pg"] = lc, pg
        return out
    MarioBase.step = hooked
    xmax = 0
    for t in range(6000):
        ot, _ = m.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(m.policy.get_distribution(ot).distribution.probs.argmax())
        o, r, term, trunc, info = env.step(a)
        xmax = max(xmax, int(R[0x6D]) * 256 + int(R[0x86]))
        if info.get("flag_get") or (info.get("world"), info.get("stage")) != tuple(map(int, STAGE.split("-"))):
            return phase, "通关", xmax, events
        if term or trunc:
            return phase, f"死/超时 x={int(R[0x6D]) * 256 + int(R[0x86])}", xmax, events
    return phase, "步数上限", xmax, events


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=NPH) as pool:
        for phase, end, xmax, ev in pool.map(run, range(NPH)):
            print(f"相位 {phase}: {end}  最远 x={xmax}")
            for e in ev[:20]:
                print("   ", e)
