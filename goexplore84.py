"""8-4 分房间找路：从已录路线的终点开局，Go-Explore 式探索（存"格子 → 帧级按键前缀"，挑格子重放后随机按键扩展）。

nes-py 只有一个存档槽，所以格子存的是"从房间起点开始的逐帧手柄字节"，每次扩展先 _restore 回房间起点再重放前缀。
格子 = (x//16, y//16, 是否站地)。往回传送（同区域 x 一帧掉 >200，即循环或错管道）当作死。
成功 = 马里奥页号一帧内往前跳 ≥2 页、或换了区域（AreaAddrsLOffset $074F 变化）且页号没往回掉。
找到后把"旧路线 + 新一段"存成新的路线 npz（drive 不变，frames 接上），供 make_env_route 开下一个房间。

用法: MARIO_COMPLEX=1 python goexplore84.py <旧路线.npz> <新路线.npz> [最多扩展次数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time
os.environ["MARIO_COMPLEX"] = "1"
import numpy as np

# 手柄字节：right 0x80 left 0x40 down 0x20 up 0x10 B 0x02 A 0x01
MACROS = [0x80, 0x82, 0x81, 0x83, 0x40, 0x42, 0x41, 0x43, 0x01, 0x00, 0x20]
HOOK = os.environ.get("MARIO_GE_ROOM2") == "1"
DBG = [0, 0]


def try_room2(n, R, X, pre, pg):
    """站在 x≈2384-2420 地面上：返回成功的完整前缀（pre + 脚本帧），失败返回 None（调用方负责恢复状态）。
    楼层（马里奥站立时的 Y）：地面 176；第 7 行的隐藏砖 / 实心砖顶 80；立在砖上的 2 格管道顶 48。
    步骤：挪到隐藏砖正下方 x≈2400 原地跳顶出砖 → 退到 x≈2380 跳上砖（Y=80）→ 从砖上跳上管道顶（Y=48）
    → 在管道顶挪进入口窗口 2436-2444 → 按下。中间等待帧数 wait 用来错开跳来跳去的飞龟。"""
    def dead():
        return int(R[0x0E]) == 0x0B or int(R[0xB5]) > 1
    Y = lambda: int(R[0xCE])
    ground = lambda: int(R[0x1D]) == 0
    for h1 in (10, 16):
      for wait in (0, 20, 40, 60, 90):
        for h2 in (12, 18, 24):
          for h3 in (8, 14, 20):
            n._restore(); n.done = False
            for b in pre:
                n._frame_advance(b)
            seq = []
            def adv(b):
                n._frame_advance(b); seq.append(b); return not dead()
            def goto(tx, tol, ylvl, limit=120):
                for _ in range(limit):
                    x = X()
                    if abs(x - tx) <= tol and ground() and Y() == ylvl:
                        return True
                    b = 0x80 if x < tx - tol else (0x40 if x > tx + tol else 0)
                    if not adv(b):
                        return False
                return False
            ok = all(adv(0) for _ in range(wait))
            ok = ok and goto(2400, 2, 176)
            if not ok: continue
            for _ in range(h1):
                ok = ok and adv(0x01)
            for _ in range(40):
                if not ok or (ground() and Y() == 176): break
                ok = adv(0)
            ok = ok and goto(2378, 3, 176)
            if not ok: continue
            for _ in range(h2):
                ok = ok and adv(0x81)
            for _ in range(50):
                if not ok or ground(): break
                ok = adv(0x80 if X() < 2398 else (0x40 if X() > 2408 else 0))
            if DBG[0] < 12 and h2 == 12:
                DBG[0] += 1
                ens = [(hex(int(R[0x16 + i])), int(R[0x6E + i]) * 256 + int(R[0x87 + i]), int(R[0xCF + i])) for i in range(5) if R[0x0F + i]]
                print(f"  dbg 上砖后 wait={wait} h1={h1} h2={h2}: x={X()} y={Y()} 站地={ground()} 活={ok} 状态{int(R[0x0E]):#x} 敌人 {ens}", flush=True)
            if not ok or not (ground() and Y() == 80): continue
            for _ in range(h3):
                ok = ok and adv(0x81)
            for _ in range(50):
                if not ok or ground(): break
                ok = adv(0x80 if X() < 2438 else (0x40 if X() > 2442 else 0))
            if DBG[1] < 30:
                DBG[1] += 1
                print(f"  dbg 上管道后 h3={h3}: x={X()} y={Y()} 站地={ground()} 活={ok}", flush=True)
            if not ok or not (ground() and Y() == 48): continue
            if not goto(2440, 4, 48, 60): continue
            for _ in range(60):
                n._frame_advance(0x20); seq.append(0x20)
                if int(R[0x6D]) >= pg + 2:
                    return list(pre) + seq
            if DBG[1] < 20:
                DBG[1] += 1
                print(f"  dbg 管道顶按下没反应 x={X()} y={Y()}", flush=True)
    return None


def main():
    from make_env import MarioBase, _nes_of, SkipFrame, GrayResize, FrameStack
    src, dst = sys.argv[1], sys.argv[2]
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    d = np.load(src)
    env = FrameStack(GrayResize(SkipFrame(MarioBase(stages=["8-4"]), k=4), 84))
    env.reset(); n = _nes_of(env); R = n.ram
    for a in d["drive"]:
        env.step(int(a))
    for b in d["frames"]:
        n._frame_advance(int(b))
    for _ in range(30):
        n._frame_advance(0)
    base_frames = list(d["frames"]) + [0] * 30
    n._backup()
    X = lambda: int(R[0x6D]) * 256 + int(R[0x86])
    area0 = int(R[0x74F])
    print(f"房间起点 x={X()} 区域 {area0:#x}", flush=True)

    def dead():
        return int(R[0x0E]) == 0x0B or int(R[0xB5]) > 1

    def cell():
        return (X() // 16, int(R[0xCE]) // 16, int(R[0x1D]) == 0)

    archive = {cell(): {"pre": [], "visits": 0}}
    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(budget):
        keys = list(archive)
        w = np.array([1.0 / np.sqrt(1 + archive[k]["visits"]) for k in keys])
        # 偏向更靠右的格子
        w *= np.array([1.0 + k[0] / 50 for k in keys])
        k = keys[int(rng.choice(len(keys), p=w / w.sum()))]
        archive[k]["visits"] += 1
        n._restore(); n.done = False
        for b in archive[k]["pre"]:
            n._frame_advance(b)
        pre = list(archive[k]["pre"])
        px, pg = X(), int(R[0x6D])
        # 第二房间：x=2432 的悬空管道要先顶出 x=2400 的隐藏砖当台阶。随机按键几乎碰不到这个组合，
        # 所以站在隐藏砖附近的格子上额外试一套脚本：原地跳顶砖 → 跳上砖 → 跳上管道 → 挪进入口窗口按下。
        if HOOK and 2384 <= X() <= 2420 and int(R[0x1D]) == 0:
            res = try_room2(n, R, X, pre, pg)
            if res is not None:
                np.savez(dst, drive=d["drive"], frames=np.array(base_frames + res, np.uint8))
                print(f"找到出口（脚本）：第 {it} 次扩展，x={X()} 页 {int(R[0x6D])}，存 {dst}", flush=True)
                return
            n._restore(); n.done = False
            for b in archive[k]["pre"]:
                n._frame_advance(b)
        for seg in range(int(rng.integers(1, 6))):
            b = int(MACROS[int(rng.integers(len(MACROS)))]); dur = int(rng.integers(4, 40))
            ok = True
            for _ in range(dur):
                n._frame_advance(b); pre.append(b)
                x, p, ar = X(), int(R[0x6D]), int(R[0x74F])
                if p >= pg + 2 or (ar != area0 and p >= pg):
                    route = {"drive": d["drive"], "frames": np.array(base_frames + pre, np.uint8)}
                    np.savez(dst, **route)
                    print(f"找到出口：第 {it} 次扩展，x {px}→{x} 页 {pg}→{p} 区域 {ar:#x}，"
                          f"新增 {len(pre)} 帧，用时 {time.time()-t0:.0f}s，存 {dst}", flush=True)
                    return
                if dead() or (x < px - 200 and ar == area0):
                    ok = False; break
                px, pg = x, p
            if not ok:
                break
            c = cell()
            if c not in archive or len(pre) < len(archive[c]["pre"]):
                archive[c] = {"pre": list(pre), "visits": archive.get(c, {"visits": 0})["visits"]}
        if it % 500 == 0:
            far = max(archive)[0] * 16
            print(f"[{it}] 格子 {len(archive)}，最远 x≈{far}，{time.time()-t0:.0f}s", flush=True)
    print("预算用完，没找到出口")


if __name__ == "__main__":
    main()
