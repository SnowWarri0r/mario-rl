"""8-4 第二房间路线：从 x≈1848 起，顶出隐藏砖 → 助跑跳上砖 → 跳上悬空管道 → 入口窗口按下 → 第 12 页（x≈3128）。

实测（逐个 x 原地跳看金币数 / 头顶碰撞高度）：隐藏砖在马里奥 x 2372-2388 处顶得出，不是关卡解码给的 2400；
小马里奥原地跳最高 Y≈115，够不到砖顶（站上去 Y=112），要从 x≈2240 助跑（右+B）到 x≈2344 起跳；
管道顶站立 Y=64，入口窗口 x 2436-2444。前提：x≈2390/2431 原地跳的两只飞龟已被踩掉（dbg_room2_pre.npy
是 goexplore84.py 搜到的、飞龟已清、站在隐藏砖附近的前缀）。产物 route84_room2.npz 供 make_env_route 开第三房间。
"""
import os; os.environ["MARIO_COMPLEX"]="1"
import numpy as np
from make_env import MarioBase, _nes_of, SkipFrame, GrayResize, FrameStack
d=np.load("route84_room1.npz"); pre=[int(b) for b in np.load("dbg_room2_pre.npy")]
env=FrameStack(GrayResize(SkipFrame(MarioBase(stages=["8-4"]),k=4),84)); env.reset(); n=_nes_of(env); R=n.ram
for a in d["drive"]: env.step(int(a))
n._backup()                                   # 存档：第二房间起点之前（drive 结束）
X=lambda: int(R[0x6D])*256+int(R[0x86]); Y=lambda: int(R[0xCE]); G=lambda: int(R[0x1D])==0
dead=lambda: int(R[0x0E])==0x0B or int(R[0xB5])>1
def pir():
    for i in range(5):
        if R[0x0F+i] and int(R[0x16+i])==0x0D: return int(R[0xCF+i])
    return 255
SEQ=[]
def fa(b): n._frame_advance(b); SEQ.append(b)
def replay(frames):
    n._restore(); n.done=False; SEQ.clear()
    for b in frames: fa(b)
def goto(tx):
    for _ in range(200):
        x=X()
        if abs(x-tx)<=1 and G(): return
        fa(0x80 if x<tx-1 else 0x40)
replay(pre)
goto(2380)
for _ in range(14): fa(0x01)
for _ in range(40): fa(0)
print("coins", int(R[0x75E]), flush=True)
BUMP=list(SEQ); pg0=int(R[0x6D])
for x0 in range(2340,2352,2):
  for h in (20,26,32):
    replay(BUMP); goto(2240)
    for _ in range(200):
        if X()>=x0: break
        fa(0x82)
    for _ in range(h): fa(0x83)
    for _ in range(60):
        if G(): break
        fa(0x80 if X()<2374 else (0x40 if X()>2386 else 0))
    if not (G() and Y()==112 and not dead()): continue
    ONB=list(SEQ)
    print("on block", x0, h, X(), flush=True)
    for hj in (6,10,14,18,22):
      for wp in (0,1):
        replay(ONB)
        for _ in range(200):
            if not wp or pir()>=96: break
            fa(0)
        for _ in range(hj): fa(0x81)
        for _ in range(60):
            if G(): break
            fa(0x80 if X()<2437 else (0x40 if X()>2443 else 0))
        if not (G() and Y()<112 and not dead()):
            continue
        yt=Y()
        for _ in range(60):
            x=X()
            if 2436<=x<=2444 and G(): break
            fa(0x80 if x<2436 else 0x40)
        for _ in range(80):
            fa(0x20)
            if int(R[0x6D])>=pg0+2: break
        print("  try hj",hj,"wp",wp,"pipe-top y",yt,"x",X(),"page",int(R[0x6D]), flush=True)
        if int(R[0x6D])>=pg0+2:
            np.savez("route84_room2.npz", drive=d["drive"], frames=np.array(SEQ, np.uint8))
            print("SAVED", len(SEQ)); raise SystemExit
print("none")
