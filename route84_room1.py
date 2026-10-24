"""8-4 第一房间路线搜索：坑前存档 → 等升降台（敌人 $28，x 1067-1120 来回）→ 落上去 → 跟到右端起跳过 144px 岩浆坑
→ 往 x=1296 管道走（马里奥 x≈1282 时第 6 页循环检查必然触发，全体后退 4 页、画面不变）→ 站上同一根管道（循环后 x 277-283）按下 → 第 7 页。
找到后把整条路线（智能体动作 drive + 逐帧手柄 frames）存成 route84_room1.npz，供 make_env_route 重放开局。
关卡数据解码见提交说明（smbdis.asm L_CastleArea6 / E_CastleArea6 / ProcLoopCommand / HandlePipeEntry）。
"""
import os; os.environ["MARIO_COMPLEX"]="1"; os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, torch as th
from stable_baselines3 import PPO; import wide_cnn
from make_env import MarioBase, _nes_of, SkipFrame, GrayResize, FrameStack
m=PPO.load("checkpoints_mario_84_w/mario_84_w_7000000_steps.zip",device="cpu")
env=FrameStack(GrayResize(SkipFrame(MarioBase(stages=["8-4"]),k=4),84)); o,_=env.reset(); n=_nes_of(env); R=n.ram
X=lambda: int(R[0x6D])*256+int(R[0x86])
PREFIX=[]; DRV=[]
_fa=n._frame_advance
def _rec(b):
    PREFIX.append(int(b)); return _fa(b)
n._frame_advance=_rec
while X()<990:
    ot,_=m.policy.obs_to_tensor(o)
    with th.no_grad(): a=int(m.policy.get_distribution(ot).distribution.probs.argmax())
    DRV.append(a); o,*_=env.step(a)
while X()<1000: n._frame_advance(0x80)
for _ in range(30): n._frame_advance(0)
print("save", X(), int(R[0xCE]), int(R[0x1D]), flush=True)
n._frame_advance=_fa
PRE=list(PREFIX)
n._backup()
def dead(): return int(R[0xE])==0x0B or int(R[0xB5])>1
def lift():
    for i in range(6):
        if R[0xF+i] and int(R[0x16+i])==0x28: return int(R[0x6E+i])*256+int(R[0x87+i])
    return None
def piranha_y():
    for i in range(6):
        if R[0xF+i] and int(R[0x16+i])==0x0D: return int(R[0xCF+i])
    return None
RIGHT,RB,A,LEFT,DOWN=0x80,0x82,0x01,0x40,0x20
log=[]
for w in range(0,240,6):
  for hop in (0,4,8,12,16):
    for go in (1100,1108,1114,1118):
      n._restore(); n.done=False; ok=True; frames=[]
      def step(b):
          n._frame_advance(b); frames.append(b); return not dead()
      for _ in range(w):
          if not step(0): ok=False; break
      if not ok: continue
      for _ in range(hop):
          if not step(RIGHT|A): ok=False; break
      for _ in range(80):                      # 往右走下台，落到升降台上
          if not ok: break
          if int(R[0x1D])==0 and int(R[0xCE])>=170 and X()>1060: break
          if not step(RIGHT if X()<1062 else 0): ok=False
      if not ok or int(R[0xCE])<170 or X()<=1060: continue
      for _ in range(200):                     # 站在升降台上等它往右
          l=lift()
          if l is not None and l>=go: break
          if not step(0): ok=False; break
      if not ok: continue
      for _ in range(24):
          if not step(RB|A): ok=False; break
      for _ in range(40):
          if not ok: break
          if not step(RB): ok=False
      if not ok or X()<1184: continue
      # 第 6 页循环检查在马里奥 x≈1282 必然触发（全体后退 4 页，画面不变），之后站上同一根管道
      # （循环后 x 窗口 276-284 = 管道左缘 +4..+12，HandlePipeEntry 要求左脚踩 $10 右脚踩 $11）再按下。
      for _ in range(400):
          if not ok: break
          if X() < 700: break
          py=piranha_y()
          if X()>=1270 and py is not None and py<126:
              if not step(0): ok=False
              continue
          if not step(RIGHT|A if X()>=1266 and int(R[0x1D])==0 else RIGHT): ok=False
      if not ok or X()>=700: continue
      for _ in range(240):
          xx=X()
          if 277<=xx<=283 and int(R[0x1D])==0 and int(R[0xCE])<=150: break
          py=piranha_y()
          if py is not None and py<126 and int(R[0x1D])==0:
              if not step(0): ok=False; break
              continue
          if int(R[0x1D])==0 and int(R[0xCE])>150:
              b=(RIGHT|A) if xx<270 else (LEFT|A)
          else:
              b=RIGHT if xx<277 else (LEFT if xx>283 else 0)
          if not step(b): ok=False; break
      if not ok: continue
      pg=int(R[0x6D])
      for _ in range(80): step(DOWN)
      log.append((w,hop,go,X(),int(R[0x6D])))
      if int(R[0x6D])>=7 or int(R[0x750])!=0x65:
          print("FOUND", (w,hop,go), "x", X(), "frames", len(frames))
          np.savez("route84_room1.npz", drive=np.array(DRV, np.uint8), frames=np.array(PRE+frames, np.uint8))
          print("prefix frames", len(PRE)+len(frames))
          raise SystemExit
print("no route; last", log[-5:], "n", len(log))
