"""开训前的秒级判据：骨干能不能把一小撮数据**过拟合**掉。

**为什么必须有这一步。** ImpalaCNN 第一版训满 32 个 epoch、烧掉 5 小时，
结果 loss 从 epoch 2 起钉死在 1.8557 —— 那正好是动作边缘分布的熵，
网络退化成了常数函数（八个不同输入输出差异为 0）。这种塌掉在**第一分钟**
就已经定局，却要等五小时才看见。

判据选"过拟合 2048 帧"而不是"跑一个 epoch 看 loss 降没降"：
后者在大数据上本来就降得慢，分不清"学得慢"和"学不动"；
前者是充分必要的体检——2048 帧对 1600 万参数是压倒性的过参数化，
**学不动一小撮就一定学不动 270 万帧**，而且只要几十秒。

⚠️ 必须走 sb3 的真实构造路径（PPO(...) 而不是裸 ImpalaCNN）：
把上一版炸掉的 gain=√2 正交初始化是 sb3 在骨干文件之外加的。

用法: MARIO_BACKBONE=impala MARIO_SCALE=64 python selftest_backbone_learns.py [数据npz]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, glob, time
import numpy as np, torch as th
from stable_baselines3 import PPO
from stub_env import make_stub_env

BACKBONE = os.environ.get("MARIO_BACKBONE", "wide")
SCALE = int(os.environ.get("MARIO_SCALE", "64"))
NFRAME = int(os.environ.get("MARIO_NFRAME", "2048"))
STEPS = int(os.environ.get("MARIO_STEPS", "300"))
LR = float(os.environ.get("MARIO_LR", "2.5e-4"))
REF = float(os.environ.get("MARIO_REF", "0.4651"))   # WideNatureCNN 2048帧x300步@2.5e-4 实测
DEV = "cuda" if th.cuda.is_available() else "cpu"

if BACKBONE == "impala":
    from impala_cnn import ImpalaCNN as EX, activation_scale
    EXTRA = dict(scale=SCALE)
else:
    from wide_cnn import WideNatureCNN as EX
    EXTRA, activation_scale = {}, None

src = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("distill_data_29/*.npz"))[0]
d = np.load(src)
obs = th.from_numpy(d["obs"][:NFRAME]).to(DEV)
prob = th.as_tensor(d["probs"][:NFRAME], dtype=th.float32, device=DEV)
# 两条基准线，别混。⚠️ 第一版把 FLOOR 写成了"常数解"，标反了：
# FLOOR = 每样本熵的均值 = 学生完全学对时的 loss 下界（谁也到不了它以下）
# CONST = 边缘分布的熵   = 学生退化成常数函数时的 loss（**塌掉的特征值**）
# 判据看的是离 CONST 多远，不是离 FLOOR 多远。标反的那一版把 0.166 当成了塌掉线，
# 而真正的塌掉线是 1.7852 —— 扫参时 1e-3 以上全部精确停在这个数，一眼可辨。
FLOOR = float(-(prob * th.log(prob.clamp_min(1e-9))).sum(1).mean())
_pm = prob.mean(0)
CONST = float(-(_pm * th.log(_pm.clamp_min(1e-9))).sum())
print(f"=== 骨干体检 | {BACKBONE} scale={SCALE} | {len(obs)} 帧 × {STEPS} 步 | lr={LR} ===")
print(f"    数据 {src}")
print(f"    **常数解 {CONST:.4f}** —— loss 停这儿＝退化成常数函数、根本没看画面")
print(f"      完美解 {FLOOR:.4f} —— 下界，仅供参考")
print(f"      对照   {REF:.4f} —— WideNatureCNN 同预算实测")

student = PPO("CnnPolicy", make_stub_env(), device=DEV, n_steps=64, verbose=0,
              policy_kwargs=dict(features_extractor_class=EX, normalize_images=False,
                                 features_extractor_kwargs=dict(features_dim=1024, **EXTRA)))
nparam = sum(p.numel() for p in student.policy.parameters())
print(f"    实际骨干 {type(student.policy.features_extractor).__name__}  参数 {nparam/1e6:.2f}M")
if activation_scale:
    rows = activation_scale(student.policy)
    grow = [v for n, v in rows if n.startswith("stage")][-1] / rows[0][1]
    print(f"    初始激活放大 输入→stage4 = {grow:.1f}×" + ("  ⚠️ 偏大" if grow > 20 else ""))

opt = th.optim.Adam(student.policy.parameters(), lr=LR)
t0, curve = time.time(), []
for s in range(STEPS):
    lg = student.policy.get_distribution(obs).distribution.logits
    loss = -(prob * lg).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 50 == 0 or s == STEPS - 1:
        curve.append((s, float(loss)))
        print(f"    step {s:4d}  loss {float(loss):.4f}", flush=True)
final = curve[-1][1]
print(f"\n    {time.time()-t0:.0f}s   末值 {final:.4f}  (常数解 {CONST:.4f} / 完美解 {FLOOR:.4f})")
# 过参数化到这个程度，过拟合 2048 帧应该轻松打到熵的一半以下
# 判据用**与 Wide 同预算的实测值**做参照，不拍绝对阈值：
# 绝对阈值试过一次 FLOOR*0.5=0.083，结果 Wide 自己也过不了 —— 那种判据等于没有。
assert final < CONST * 0.8, (f"塌了：末值 {final:.4f} 贴着常数解 {CONST:.4f}，"
                             f"网络没在看画面——别开整轮训练")
if final > REF:
    print(f"    ⚠️ 活着，但不如 Wide（{final:.4f} > {REF:.4f}）：换大骨干这笔买卖不成立")
else:
    print(f"    ✓ 活着且优于 Wide（{final:.4f} < {REF:.4f}）")
