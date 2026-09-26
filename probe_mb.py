"""贴合真实训练条件的小批量探针：跨关分层取样 + batch 512 + 真实步数尺度。

**为什么不能用 selftest_backbone_learns.py 定生死。** 那个探针是单关 2048 帧的
**全批量**梯度，而真实蒸馏是 270 万帧跨 28 关、batch 512 的小批量。
实测 ImpalaCNN scale=32 在全批量探针上一路降到 0.168（跟 Wide 持平），
换到真实训练第一个 epoch 就是 1.8561 —— 正好是常数解。
两者差的是**梯度噪声**，而扫参已经表明这个骨干对有效步长极敏感（lr 1e-3 秒塌）。
探针不复现噪声，就测不出会不会塌。

用法: MARIO_SCALE=32 MARIO_LR=1e-4 MARIO_CLIP=0.5 python probe_mb.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, glob, time
import numpy as np, torch as th
from stable_baselines3 import PPO
from stub_env import make_stub_env

SCALE = int(os.environ.get("MARIO_SCALE", "32"))
LR    = float(os.environ.get("MARIO_LR", "2.5e-4"))
CLIP  = float(os.environ.get("MARIO_CLIP", "0"))       # 0 = 不裁
WARM  = int(os.environ.get("MARIO_WARMUP", "0"))       # 线性 warmup 步数
STEPS = int(os.environ.get("MARIO_STEPS", "1500"))
BATCH = int(os.environ.get("MARIO_BATCH", "512"))
PERF  = int(os.environ.get("MARIO_PERFILE", "1500"))   # 每个 npz 取多少帧，保证跨关覆盖
BACK  = os.environ.get("MARIO_BACKBONE", "impala")
SEED  = int(os.environ.get("MARIO_SEED", "0"))
DIRS  = os.environ.get("MARIO_DATA_DIRS", "distill_data_29").split(",")
DEV = "cuda"

VARIANT = os.environ.get("MARIO_VARIANT", "v1")
if BACK == "impala":
    from impala_cnn import ImpalaCNN as EX; EXTRA = dict(scale=SCALE)
elif BACK == "big":
    from big_cnn import BigCNN as EX; EXTRA = dict(variant=VARIANT)
else:
    from wide_cnn import WideNatureCNN as EX; EXTRA = {}

files = [f for d in DIRS for f in sorted(glob.glob(f"{d.strip()}/*.npz"))]
ob_l, pb_l = [], []
for f in files:
    d = np.load(f)
    k = min(PERF, len(d["probs"]))
    sel = np.linspace(0, len(d["probs"]) - 1, k).astype(int)
    ob_l.append(d["obs"][sel]); pb_l.append(d["probs"][sel])
obs = th.from_numpy(np.concatenate(ob_l)).to(DEV)
prob = th.as_tensor(np.concatenate(pb_l), dtype=th.float32, device=DEV)
N = len(obs)
pm = prob.mean(0); CONST = float(-(pm * th.log(pm.clamp_min(1e-9))).sum())
FLOOR = float(-(prob * th.log(prob.clamp_min(1e-9))).sum(1).mean())
print(f"=== 小批量探针 | {BACK}{VARIANT if BACK=='big' else SCALE} lr={LR} clip={CLIP} warmup={WARM} seed={SEED} ===")
print(f"    {len(files)} 个 npz 分层取 {N} 帧 | batch {BATCH} × {STEPS} 步")
print(f"    常数解 {CONST:.4f}（塌掉就停这儿） / 完美解 {FLOOR:.4f}", flush=True)

m = PPO("CnnPolicy", make_stub_env(), device=DEV, n_steps=64, verbose=0, seed=SEED,
        policy_kwargs=dict(features_extractor_class=EX, normalize_images=False,
                           features_extractor_kwargs=dict(features_dim=1024, **EXTRA)))
opt = th.optim.Adam(m.policy.parameters(), lr=LR)
t0, run = time.time(), []
for s in range(STEPS):
    if WARM and s < WARM:
        for g in opt.param_groups: g["lr"] = LR * (s + 1) / WARM
    b = th.randint(0, N, (BATCH,), device=DEV)
    lg = m.policy.get_distribution(obs[b]).distribution.logits
    loss = -(prob[b] * lg).sum(1).mean()
    opt.zero_grad(); loss.backward()
    gn = th.nn.utils.clip_grad_norm_(m.policy.parameters(), CLIP if CLIP > 0 else 1e9)
    opt.step()
    run.append(float(loss))
    if s % 150 == 0 or s == STEPS - 1:
        print(f"    step {s:5d}  loss {np.mean(run[-150:]):.4f}  |grad| {float(gn):8.2f}", flush=True)
fin = float(np.mean(run[-150:]))
print(f"\n    {time.time()-t0:.0f}s  末值 {fin:.4f}  vs 常数解 {CONST:.4f}")
print("    " + ("✗ 塌了" if fin > CONST * 0.9 else "✓ 在学"), flush=True)
