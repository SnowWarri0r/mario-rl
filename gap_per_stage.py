"""逐关量"学生离下界还有多远"，而不是只看总 loss。

**为什么需要它。** 蒸馏目标是老师的概率分布，所以 loss 有一条硬下界 =
老师分布熵的均值（全量 FLOOR=0.8346）。v32 实测 0.8600，只差 0.025，
已吃掉常数解(1.8556)到下界之间 97.5% 的空间 —— 也就是说
"数据涨 3.3 倍 loss 只从 0.8859 挪到 0.8570"是**在贴渐近线**，不是欠拟合。
这条下界跟模型大小、数据量都无关，换多大骨干都压不下去（实测 BigCNN v1
6.34M 得 0.8618、Wide 6.86M 得 0.8600，差异在噪声里）。

但总量会掩盖结构：如果 0.025 的缺口全压在少数几关上，那几关仍有可为。
这里按关拆开，报 学生loss / 该关下界 / 缺口。缺口大的关才是还能靠
容量或数据改善的；缺口已经贴地的关，剩下的差距只能是分布漂移（学生走到
老师数据里没有的状态），处方是 DAgger 而不是更大的网络。

用法: python gap_per_stage.py <模型.zip> [数据目录逗号分隔]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, glob, re, collections
import numpy as np, torch as th
from stable_baselines3 import PPO
import wide_cnn, big_cnn  # noqa: F401  注册骨干类，PPO.load 要用

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_v32.zip"
DIRS = (sys.argv[2] if len(sys.argv) > 2 else
        "distill_data_29,distill_data_dagger_29,distill_data_dagger_30,distill_data_dagger_31").split(",")
BATCH = 2048
DEV = "cuda" if th.cuda.is_available() else "cpu"

m = PPO.load(MODEL, device=DEV)
print(f"=== 逐关缺口 | {MODEL} | {type(m.policy.features_extractor).__name__} "
      f"{sum(p.numel() for p in m.policy.parameters())/1e6:.2f}M ===", flush=True)

agg = collections.defaultdict(lambda: [0.0, 0.0, 0])   # stage -> [loss和, 下界和, 帧数]
files = [f for d in DIRS for f in sorted(glob.glob(f"{d.strip()}/*.npz"))]
for f in files:
    st = re.match(r"(\d+-\d+)", os.path.basename(f))
    if not st:
        continue
    st = st.group(1)
    d = np.load(f)
    obs, prob = d["obs"], d["probs"].astype(np.float32)
    for i in range(0, len(obs), BATCH):
        ob = th.from_numpy(obs[i:i+BATCH]).to(DEV)
        pb = th.from_numpy(prob[i:i+BATCH]).to(DEV)
        with th.no_grad():
            lg = m.policy.get_distribution(ob).distribution.logits
        l = float(-(pb * lg).sum(1).sum())
        fl = float(-(pb * th.log(pb.clamp_min(1e-9))).sum(1).sum())
        a = agg[st]; a[0] += l; a[1] += fl; a[2] += len(ob)

rows = []
for st, (l, fl, n) in agg.items():
    rows.append((l/n - fl/n, st, l/n, fl/n, n))
rows.sort(reverse=True)
print(f"\n{'关卡':6s} {'学生loss':>9s} {'下界':>8s} {'缺口':>8s} {'帧数':>9s}")
for gap, st, l, fl, n in rows:
    print(f"{st:6s} {l:9.4f} {fl:8.4f} {gap:8.4f} {n:9d}")
tl = sum(r[2]*r[4] for r in rows); tf = sum(r[3]*r[4] for r in rows); tn = sum(r[4] for r in rows)
print(f"\n合计   {tl/tn:9.4f} {tf/tn:8.4f} {tl/tn-tf/tn:8.4f} {tn:9d}")
print("\n>>> 读法：缺口大的关＝学生还没学会老师在那关的行为，加容量/加数据有指望；"
      "\n    缺口贴地的关＝蒸馏这一侧已经做完，实战仍差就是分布漂移，只能 DAgger。", flush=True)
