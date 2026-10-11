"""换了某关老师后，把现有蒸馏数据里这一关的帧用新老师重打标签（只换 probs，obs 不动）。

输出文件名 = rl_<来源目录>__<原文件名>，放进一个新目录；配合 distill_all12_gpu.py 的
MARIO_EXCLUDE='(来源目录…)/2-2_' 按路径剔掉旧标签。

用法: python relabel_stage.py <关卡如 2-2> <新老师.zip> <输出目录> <来源目录逗号分隔>
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, glob
import numpy as np
import torch as th
from stable_baselines3 import PPO
import wide_cnn, big_cnn, impala_cnn  # noqa: F401

STAGE, TEACHER, OUT, SRC = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
dev = "cuda" if th.cuda.is_available() else "cpu"
m = PPO.load(TEACHER, device=dev)
os.makedirs(OUT, exist_ok=True)
tot = 0
for d in SRC:
    for f in sorted(glob.glob(f"{d}/{STAGE}_*.npz")):
        z = np.load(f); obs = z["obs"]; old = z["probs"]
        ps = []
        with th.no_grad():
            for i in range(0, len(obs), 2048):
                x = th.as_tensor(obs[i:i + 2048], device=dev)
                ps.append(m.policy.get_distribution(x).distribution.probs.float().cpu().numpy())
        p = np.concatenate(ps).astype(old.dtype)
        agree = (p.argmax(1) == old.argmax(1)).mean()
        np.savez_compressed(f"{OUT}/rl_{os.path.basename(d)}__{os.path.basename(f)}", obs=obs, probs=p)
        tot += len(obs)
        print(f"{f}: {len(obs)} 帧，新旧老师 argmax 一致 {agree:.1%}", flush=True)
print(f"共重标 {tot} 帧 → {OUT}")
