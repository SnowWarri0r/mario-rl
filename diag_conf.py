"""按老师的置信度分桶，看 argmax 分歧到底落在哪儿。

**为什么这一步不能省。** 实测学生在老师自己的数据上就有 8.13% 的帧 argmax 不一致，
听着很大；但老师的平均输出熵是 0.83 nats（7 个动作），说明它很多帧本来就在两三个
动作之间五五开 —— 那种帧上学生选哪个都行，不一致率再高也不疼。
只有**老师确信**的帧上翻了，才是真的错。

这也是"硬标签反而更差"（746→706）的候选解释：近似平局帧的 argmax 本身是任意的，
把它当成硬目标是在学噪声。

判读：
  若分歧几乎全在低置信桶 ⇒ 8.13% 是假警报，蒸馏侧真的做完了；
  若高置信桶里还有显著分歧 ⇒ 那些帧才是要针对的，且数量应该很小、可以定点处理。

用法: python diag_conf.py <学生.zip>
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, glob, re
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

STUDENT = sys.argv[1] if len(sys.argv) > 1 else "mario_v32.zip"
DIRS = "distill_data_29,distill_data_dagger_29,distill_data_dagger_30,distill_data_dagger_31".split(",")
EDGES = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01]
STAGES = sorted({re.match(r"(\d+-\d+)", os.path.basename(f)).group(1)
                 for f in glob.glob("distill_data_29/*.npz")},
                key=lambda s: tuple(map(int, s.split("-"))))


def work(stage):
    import torch as th; th.set_num_threads(2)
    from stable_baselines3 import PPO
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    m = PPO.load(STUDENT, device="cpu")
    nb = len(EDGES) - 1
    cnt = np.zeros(nb, dtype=np.int64); bad = np.zeros(nb, dtype=np.int64)
    for f in [x for d in DIRS for x in glob.glob(f"{d}/{stage}_*.npz")]:
        d = np.load(f)
        obs, pr = d["obs"], d["probs"]
        for i in range(0, len(obs), 1024):
            ob = th.from_numpy(obs[i:i+1024])
            with th.no_grad():
                sa = m.policy.get_distribution(ob).distribution.probs.argmax(1).numpy()
            p = pr[i:i+1024]
            conf = p.max(1); ta = p.argmax(1)
            b = np.digitize(conf, EDGES) - 1
            wrong = (sa != ta)
            for k in range(nb):
                sel = b == k
                cnt[k] += int(sel.sum()); bad[k] += int((sel & wrong).sum())
    return cnt, bad


def main():
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    nb = len(EDGES) - 1
    C = np.zeros(nb, dtype=np.int64); B = np.zeros(nb, dtype=np.int64)
    with ProcessPoolExecutor(max_workers=28) as pool:
        for c, b in pool.map(work, STAGES):
            C += c; B += b
    tot = C.sum(); totbad = B.sum()
    print(f"=== 按老师置信度分桶 | {STUDENT} | {tot} 帧 ===")
    print(f"{'老师最大概率':>14s} {'帧数':>10s} {'占比':>7s} {'分歧率':>8s} {'占全部分歧':>10s}")
    for k in range(nb):
        if not C[k]:
            continue
        print(f"  {EDGES[k]:.2f}-{EDGES[k+1]:.2f}   {C[k]:10d} {C[k]/tot*100:6.1f}% "
              f"{B[k]/C[k]*100:7.2f}% {B[k]/max(totbad,1)*100:9.1f}%")
    print(f"\n合计分歧 {totbad}/{tot} = {totbad/tot*100:.2f}%")
    hi = B[4:].sum(); hic = C[4:].sum()
    print(f"\n>>> 老师确信(>0.85)的帧：{hic} 帧（{hic/tot*100:.1f}%），学生翻掉 {hi} 帧 "
          f"= 该桶 {hi/max(hic,1)*100:.2f}%，占全部分歧的 {hi/max(totbad,1)*100:.1f}%")
    print(">>> 判读：分歧若几乎全在低置信桶 ⇒ 8% 是假警报；"
          "高置信桶仍有显著分歧 ⇒ 那才是该针对的帧。", flush=True)


if __name__ == "__main__":
    main()
