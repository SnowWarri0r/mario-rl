"""跨架构搬技能用的数据：一份轨迹里同时存 4 帧和 8 帧两种观测。

为什么需要它：换叠帧数就换了第一层卷积的输入通道，没法 warm-start，而从零训 2-2 实测
12M 步只能到 3-4%（平均最远 x≈1700，旗杆在 3161）——那条又土又长的 warm-start 链条
（塑形→checkpoint→梯子）不是绕弯路，它把不可能的探索问题拆成了几步可行的。
用蒸馏当桥：老师看它自己的 4 帧出动作概率，我们把同一时刻的 8 帧观测存下来给学生，
纯监督几分钟就能把"怎么游完这一关"灌进任意架构，再带抖动 RL 微调。
（同一个技巧裁 HUD 那轮用过：老师看未裁图当标注 oracle、存裁过的图给学生。）

在 noop=0 下收：老师是背轨迹的，抖动下只有 7%，拿它的失败轨迹没意义；
先把它最好状态下的通关技能搬过来，反应能力交给后面的抖动微调去练。

用法: python collect_distill_stack.py [总帧数] [分片数]   默认 150000 / 4
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, collections
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TEACHER = "mario_22ladder_final.zip"
OUTDIR = "distill_data_stack22"


def _prep(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA).astype(np.uint8)


def collect(job):
    shard, n = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import MarioBase, SkipFrame

    teacher = PPO.load(TEACHER, device="cpu")
    raw = MarioBase(stages=["2-2"])
    env = SkipFrame(raw, k=4)
    d4, d8 = collections.deque(maxlen=4), collections.deque(maxlen=8)

    frame, _ = env.reset()
    f = _prep(frame)
    for _ in range(8):
        d8.append(f)
    for _ in range(4):
        d4.append(f)

    o4_buf, o8_buf, p_buf = [], [], []
    while len(o4_buf) < n:
        ot, _ = teacher.policy.obs_to_tensor(np.stack(d4, 0))     # 老师只认 4 帧
        with th.no_grad():
            dist = teacher.policy.get_distribution(ot)
            probs = dist.distribution.probs.cpu().numpy()[0]
            a = int(dist.sample().cpu().numpy()[0])
        o4_buf.append(np.stack(d4, 0)); o8_buf.append(np.stack(d8, 0))
        p_buf.append(probs.astype(np.float32))
        frame, r, term, trunc, info = env.step(a)
        f = _prep(frame); d4.append(f); d8.append(f)
        if term or trunc:
            frame, _ = env.reset(); f = _prep(frame)
            for _ in range(8):
                d8.append(f)
            for _ in range(4):
                d4.append(f)
    env.close()
    path = f"{OUTDIR}/s{shard}.npz"
    np.savez_compressed(path, obs4=np.array(o4_buf[:n], np.uint8),
                        obs8=np.array(o8_buf[:n], np.uint8),
                        probs=np.array(p_buf[:n], np.float32))
    return shard, n, path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    per = TOTAL // SHARDS
    print(f"=== 收 2-2 双视角数据：{TOTAL} 帧（4 帧 + 8 帧同时存），老师 {TEACHER}，noop=0 ===", flush=True)
    with ProcessPoolExecutor(max_workers=SHARDS) as pool:
        for shard, n, path in pool.map(collect, [(k, per) for k in range(SHARDS)]):
            print(f">>> 分片{shard} 收完 {n} 帧 → {path}", flush=True)
    print(f">>> 共 {per * SHARDS} 帧 → {OUTDIR}/", flush=True)


if __name__ == "__main__":
    main()
