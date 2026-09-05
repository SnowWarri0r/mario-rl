"""对照：存档开局本身是不是个畸形状态？

起因：定点 DAgger 收数据时，学生从 x≈1408 起步只有 13% 通关，而它从整关开头起步是 56%，
平均每局仅 75 帧（1408→旗杆 3161 至少要 400 步）。**离终点更近反而更差**，不合常理。

怀疑的机制：`ArchiveStart` 恢复快照后 `FrameStack` 被重置成"同一帧复制四份"，
观测里的**速度信息全没了**。2-2 是水下关，策略靠 4 帧差分判断鱼往哪游、多快，
看到一堆"静止的鱼"就会撞上去。若成立，那 16 万帧带着真实游戏里不存在的畸形，蒸进去可能有害。

判据：让**老师**从同一批存档点开局。它整关 86%——
  · 若它也掉到 15% 上下 → 是开局机制的问题，数据要么丢掉要么修（预热几帧再开始记录）
  · 若它仍有 80%+      → 存档点没问题，13% 是学生的真实后半段水平，数据可用

用法: MARIO_ARCHIVE=states22_back.npz MARIO_ARCHIVE_P=1.0 python diag_archive_start.py [每模型局数] [并发]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MARIO_ARCHIVE_P", "1.0")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
MODELS = os.environ.get("MARIO_MODELS", "mario_22champ.zip,mario_v7_wide.zip").split(",")


def run(job):
    model_path, n, det = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env_stage22_archive
    import wide_cnn  # noqa: F401

    m = PPO.load(model_path, device="cpu")
    env = make_env_stage22_archive()
    cleared, steps, starts, ends = 0, [], [], []
    for _ in range(n):
        o, _ = env.reset()
        done, k, x0, last_x = False, 0, None, 0
        while not done:
            ot, _ = m.policy.obs_to_tensor(o)
            with th.no_grad():
                p = m.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
            a = int(np.argmax(p)) if det else int(np.random.choice(len(p), p=p / p.sum()))
            o, r, term, trunc, info = env.step(a)
            done = term or trunc
            k += 1
            if x0 is None:
                x0 = int(info.get("x_pos", 0))
            if not done:
                last_x = int(info.get("x_pos", last_x))
        cleared += bool(info.get("flag_get") or (info.get("world"), info.get("stage")) != (2, 2))
        steps.append(k); starts.append(x0 or 0); ends.append(last_x)
    env.close()
    return cleared, n, steps, starts, ends


def main():
    print(f"=== 存档开局对照：{MODELS}，各 {N} 局，存档 {os.environ.get('MARIO_ARCHIVE')} ===", flush=True)
    per = max(N // WORKERS, 1)
    for mp in MODELS:
        for det in (True, False):
            jobs = [(mp, per, det) for _ in range(WORKERS)]
            C = A = 0; S = []; X0 = []; XE = []
            with ProcessPoolExecutor(max_workers=WORKERS) as pool:
                for c, a, st, x0, xe in pool.map(run, jobs):
                    C += c; A += a; S += st; X0 += x0; XE += xe
            print(f"  {os.path.basename(mp):24s} {'argmax' if det else '采样  '}  "
                  f"通关 {C}/{A} = {C/A*100:4.0f}%   开局 x 中位 {int(np.median(X0))}   "
                  f"局长中位 {int(np.median(S))} 帧   终点 x 中位 {int(np.median(XE))}", flush=True)


if __name__ == "__main__":
    main()
