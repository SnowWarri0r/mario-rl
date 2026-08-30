"""把"学生模仿保真度不够"落到具体位置上：学生和老师在哪些 x 上做了不同的选择。

**为什么需要这个诊断。** 2-2 的师生落差 30pp（老师 86%、学生 56%），已经排除了两种解释：
不是容量/干扰（把整张网专供给 2-2 也只买到 8pp），不是价值头或不动点（RL 微调和 KL 锚都试过）。
剩下"模仿保真度不够"这个说法太笼统，没法据它决定下一步——所以要问得更具体：
**分歧集中在鱼缝那一下（精度问题，该加密那一段的示范），还是散落全程（表示问题，MoE 才有戏）？**

做法：让**老师**开车（argmax，它 86% 的那条轨迹就是我们要学生复制的目标），
每一帧同时记老师和学生的 argmax，按 x 分桶统计不一致率。老师开车而不是学生开车，
是因为我们要问的是"学生在老师会走的那条路上抄得像不像"，而不是"学生自己会走到哪儿"。
另外顺带记一份学生开车时的死亡 x 分布，两张图对上才算定位。

用法: MARIO_NOOP=30 python diag_agreement22.py [局数] [并发]   默认 60 / 30
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
TEACHER = os.environ.get("MARIO_TEACHER", "mario_22champ.zip")
STUDENT = os.environ.get("MARIO_STUDENT", "mario_v7_wide.zip")
STAGE = os.environ.get("MARIO_EVAL_STAGE", "2-2")
NOOP = int(os.environ.get("MARIO_NOOP", "30"))
BUCKET = 200          # x 分桶宽度；2-2 旗杆在 ~3161


def run(seed_batch):
    n = seed_batch
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    teacher = PPO.load(TEACHER, device="cpu")
    student = PPO.load(STUDENT, device="cpu")
    env = make_env(stages=[STAGE], noop=NOOP)

    nb = 3400 // BUCKET + 1
    disagree = np.zeros(nb, np.int64)     # 老师开车时两者 argmax 不同的帧数
    total = np.zeros(nb, np.int64)
    kl_sum = np.zeros(nb, np.float64)     # 分布层面的差异，比 argmax 更细
    deaths = []                           # 学生开车时的死亡 x

    for ep in range(n):
        # —— 第一趟：老师开车，逐帧比对 ——
        o, _ = env.reset()
        done = False
        while not done:
            ot, _ = teacher.policy.obs_to_tensor(o)
            st, _ = student.policy.obs_to_tensor(o)
            with th.no_grad():
                tp = teacher.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
                sp = student.policy.get_distribution(st).distribution.probs.cpu().numpy()[0]
            o, r, term, trunc, info = env.step(int(np.argmax(tp)))
            done = term or trunc
            b = min(int(info.get("x_pos", 0)) // BUCKET, nb - 1)
            total[b] += 1
            disagree[b] += int(np.argmax(tp) != np.argmax(sp))
            kl_sum[b] += float((tp * (np.log(tp + 1e-9) - np.log(sp + 1e-9))).sum())

        # —— 第二趟：学生开车，记它死在哪 ——
        o, _ = env.reset()
        done, last_x = False, 0
        while not done:
            st, _ = student.policy.obs_to_tensor(o)
            with th.no_grad():
                a = int(np.argmax(student.policy.get_distribution(st)
                                  .distribution.probs.cpu().numpy()[0]))
            o, r, term, trunc, info = env.step(a)
            done = term or trunc
            # ⚠️掉命那一步 info 已是复活后的状态（x=40/time=400），死亡位置要取上一步
            if not done:
                last_x = int(info.get("x_pos", last_x))
        cleared = bool(info.get("flag_get")) or (info.get("world"), info.get("stage")) != (
            int(STAGE.split("-")[0]), int(STAGE.split("-")[1]))
        if not cleared:
            deaths.append(last_x)
    env.close()
    return disagree, total, kl_sum, deaths


def main():
    per = max(EPISODES // WORKERS, 1)
    print(f"=== {STAGE} 保真度诊断：老师 {TEACHER} 开车逐帧比对学生 {STUDENT}，"
          f"{per*WORKERS} 局，抖动 0-{NOOP} ===", flush=True)
    nb = 3400 // BUCKET + 1
    D = np.zeros(nb, np.int64); T = np.zeros(nb, np.int64); K = np.zeros(nb, np.float64)
    deaths = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for d, t, k, dx in pool.map(run, [per] * WORKERS):
            D += d; T += t; K += k; deaths.extend(dx)

    print(f"\n{'x 区间':>12} {'帧数':>8} {'argmax 不一致':>14} {'平均KL':>9}   {'学生死亡数':>10}")
    dh, _ = np.histogram(deaths, bins=np.arange(0, (nb + 1) * BUCKET, BUCKET))
    for b in range(nb):
        if T[b] == 0 and dh[b] == 0:
            continue
        rate = D[b] / T[b] * 100 if T[b] else 0
        kl = K[b] / T[b] if T[b] else 0
        bar = "#" * int(rate / 2)
        print(f"{b*BUCKET:5d}-{(b+1)*BUCKET:5d} {T[b]:8d} {rate:12.1f}% {kl:9.3f}   "
              f"{dh[b]:8d}  {bar}", flush=True)
    tot_rate = D.sum() / T.sum() * 100
    print(f"\n>>> 全程 argmax 不一致率 {D.sum()}/{T.sum()} = {tot_rate:.1f}%"
          f"，平均 KL {K.sum()/T.sum():.3f}", flush=True)
    print(f">>> 学生开车死亡 {len(deaths)} 次，中位 x = "
          f"{int(np.median(deaths)) if deaths else '-'}", flush=True)
    print(">>> 读法：不一致率若集中在某一两个桶＝精度问题（加密那一段的示范）；"
          "若全程均匀＝表示/容量问题（MoE 或更大网络才有戏）", flush=True)


if __name__ == "__main__":
    main()
