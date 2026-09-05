"""收 2-2 的"硬点前"动作前缀，给 Backplay / Go-Explore 式训练当起点档案。

思路：让最好的老师（argmax + 抖动）去跑，把它每局走到 x∈[LO, HI] 那一刻之前的动作序列存下来。
之后训练时重放这段前缀 → _backup() → 每回合 _restore()，等于让 agent 每次开局就站在鱼缝前面，
把"要先游 2000 像素才能开始练那一下"变成"每回合都在练那一下"。

存动作序列而不是存档文件，因为模拟器只有单槽存档（存不下一整个档案库）；
重放成本 ~0.9s，摊到 20 个回合上可以接受。前缀之间相位不同，档案自带多样性。

用法: python collect_states22.py [条数] [下界x] [上界x]   默认 240 / 1850 / 2050（鱼缝在 ~2095）
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 240
LO = int(sys.argv[2]) if len(sys.argv) > 2 else 1850
HI = int(sys.argv[3]) if len(sys.argv) > 3 else 2050
TEACHER = os.environ.get("MARIO_TEACHER", "mario_22cur_30.zip")
WORKERS = int(os.environ.get("MARIO_WORKERS", "12"))
OUT = os.environ.get("MARIO_OUT", "states22_prefixes.npz")


def worker(job):
    wid, want = job
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env

    teacher = PPO.load(TEACHER, device="cpu")
    env = make_env(stages=["2-2"], noop=30)          # 抖动开局 → 前缀覆盖不同相位
    got = []
    tries = 0
    while len(got) < want and tries < want * 12:
        tries += 1
        o, _ = env.reset()
        acts, done, recorded = [], False, False
        while not done:
            ot, _ = teacher.policy.obs_to_tensor(o)
            with th.no_grad():
                a = int(np.argmax(teacher.policy.get_distribution(ot)
                                  .distribution.probs.cpu().numpy()[0]))
            acts.append(a)
            o, r, term, trunc, info = env.step(a)
            done = term or trunc
            x = info.get("x_pos", 0)
            if not recorded and LO <= x <= HI:
                got.append(np.array(acts, dtype=np.int8))   # 存到刚进入区间为止
                recorded = True
                break                                        # 这一局的任务完成
    env.close()
    return wid, got, tries


def main():
    per = max(1, WANT // WORKERS)
    print(f"=== 收 2-2 前缀：目标 {WANT} 条，落点 x∈[{LO},{HI}]，老师 {TEACHER} argmax+抖动 ===", flush=True)
    allp = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for wid, got, tries in pool.map(worker, [(k, per) for k in range(WORKERS)]):
            allp.extend(got)
            print(f">>> worker{wid}: {len(got)} 条 / 试了 {tries} 局", flush=True)
    lens = [len(p) for p in allp]
    np.savez_compressed(OUT, prefixes=np.array(allp, dtype=object))
    print(f">>> 共 {len(allp)} 条前缀 → {OUT}；长度 {min(lens)}-{max(lens)} 步（中位 {int(np.median(lens))}）",
          flush=True)


if __name__ == "__main__":
    main()
