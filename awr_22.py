"""2-2 的离线 RL（AWR，advantage-weighted regression）：只用录下来的数据改进老师，不在线跑 PPO。

**为什么是离线。** 这个项目里"蒸馏/收敛后接着在线 RL"试过四次都退化：学生 2-2 51%→0%（补了价值头也一样）、
KL 锚不崩但回不到起点、2-2 老师 22robust 续训 80→14/121（121 相位单命）。共同原因：PPO 会把策略拖向
奖励最优点，而途中要穿过一片更差的区域；argmax + 确定性模拟器下每个相位要么全过要么全灭，参数一动
大量相位同时翻，表现为断崖。AWR 只在已录数据上做加权模仿，策略不会被拖着走。

做法：
  collect：老师在 2-2 上打约 2400 局，相位随机 0-120；每局随机定采样比例 p∈[0.2,1]（p 的步用采样、其余 argmax），
           既覆盖 argmax 实际走的路线，又有从它偏离出去的动作。回报 = 原版奖励 + 通关 +100 + 死亡 -50，γ=0.99。
  train：先训价值网络 V(s)（回归蒙特卡洛回报），再从老师出发做加权模仿：
         loss = -mean(w · log π(a|s))，w = exp(clip(A/β)), A = G - V(s)（按标准差归一），权重截断 ≤ WMAX。
  评测：diag_progress 单命 argmax 121 相位，**超过老师的 80/121 才算成功**。

**结果（2026-09-30）：未达标，按预设规则停。** 采集 2400 局 / 98 万帧（老师混合采样通关 56.4%），
V 的归一化 MSE 0.50，权重均值 1.57、截断 0.1%。9 个档单命 argmax：
  相位 0-60（选择集）：老师 48/61，AWR 35-43 —— 全部变差
  相位 61-120（验证集）：老师 32/60，AWR 33-42 —— 全部变好
  按选择集挑出的第 2 档：验证 35/60，合计 78/121 < 老师 80。
  合计最好的 06/07 档是 83，但不是按规则挑出的，且单模型 121 相位标准误约 ±5 局。
解读：离线避免了崩塌（最差 72，在线 PPO 是 0-14），但确定性下参数一动就重新洗牌哪些相位能过，
没有整体变强。

用法: python awr_22.py collect <输出目录> [局数] [worker]
      python awr_22.py train <数据目录> <输出前缀>
"""
import os, sys, glob, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np

TEACHER = os.environ.get("MARIO_22_TEACHER", "mario_22robust.zip")
GAMMA, CLEAR_BONUS, DEATH_PEN = 0.99, 100.0, -50.0
BETA = float(os.environ.get("MARIO_AWR_BETA", "1.0"))
WMAX = float(os.environ.get("MARIO_AWR_WMAX", "20"))
LR = float(os.environ.get("MARIO_AWR_LR", "1e-5"))
EPOCHS = int(os.environ.get("MARIO_AWR_EPOCHS", "3"))
BATCH = 512


def _collect_worker(args):
    wid, n_ep, outdir = args
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401
    rng = np.random.default_rng(1000 + wid)
    m = PPO.load(TEACHER, device="cpu")
    O, A, G, stats = [], [], [], [0, 0]
    for ep in range(n_ep):
        env = make_env(stages=["2-2"], noop=int(rng.integers(0, 121)), exact=True)
        o, _ = env.reset()
        p = float(rng.uniform(0.2, 1.0))
        obs, acts, rews = [], [], []
        cleared = False
        for _ in range(3000):
            ot, _ = m.policy.obs_to_tensor(o)
            with th.no_grad():
                d = m.policy.get_distribution(ot).distribution
                a = int(d.sample()[0]) if rng.random() < p else int(d.probs.argmax())
            obs.append(o.astype(np.uint8)); acts.append(a)
            o, r, term, trunc, info = env.step(a)
            if info.get("flag_get") or (info.get("world"), info.get("stage")) != (2, 2):
                rews.append(r + CLEAR_BONUS); cleared = True; break
            if term or trunc:
                rews.append(r + DEATH_PEN); break
            rews.append(r)
        env.close()
        g, ret = 0.0, np.zeros(len(rews), np.float32)
        for t in range(len(rews) - 1, -1, -1):
            g = rews[t] + GAMMA * g; ret[t] = g
        O += obs; A += acts; G.append(ret); stats[0] += 1; stats[1] += cleared
    np.savez_compressed(f"{outdir}/awr_w{wid:03d}.npz", obs=np.array(O, np.uint8),
                        act=np.array(A, np.int8), ret=np.concatenate(G))
    return stats, len(O)


def collect(outdir, n_ep=2400, workers=100):
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(outdir, exist_ok=True)
    per = max(1, n_ep // workers)
    tot, ok, frames = 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (n, c), f in pool.map(_collect_worker, [(w, per, outdir) for w in range(workers)]):
            tot += n; ok += c; frames += f
    print(f">>> 采集完：{tot} 局，老师(混合采样)通关 {ok} 局 = {ok/tot*100:.1f}%，共 {frames:,} 帧", flush=True)


def train(datadir, outp):
    import torch as th
    from stable_baselines3 import PPO
    from stub_env import make_stub_env
    from wide_cnn import WideNatureCNN
    dev = "cuda"
    files = sorted(glob.glob(f"{datadir}/*.npz"))
    obs = np.concatenate([np.load(f)["obs"] for f in files])
    act = np.concatenate([np.load(f)["act"] for f in files]).astype(np.int64)
    ret = np.concatenate([np.load(f)["ret"] for f in files]).astype(np.float32)
    N = len(obs)
    print(f">>> 数据 {N:,} 帧，回报均值 {ret.mean():.1f} 标准差 {ret.std():.1f}", flush=True)
    O = th.from_numpy(obs).to(dev); Aa = th.from_numpy(act).to(dev); R = th.from_numpy(ret).to(dev)
    del obs

    # —— 1. 价值网络：独立一份，不和策略共享主干（共享的话回归 V 的梯度会改到策略） ——
    from gymnasium import spaces
    sp = spaces.Box(0, 255, (4, 84, 84), np.uint8)
    vnet = th.nn.Sequential(WideNatureCNN(sp, features_dim=512), th.nn.Linear(512, 1)).to(dev)
    vo = th.optim.Adam(vnet.parameters(), lr=2.5e-4)
    rm, rs = float(R.mean()), float(R.std())
    for ep in range(4):
        idx = th.randperm(N, device=dev); tot = 0.0
        for i in range(0, N, BATCH):
            b = idx[i:i + BATCH]
            v = vnet(O[b]).squeeze(1)
            l = ((v - (R[b] - rm) / rs) ** 2).mean()
            vo.zero_grad(); l.backward(); vo.step(); tot += l.item() * len(b)
        print(f"   V epoch {ep+1}: 归一化 MSE {tot/N:.4f}（1.0 = 只会猜均值）", flush=True)
    with th.no_grad():
        V = th.cat([vnet(O[i:i + 4096]).squeeze(1) for i in range(0, N, 4096)]) * rs + rm
    Adv = R - V
    An = Adv / (Adv.std() + 1e-8)
    W = th.exp(th.clamp(An / BETA, max=np.log(WMAX)))
    print(f">>> 优势：std {float(Adv.std()):.1f}；权重 均值 {float(W.mean()):.2f} 中位 {float(W.median()):.2f} "
          f"截断占比 {float((W >= WMAX - 1e-3).float().mean())*100:.1f}%", flush=True)
    del vnet

    # —— 2. 从老师出发做加权模仿 ——
    m = PPO.load(TEACHER, device=dev)
    opt = th.optim.Adam(m.policy.parameters(), lr=LR)
    save_every = max(1, (N // BATCH) // 3)
    os.makedirs(f"ckpt_{outp}", exist_ok=True)
    k = 0
    for ep in range(EPOCHS):
        idx = th.randperm(N, device=dev); tot = 0.0; t0 = time.time()
        for j, i in enumerate(range(0, N, BATCH)):
            b = idx[i:i + BATCH]
            logp = m.policy.get_distribution(O[b]).distribution.log_prob(Aa[b])
            l = -(W[b] * logp).mean()
            opt.zero_grad(); l.backward(); opt.step(); tot += l.item() * len(b)
            if (j + 1) % save_every == 0:
                k += 1; m.save(f"ckpt_{outp}/{outp}_{k:02d}_steps.zip")
        print(f"   π epoch {ep+1}: 加权 NLL {tot/N:.4f}  {time.time()-t0:.0f}s", flush=True)
    print(f">>> 存了 {k} 个档 → ckpt_{outp}/", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2400,
                int(sys.argv[4]) if len(sys.argv) > 4 else 100)
    else:
        train(sys.argv[2], sys.argv[3])
