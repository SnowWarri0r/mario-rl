"""2-2：在老师会死的地方，用模拟器存档搜出能活下来的按键，当硬标签补给老师（搜索当老师）。

为什么：换随机数续训（diag_22_rng.py）治好了"背鱼群"，但没见过的鱼群下老师躲鱼只有约五成，
连打进关 41% → 50%。RL 自己在 2-2 上反复续训都学不出更好的躲法；存档搜索能直接给出"这里该怎么按"。
开源的存档最佳优先搜索在 2-2 上实测能搜通，说明活路一般存在。

collect：老师 argmax 打 2-2（每局随机开局相位 + 训练池里随机一套游戏随机数）。死在第 T 步时，
  退回 T-back 步存档（back 依次试 6/12/20），从那里起采样一批候选：前 back+4 步按老师概率加温采样
  （温度 1/2/4 轮换），之后老师 argmax 接管；能比原死亡点多活 40 步（或过关）算成功，
  成功里挑偏离老师 argmax 次数最少的那条（改动最小 = 最像老师自己会做的事，像素上更可能学得会）。
  把这段偏离的观测和动作记成硬标签，然后沿成功轨迹继续让老师打，再死再搜，每局最多 4 次。
  老师一路上经过的状态每 4 步留一个，记老师自己的概率，微调时当锚（防止改着改着把别处带坏）。
finetune：硬标签交叉熵 + 锚点 KL(老师旧分布 || 新分布)，只训策略头和骨干，存成 PPO zip。

⚠️ 存档：nes-py 只有一个备份槽，而且 reset 会恢复到这个槽。所以每次要回到 t0 都新建环境、
从头重放动作前缀（确定性，重放后核对 x 坐标），在 t0 `_backup()` 之后这个环境就只用于搜索，不再 reset。
恢复存档后要手动清 `done`（nes-py 的 _restore 不清，踩过）并还原 FrameStack 的帧栈。

结果（2026-10-09，一轮，未过预设标准）：老师 = 随机数+粘滞续训的 B（750 万步档）。
采集 1200 局：不修自过 570，修补后过关 1147；死亡点里约 92% 搜得出活路（退 6 步 690 次、12 步 98、20 步 10），
得硬标签 8708、锚点 15.3 万。微调 20 轮：s1（lr 1e-5，硬标签命中 0.75→0.84）、s3（lr 3e-5，→0.91）。
单关留出随机数（新种子 5000）：B 63 / s1 65 / s3 65（/121）。
连打首次进 2-2：s1 121/243=49.8%、s3 107/242=44.2%，同批复测 B 123/253=48.6%（两次合计 B 242/489=49.5%）。
⇒ 搜出来的活路学进去了（训练集命中率涨），但换了鱼群不迁移；lr 大的 s3 反而更差。
推断：活路依赖搜索看得到的完整状态（含随机数、鱼还没出屏），像素上学不出来。

用法:
  python search_22.py collect <老师.zip> <局数> <并发> <输出.npz>
  python search_22.py finetune <老师.zip> <数据.npz> <输出.zip> [轮数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, collections
os.environ.setdefault("OMP_NUM_THREADS", "1")
assert not os.environ.get("MARIO_RNG_POOL"), "随机数由本脚本自己写"
import numpy as np

POOL = os.environ.get("MARIO_RNG_TRAIN_POOL", "rng22_pool.npy")   # 只用偶数行（训练半边）
BACKS = (6, 12, 20)
EXTRA = 40           # 比原死亡点多活多少步才算活路
NCAND = int(os.environ.get("MARIO_NCAND", "48"))
MAXFIX = 4
ANCHOR_EVERY = 4


def _load(path):
    import torch as th; th.set_num_threads(1)
    from stable_baselines3 import PPO
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    return PPO.load(path, device="cpu")


def _probs(m, o):
    import torch as th
    ot, _ = m.policy.obs_to_tensor(o)
    with th.no_grad():
        return m.policy.get_distribution(ot).distribution.probs[0].numpy()


def _fresh(phase, row):
    from make_env import make_env, _nes_of
    env = make_env(stages=["2-2"], noop=phase, exact=True)
    o, _ = env.reset()
    n = _nes_of(env)
    n.ram[0x07A7:0x07AE] = row[:7]
    n.ram[0x09] = row[7]
    return env, n, o


def _x(n):
    return int(n.ram[0x6D]) * 256 + int(n.ram[0x86])


def _fs(env):
    e = env
    while not hasattr(e, "frames"):
        e = e.env
    return e


def _done(info, term, trunc):
    """返回 'clear' / 'dead' / None"""
    if info.get("flag_get") or (info.get("world"), info.get("stage")) != (2, 2):
        return "clear"
    if term or trunc:
        return "dead"
    return None


def _play(m, env, o, limit, first=None, temp=None, rng=None):
    """从当前状态走：first 步按 temp 加温采样（之后 argmax）。返回 (结局, 步数, 动作, 观测, 偏离数)。"""
    acts, obs, dev = [], [], 0
    for t in range(limit):
        p = _probs(m, o)
        a = int(p.argmax())
        if first is not None and t < first:
            q = p ** (1.0 / temp); q /= q.sum()
            a = int(rng.choice(len(q), p=q))
            dev += a != int(p.argmax())
        obs.append(o); acts.append(a)
        o, r, term, trunc, info = env.step(a)
        d = _done(info, term, trunc)
        if d:
            return d, t + 1, acts, obs, dev, o
    return None, limit, acts, obs, dev, o


def _replay(phase, row, actions):
    env, n, o = _fresh(phase, row)
    for a in actions:
        o, r, term, trunc, info = env.step(a)
        assert not _done(info, term, trunc), "重放提前结束：环境不确定？"
    return env, n, o


def _search(m, phase, row, hist, xs, rng):
    """hist：到死亡为止的完整动作序列。返回 (成功后的完整动作前缀, 硬标签观测, 硬标签动作) 或 None。"""
    T = len(hist)
    for back in BACKS:
        t0 = max(0, T - back)
        env, n, o0 = _replay(phase, row, hist[:t0])
        if t0 and _x(n) != xs[t0 - 1]:
            env.close(); raise RuntimeError(f"重放 x 对不上 {_x(n)} vs {xs[t0 - 1]}")
        n._backup()
        fs = _fs(env); frames0 = list(fs.frames)
        best = None
        for c in range(NCAND):
            n._restore(); n.done = False
            fs.frames.clear(); fs.frames.extend(frames0)
            temp = (1.0, 2.0, 4.0)[c % 3]
            d, steps, acts, obs, dev, _ = _play(m, env, o0, (T - t0) + EXTRA, first=back + 4, temp=temp, rng=rng)
            if d == "dead":
                continue
            if best is None or dev < best[0]:
                best = (dev, acts, obs, d)
            if dev <= 1:
                break
        env.close()
        if best is not None:
            dev, acts, obs, d = best
            k = back + 4                         # 硬标签 = 加温段（含和老师 argmax 相同的步）
            return hist[:t0] + acts, obs[:k], acts[:k], back, d == "clear"
    return None


def collect_one(job):
    teacher, ep, seed = job
    rng = np.random.default_rng(seed)
    m = _load(teacher)
    pool = np.load(POOL)[0::2]
    phase = int(rng.integers(0, 121)); row = pool[int(rng.integers(len(pool)))]
    env, n, o = _fresh(phase, row)
    hist, xs = [], []
    lab_o, lab_a, anc_o, anc_p = [], [], [], []
    fixes, outcome, backs = 0, None, []
    for t in range(4000):
        p = _probs(m, o)
        if t % ANCHOR_EVERY == 0:
            anc_o.append(o); anc_p.append(p)
        a = int(p.argmax())
        o, r, term, trunc, info = env.step(a)
        hist.append(a); xs.append(_x(n))
        d = _done(info, term, trunc)
        if d == "clear":
            outcome = "clear"; break
        if d == "dead":
            env.close()
            if fixes >= MAXFIX:
                outcome = "dead"; break
            res = _search(m, phase, row, hist, xs, rng)
            fixes += 1
            if res is None:
                outcome = "nofix"; break
            prefix, so, sa, back, cleared = res
            lab_o += so; lab_a += sa; backs.append(back)
            if cleared:                              # 候选一路打到了旗杆
                outcome = "clear"; break
            env, n, o = _fresh(phase, row)          # 重放到成功轨迹末尾，顺便补 x 记录
            hist, xs = list(prefix), []
            for a2 in prefix:
                o, *_ = env.step(a2); xs.append(_x(n))
    else:
        outcome = "timeout"
    try:
        env.close()
    except Exception:
        pass
    return dict(ep=ep, outcome=outcome, fixes=fixes, backs=backs, x=xs[-1] if xs else 0,
                lab_o=np.array(lab_o, np.uint8).reshape(-1, 4, 84, 84), lab_a=np.array(lab_a, np.int64),
                anc_o=np.array(anc_o, np.uint8).reshape(-1, 4, 84, 84), anc_p=np.array(anc_p, np.float32).reshape(-1, 7))


def collect(teacher, n_ep, workers, out):
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(collect_one, [(teacher, e, 7000 + e) for e in range(n_ep)], chunksize=1)):
            res.append(r)
            if (i + 1) % 50 == 0:
                c = collections.Counter(x["outcome"] for x in res)
                print(f"[{i+1}/{n_ep}] {dict(c)}  标签 {sum(len(x['lab_a']) for x in res)}  "
                      f"{time.time()-t0:.0f}s", flush=True)
    c = collections.Counter(x["outcome"] for x in res)
    fx = collections.Counter(b for x in res for b in x["backs"])
    print(f"结局 {dict(c)}；修补次数分布 {collections.Counter(x['fixes'] for x in res)}；成功修补的退回步数 {dict(fx)}")
    np.savez_compressed(out,
                        lab_o=np.concatenate([x["lab_o"] for x in res]), lab_a=np.concatenate([x["lab_a"] for x in res]),
                        anc_o=np.concatenate([x["anc_o"] for x in res]), anc_p=np.concatenate([x["anc_p"] for x in res]))
    print(f"存 {out}", flush=True)


def finetune(teacher, data, out, epochs=4):
    import torch as th
    import torch.nn.functional as F
    from stable_baselines3 import PPO
    import wide_cnn, big_cnn, impala_cnn  # noqa: F401
    dev = "cuda" if th.cuda.is_available() else "cpu"
    m = PPO.load(teacher, device=dev)
    pol = m.policy; pol.train()
    d = np.load(data)
    lo, la, ao, ap = d["lab_o"], d["lab_a"], d["anc_o"], d["anc_p"]
    print(f"硬标签 {len(la)}，锚点 {len(ap)}，动作分布 {np.bincount(la, minlength=7)}", flush=True)
    LR = float(os.environ.get("MARIO_FT_LR", "1e-5")); W = float(os.environ.get("MARIO_FT_ANCHOR", "1.0"))
    opt = th.optim.Adam(pol.parameters(), lr=LR)
    rng = np.random.default_rng(0)
    B = 256
    steps = max(1, epochs * len(la) // (B // 2))
    for s in range(steps):
        i = rng.integers(len(la), size=B // 2); j = rng.integers(len(ap), size=B // 2)
        xo = th.as_tensor(lo[i], device=dev); ya = th.as_tensor(la[i], device=dev)
        xa = th.as_tensor(ao[j], device=dev); pa = th.as_tensor(ap[j], device=dev)
        lg_o = pol.get_distribution(xo).distribution.logits
        lg_a = pol.get_distribution(xa).distribution.logits
        ce = F.cross_entropy(lg_o, ya)
        kl = (pa * (th.log(pa.clamp_min(1e-8)) - F.log_softmax(lg_a, -1))).sum(-1).mean()
        loss = ce + W * kl
        opt.zero_grad(); loss.backward(); th.nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
        if s % 200 == 0 or s == steps - 1:
            acc = (lg_o.argmax(-1) == ya).float().mean().item()
            print(f"step {s}/{steps} ce {ce.item():.3f} kl {kl.item():.4f} 硬标签命中 {acc:.2f}", flush=True)
    m.save(out)
    print(f"存 {out}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "collect":
        collect(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    elif mode == "finetune":
        finetune(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 4)
