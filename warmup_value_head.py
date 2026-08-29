"""给蒸馏出来的学生补一个价值头，然后才允许它进 RL 微调。

**这是为一个摔过两次的坑写的**：蒸馏只训策略头（损失是 -Σ p_teacher·log q_student，
根本没碰 value_net），所以学生的 V 是随机初始化的。直接拿去 PPO 微调，前几十次更新的
advantage = r + γV(s') − V(s) 全是噪声，等于朝随机方向推一个本来很好的策略——
实测第一级 1M 步就把 52% 打成 1%（叠帧桥那次），W1 pilot 那次是 55%→1%。

修法：先用学生自己跑出来的轨迹，把**蒙特卡洛回报**回归进 value_net，策略一动不动。
关键是**冻住共享的 CNN 主干**——sb3 的 `CnnPolicy` 默认 pi/vf 共用 features_extractor，
不冻的话补 V 的梯度会顺着主干爬回去改策略（光冻 action_net 没用，这点上次记反了）。
冻住之后这一步退化成"在固定特征上拟合一个线性头"，几分钟就收敛，且**策略逐字不变**。

注意 VecNormalize：回报必须在**归一化后**的奖励尺度上算，因为 PPO 微调时 V 面对的就是那个尺度。
所以这里直接复用 base 的 pkl（缺了就别跑——这也是摔过两次的地方）。

用法: MARIO_BASE=mario_v5_wide.zip MARIO_BASE_VECN=... MARIO_OUT=mario_v5_vwarm \
      MARIO_STAGES=2-2 python warmup_value_head.py [采样步数] [并行环境数] [epochs]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from make_env import make_env, NOOP_JITTER

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
N_ENVS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
BASE = os.environ.get("MARIO_BASE", "mario_v5_wide.zip")
BASE_VECN = os.environ.get("MARIO_BASE_VECN", "")
OUT = os.environ.get("MARIO_OUT", "mario_v5_vwarm")
STAGES = os.environ.get("MARIO_STAGES", "2-2").split(",")
DEVICE = os.environ.get("MARIO_DEVICE", "cuda")
GAMMA = 0.99


# ⚠️补 V 用的环境必须跟后面微调用的**逐字相同**，尤其是奖励塑形：
# 在裸 2-2 上拟合的 V 拿到带梯子 bonus 的环境里，尺度直接对不上，等于没补。
FACTORY = os.environ.get("MARIO_FACTORY", "plain")


def factory():
    if FACTORY == "ladder22":
        from make_env import make_env_stage22_ladder_noop
        return make_env_stage22_ladder_noop()
    return make_env(stages=STAGES)


def main():
    assert NOOP_JITTER, "补 V 也要在抖动分布上补，记得 MARIO_NOOP=30"
    per_env = STEPS // N_ENVS
    venv = make_vec_env(factory, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    if BASE_VECN and os.path.exists(BASE_VECN):
        venv = VecNormalize.load(BASE_VECN, venv)
        venv.training = False          # 采样期间别再动统计，V 要拟合的就是当前这套尺度
        venv.norm_reward = True
        print(f">>> 载入 {BASE_VECN}（奖励尺度沿用 base）", flush=True)
    else:
        # 没有 pkl 就现场估一段统计。⚠️这不如复用 base 的，但比"静默新建然后训崩"好，至少这里说出来了
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
        print(f"!!! 没有 {BASE_VECN!r}，现场新建奖励统计——微调时的 advantage 尺度会和 base 对不上", flush=True)

    model = PPO.load(BASE, env=venv, device=DEVICE)
    pol = model.policy
    print(f">>> 用 {BASE} 在 {STAGES} 上跑 {STEPS:,} 步收回报（{N_ENVS} 环境 × {per_env} 步）", flush=True)

    obs_l, rew_l, done_l = [], [], []
    o = venv.reset()
    for t in range(per_env):
        with th.no_grad():
            a, _ = model.predict(o, deterministic=False)
        obs_l.append(o.astype(np.uint8))
        o, r, d, _ = venv.step(a)
        rew_l.append(r.astype(np.float32)); done_l.append(d.astype(bool))
        if (t + 1) % max(per_env // 10, 1) == 0:
            print(f"    采样 {(t+1)*N_ENVS:,}/{STEPS:,}", flush=True)
    with th.no_grad():
        last_v = pol.predict_values(pol.obs_to_tensor(o)[0]).cpu().numpy().flatten()
    # ⚠️必存：补 V 用的是这套奖励尺度，后面 PPO 微调必须接着用同一套，否则 V 又对不上了
    vecn_out = f"vecnormalize_{os.path.basename(OUT)}.pkl"
    venv.save(vecn_out)
    print(f">>> 奖励统计存为 {vecn_out}（微调时用 MARIO_BASE_VECN 指向它）", flush=True)
    venv.close()

    # 逆序算折扣回报；截断处（时间上限）用 bootstrap，真结束处置零。
    # 这里不区分 terminated/truncated（VecEnv 只给一个 done），对补 V 的精度影响小于噪声。
    rews = np.array(rew_l, np.float32); dones = np.array(done_l, bool)
    rets = np.zeros_like(rews)
    run = last_v.copy()
    for t in range(per_env - 1, -1, -1):
        run = rews[t] + GAMMA * run * (~dones[t])
        rets[t] = run
    obs = np.concatenate([o[None] for o in obs_l], 0).reshape(-1, *obs_l[0].shape[1:])
    y = rets.reshape(-1)
    print(f">>> {len(y):,} 条回报 | 均值 {y.mean():.2f} 方差 {y.var():.2f} "
          f"| 当前 V 预测均值 ?（下面 epoch0 的 loss 就是它的起点）", flush=True)

    # 冻住除 value_net 之外的一切：共享主干也要冻，否则补 V 会顺着主干改掉策略
    for p in pol.parameters():
        p.requires_grad_(False)
    for p in pol.value_net.parameters():
        p.requires_grad_(True)
    if hasattr(pol.mlp_extractor, "value_net"):
        for p in pol.mlp_extractor.value_net.parameters():
            p.requires_grad_(True)
    trainable = [p for p in pol.parameters() if p.requires_grad]
    print(f">>> 只训 {sum(p.numel() for p in trainable):,} 个参数（共 "
          f"{sum(p.numel() for p in pol.parameters()):,}）", flush=True)

    dev = model.device
    obs_g = th.from_numpy(obs).to(dev)
    y_g = th.as_tensor(y, dtype=th.float32, device=dev)
    opt = th.optim.Adam(trainable, lr=1e-3)
    # 存一份策略输出，训完逐字核对策略确实没被动过
    probe = obs_g[:256]
    with th.no_grad():
        before = pol.get_distribution(probe).distribution.probs.clone()

    N, B = len(y_g), 1024
    for ep in range(EPOCHS):
        idx = th.randperm(N, device=dev); tot = 0.0
        for i in range(0, N, B):
            b = idx[i:i+B]
            with th.no_grad():
                feat = pol.extract_features(obs_g[b])
                if not pol.share_features_extractor:
                    feat = feat[1]
            v = pol.value_net(pol.mlp_extractor.forward_critic(feat)).flatten()
            loss = th.nn.functional.mse_loss(v, y_g[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"epoch {ep+1}/{EPOCHS}  value MSE {tot/N:.4f}", flush=True)

    with th.no_grad():
        after = pol.get_distribution(probe).distribution.probs
    drift = (after - before).abs().max().item()
    print(f">>> 策略最大漂移 {drift:.2e}（应该是 0；非零说明有梯度漏回主干了）", flush=True)
    assert drift < 1e-6, "策略被动了，检查冻结逻辑"

    model.save(OUT)
    print(f">>> 存为 {OUT}.zip —— 现在才可以拿去 PPO 微调", flush=True)


if __name__ == "__main__":
    main()
