"""2-2 · 带 no-op starts 重训：第一次真正要求它「看着鱼做决定」而不是背舞步。

背景：2-2 那个 70%（梯子塑形拿的）在开局抖 2 帧后腰斩、抖 30 帧只剩 4%——整套动作跟鱼的位置
是帧级锁死的。加上 no-op starts（每局随机空按 0-30 帧）后固定序列彻底失效，梯度只剩一条路：
学会看画面里鱼在哪。梯子奖励（2100/2400/2700/2900 各 +50）保留，密集信号还是要的。

warm-start 自梯子专家：它已经会游泳、会往右、会走完全程，只是不会应变；从零开始要重付那部分学费。
用法: python train_2_2_noop.py [步数] [并行环境数]   默认 500 万 / 32
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env_stage22_ladder_noop, make_env_stage22_archive, NOOP_JITTER, STACK_FRAMES

DEVICE = os.environ.get("MARIO_DEVICE", "cpu")
ARCH = os.environ.get("MARIO_ARCH", "ff")        # ff=前馈 CnnPolicy；lstm=RecurrentPPO CnnLstmPolicy   # PPO 的梯度更新放 GPU 上能快不少，采样还是 CPU 在跑
CKPT_DIR = os.environ.get("MARIO_CKPT_DIR", "./checkpoints_22noop")
ENT = os.environ.get("MARIO_ENT")      # 想把策略从"背下来的解"里踹出来时调高(0.05)，默认沿用存档里的 0.02
OUT = os.environ.get("MARIO_OUT", "mario_22noop_final")

BASE_MODEL = os.environ.get("MARIO_BASE", "mario_22ladder_final.zip")
BASE_VECN = os.environ.get("MARIO_BASE_VECN", "vecnormalize_22ladder.pkl")


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 32       # 192 核的机器，并行环境可以开大
    # MARIO_FACTORY=archive → 从硬点前的存档点开局（Backplay），否则从关卡开头
    factory = (make_env_stage22_archive if os.environ.get("MARIO_FACTORY") == "archive"
               else make_env_stage22_ladder_noop)
    venv = make_vec_env(factory, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    if os.path.exists(BASE_VECN):
        venv = VecNormalize.load(BASE_VECN, venv)                # 接着上一阶段的奖励统计，别从头估
        venv.training = True; venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if ARCH == "lstm":
        from sb3_contrib import RecurrentPPO
        Algo, policy = RecurrentPPO, "CnnLstmPolicy"
    else:
        Algo, policy = PPO, "CnnPolicy"

    if os.path.exists(BASE_MODEL):
        model = Algo.load(BASE_MODEL, env=venv, device=DEVICE)
    else:
        # 从零训：换架构（叠帧数变了 / 上 LSTM）没法 warm-start，第一层卷积和网络结构都不一样。
        # 沿用梯子专家那套超参，便于跟 warm-start 那条线对比。
        if ARCH == "lstm":
            # LSTM 的 n_steps 同时是截断 BPTT 的长度，照抄前馈的 512 会让一次更新变成
            # 96 个 minibatch × 10 epoch = 960 次沿 512 步展开的序列前反向，慢到卡死（踩过）。
            # 缩短展开长度 + 加大 minibatch + 减 epoch：每次更新只 12 次 BPTT。
            kw = dict(n_steps=128, batch_size=2048, n_epochs=4)
        else:
            kw = dict(n_steps=512, batch_size=256)
        model = Algo(policy, venv, device=DEVICE, ent_coef=0.02, learning_rate=2.5e-4,
                     verbose=1, **kw)
        print(f">>> 从零训 {ARCH} | 叠 {STACK_FRAMES} 帧 | {kw}", flush=True)
    model.learning_rate = 2.5e-4
    if ENT:
        model.ent_coef = float(ENT)
    # ent_coef 默认沿用存档里的 0.02（当年 0.05 太高不收敛、0.01 太低会过早确定化，0.02 是试出来的档）。
    # n_steps 保持存档值 512：PPO.load 已按它建好 rollout buffer，事后改字段 buffer 不会跟着重建，
    # 凑不满就 assert self.full 失败（踩过）。32 环境 × 512 = 每轮 16384 步，5M 步约 305 次更新。
    print(f">>> 2-2 no-op 重训 {total:,} 步 | {n_envs} 环境 | 续 {BASE_MODEL} | 开局随机空按 0-{NOOP_JITTER} 帧 | "
          f"ent_coef {model.ent_coef} | n_steps {model.n_steps} (每轮 {n_envs * model.n_steps} 步) | device {DEVICE}", flush=True)

    ckpt = CheckpointCallback(save_freq=max(250_000 // n_envs, 1),
                              save_path=CKPT_DIR, name_prefix="mario_22noop")
    model.learn(total_timesteps=total, callback=ckpt, reset_num_timesteps=True)
    model.save(OUT)
    venv.save(f"vecnormalize_{os.path.basename(OUT)}.pkl")
    print(f">>> 2-2 no-op 重训结束，存为 {OUT}.zip", flush=True)


if __name__ == "__main__":
    main()
