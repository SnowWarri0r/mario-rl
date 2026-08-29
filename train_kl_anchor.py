"""带 KL 锚的 PPO 微调：给"先蒸后 RL 必炸"这个坑做最后一个候选解释的判决实验。

**为什么需要它。** 蒸馏出来的学生在 2-2 上有 51%(argmax)/39%(采样)，但只要拿去 PPO 微调，
25 万步就归零——补了价值头照样归零（EV≈0.48，不是随机 V 了），压到 lr=1e-5 也挡不住。
排除掉存取往返、价值头、冻结逻辑之后，剩下的解释是结构性的：
**蒸馏策略不是这个 RL 目标的不动点**。champ 能在同样的手术下 34%→88%，是因为它本来就停在
那个目标的驻点附近；v5 只是在模仿 champ 的行为，对"带梯子塑形的 2-2 奖励"而言它站在一个
任意位置，PPO 会拖着它走向那个目标的最优点，而这条路要穿过一大片很差的区域。
lr 小只是让它走得慢，不改变方向——64 环境 × n_steps 64 × 10 epoch，25 万步已经是近万次
Adam 更新，对 686 万参数足够走很远。

**修法**＝RLHF 那套的 KL 锚：损失里显式加一项 `β·KL(π_ref ‖ π)`，把策略钉在参考策略附近，
让 RL 只能在它周围找改进，不许走远。这是可判决的：加了锚还炸，说明我这条归因也错了。

**实现取巧的地方（重要，别看漏）**：没有重写 sb3 的 `PPO.train`（那是照抄一大段实现，跨版本易碎），
而是利用 sb3 的这一行——
    loss = policy_loss + self.ent_coef * (-mean(entropy)) + vf_coef * value_loss
把 `evaluate_actions` 返回的 `entropy` 换成 `-KL(π_ref‖π)`，那一项就正好变成 `+ent_coef·mean(KL)`。
于是 **`ent_coef` 在这个脚本里是 KL 系数 β，不是熵系数**；熵奖励本身被完全去掉（2-2 本来就要它为零）。
日志里 sb3 打的 `entropy_loss` 也随之变成 mean(KL)，别读错。

用法: MARIO_NOOP=30 MARIO_BASE=x.zip MARIO_BASE_VECN=x.pkl MARIO_BETA=0.1 \
      python train_kl_anchor.py [步数] [并行环境数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, copy
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from make_env import make_env, make_env_stage22_ladder_noop, NOOP_JITTER

DEVICE = os.environ.get("MARIO_DEVICE", "cuda")
BASE = os.environ.get("MARIO_BASE", "mario_v5_vwarm_ladder.zip")
BASE_VECN = os.environ.get("MARIO_BASE_VECN", "vecnormalize_mario_v5_vwarm_ladder.pkl")
OUT = os.environ.get("MARIO_OUT", "mario_v5_22kl")
CKPT_DIR = os.environ.get("MARIO_CKPT_DIR", f"./checkpoints_{OUT}")
BETA = float(os.environ.get("MARIO_BETA", "0.1"))       # KL 系数；0 就退化成普通 PPO（当对照用）
LR = float(os.environ.get("MARIO_LR", "1e-5"))
SAVE_FREQ = int(os.environ.get("MARIO_SAVE_FREQ", "250000"))
FACTORY = os.environ.get("MARIO_FACTORY", "ladder22")
STAGES = os.environ.get("MARIO_STAGES", "2-2").split(",")


def factory():
    return make_env_stage22_ladder_noop() if FACTORY == "ladder22" else make_env(stages=STAGES)


def attach_kl_anchor(model, ref_policy, beta):
    """把 evaluate_actions 的第三个返回值从 entropy 换成 -KL(ref‖pi)。"""
    orig = model.policy.evaluate_actions

    def patched(obs, actions):
        values, log_prob, _entropy = orig(obs, actions)
        with th.no_grad():
            ref_logits = ref_policy.get_distribution(obs).distribution.logits
            ref_p = th.softmax(ref_logits, dim=-1)
            ref_logp = th.log_softmax(ref_logits, dim=-1)
        cur_logp = th.log_softmax(
            model.policy.get_distribution(obs).distribution.logits, dim=-1)
        kl = (ref_p * (ref_logp - cur_logp)).sum(-1)     # KL(ref ‖ pi)，逐样本
        return values, log_prob, -kl                     # 负号：sb3 会再取一次负

    model.policy.evaluate_actions = patched
    model.ent_coef = beta


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    assert NOOP_JITTER, "记得 MARIO_NOOP=30"

    venv = make_vec_env(factory, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    assert os.path.exists(BASE_VECN), f"缺 {BASE_VECN}——奖励尺度对不上会白跑一轮，这坑摔过三次了"
    venv = VecNormalize.load(BASE_VECN, venv)
    venv.training = True; venv.norm_reward = True

    model = PPO.load(BASE, env=venv, device=DEVICE)
    model.learning_rate = LR
    # 参考策略＝base 的一份冻结拷贝，训练全程不动
    ref = copy.deepcopy(model.policy).to(model.device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    attach_kl_anchor(model, ref, BETA)

    print(f">>> KL 锚微调 {total:,} 步 | {n_envs} 环境 | 续 {BASE} | β(KL) {BETA} | lr {LR} | "
          f"抖动 0-{NOOP_JITTER} | 环境 {FACTORY} | device {DEVICE}", flush=True)
    print(f"    注意：sb3 日志里的 entropy_loss 现在是 mean(KL)，不是熵", flush=True)

    ckpt = CheckpointCallback(save_freq=max(SAVE_FREQ // n_envs, 1),
                              save_path=CKPT_DIR, name_prefix=OUT)
    model.learn(total_timesteps=total, callback=ckpt, reset_num_timesteps=True)
    # ⚠️存之前把 patch 摘掉：evaluate_actions 上挂着闭包（引用了 ref），pickle 不了
    del model.policy.evaluate_actions
    model.save(OUT)
    venv.save(f"vecnormalize_{os.path.basename(OUT)}.pkl")
    print(f">>> 存为 {OUT}.zip", flush=True)


if __name__ == "__main__":
    main()
