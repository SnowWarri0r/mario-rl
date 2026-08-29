"""KL 锚的自检：确认那个"把 entropy 换成 -KL"的取巧写法真的接对了。

查三件事：
  ① 刚挂上时 ref 和 pi 逐字相同 → KL 必须是 0（不是"很小"，是 0）
  ② 手动把策略推歪一点 → KL 必须变正，且等于手算的 Σ p_ref·(log p_ref − log p_pi)
  ③ 摘掉 patch 之后 evaluate_actions 恢复成原来的（返回真熵，是正数）
不写成 heredoc 是因为 SubprocVecEnv 的 forkserver 子进程会重新 import __main__，
`python - <<EOF` 会炸 FileNotFoundError: '<stdin>'（踩过）。
"""
import warnings; warnings.filterwarnings("ignore")
import os, copy
import numpy as np
import torch as th
from stable_baselines3 import PPO
from train_kl_anchor import attach_kl_anchor

BASE = os.environ.get("MARIO_BASE", "mario_v5_vwarm_ladder.zip")
model = PPO.load(BASE, device="cuda")
ref = copy.deepcopy(model.policy).eval()
for p in ref.parameters():
    p.requires_grad_(False)
attach_kl_anchor(model, ref, 0.1)

obs = th.randint(0, 255, (64, 4, 84, 84), dtype=th.uint8, device=model.device)
acts = th.randint(0, 7, (64,), device=model.device)

_, _, neg_kl = model.policy.evaluate_actions(obs, acts)
print(f"① 挂上时 max|KL| = {neg_kl.abs().max().item():.3e}（应为 0）")
assert neg_kl.abs().max().item() == 0.0, "起点 KL 不为 0，ref 拷贝或计算接错了"

with th.no_grad():                       # 把最后一层偏置推歪，制造真实差异
    model.policy.action_net.bias += th.randn_like(model.policy.action_net.bias) * 0.5
_, _, neg_kl2 = model.policy.evaluate_actions(obs, acts)
with th.no_grad():
    rp = th.softmax(ref.get_distribution(obs).distribution.logits, -1)
    rlp = th.log_softmax(ref.get_distribution(obs).distribution.logits, -1)
    clp = th.log_softmax(model.policy.get_distribution(obs).distribution.logits, -1)
    manual = (rp * (rlp - clp)).sum(-1)
print(f"② 推歪后 mean(KL) = {(-neg_kl2).mean().item():.4f}，手算 {manual.mean().item():.4f}")
assert (-neg_kl2).mean() > 1e-4, "推歪了 KL 还是 0"
assert th.allclose(-neg_kl2, manual, atol=1e-6), "跟手算对不上"

# sb3 会算 entropy_loss = -mean(entropy)，再加 ent_coef*entropy_loss 进总损失
print(f"   → 进损失的那一项 = ent_coef·mean(KL) = {0.1 * (-neg_kl2).mean().item():.5f}（应为正＝惩罚）")
assert 0.1 * (-neg_kl2).mean().item() > 0

del model.policy.evaluate_actions
_, _, ent = model.policy.evaluate_actions(obs, acts)
print(f"③ 摘掉 patch 后返回真熵 mean = {ent.mean().item():.4f}（应为正）")
assert ent.mean() > 0, "摘不干净，存档时会带着闭包 pickle 失败"
print(">>> 三项全过")
