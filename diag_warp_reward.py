"""迷宫回卷那一步，奖励到底是多少：环里绕圈是不是净赚。

4-4 的 ep_rew_mean 是 7.6e3，别的关只有 2-3e3；同时实测它每局 x 都要回撤 ~1000。
一个可疑的机制是：原生 SMB 奖励是 delta-x，而 delta-x **按帧截断在 ±15**——
往前走 1000 是分 ~80 步慢慢挣的（每步 ~12，不触顶），被传回起点却是**一步**掉 1000，
截断后只扣 15。那样一圈净赚 ~985，agent 会学会在环里绕圈刷分而不是通关。

但这只是个说得通的故事，这个项目上这类故事命中率是八分之一。所以直接量：
**逮住 x 大幅回退的那一步，打印它的即时奖励**，再统计整局的正负奖励收支。

判据：回卷步的奖励若约等于 -(回退量)＝奖励是对称的，绕圈不赚，塑形没问题；
若回卷步只扣个位数/十几分＝确认被截断，环是可刷的，必须换奖励（用 max-x 单调势能，
即只对"刷新历史最远"发奖，回退不扣也不赚）。

用法: python diag_warp_reward.py <模型> [关卡] [局数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_w4.zip"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "4-4"
EPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAXSTEP = int(os.environ.get("MARIO_MAXSTEP", "2500"))
DROP = int(os.environ.get("MARIO_DROP", "200"))     # x 一步掉这么多算回卷


def main():
    import torch as th
    from stable_baselines3 import PPO
    from make_env import make_env
    import wide_cnn  # noqa: F401

    model = PPO.load(MODEL, device="cpu")
    env = make_env(stages=[STAGE], noop=30)
    print(f"=== {STAGE} 回卷步的奖励 | {MODEL} | {EPS} 局 ===", flush=True)

    for ep in range(EPS):
        o, _ = env.reset()
        prev_x, tot, pos, neg, warps = None, 0.0, 0.0, 0.0, []
        for t in range(MAXSTEP):
            ot, _ = model.policy.obs_to_tensor(o)
            with th.no_grad():
                p = model.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
            o, r, term, trunc, info = env.step(int(np.random.choice(len(p), p=p / p.sum())))
            x = int(info.get("x_pos", 0))
            tot += r
            pos += max(r, 0.0); neg += min(r, 0.0)
            if prev_x is not None and prev_x - x >= DROP:
                warps.append((t, prev_x, x, prev_x - x, r))
            prev_x = x
            if term or trunc:
                break
        print(f"\n第{ep+1}局：{t+1} 步，总奖励 {tot:8.1f}（正 {pos:8.1f} / 负 {neg:8.1f}），"
              f"回卷 {len(warps)} 次")
        for t_, a, b, d, r_ in warps:
            print(f"   步{t_:5d}  x {a} → {b}（退 {d}）  这一步奖励 = {r_:+8.2f}"
                  f"   {'← 被截断，环可刷' if abs(r_) < d * 0.5 else '← 对称扣回，环不可刷'}")
    env.close()
    print("\n>>> 若回卷步的扣分远小于回退量，x 塑形就在给「绕圈」发钱，"
          "得换成只奖励刷新历史最远 x 的单调势能", flush=True)


if __name__ == "__main__":
    main()
