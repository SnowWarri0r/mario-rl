"""证明 MaxXReward 真的让"绕圈"不赚钱。

不验这一步就开训，等于再赌一次机制故事。判据是可证伪的：
同一条轨迹（奖励不影响动力学），**回卷之后重复跑同一段的收益必须从正变负**。

⚠️ 两个判据上的坑，都踩过：
① 别要求"回卷步奖励 ≤0"。MaxXReward 挂在 SkipFrame **之前**，一个 step 汇总 4 个模拟器帧，
   含回卷的那个窗口里往往还夹着真实的前进（1045→1049），所以会是小正数。
   实测 +4.6 对应 1017 的回退量＝0.4%，是正常的。判据要用「相对回退量足够小」。
② 两侧必须跑**同一条轨迹**才可比。`NoopReset` 自带 `default_rng`，跟 `np.random.seed` 无关，
   所以设了 seed 两侧的相位仍然不同（实测一侧回卷 1 次、另一侧 2 次，根本不是同一条轨迹）。
   这里两侧都传 exact=<相位>（不是 MARIO_NOOP=0——相位 0 走不到环），并断言两侧回卷位置一致。

⚠️ 第一版**空过**了：喂的模型根本走不到环（0 次回卷），`all(...)` 对空列表恒真，
打印出一个绿色的 ✓。断言必须先断言"这次真的观察到了回卷"，否则测的是空气。
所以下面会自动换 seed 找出一条会回卷的轨迹，找不到就报失败而不是通过。

③ 还要验**脏读上限**：`x_pos` 偶尔读出 65535（16 位下溢），无截断的势能会为这一步发 +65535。
   实测中招时每局奖励双峰——正常局 100-900、中招局 65,000——`ep_rew_mean` 被拉到 2.3e4，
   而单局探针跑出来只有 850，两者差 27 倍我才发现。所以最后跑一遍**真·vec env**，
   断言没有任何一局的奖励超过关卡长度的合理上界。单环境探针看不出来，必须过 Monitor。

用法: ./venv/bin/python selftest_maxx.py <能走到环的模型>
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mario_w4.zip"
STAGE = os.environ.get("MARIO_STAGE", "4-4")
MAXSTEP = 700
DROP = 200
# 扫相位而不是扫 seed：走不走得到环由相位决定，而相位可以钉死，两侧才可比


def rollout(maze, phase):
    import torch as th
    from stable_baselines3 import PPO
    from make_env import make_env, build_maze_env
    import wide_cnn  # noqa: F401
    os.environ["MARIO_STAGE"] = STAGE
    env = (build_maze_env(STAGE, noop=phase, exact=True) if maze
           else make_env(stages=[STAGE], noop=phase, exact=True))
    model = PPO.load(MODEL, device="cpu")
    np.random.seed(0)      # 相位钉死后，动作采样也钉死
    o, _ = env.reset()
    prev_x, tot, warps, after_first = None, 0.0, [], 0.0
    seen_warp = False
    for t in range(MAXSTEP):
        ot, _ = model.policy.obs_to_tensor(o)
        with th.no_grad():
            p = model.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
        o, r, term, trunc, info = env.step(int(np.random.choice(len(p), p=p / p.sum())))
        x = int(info.get("x_pos", 0))
        tot += r
        if seen_warp:
            after_first += r                 # 第一次回卷之后挣的钱＝"重复跑同一段"的收益
        if prev_x is not None and prev_x - x >= DROP:
            warps.append((prev_x, x, r)); seen_warp = True
        prev_x = x
        if term or trunc:
            break
    env.close()
    return tot, warps, after_first


def main():
    print(f"=== MaxXReward 自检 | {STAGE} | {MODEL} ===", flush=True)
    # 先找一条真的会回卷的轨迹，否则这个自检没有被测对象
    seed = None
    for k in range(0, 31, 2):
        _, w, _ = rollout(False, k)
        print(f"  相位 {k}: 回卷 {len(w)} 次", flush=True)
        if w:
            seed = k; break
    assert seed is not None, (
        f"{MODEL} 在 {STAGE} 上扫过的相位都没走到环，这个自检无对象可测。"
        "换一个能跑到 x>1012 的档再来（别让它空过）。")
    print(f">>> 用相位 {seed}（这条轨迹会回卷）\n")

    for tag, maze in (("原生 delta-x", False), ("max-x 势能", True)):
        tot, warps, after = rollout(maze, seed)
        ws = ", ".join(f"{a}→{b} 得 {r:+.1f}" for a, b, r in warps) or "本局没回卷"
        print(f"\n{tag}：总奖励 {tot:8.1f} | 回卷 {len(warps)} 次 | "
              f"第一次回卷之后又挣了 {after:8.1f}")
        print(f"  回卷步：{ws}")
        base = (tot, warps, after) if not maze else base
        if maze:
            assert warps, "这一侧没观察到回卷，说明 make_env_maze 改了动力学"
            assert [(a, b) for a, b, _ in warps] == [(a, b) for a, b, _ in base[1]], (
                f"两侧回卷位置不同，不是同一条轨迹，不可比：{warps} vs {base[1]}")
            worst = max(r / (a - b) for a, b, r in warps)
            assert worst < 0.05, f"回卷步还在按回退量的 {worst:.1%} 发钱，势能没生效"
            assert after < 0, f"回卷后还挣了 {after:.0f}（原生 {base[2]:.0f}），重复段仍然赚钱"
            print(f"  ✓ 回卷步收益 ≤ 回退量的 {worst:.2%}；"
                  f"重复段收益 {base[2]:+.0f} → {after:+.0f}，环不可刷了")


def check_vecenv():
    """过一遍真正的训练 vec env，看 Monitor 记的每局奖励有没有异常尖峰。"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from make_env import make_env_maze
    import wide_cnn  # noqa
    os.environ["MARIO_STAGE"] = STAGE
    venv = make_vec_env(make_env_maze, n_envs=8, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
    m = PPO.load(MODEL, env=venv, device="cpu")
    obs = venv.reset(); eps = []
    for _ in range(4000):
        a, _ = m.predict(obs, deterministic=False)
        obs, r, done, infos = venv.step(a)
        eps += [i["episode"]["r"] for i in infos if "episode" in i]
        if len(eps) >= 25:
            break
    venv.close()
    hi = max(eps)
    print(f"\n真 vec env {len(eps)} 局：最大单局奖励 {hi:.0f}，中位 {np.median(eps):.0f}")
    assert hi < 20000, (
        f"有单局奖励 {hi:.0f}，远超关卡长度——x_pos 脏读又漏进来了（65535 那个）")
    print("  ✓ 没有 65535 量级的尖峰")


if __name__ == "__main__":
    main()
    check_vecenv()
