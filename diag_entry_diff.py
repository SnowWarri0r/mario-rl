"""把"连打进关"和"单关开局"这两种观测直接 dump 出来逐像素 diff。

**为什么写这个。** 这条流水线上逐关能力稳在 89-95%，而连打全通率在 1.0%-21.5% 之间乱跳，
两者解耦。我为这个落差提过八个假设（相位 / 帧栈 / 步数上限 / 马里奥形态 / 数据覆盖 /
sticky 配比 / HUD / 老师代际），逐个测下来**没有一个能在两种配方下保持同号**——
每次都能给出听起来合理的机制解释，然后被下一次数据推翻。那是没有机制、只在拟合噪声的样子。

所以这次不猜了，直接量：**同一个模型、同一关，「从 1-1 打过来」和「单关 reset」
那一刻的观测张量，到底差在哪几个像素。** 差在哪就是哪。

选 1-3 当靶子：每局连打必经（两关就到，Mac 上也跑得动），而且落差大
（v10base 单关 argmax 93%，连打里 57%）。

输出三样：
  ① 观测的逐行带差异——HUD 带（顶部 40 行缩放后≈前 14 行）vs 游戏区，看差异集中在哪
  ② info 里的标量差异（分数 / 硬币 / 命数 / 时间 / 形态）
  ③ 把两张平均图存成 png，可以直接看

用法: ./venv/bin/python diag_entry_diff.py [每种采多少次] [模型]
      默认 12 / mario_v7_wide.zip；MARIO_NOOP 控制单关那侧的抖动（默认 30）
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MODEL = sys.argv[2] if len(sys.argv) > 2 else "mario_v7_wide.zip"
TARGET = os.environ.get("MARIO_TARGET", "1-3")
NOOP = int(os.environ.get("MARIO_NOOP", "30"))
HUD_BAND = 14          # 原图顶部 40 行 / 240 × 84 ≈ 14 行
KEYS = ("score", "coins", "life", "time", "status", "x_pos", "y_pos")


def _info_row(info):
    return {k: info.get(k) for k in KEYS}


def collect_playthrough(model, n):
    """从 1-1 打起，记录**刚进入目标关**那一帧的观测与 info"""
    import torch as th
    from make_env import make_env
    tw, ts = (int(x) for x in TARGET.split("-"))
    obs_list, infos = [], []
    env = make_env(stages=None, noop=NOOP)          # 完整游戏
    while len(obs_list) < n:
        o, _ = env.reset()
        ws = None
        for _ in range(20000):
            ot, _ = model.policy.obs_to_tensor(o)
            with th.no_grad():
                p = model.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
            # ⚠️两侧都用**采样**：纯 argmax 下 v7 的 1-1 只有 44%，绝大多数局在 1-1 就 game over，
            # 根本走不到目标关（真实连打用的是 auto 打法 + 死后回退采样）。
            # 采什么模式不影响"入场那一帧长什么样"，但必须两侧一致。
            o, r, term, trunc, info = env.step(int(np.random.choice(len(p), p=p / p.sum())))
            cur = (info.get("world"), info.get("stage"))
            if ws is not None and cur != ws and cur == (tw, ts):
                obs_list.append(o.copy()); infos.append(_info_row(info))
                break
            ws = cur
            if term or trunc:
                break
        # 打印这一局停在哪：第一次写这个诊断时卡在"只采到 1 个样本"，
        # 盲猜了两轮（推理模式、步数）都不对，加上这行立刻就看见原因了
        print(f"  连打样本 {len(obs_list)}/{n}  （本局止于 {info.get('world')}-{info.get('stage')} "
              f"x={info.get('x_pos')} 命={info.get('life')} 步={_+1}）", flush=True)
    env.close()
    return np.array(obs_list), infos


def collect_standalone(model, n):
    """单关 reset 后走一步（跟连打那侧对齐：都是"进关后的第一帧"）"""
    import torch as th
    from make_env import make_env
    obs_list, infos = [], []
    env = make_env(stages=[TARGET], noop=NOOP)
    for _ in range(n):
        o, _ = env.reset()
        ot, _ = model.policy.obs_to_tensor(o)
        with th.no_grad():
            p = model.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
        o, r, term, trunc, info = env.step(int(np.random.choice(len(p), p=p / p.sum())))
        obs_list.append(o.copy()); infos.append(_info_row(info))
    env.close()
    return np.array(obs_list), infos


def main():
    import torch as th  # noqa: F401
    from stable_baselines3 import PPO
    import wide_cnn  # noqa: F401
    model = PPO.load(MODEL, device="cpu")
    print(f"=== {TARGET} 的入场观测 diff：{MODEL}，各采 {N} 次，抖动 0-{NOOP} ===", flush=True)

    print("采连打样本（从 1-1 打过来）…", flush=True)
    pt_obs, pt_info = collect_playthrough(model, N)
    print("采单关样本（直接 reset）…", flush=True)
    sa_obs, sa_info = collect_standalone(model, N)

    # ① 逐行带差异。⚠️必须分两种口径看：
    #   「四帧平均」里，连打那侧的栈还混着上一关的画面——那是已经测过、且证明无关紧要的
    #   帧栈问题（flush/prime/不处理在 N=288 下打平），会把真实差异盖住；
    #   「只比最新一帧」才是两种入场方式下"这一关的第一帧"的真实差异。
    for tag, sl in (("四帧平均（含帧栈污染）", slice(None)), ("只比最新一帧", -1)):
        a = pt_obs[:, sl]; b = sa_obs[:, sl]
        pt_m = a.mean(0) if a.ndim == 3 else a.mean((0, 1))
        sa_m = b.mean(0) if b.ndim == 3 else b.mean((0, 1))
        dd = np.abs(pt_m - sa_m)
        print(f"\n①{tag}：HUD 带 {dd[:HUD_BAND].mean():5.2f} / "
              f"游戏区 {dd[HUD_BAND:].mean():5.2f} / 全图 {dd.mean():5.2f}（最大 {dd.max():.1f}）",
              flush=True)
    pt_m, sa_m = pt_obs[:, -1].mean(0), sa_obs[:, -1].mean(0)   # 下面的逐行明细用"最新一帧"
    d = np.abs(pt_m - sa_m)
    print(f"\n① 最新一帧的逐像素绝对差（0-255 灰度）")
    print(f"   全图      {d.mean():6.2f}   最大 {d.max():6.1f}")
    print(f"   HUD 带（前 {HUD_BAND} 行）  {d[:HUD_BAND].mean():6.2f}   最大 {d[:HUD_BAND].max():6.1f}")
    print(f"   游戏区（其余 {84-HUD_BAND} 行） {d[HUD_BAND:].mean():6.2f}   最大 {d[HUD_BAND:].max():6.1f}")
    print("   逐行（每 6 行一档）:")
    for r0 in range(0, 84, 6):
        band = d[r0:r0+6]
        bar = "#" * int(band.mean() * 2)
        tag = "HUD" if r0 < HUD_BAND else "   "
        print(f"     行 {r0:2d}-{r0+5:2d} {tag} {band.mean():6.2f} {bar}")

    # ② 标量差异
    print(f"\n② info 标量（连打 vs 单关，各 {N} 次的取值）")
    for k in KEYS:
        pv = [i[k] for i in pt_info]
        sv = [i[k] for i in sa_info]
        def fmt(v):
            u = sorted(set(map(str, v)))
            return u[0] if len(u) == 1 else f"{u[0]}…{u[-1]}({len(u)}种)"
        print(f"   {k:8s} 连打 {fmt(pv):24s} 单关 {fmt(sv)}")

    # ③ 存图
    try:
        import cv2
        cv2.imwrite("entry_playthrough.png", pt_m.astype(np.uint8))
        cv2.imwrite("entry_standalone.png", sa_m.astype(np.uint8))
        cv2.imwrite("entry_absdiff.png", (d / max(d.max(), 1) * 255).astype(np.uint8))
        print("\n③ 已存 entry_playthrough.png / entry_standalone.png / entry_absdiff.png")
    except Exception as e:
        print(f"\n③ 存图失败: {e}")

    print("\n>>> 读法：差异若压倒性集中在 HUD 带＝入场差别就是那几个数字；"
          "若游戏区也差很多＝进关时马里奥的位置/速度/画面滚动状态本身就不同", flush=True)


if __name__ == "__main__":
    main()
