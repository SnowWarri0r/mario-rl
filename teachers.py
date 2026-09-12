"""全部关卡的老师名册——**唯一事实来源**。

为什么要有这个文件：这份名册目前被抄在 8 个脚本里（collect_distill_v5 / collect_dagger_v6 /
collect_distill_crop / collect_dagger_crop / collect_distill_noop / collect_dagger_all12 /
eval_noop_audit / run_fullgame），十二关时勉强能对得上，扩到 32 关必然漂移——
而漂移的后果不是报错，是**悄悄用旧老师收了一批数据**，要等蒸出来学生变差才发现。
新代码一律从这里 import，别再抄。

每条记录 = (关卡, 模型路径, 主指标 0-30, 鲁棒性 0-120)。
分数都是 `diag_progress.py` 逐相位枚举实测的（argmax，31 或 121 个确切相位），
不是 `eval_noop_audit.py` 的随机抽样口径——后者会虚高，见 [[feedback_rl_eval_effective_sample_size]]。
`None` = 还没测那个口径。

⚠️ 换老师的规矩：**只按实测通关率挑，不看 ep_rew_mean，也不取最终档。**
混训里各关峰值不同步，实测 W4 最终档把 4-2 拉到 29/31 的同时把 4-1 摔到 17/31。
"""

# 关卡 -> (模型, 主指标 %, 鲁棒 %)
TEACHERS = {
    # --- World 1-3：熵归零手术后的班底（2026-09-03 定稿）---
    "1-1": ("mario_11robust.zip",                                   100, 100),
    "1-2": ("mario_12robust.zip",                                    95,  94),
    "1-3": ("checkpoints_w1ent0/w1ent0_6250000_steps.zip",          100, 100),
    "1-4": ("mario_14robust.zip",                                   100, 100),
    "2-1": ("checkpoints_s21ent0/s21ent0_3500000_steps.zip",        100,  98),
    "2-2": ("mario_22robust.zip",                                    84,  60),   # 唯一没收口的老关
    "2-3": ("mario_23robust.zip",                                    91,  91),
    "2-4": ("checkpoints_mario_24fine/mario_24fine_99840_steps.zip", 100, 100),
    "3-1": ("checkpoints_w3ent0/w3ent0_2250000_steps.zip",          100, 100),
    "3-2": ("mario_32robust.zip",                                   100, 100),
    "3-3": ("checkpoints_w3ent0/w3ent0_2250000_steps.zip",          100, 100),
    "3-4": ("mario_34robust.zip",                                   100, 100),

    # --- World 4（2026-09-04/06）---
    # 4-1/4-2 出自混训最终档，4-3 是专家档：混训里 4-3 打满 6M 一直 0/31，
    # 单关专训在 3.75M 处突变到 29/31 —— 是被混训饿着了，不是关卡需要新机制。
    "4-1": ("checkpoints_mario_41exp/mario_41exp_5249664_steps.zip",  94, None),
    "4-2": ("checkpoints_mario_w4/mario_w4_5999616_steps.zip",        94, None),
    "4-3": ("checkpoints_mario_43exp/mario_43exp_4499712_steps.zip", 100, 100),

    # --- World 5（2026-09-06）---
    # 四关全部出自专家续训。混训 8M 的成绩是 42/0/0/29，专家之后 84/90/100/100，
    # 「混训打底 + 逐关专家」这条路到此有两个世界的完整证据。
    "5-1": ("checkpoints_mario_51exp/mario_51exp_5749632_steps.zip",  84, None),  # 还在震荡，待熵归零
    "5-2": ("checkpoints_mario_52exp/mario_52exp_5749632_steps.zip",  90, None),
    "5-3": ("checkpoints_mario_53exp/mario_53exp_5499648_steps.zip", 100, None),
    "5-4": ("checkpoints_mario_54exp/mario_54exp_3749760_steps.zip", 100, None),
    # --- World 6（2026-09-06）---
    # 混训 8M：6-1 100 / 6-2 55 / 6-3 0 / 6-4 23，又是同一个形状。6-2/6-3/6-4 专家训练中。
    "6-1": ("checkpoints_mario_w6/mario_w6_4499712_steps.zip",       100, None),
}

# 还没有老师的关。迷宫城堡单列，它们卡在同一个病上：
# 上下两条走廊 x 区间相同，max-x 势能与 per-cell novelty 对二者都对称，
# 中间没有任何梯度区分，要等跨回合的持久访问计数 / Go-Explore。
MAZE_STAGES = ["4-4", "7-4", "8-4"]
TODO_STAGES = (["6-2", "6-3", "6-4"]
               + [f"7-{i}" for i in range(1, 4)]
               + [f"8-{i}" for i in range(1, 4)]
               + MAZE_STAGES)

ALL_STAGES = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]


def path(stage):
    """老师的模型路径；没有老师就抛错，别让调用方拿到 None 悄悄跑下去。"""
    if stage not in TEACHERS:
        raise KeyError(f"{stage} 还没有老师（待办：{TODO_STAGES}）")
    return TEACHERS[stage][0]


def lineup(stages=None):
    """{关卡: 模型路径}，给收数据脚本用。"""
    return {s: TEACHERS[s][0] for s in (stages or TEACHERS)}


def weights(stages=None, floor=1, scale=100):
    """按"离满分还差多少"给采样权重：越弱的关配越多数据，强关给个地板值防重蒸时漂移掉。
    这是 collect_dagger_v6 里那套手工配比的公式化——手工表每换一次学生就要重排一次，
    而且很容易忘记跟着实测分数更新。"""
    out = {}
    for s in (stages or TEACHERS):
        main = TEACHERS[s][1]
        out[s] = max(floor, round((100 - main) / 100 * scale))
    return out


if __name__ == "__main__":
    done, todo = len(TEACHERS), len(TODO_STAGES)
    print(f"=== 老师名册：{done}/32 关有老师，待办 {todo} 关 ===")
    for s in ALL_STAGES:
        if s in TEACHERS:
            p, m, r = TEACHERS[s]
            tag = "迷宫" if s in MAZE_STAGES else ""
            print(f"  {s}  主 {m:3d}%  鲁棒 {str(r) + '%' if r else '  — ':>4s}  {p} {tag}")
        else:
            print(f"  {s}  —— 待办 {'（迷宫）' if s in MAZE_STAGES else ''}")
    mains = [v[1] for v in TEACHERS.values()]
    print(f"\n已有老师的 {done} 关主指标均值 {sum(mains)/len(mains):.1f}%")
    print(f"DAgger 建议配比（越弱配越多）：{weights()}")
