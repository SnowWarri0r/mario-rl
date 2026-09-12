"""按世界混训 · 带 no-op starts：把 W1 pilot 验证过的做法推到 W2/W3。

为什么要验这个：抖动下的诚实基线是学生 54% / 老师平均 56%——学生贴着老师的天花板，
所以换抖动数据重蒸没用（实测 53%→54%）。要抬高天花板只能动老师，也就是让 RL 训练本身
带 no-op starts。这一路先拿 World 1 当 pilot，成了再推到其余世界。

warm-start 自现有 W1 混训老师（它在抖动下：1-1 44% / 1-2 60% / 1-4 58%）。
用法: MARIO_NOOP=30 python train_world_noop.py <w1|w2|w3> [步数] [并行环境数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn
from make_env import (make_env_world1, make_env_world2_land, make_env_world3,
                      make_env_stage21, make_env_stage13, make_env_stage12,
                      make_env_stage23, make_env_single, make_env_multi,
                      make_env_maze, make_env_shaped, NOOP_JITTER)

DEVICE = os.environ.get("MARIO_DEVICE", "cpu")
OUT = os.environ.get("MARIO_OUT", "mario_w1noop")
BASE_MODEL = os.environ.get("MARIO_BASE", "mario_w1c_final.zip")
BASE_VECN = os.environ.get("MARIO_BASE_VECN", "vecnormalize_w1c.pkl")
# 2-2 那套手术的三个开关：熵归零（熵奖励会持续溶解确定性解）、低 lr、密存档（刀锋解只能靠模型选择抓）
ENT = os.environ.get("MARIO_ENT")
LR = float(os.environ.get("MARIO_LR", "2.5e-4"))
SAVE_FREQ = int(os.environ.get("MARIO_SAVE_FREQ", "500000"))


FACTORY = {"w1": "make_env_world1", "w2": "make_env_world2_land", "w3": "make_env_world3"}


def main():
    world = sys.argv[1] if len(sys.argv) > 1 else "w1"
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 5_000_000
    n_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    factory = {"w1": make_env_world1, "w2": make_env_world2_land, "w3": make_env_world3,
               "s21": make_env_stage21, "s13": make_env_stage13,
               "s12": make_env_stage12,
               "s23": make_env_stage23,
               "single": make_env_single,          # 配 MARIO_STAGE=2-4
               "multi": make_env_multi,           # 配 MARIO_STAGES=4-1,4-2,4-3
               "maze": make_env_maze,             # 配 MARIO_STAGE=4-4，换 max-x 势能奖励
               "shaped": make_env_shaped}[world]  # 配 MARIO_STAGE + MARIO_CKPTS，死一次重罚
    assert NOOP_JITTER, "这个脚本的意义就在抖动，记得 MARIO_NOOP=30"
    venv = make_vec_env(factory, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    # ⚠️ 文件名规律是 `vecnormalize_{OUT}.pkl`（OUT 含 "mario_" 前缀时也带上），
    # 所以 MARIO_OUT=mario_w4 存出来的是 `vecnormalize_mario_w4.pkl` 而不是 `vecnormalize_w4.pkl`。
    # 这个名字我已经猜错五次。原来「文件不存在就新建一个」是静默的，后果不是报错而是**悄悄跑歪**：
    # warm-start 来的价值头是在旧的 reward 归一化尺度上拟合的，配一个全新的 running stats
    # 等于价值头一开始就读错刻度。所以显式指定了却找不到＝直接炸，只有不指定才允许新建。
    if os.environ.get("MARIO_BASE_VECN"):
        assert os.path.exists(BASE_VECN), (
            f"指定了 MARIO_BASE_VECN={BASE_VECN} 但文件不存在。"
            f"命名是 vecnormalize_<MARIO_OUT>.pkl，现有的："
            f"{sorted(f for f in os.listdir('.') if f.startswith('vecnormalize'))}")
    if os.path.exists(BASE_VECN):
        venv = VecNormalize.load(BASE_VECN, venv)
        venv.training = True; venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO.load(BASE_MODEL, env=venv, device=DEVICE)
    # ⚠️ `PPO.load` 会把存档里的超参一起恢复，光赋值 `model.learning_rate` **完全无效**：
    # sb3 每次更新读的是 `self.lr_schedule(progress)`，那是从 zip 里 unpickle 出来的 callable。
    # 踩过：从「熵归零手术档」(lr=1e-5) warm-start 去学 World 4，banner 打印 lr 0.00025，
    # 实际 6M 步全程 1e-5，慢了 25 倍。ent_coef 是普通属性所以赋值有效，lr 不是——
    # 一个坑修了一半比没修更骗人，所以下面连 schedule 一起换掉，并当场断言。
    model.learning_rate = LR
    model.lr_schedule = get_schedule_fn(LR)
    assert model.lr_schedule(1.0) == LR, "lr 没生效"
    if ENT is not None:
        model.ent_coef = float(ENT)
    print(f">>> {world} 抖动重训 {total:,} 步 | {n_envs} 环境 | 续 {BASE_MODEL} | "
          f"开局随机空按 0-{NOOP_JITTER} 帧 | ent_coef {model.ent_coef} | "
          f"lr {model.lr_schedule(1.0)}（实测生效值，非入参）| device {DEVICE}", flush=True)

    ckpt = CheckpointCallback(save_freq=max(SAVE_FREQ // n_envs, 1),
                              save_path=f"./checkpoints_{os.path.basename(OUT)}", name_prefix=OUT)
    model.learn(total_timesteps=total, callback=ckpt, reset_num_timesteps=True)
    model.save(OUT)
    venv.save(f"vecnormalize_{os.path.basename(OUT)}.pkl")
    print(f">>> {world} 抖动重训结束，存为 {OUT}.zip", flush=True)


if __name__ == "__main__":
    main()
