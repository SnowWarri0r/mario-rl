"""DAgger：学生自己开车，老师在副驾报"该按啥"。
收"学生会走到、但之前没覆盖"的状态 + 老师标签，加进 distill_data/ 让下一轮蒸馏补上。
配比调匀：1-3 已强少收，1-2/1-4 掉得狠多收。存 <stage>_dagger.npz。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env

STUDENT = "mario_distilled.zip"
JOBS = [
    ("1-3", "mario_13expert_final.zip", 20000),   # 已 87%, 少收
    ("1-1", "mario_w1c_final.zip",      30000),
    ("1-2", "mario_w1c_final.zip",      40000),   # 掉最狠, 多收
    ("1-4", "mario_w1c_final.zip",      40000),
]


def collect(stage, teacher_path, n):
    student = PPO.load(STUDENT, device="cpu")     # 开车的(决定走哪)
    teacher = PPO.load(teacher_path, device="cpu")# 报答案的(给标签)
    env = make_env(stages=[stage])
    obs_buf, prob_buf = [], []
    o, _ = env.reset()
    while len(obs_buf) < n:
        # 老师对"学生现在所处画面"给出正确动作分布 = 标签
        ot, _ = teacher.policy.obs_to_tensor(o)
        with th.no_grad():
            tprobs = teacher.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
        # 学生自己决定怎么走(沿学生的轨迹 → 收的是学生会遇到的状态)
        st, _ = student.policy.obs_to_tensor(o)
        with th.no_grad():
            a = int(student.policy.get_distribution(st).distribution.sample().cpu().numpy()[0])
        obs_buf.append(o.astype(np.uint8)); prob_buf.append(tprobs.astype(np.float32))
        o, r, term, trunc, info = env.step(a)     # 学生开车
        if term or trunc:
            o, _ = env.reset()
        if len(obs_buf) % 10000 == 0:
            print(f"  {stage}: {len(obs_buf)}/{n}")
    env.close()
    np.savez_compressed(f"distill_data/{stage}_dagger.npz",
                        obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    print(f">>> {stage}_dagger: 存了 {n} 条")


if __name__ == "__main__":
    for stage, tp, n in JOBS:
        print(f"=== DAgger 收 {stage} ===")
        collect(stage, tp, n)
    print(">>> DAgger 数据收完 → distill_data/")
