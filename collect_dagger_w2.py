"""World 2 DAgger：student 自己开车，老师在它走偏的状态上报正确动作分布。
收"student 会走到、但老师示范没覆盖"的状态 → 补进 distill_data_w2/ 重蒸。
2-1/2-2 崩得狠多收，2-3/2-4 接住了少收。存 <stage>_dagger.npz。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env

STUDENT = "mario_w2_distilled.zip"
JOBS = [
    ("2-1", "mario_21expert_v2.zip",    50000),   # 崩到 4%, 多收
    ("2-2", "mario_22ladder_final.zip", 50000),   # 崩到 4%, 多收
    ("2-3", "mario_w2land_final.zip",   20000),   # 60% 接住了, 少收
    ("2-4", "mario_w2land_final.zip",   30000),
]


def collect(stage, teacher_path, n):
    student = PPO.load(STUDENT, device="cpu")
    teacher = PPO.load(teacher_path, device="cpu")
    env = make_env(stages=[stage])
    obs_buf, prob_buf = [], []
    o, _ = env.reset()
    while len(obs_buf) < n:
        ot, _ = teacher.policy.obs_to_tensor(o)
        with th.no_grad():
            tprobs = teacher.policy.get_distribution(ot).distribution.probs.cpu().numpy()[0]
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
    np.savez_compressed(f"distill_data_w2/{stage}_dagger.npz",
                        obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    print(f">>> {stage}_dagger: 存了 {n} 条")


if __name__ == "__main__":
    for stage, tp, n in JOBS:
        print(f"=== DAgger 收 {stage} ===")
        collect(stage, tp, n)
    print(">>> World 2 DAgger 数据收完 → distill_data_w2/")
