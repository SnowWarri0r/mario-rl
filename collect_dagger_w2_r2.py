"""World 2 DAgger round 2：WideCNN student 开车，老师标它当前走偏的状态。
按 WideCNN 现状配比：2-1(12%)最弱多收、2-4(52%掉了)多收、2-2(28%)中、2-3(64%)少。
存 <stage>_dagger2.npz → distill_data_w2/，重蒸 WideCNN 自动吃进去。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch as th
from stable_baselines3 import PPO
from make_env import make_env
import wide_cnn  # noqa: 注册 WideNatureCNN 供 student 反序列化

STUDENT = "mario_w2_wide.zip"
JOBS = [
    ("2-1", "mario_21expert_v2.zip",    50000),   # 12% 最弱
    ("2-4", "mario_w2land_final.zip",   45000),   # 76→52 掉了, 多补
    ("2-2", "mario_22ladder_final.zip", 35000),   # 28% 中
    ("2-3", "mario_w2land_final.zip",   15000),   # 64% 稳, 少收
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
        o, r, term, trunc, info = env.step(a)
        if term or trunc:
            o, _ = env.reset()
        if len(obs_buf) % 10000 == 0:
            print(f"  {stage}: {len(obs_buf)}/{n}")
    env.close()
    np.savez_compressed(f"distill_data_w2/{stage}_dagger2.npz",
                        obs=np.array(obs_buf[:n], dtype=np.uint8),
                        probs=np.array(prob_buf[:n], dtype=np.float32))
    print(f">>> {stage}_dagger2: 存了 {n} 条")


if __name__ == "__main__":
    for stage, tp, n in JOBS:
        print(f"=== DAgger2 收 {stage} ===")
        collect(stage, tp, n)
    print(">>> World 2 DAgger round 2 数据收完 → distill_data_w2/")
