"""逐档评测 4-4 第二段专项：从关卡**起点**走真实流程——44p 前缀 → 脚本入口 → 待评档接手。

训练时每回合从底路入口开局，训练曲线的 ep_rew 只说明那一段在变好；
老师的分数必须按完整关卡、31 个确切相位量（项目规矩：只按实测通关率挑档，不看 ep_rew，不取最终档）。

用法: python eval_44b.py <档目录或逗号分隔的档> [相位数]
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, glob, re
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
_argv, sys.argv = sys.argv, sys.argv[:1]   # demo_44_route 在 import 时解析 argv（当相位列表），先挡掉
import demo_44_route as D
sys.argv = _argv

spec = sys.argv[1]
CKPTS = (sorted(glob.glob(f"{spec}/*.zip"), key=lambda p: int(re.search(r"(\d+)_steps", p).group(1)))
         if os.path.isdir(spec) else spec.split(","))
NPH = int(sys.argv[2]) if len(sys.argv) > 2 else 31


def job(a):
    ck, phase = a
    D.SUFFIX = ck
    _, res, _ = D.run(phase)
    return ck, phase, res


def main():
    jobs = [(c, p) for c in CKPTS for p in range(NPH)]
    tab = {c: [0, 0] for c in CKPTS}
    with ProcessPoolExecutor(max_workers=min(len(jobs), 150)) as pool:
        for ck, phase, res in pool.map(job, jobs, chunksize=1):
            tab[ck][0] += any(s[0] == "过了第9页检查" for s in res.get("log", []))
            tab[ck][1] += bool(res.get("cleared"))
    print(f"{'档':44s} {'过第9页检查':>10s} {'通关':>8s}")
    for c in CKPTS:
        print(f"{os.path.basename(c):44s} {tab[c][0]:6d}/{NPH:<3d} {tab[c][1]:5d}/{NPH}")


if __name__ == "__main__":
    main()
