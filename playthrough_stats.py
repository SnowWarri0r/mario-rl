"""从 run_fullgame 日志还原连打"通关关数"的完整分布，算均值和标准误。

日志只列最好的几局，逐局数据不在里面。但连打是按顺序串的：
第 k 关被清掉的次数 = 至少过了 k 关的局数，于是 E[X]=Σ P(X≥k)、E[X²]=Σ(2k−1)P(X≥k)。

用法: python playthrough_stats.py <fg日志> ...
"""
import re, math, sys
order = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]   # 连打按这个顺序串
for f in sys.argv[1:]:
    sec = open(f).read().split("逐关真实成功率")[1].split("按进关形态")[0]
    rows = {m.group(1): (int(m.group(2)), int(m.group(3)), int(m.group(4)))
            for m in re.finditer(r"(\d-\d): 进 +(\d+) 次  清 +(\d+) 次 = +\d+%   死 +(\d+) 次", sec)}
    N = rows["1-1"][0]
    ge = [rows.get(s, (0, 0, 0))[1] for s in order]
    m = sum(ge) / N; ex2 = sum((2*k-1)*g for k, g in enumerate(ge, 1)) / N
    sd = math.sqrt(ex2 - m*m)
    print(f"{f}: N={N} 均值 {m:.2f} ± {sd/math.sqrt(N):.2f}")
    print("   " + "  ".join(f"{s}:{rows[s][1]}/{rows[s][0]}(死{rows[s][2]})" for s in order if s in rows))
