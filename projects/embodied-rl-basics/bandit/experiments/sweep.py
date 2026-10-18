"""
sweep.py — 讲义第 7 节: 每种方法对自己的超参数有多敏感

指标: 前 1000 步的平均收益 (2000 局), 即 Sutton & Barto 图 2.6 的画法。
每种方法扫一个参数, 参数按 2 的幂取值:
  ε-greedy        ε  ∈ 1/128 … 1/4
  UCB             c  ∈ 1/16 … 4
  乐观初始值      Q0 ∈ 1/4 … 4   (ε = 0, 固定步长 0.1)

运行: python experiments/sweep.py   (本机约 10 秒)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import UCB, EpsilonGreedy  # noqa: E402
from runner import run, save  # noqa: E402

FAMILIES = {
    "eps-greedy": ([2.0 ** p for p in range(-7, -1)], lambda v: (lambda R, k: EpsilonGreedy(R, k, eps=v))),
    "UCB": ([2.0 ** p for p in range(-4, 3)], lambda v: (lambda R, k: UCB(R, k, c=v))),
    "optimistic": ([2.0 ** p for p in range(-2, 3)],
                   lambda v: (lambda R, k: EpsilonGreedy(R, k, eps=0.0, q0=v, alpha=0.1))),
}

if __name__ == "__main__":
    out = {}
    for fam, (values, make) in FAMILIES.items():
        pts = []
        for v in values:
            r = run(make(v), 1000, 2000)
            pts.append([v, round(float(np.mean(r["avg_reward"])), 4)])
        out[fam] = pts
        print(fam, "  ".join(f"{v:g}:{m:.3f}" for v, m in pts))
    save("sweep", out)
