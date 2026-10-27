"""
explore.py — 讲义第 5~7 节: 几种探索方法放在同一台机器上比

方法:
  greedy              ε = 0
  eps=0.1             ε-greedy, 样本平均
  eps=0.01
  optimistic Q0=5     ε = 0, 初始估计 5, 固定步长 0.1   (Sutton & Barto 图 2.3 的设置)
  realistic eps=0.1   ε = 0.1, 初始估计 0, 固定步长 0.1 (乐观初始值的对照组)
  UCB c=2

两组测量:
  1. 1000 步, 3 套机器 (env_seed = 0, 1, 2), 看结论是否依赖某一套机器; 曲线只存 seed 0;
  2. 10000 步, 只看 ε=0.1 / ε=0.01 / UCB: 短期赢家长期还是赢家吗 (曲线每 10 步存一个点)。

运行: python experiments/explore.py   (本机约 25 秒)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import UCB, EpsilonGreedy  # noqa: E402
from runner import run, save, summary  # noqa: E402

METHODS = {
    "greedy": lambda R, k: EpsilonGreedy(R, k, eps=0.0),
    "eps=0.1": lambda R, k: EpsilonGreedy(R, k, eps=0.1),
    "eps=0.01": lambda R, k: EpsilonGreedy(R, k, eps=0.01),
    "optimistic Q0=5": lambda R, k: EpsilonGreedy(R, k, eps=0.0, q0=5.0, alpha=0.1),
    "realistic eps=0.1": lambda R, k: EpsilonGreedy(R, k, eps=0.1, q0=0.0, alpha=0.1),
    "UCB c=2": lambda R, k: UCB(R, k, c=2.0),
}
LONG = ["eps=0.1", "eps=0.01", "UCB c=2"]


def thin(result, every):
    """长曲线每 every 步留一个点, 控制 JSON 大小。"""
    return {key: (val[every - 1::every] if isinstance(val, list) else val) for key, val in result.items()}


if __name__ == "__main__":
    curves, per_seed = {}, {name: [] for name in METHODS}
    for seed in (0, 1, 2):
        for name, mk in METHODS.items():
            r = run(mk, 1000, 2000, env_seed=seed)
            per_seed[name].append(summary(r))
            if seed == 0:
                curves[name] = r
    print(f"{'方法':20s} 最后100步最优臂比例 (3 套机器)     总遗憾 (3 套机器)")
    table = {}
    for name, rows in per_seed.items():
        opt = [x["optimal_last"] for x in rows]
        reg = [x["regret_total"] for x in rows]
        table[name] = {"optimal_last_mean": round(float(np.mean(opt)), 3),
                       "optimal_last_range": [min(opt), max(opt)],
                       "regret_mean": round(float(np.mean(reg)), 1),
                       "regret_range": [min(reg), max(reg)],
                       "reward_last_mean": round(float(np.mean([x["reward_last"] for x in rows])), 3),
                       "locked_mean": round(float(np.mean([x["locked_frac"] for x in rows])), 4)}
        print(f"{name:20s} {table[name]['optimal_last_mean']:.3f} {opt}   "
              f"{table[name]['regret_mean']:7.1f} {reg}")

    long_curves, long_table = {}, {}
    for name in LONG:
        r = run(METHODS[name], 10000, 2000, env_seed=0)
        long_table[name] = summary(r, last=1000)
        long_curves[name] = thin(r, 10)
        print(f"10000 步 {name:10s} {long_table[name]}")

    save("explore", {"curves": curves, "per_seed": per_seed, "summary": table,
                     "long": {"steps": 10000, "every": 10, "curves": long_curves, "summary": long_table}})
