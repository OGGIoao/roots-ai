"""
nonstationary.py — 讲义第 8 节 "改一处会怎样": 机器的收益在悄悄变

环境: 10 个臂的真实均值都从 0 出发, 每步各自随机游走 N(0, 0.01²) (Sutton & Barto 练习 2.5)。
比较 ε = 0.1 下的两种更新:
  sample-average   步长 1/N: 第 1 次和第 10000 次的收益权重一样
  alpha=0.1        固定步长: 越新的收益权重越大, 旧收益按 0.9 的幂衰减
外加 UCB c=2 (它也用样本平均) 看看它受不受影响。

运行: python experiments/nonstationary.py   (本机约 15 秒)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import UCB, EpsilonGreedy  # noqa: E402
from runner import run, save, summary  # noqa: E402

STEPS = 10000
METHODS = {
    "sample-average": lambda R, k: EpsilonGreedy(R, k, eps=0.1),
    "alpha=0.1": lambda R, k: EpsilonGreedy(R, k, eps=0.1, alpha=0.1),
    "UCB c=2": lambda R, k: UCB(R, k, c=2.0),
}

if __name__ == "__main__":
    curves, table = {}, {}
    for name, mk in METHODS.items():
        r = run(mk, STEPS, 2000, nonstationary=True)
        table[name] = summary(r, last=1000)
        curves[name] = {key: (val[9::10] if isinstance(val, list) else val) for key, val in r.items()}
        print(f"{name:15s} {table[name]}")
    save("nonstationary", {"steps": STEPS, "every": 10, "curves": curves, "summary": table})
