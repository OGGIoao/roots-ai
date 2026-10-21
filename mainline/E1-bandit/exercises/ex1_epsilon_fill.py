"""
练习 1 (① 填空): 补全 ε-greedy

把下面两处 TODO 补全, 让 EpsilonGreedy 在贪心、ε-greedy、乐观初始值、固定步长四种配置下,
与参考实现 (../agents.py) 每一步选的臂完全相同、估计值逐数值一致。

验收:
    cd mainline/E1-bandit/exercises
    python -m pytest test_ex1.py -q

要"逐步相同", 随机数的调用顺序必须和参考实现一样 (见 select 的注释)。
提示规则: 卡住了可以要提示, 按"方向 → 哪一行 → 代码"分级给, 每次只升一级。
写完再去对照 ../agents.py。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import argmax_random_tie  # noqa: E402  打破平局的工具直接给你用


class EpsilonGreedy:
    def __init__(self, n_runs, k, eps=0.1, q0=0.0, alpha=None, seed=1):
        self.eps, self.alpha = eps, alpha
        self.rng = np.random.default_rng(seed)
        self.Q = np.full((n_runs, k), float(q0))     # (R, k) 每个臂收益的估计
        self.N = np.zeros((n_runs, k), dtype=int)    # (R, k) 每个臂被拉过几次

    def select(self):
        """返回 (R,) 的动作。

        随机数按这个顺序各用一次 (和参考实现保持一致):
          1. self.rng.random(R)             → 每局是否探索 (小于 eps 就探索)
          2. self.rng.integers(k, size=R)   → 探索时拉的臂
          3. argmax_random_tie(self.Q, self.rng) → 利用时拉的臂
        注意: 不管这一局探不探索, 2 和 3 都要算 (这样随机数的消耗才和参考实现一样)。
        """
        R, k = self.Q.shape
        # TODO 1: 按上面的顺序算出 explore、random_arm、greedy_arm, 组合成 (R,) 的动作
        raise NotImplementedError

    def update(self, actions, rewards):
        """actions, rewards: (R,)。更新被拉的那个臂的 N 和 Q (其它臂不动)。"""
        # TODO 2: 先给被拉的臂计数, 再做增量更新。
        #   alpha 为 None: 步长 1/N (结果等于样本平均); 否则步长固定为 alpha
        #   提示: 第 r 局被拉的是 actions[r], 用 (行下标, 列下标) 两个数组一起索引
        raise NotImplementedError
