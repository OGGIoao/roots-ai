"""
练习 2 (② 接口): 从签名写出 UCB

只给接口和测试, 整个类自己写。公式见讲义第 6 节:

    A_t = argmax_a [ Q(a) + c · sqrt( ln t / N(a) ) ]      没被拉过的臂 (N=0) 优先

验收:
    cd mainline/E1-bandit/exercises
    python -m pytest test_ex2.py -q

测试检查四件事:
  1. 形状: select() 返回 (R,) 的整数数组;
  2. 开头 k 步每局把每个臂各试一次 (顺序随你);
  3. 给定一个 Q、N、t 的状态, select() 选的正是公式算出来的那个臂;
  4. 在 10 臂测试台上 (2000 局, 1000 步) 最后 100 步选中最优臂的比例 ≥ 0.80,
     总遗憾 ≤ 170 (参考实现实测: 0.85~0.86, 148 左右)。
和练习 1 不同, 这里不要求和参考实现逐步相同: 平局怎么打破、随机数怎么用, 你自己定。
"""

import numpy as np  # noqa: F401


class UCB:
    """R 局并行的 UCB。

    必须有的属性 (测试会直接读写它们):
        Q  (R, k) float   每个臂收益的估计 (样本平均)
        N  (R, k) int     每个臂被拉过几次
        t  int            select() 被调用过几次; 每次 select() 一开始先加 1, 再用它算 ln t
        c  float
    """

    def __init__(self, n_runs, k, c=2.0, seed=1):
        raise NotImplementedError

    def select(self):
        """→ (R,) 每局这一步拉的臂。"""
        raise NotImplementedError

    def update(self, actions, rewards):
        """actions, rewards: (R,)。"""
        raise NotImplementedError
