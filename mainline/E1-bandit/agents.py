"""
agents.py — 老虎机的三种参考策略: 贪心 / ε-greedy (含乐观初始值、固定步长) / UCB

所有策略都是 R 局并行的, 只需满足两个方法:

    select() -> actions (R,)       这一步每局拉哪个臂
    update(actions, rewards)       用刚拿到的收益更新自己的估计

形状记号同 env.py: R 局数, k 臂数。估计值 Q (R, k), 每个臂被拉的次数 N (R, k)。

讲义: TUTORIAL.md 第 3~6 节。这里也是练习 1、2 的标准答案。
"""

import numpy as np


def argmax_random_tie(values, rng):
    """按行取最大值的下标; 有并列时在并列者里随机选一个。values: (R, k) → (R,)。

    为什么不直接用 values.argmax(axis=1): np.argmax 遇到并列永远返回最靠前的下标。
    开局时 Q 全是 0, 所有臂并列, 直接 argmax 会让每一局都先拉臂 0, 结果依赖臂的编号顺序。
    在本测试台上 (臂的均值与编号无关) 实测几乎没有差别, 但臂的编号一旦有含义就会出问题
    (讲义 9.1 节)。
    """
    is_best = values == values.max(axis=1, keepdims=True)          # (R, k) 并列最大的位置为 True
    # 给每个位置一个 [0,1) 随机数, 只在并列最大的位置上保留, 再取 argmax
    return (is_best * rng.random(values.shape)).argmax(axis=1)


class EpsilonGreedy:
    """以概率 ε 随机拉一个臂 (探索), 否则拉当前估计最高的臂 (利用)。

    eps=0 就是纯贪心。
    q0:   估计值的初始值。设成远大于真实收益的数 (如 5) 就是"乐观初始值" (讲义第 5 节)。
    alpha: None 时用样本平均 (步长 1/N); 给一个数时用固定步长 (讲义第 8 节)。
    rng 调用顺序固定为: 先抽"是否探索", 再抽"探索时拉哪个臂", 最后才是打破平局。
    """

    def __init__(self, n_runs, k, eps=0.1, q0=0.0, alpha=None, seed=1):
        self.eps, self.alpha = eps, alpha
        self.rng = np.random.default_rng(seed)
        self.Q = np.full((n_runs, k), float(q0))     # (R, k) 每个臂收益的估计
        self.N = np.zeros((n_runs, k), dtype=int)    # (R, k) 每个臂被拉过几次

    def select(self):
        R, k = self.Q.shape
        explore = self.rng.random(R) < self.eps                     # (R,) 这一局本步是否探索
        random_arm = self.rng.integers(k, size=R)                   # (R,) 探索时拉的臂
        greedy_arm = argmax_random_tie(self.Q, self.rng)            # (R,) 利用时拉的臂
        return np.where(explore, random_arm, greedy_arm)

    def update(self, actions, rewards):
        rows = np.arange(len(actions))
        self.N[rows, actions] += 1
        # 增量更新: 新估计 = 旧估计 + 步长 × (新收益 − 旧估计)   (讲义 3.3 节)
        # 步长取 1/N 时, 结果恰好等于这个臂所有收益的平均值
        step = 1.0 / self.N[rows, actions] if self.alpha is None else self.alpha
        self.Q[rows, actions] += step * (rewards - self.Q[rows, actions])


class UCB:
    """置信上界: 拉 "估计值 + 不确定性奖励" 最高的臂。

        A_t = argmax_a [ Q(a) + c · sqrt( ln t / N(a) ) ]

    t 是已经拉过的总次数 + 1。没被拉过的臂 (N=0) 奖励视为无穷大, 所以开头 k 步
    会把每个臂各试一次 (讲义第 6 节)。
    """

    def __init__(self, n_runs, k, c=2.0, seed=1):
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_runs, k))
        self.N = np.zeros((n_runs, k), dtype=int)
        self.t = 0                                    # 所有局同步, 所以 t 是一个数

    def select(self):
        self.t += 1
        untried = self.N == 0                                        # (R, k)
        # N=0 的位置会除以 0 (t=1 时 ln 1 = 0, 还是 0/0), 结果被下一行的 np.where 覆盖, 不影响选择
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = self.c * np.sqrt(np.log(self.t) / self.N)        # (R, k) 不确定性奖励
        scores = np.where(untried, np.inf, self.Q + bonus)
        return argmax_random_tie(scores, self.rng)

    def update(self, actions, rewards):
        rows = np.arange(len(actions))
        self.N[rows, actions] += 1
        self.Q[rows, actions] += (rewards - self.Q[rows, actions]) / self.N[rows, actions]
