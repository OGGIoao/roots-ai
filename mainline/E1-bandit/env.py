"""
env.py — 10 臂老虎机测试台 (Sutton & Barto 第 2 章)

一台"老虎机"有 k 个拉杆 (臂)。第 a 个臂有一个真实的平均收益 q*(a), 玩家看不到;
每拉一次, 拿到的收益是 q*(a) 加上标准差为 1 的噪声。

为了让结论不依赖某一台机器的运气, 一次同时开 R 局 (R 台独立抽出来的机器),
所有局在同一步一起拉杆, 用 numpy 一次算完。全文的形状记号:

    R  局数 (n_runs, 默认 2000)      k  臂数 (默认 10)

讲义: TUTORIAL.md 第 2 节。
运行: python env.py   (打印一台机器的真实收益, 看看问题长什么样)
"""

import numpy as np


class BanditTestbed:
    """R 局并行的 k 臂老虎机。

    平稳 (默认): 每局的 q*(a) 从 N(0, 1) 里抽一次, 之后不再变。
    非平稳 (nonstationary=True): 所有臂的 q*(a) 从 0 出发, 每拉一次杆后各自做一步
        标准差为 walk_std 的随机游走 (讲义第 8 节"改一处会怎样")。
    """

    def __init__(self, n_runs=2000, k=10, seed=0, nonstationary=False, walk_std=0.01):
        self.n_runs, self.k = n_runs, k
        self.nonstationary, self.walk_std = nonstationary, walk_std
        # 环境有自己的随机数发生器, 和玩家的分开:
        # 这样换一种策略时, 机器本身 (q*) 完全相同, 比较才公平
        self.rng = np.random.default_rng(seed)
        if nonstationary:
            self.q_star = np.zeros((n_runs, k))                       # (R, k) 起点全相同
        else:
            self.q_star = self.rng.standard_normal((n_runs, k))      # (R, k)

    def pull(self, actions):
        """每局拉一个臂。actions: (R,) 整数 → 收益 (R,)。"""
        rows = np.arange(self.n_runs)
        # 收益 = 这一局被拉的那个臂的真实均值 + N(0, 1) 噪声
        rewards = self.q_star[rows, actions] + self.rng.standard_normal(self.n_runs)
        if self.nonstationary:
            # 拉完之后, 所有臂的真实均值都漂一点 (玩家不知道它们在变)
            self.q_star += self.walk_std * self.rng.standard_normal(self.q_star.shape)
        return rewards

    def optimal_arm(self):
        """每局当前真实均值最高的臂, (R,)。非平稳时它会随时间变。"""
        return self.q_star.argmax(axis=1)

    def regret_of(self, actions):
        """这一步的(期望)遗憾: 最优臂的真实均值 − 被拉臂的真实均值, (R,)。

        用真实均值而不是实际收益来算, 去掉了噪声, 曲线更平滑 (讲义 2.4 节)。
        """
        rows = np.arange(self.n_runs)
        return self.q_star.max(axis=1) - self.q_star[rows, actions]


if __name__ == "__main__":
    env = BanditTestbed(n_runs=1, seed=0)
    print("第 0 局的真实收益 q*(a):")
    for a, q in enumerate(env.q_star[0]):
        print(f"  臂 {a}: {q:+.3f}")
    print(f"最优臂: {env.optimal_arm()[0]}")
    rewards = [env.pull(np.array([0]))[0] for _ in range(5)]
    print("连拉 5 次臂 0 的收益:", " ".join(f"{r:+.2f}" for r in rewards))
