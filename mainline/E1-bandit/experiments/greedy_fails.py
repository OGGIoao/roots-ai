"""
greedy_fails.py — 讲义第 4 节: 贪心为什么会锁死在次优臂上

三组测量 (2000 局, 每局 1000 步, 机器由 env_seed=0 决定):
  1. 曲线: 贪心 vs ε=0.01 vs ε=0.1 的平均收益与最优臂比例;
  2. 锁死在哪: 贪心最后 100 步最常拉的臂, 在这局 10 个臂里按真实均值排第几;
     以及每局最后一次"换臂"发生在第几步;
  3. 打破平局的方式: 并列时随机选 vs 直接 np.argmax (永远选最靠前的),
     先在 env_seed=0 上比, 再在 20 套新机器上比差距是否稳定。

运行: python experiments/greedy_fails.py   (本机约 20 秒)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import EpsilonGreedy  # noqa: E402
from env import BanditTestbed  # noqa: E402
from runner import run, save, summary  # noqa: E402

STEPS, RUNS = 1000, 2000


class GreedyPlainArgmax(EpsilonGreedy):
    """贪心, 但并列时直接用 np.argmax (总是选下标最小的那个)。只用于第 3 组对照。"""

    def select(self):
        return self.Q.argmax(axis=1)


def where_does_greedy_lock():
    """单独跑一遍贪心, 记下每一步的动作, 分析它最后停在哪个臂、什么时候停下。"""
    env = BanditTestbed(RUNS, 10, seed=0)
    agent = EpsilonGreedy(RUNS, 10, eps=0.0)
    history = np.zeros((STEPS, RUNS), dtype=int)                   # (T, R) 每步每局拉的臂
    for t in range(STEPS):
        a = agent.select()
        agent.update(a, env.pull(a))
        history[t] = a
    # 每局最后 100 步里出现次数最多的臂
    tail = history[-100:]                                           # (100, R)
    final_arm = np.array([np.bincount(tail[:, r], minlength=10).argmax() for r in range(RUNS)])
    # 这个臂的真实均值在本局排第几 (1 = 最好)
    order = np.argsort(-env.q_star, axis=1)                         # (R, 10) 从好到差的臂下标
    rank = np.array([int(np.where(order[r] == final_arm[r])[0][0]) + 1 for r in range(RUNS)])
    # 最后一次换臂的步数 (从 1 数起); 从没换过的局记为 1
    switched = history[1:] != history[:-1]                          # (T-1, R)
    last_switch = np.where(switched.any(axis=0), STEPS - 1 - np.argmax(switched[::-1], axis=0), 0) + 1
    # 被锁死在次优臂上的局, 损失有多大: 最优臂均值 − 锁定臂均值
    rows = np.arange(RUNS)
    gap = env.q_star.max(axis=1) - env.q_star[rows, final_arm]
    return {
        "rank_hist": np.bincount(rank, minlength=11)[1:].tolist(),     # 排第 1..10 的局数
        "last_switch_median": int(np.median(last_switch)),
        "last_switch_p90": int(np.percentile(last_switch, 90)),
        "frac_locked_by_step_10": round(float((last_switch <= 10).mean()), 4),
        "gap_mean_when_suboptimal": round(float(gap[rank > 1].mean()), 3),
    }


if __name__ == "__main__":
    agents = {
        "greedy": lambda R, k: EpsilonGreedy(R, k, eps=0.0),
        "eps=0.01": lambda R, k: EpsilonGreedy(R, k, eps=0.01),
        "eps=0.1": lambda R, k: EpsilonGreedy(R, k, eps=0.1),
    }
    curves, table = {}, {}
    for name, mk in agents.items():
        r = run(mk, STEPS, RUNS)
        curves[name], table[name] = r, summary(r)
        print(f"{name:10s} {table[name]}")

    lock = where_does_greedy_lock()
    print("贪心最后停在第几好的臂 (1..10):", lock["rank_hist"])
    print(f"最后一次换臂: 中位数第 {lock['last_switch_median']} 步, 90% 分位第 {lock['last_switch_p90']} 步; "
          f"前 10 步内就定下来的局占 {lock['frac_locked_by_step_10']:.1%}")

    tie = {}
    for name, mk in {"random_tie": lambda R, k: EpsilonGreedy(R, k, eps=0.0),
                     "plain_argmax": lambda R, k: GreedyPlainArgmax(R, k, eps=0.0)}.items():
        tie[name] = summary(run(mk, STEPS, RUNS))
        print(f"打破平局 {name:13s} {tie[name]}")

    # 只看一套机器的差距会被偶然性误导: 再在 20 套新机器上比, 看差距是否稳定
    diffs = []
    for s in range(100, 120):
        a = summary(run(lambda R, k: EpsilonGreedy(R, k, eps=0.0), STEPS, RUNS, env_seed=s))["optimal_last"]
        b = summary(run(lambda R, k: GreedyPlainArgmax(R, k, eps=0.0), STEPS, RUNS, env_seed=s))["optimal_last"]
        diffs.append(b - a)
    diffs = np.array(diffs)
    tie["20_seeds"] = {"mean_diff": round(float(diffs.mean()), 4),
                       "se": round(float(diffs.std(ddof=1) / np.sqrt(len(diffs))), 4),
                       "plain_better": int((diffs > 0).sum()), "n": len(diffs)}
    print(f"20 套新机器 (env_seed 100~119): 最优臂比例 plain − random = {diffs.mean():+.4f} "
          f"± {tie['20_seeds']['se']:.4f} (标准误), plain 更好的有 {tie['20_seeds']['plain_better']} 套")

    save("greedy", {"steps": STEPS, "runs": RUNS, "curves": curves, "summary": table, "lock": lock, "tie": tie})
