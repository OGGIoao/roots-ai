"""练习 3 的验收测试: 3 套新机器上的最优臂比例与总遗憾。"""

import numpy as np

from _target import USE_REFERENCE, load
from runner import run, summary

if USE_REFERENCE:
    UCB = load(None, "UCB")

    def make_agent(n_runs, k):
        return UCB(n_runs, k, c=2.0)
else:
    make_agent = load("ex3_blank", "make_agent")

SEEDS = (10, 11, 12)


def test_acceptance():
    rows = [summary(run(make_agent, steps=1000, n_runs=2000, env_seed=s)) for s in SEEDS]
    optimal = float(np.mean([r["optimal_last"] for r in rows]))
    regret = float(np.mean([r["regret_total"] for r in rows]))
    print(f"\n最后 100 步最优臂比例 {optimal:.3f} (≥ 0.80), 总遗憾 {regret:.1f} (≤ 200); 每套机器: {rows}")
    assert optimal >= 0.80, f"最优臂比例 {optimal:.3f} < 0.80"
    assert regret <= 200, f"总遗憾 {regret:.1f} > 200"
