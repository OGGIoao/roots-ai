"""练习 2 的验收测试: 行为与公式, 不要求和参考实现逐步相同。"""

import math

import numpy as np

from _target import load
from runner import run, summary

UCB = load("ex2_ucb_interface", "UCB")


def test_shape_and_dtype():
    agent = UCB(5, 4)
    a = agent.select()
    assert a.shape == (5,)
    assert np.issubdtype(a.dtype, np.integer)


def test_tries_every_arm_once_first():
    R, k = 50, 10
    agent = UCB(R, k)
    seen = []
    for _ in range(k):
        a = agent.select()
        seen.append(a)
        agent.update(a, np.zeros(R))
    seen = np.stack(seen, axis=1)                      # (R, k) 每局前 k 步拉过的臂
    for r in range(R):
        assert sorted(seen[r].tolist()) == list(range(k)), f"第 {r} 局前 {k} 步没有把每个臂各试一次: {seen[r]}"
    assert agent.t == k


def test_formula_on_a_given_state():
    agent = UCB(1, 4, c=2.0)
    agent.Q = np.array([[1.0, 0.5, 0.9, -0.2]])
    agent.N = np.array([[20, 2, 10, 1]])
    agent.t = 32                                       # 下一次 select() 会先变成 33
    scores = [agent.Q[0, a] + 2.0 * math.sqrt(math.log(33) / agent.N[0, a]) for a in range(4)]
    # 手算 (ln 33 ≈ 3.497): [1.836, 3.144, 2.083, 3.540] → 臂 3; 估计值最高的臂 0 反而最低
    assert agent.select()[0] == int(np.argmax(scores))
    assert agent.t == 33


def test_beats_the_threshold_on_the_testbed():
    s = summary(run(lambda R, k: UCB(R, k, c=2.0), steps=1000, n_runs=2000, env_seed=0))
    assert s["optimal_last"] >= 0.80, s
    assert s["regret_total"] <= 170, s
