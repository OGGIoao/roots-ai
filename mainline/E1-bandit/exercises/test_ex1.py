"""练习 1 的验收测试: 与参考实现逐步比对。"""

import numpy as np
import pytest

from _target import load
from agents import EpsilonGreedy as Ref
from env import BanditTestbed

Mine = load("ex1_epsilon_fill", "EpsilonGreedy")

CONFIGS = {
    "greedy": dict(eps=0.0),
    "eps=0.1": dict(eps=0.1),
    "optimistic": dict(eps=0.0, q0=5.0, alpha=0.1),
    "eps=0.1 alpha=0.1": dict(eps=0.1, alpha=0.1),
}


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CONFIGS.keys())
def test_same_actions_and_estimates_as_reference(cfg):
    R, k, steps = 200, 10, 300
    env_ref, env_mine = BanditTestbed(R, k, seed=3), BanditTestbed(R, k, seed=3)
    ref, mine = Ref(R, k, seed=7, **cfg), Mine(R, k, seed=7, **cfg)
    for t in range(steps):
        a_ref, a_mine = ref.select(), mine.select()
        assert a_mine.shape == (R,), f"select() 应返回形状 (R,), 得到 {a_mine.shape}"
        assert np.array_equal(a_mine, a_ref), f"第 {t + 1} 步选的臂和参考实现不同 (检查随机数调用顺序)"
        ref.update(a_ref, env_ref.pull(a_ref))
        mine.update(a_mine, env_mine.pull(a_mine))
    assert np.array_equal(mine.N, ref.N)
    assert np.allclose(mine.Q, ref.Q, atol=1e-12)


def test_sample_average_equals_plain_mean():
    # 步长 1/N 的增量更新, 结果必须等于"把这个臂所有收益存下来再求平均"
    agent = Mine(1, 3, eps=0.0)
    rewards = [0.5, -1.2, 2.0, 0.3]
    for r in rewards:
        agent.update(np.array([1]), np.array([r]))
    assert agent.N[0, 1] == len(rewards)
    assert agent.Q[0, 1] == pytest.approx(np.mean(rewards), abs=1e-12)
    assert agent.Q[0, 0] == 0.0 and agent.Q[0, 2] == 0.0, "没被拉的臂不应该变"
