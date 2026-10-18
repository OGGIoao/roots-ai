"""具身线 R0 ① 老虎机: 参考实现的回归测试 (CPU, 约 5 秒)。运行: python -m pytest tests -q"""

import os
import subprocess
import sys

import numpy as np
import pytest

BANDIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "projects", "embodied-rl-basics", "bandit")
sys.path.insert(0, BANDIT)

from agents import UCB, EpsilonGreedy, argmax_random_tie  # noqa: E402
from env import BanditTestbed  # noqa: E402
from runner import run, summary  # noqa: E402


def test_tie_break_is_uniform():
    # 全部并列时, 每个臂被选中的频率应接近 1/k
    rng = np.random.default_rng(0)
    picks = argmax_random_tie(np.zeros((20000, 4)), rng)
    freq = np.bincount(picks, minlength=4) / len(picks)
    assert np.allclose(freq, 0.25, atol=0.02)


def test_env_is_shared_across_agents():
    # 同一个 env_seed 下换策略, 机器 (q*) 必须完全相同, 比较才公平
    assert np.array_equal(BanditTestbed(50, seed=0).q_star, BanditTestbed(50, seed=0).q_star)


def test_known_ordering_on_the_testbed():
    # 讲义第 4~7 节的核心结论: 贪心最差, UCB 遗憾最小
    s = {name: summary(run(mk, 1000, 1000))
         for name, mk in {"greedy": lambda R, k: EpsilonGreedy(R, k, eps=0.0),
                          "eps=0.1": lambda R, k: EpsilonGreedy(R, k, eps=0.1),
                          "ucb": lambda R, k: UCB(R, k, c=2.0)}.items()}
    assert s["greedy"]["optimal_last"] < 0.45 < s["eps=0.1"]["optimal_last"]
    assert s["ucb"]["regret_total"] < s["eps=0.1"]["regret_total"] < s["greedy"]["regret_total"]


def test_exercise_tests_pass_against_reference():
    # 练习的测试本身必须是能通过的: 让它们去检查参考实现
    env = dict(os.environ, BANDIT_SOLUTION="reference")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
                       cwd=os.path.join(BANDIT, "exercises"), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
