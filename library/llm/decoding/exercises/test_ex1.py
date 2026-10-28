"""练习 1 的验收测试: 和参考实现 (../sampling.py) 在随机打分上逐位比对。"""

import random

import pytest

from _target import load  # 先导入它: 它把上一级目录加进搜索路径
import sampling as REF  # noqa: E402

top_p = load("ex1_filters_fill", "top_p", "sampling")
min_p = load("ex1_filters_fill", "min_p", "sampling")
repetition_penalty = load("ex1_filters_fill", "repetition_penalty", "sampling")


def cases(n=200, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        kind = rng.choice(["flat", "peaked", "negative"])
        x = [rng.gauss(0, 1) for _ in range(30)]
        if kind == "peaked":
            x[rng.randrange(30)] += rng.uniform(3, 9)
        if kind == "negative":
            x = [z - 20 for z in x]                    # 全是负分 (像对数概率)
        out.append((x, [rng.randrange(30) for _ in range(rng.randint(0, 12))]))
    return out


def same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x == REF.NEG_INF) == (y == REF.NEG_INF), (a, b)
        if x != REF.NEG_INF:
            assert abs(x - y) < 1e-9


@pytest.mark.parametrize("p", [1.0, 0.95, 0.8, 0.5, 0.05])
def test_top_p(p):
    for x, _ in cases():
        same(top_p(x, p), REF.top_p(x, p))


@pytest.mark.parametrize("p", [0.02, 0.1, 0.5, 1.0])
def test_min_p(p):
    for x, _ in cases(seed=1):
        same(min_p(x, p), REF.min_p(x, p))


@pytest.mark.parametrize("penalty", [1.0, 1.1, 1.3, 2.0])
def test_repetition_penalty(penalty):
    for x, prev in cases(seed=2):
        before = list(x)
        same(repetition_penalty(x, prev, penalty), REF.repetition_penalty(x, prev, penalty))
        assert x == before, "不要改动传进来的 logits"


def test_keeps_at_least_one():
    x = [0.0] * 5
    assert sum(z != REF.NEG_INF for z in top_p(x, 0.01)) >= 1
    assert sum(z != REF.NEG_INF for z in min_p([5.0, -50.0], 0.99)) >= 1
