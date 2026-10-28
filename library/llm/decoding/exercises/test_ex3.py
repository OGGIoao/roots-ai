"""练习 3 的验收测试: 束足够宽时等于穷举; 束宽 1 等于贪心; 停止 token; 在 ToyLM 上和参考实现一致。"""

import itertools
import random

import pytest

from _target import load  # 先导入它: 它把上一级目录加进搜索路径
import beam as REF  # noqa: E402
from sampling import greedy  # noqa: E402
from toy_lm import ToyLM  # noqa: E402

beam_search = load("ex3_beam_blank", "beam_search", "beam")


def random_model(seed, V=4):
    rng = random.Random(seed)
    table = {}

    def step(ids):
        key = tuple(ids[-2:])
        if key not in table:
            table[key] = [rng.gauss(0, 2) for _ in range(V)]
        return table[key]
    return step


def total(step, prompt, seq):
    return sum(REF.log_softmax(step(prompt + list(seq[:i])))[seq[i]] for i in range(len(seq)))


@pytest.mark.parametrize("seed", range(10))
def test_wide_beam_equals_brute_force(seed):
    V, L = 4, 3
    step = random_model(seed, V)
    best = max(itertools.product(range(V), repeat=L), key=lambda s: (total(step, [0], s), [-t for t in s]))
    ids, score = beam_search(step, [0], V ** L, L)
    assert ids == list(best)
    assert abs(score - total(step, [0], best)) < 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_width_one_is_greedy(seed):
    step = random_model(seed)
    ids, _ = beam_search(step, [0], 1, 6)
    ctx = [0]
    for t in ids:
        assert t == greedy(step(ctx))
        ctx.append(t)


def test_stop_token():
    step = random_model(0)
    ids, score = beam_search(step, [0], 3, 8, stop_ids=(2,))
    ref = REF.beam_search(step, [0], 3, 8, stop_ids=(2,))
    assert (ids, round(score, 9)) == (ref[0], round(ref[1], 9))
    assert 2 not in ids[:-1]


@pytest.mark.parametrize("width", [1, 2, 4])
def test_toylm_same_as_reference(width):
    lm = ToyLM.from_corpus()

    def step(ids):
        return lm.logits(lm.decode(ids[-2:]))
    p = lm.encode("ROMEO:\n")
    mine = beam_search(step, p, width, 25)
    ref = REF.beam_search(step, p, width, 25)
    assert mine[0] == ref[0] and abs(mine[1] - ref[1]) < 1e-9
