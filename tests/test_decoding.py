"""知识库 llm/decoding (解码与采样) 的回归测试: 只用标准库, 不需要 transformers 和模型。"""

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIT = os.path.join(ROOT, "library", "llm", "decoding")
sys.path.insert(0, UNIT)

import sampling as S  # noqa: E402

INF = S.NEG_INF


def kept(x):
    return [i for i, z in enumerate(x) if z != INF]


def test_repetition_penalty_pushes_down_both_signs():
    out = S.repetition_penalty([2.0, -2.0, 1.0], [0, 1, 1], 2.0)
    assert out == [1.0, -4.0, 1.0]          # 正分除以 2, 负分乘以 2, 没出现过的不变, 出现两次也只罚一次


def test_top_k_keeps_ties():
    assert kept(S.top_k([3.0, 1.0, 1.0, 0.0], 2)) == [0, 1, 2]


def test_top_p_smallest_set():
    logits = [math_log(p) for p in (0.5, 0.3, 0.15, 0.05)]
    assert kept(S.top_p(logits, 0.75)) == [0, 1]       # 不要取正好 0.8: 0.05+0.15 在浮点里不一定等于 0.2
    assert kept(S.top_p(logits, 0.81)) == [0, 1, 2]
    assert kept(S.top_p(logits, 0.01)) == [0]     # 至少留一个


def test_min_p_relative_to_top():
    logits = [math_log(p) for p in (0.5, 0.3, 0.15, 0.05)]
    assert kept(S.min_p(logits, 0.3)) == [0, 1, 2]    # 门槛 0.15


def test_greedy_ignores_sampling_steps_but_not_penalty():
    logits = [2.0, 1.9, 0.0]
    assert S.next_token(logits, prev_ids=[], do_sample=False, rep=1.1) == 0
    assert S.next_token(logits, prev_ids=[0], do_sample=False, rep=1.1) == 1    # 2.0/1.1 < 1.9


def test_sample_reproducible_and_respects_mask():
    x = S.top_k([1.0, 0.5, 0.2, -1.0], 2)
    a = [S.sample(x, random.Random(7)) for _ in range(5)]
    b = [S.sample(x, random.Random(7)) for _ in range(5)]
    assert a == b and set(a) <= {0, 1}


def test_recorded_hf_check():
    with open(os.path.join(UNIT, "experiments", "results", "check_against_hf.json"), encoding="utf-8") as f:
        r = json.load(f)
    for name, ok in r["same"].items():
        assert ok + r["ties_only"][name] == r["n"], name


def math_log(p):
    import math
    return math.log(p)
