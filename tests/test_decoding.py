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


def test_toy_experiment_numbers_used_in_tutorial():
    with open(os.path.join(UNIT, "experiments", "results", "toy_experiments.json"), encoding="utf-8") as f:
        r = json.load(f)
    assert all(d["cycle"] for d in r["greedy_loops"])                       # 第 2 节: 5 个开头全部转圈
    rows = {x["setting"]: x for x in r["tradeoff"]}
    assert rows["贪心 + 重复惩罚 2.0"]["loop_rate"] == 1.0                     # 第 5.1 节
    assert rows["温度 1.0"]["loop_rate"] == 0.0
    shift = {x["shift"]: x["p_e_after"] for x in r["penalty_shift"]["rows"]}  # 第 5.2 节
    assert shift[0.0] > 0.3 and shift[-20.0] < 0.01 and shift[20.0] < 0.05


def test_beam_width_one_is_greedy_on_toylm():
    from beam import beam_search
    from generate import generate
    from toy_lm import ToyLM
    lm = ToyLM.from_corpus()

    def step(ids):
        return lm.logits(lm.decode(ids[-2:]))
    p = lm.encode("ROMEO:\n")
    assert beam_search(step, p, 1, 30)[0] == generate(step, p, 30, do_sample=False)[0]


def test_exercise_tests_pass_on_reference():
    import subprocess
    env = dict(os.environ, DEC_SOLUTION="reference", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
                       cwd=os.path.join(UNIT, "exercises"), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
