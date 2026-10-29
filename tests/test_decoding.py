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
    assert all(d["cycle"] for d in r["greedy_loops"])                       # 第 3 节: 5 个开头全部转圈
    rows = {x["setting"]: x for x in r["tradeoff"]}
    assert rows["贪心 + 重复惩罚 2.0"]["loop_rate"] == 1.0                     # 第 6.2 节
    assert rows["温度 1.0"]["loop_rate"] == 0.0
    shift = {x["shift"]: x["p_e_after"] for x in r["penalty_shift"]["rows"]}  # 第 6.3 节
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


def test_real_model_numbers_used_in_tutorial():
    station = os.path.join(ROOT, "projects", "car", "v0.4-text-commands", "results")
    with open(os.path.join(station, "decoding_compare.json"), encoding="utf-8") as f:
        c = json.load(f)
    s = c["settings"]
    assert s["已公开: 贪心 + 自带重复惩罚 1.1"]["success"] == 14
    assert s["贪心, 关掉重复惩罚 (1.0)"]["success"] == 18
    assert c["rerun_check"]["success_same"] == c["rerun_check"]["first_reply_same"] == 30
    with open(os.path.join(UNIT, "experiments", "results", "practice_car_confidence_Qwen2.5-1.5B-Instruct.json"),
              encoding="utf-8") as f:
        t = json.load(f)
    assert sum(r["n_flips_by_penalty"] > 0 for r in t["rows"]) == 3


def test_other_penalties():
    x = [1.0, 1.0, 1.0]
    assert S.presence_frequency_penalty(x, [0, 0, 1], presence=0.5, frequency=0.25) == [0.0, 0.25, 1.0]
    assert S.windowed_repetition_penalty([2.0, 2.0], [0, 1], 2.0, last_n=1) == [2.0, 1.0]    # 只看最近 1 个
    assert kept(S.no_repeat_ngram([0.0] * 3, [0, 1, 2, 0], 2)) == [0, 2]                      # "0 1" 出现过, 不许再接 1


def test_additive_penalty_is_shift_invariant():
    base = [0.3, -1.2, 2.0, 0.7]
    ref = S.softmax(S.presence_frequency_penalty(base, [2, 2, 0], presence=0.4, frequency=0.3))
    for c in (-30.0, 7.5, 40.0):
        got = S.softmax(S.presence_frequency_penalty([z + c for z in base], [2, 2, 0], presence=0.4, frequency=0.3))
        assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-12


def test_orders_differ_only_by_order():
    logits = [2.0, 1.5, 1.0, 0.5, -1.0, -2.0]                   # 讲义第 2.5 节的手算
    a = S.softmax(S.process_order(logits, (), "transformers", temp=0.5, p=0.8))
    b = S.softmax(S.process_order(logits, (), "llama.cpp", temp=0.5, p=0.8))
    assert kept_p(a) == [0, 1] and kept_p(b) == [0, 1, 2]
    assert round(a[0], 2) == 0.73 and round(b[0], 2) == 0.67
    assert S.process_order(logits, (), "vllm", temp=0.7, k=3, p=0.9) == S.process_order(logits, (), "transformers", temp=0.7, k=3, p=0.9)


def kept_p(probs):
    return [i for i, p in enumerate(probs) if p > 0]


def test_toy_more_numbers_used_in_tutorial():
    with open(os.path.join(UNIT, "experiments", "results", "toy_more.json"), encoding="utf-8") as f:
        r = json.load(f)
    pen = {x["setting"]: x for x in r["penalties"]}                          # 第 7.2 节
    assert pen["出现惩罚 0.5 (加法)"]["loop_rate"] == 1.0
    assert pen["频率惩罚 0.3 (加法, 按次数)"]["loop_rate"] == 0.0
    assert pen["禁止重复 4 元组"]["distinct4"] == 1.0
    assert len({x["presence_0.5"] for x in r["shift"]["rows"]}) == 1         # 第 6.3 节: 减法是一条平线
    o = r["orders"][0]                                                       # 专题 A 第一行
    assert o["kept_set_differs_from_transformers"] == {"vllm": 0, "llama.cpp": 1640}
    assert [b["score"] for b in r["beam"]][2:] == [-39.79] * 4               # 第 9.2 节
    assert all(b["loops"] for b in r["beam"])


def test_real_model_generic_numbers_used_in_tutorial():
    def load(m):
        with open(os.path.join(UNIT, "experiments", "results", f"real_model_Qwen2.5-{m}-Instruct.json"), encoding="utf-8") as f:
            return json.load(f)["summary"]
    a, b = load("0.5B"), load("1.5B")                                         # 专题 B、C
    assert (a["reply_changed_by_penalty"], a["first_token_changed_by_penalty"]) == (7, 0)
    assert (b["reply_changed_by_penalty"], b["first_token_changed_by_penalty"]) == (6, 3)
    assert (b["calls_correct"], b["calls_correct_with_penalty"]) == (11, 10)
    assert a["bf16_reply_differs"] == 1 and a["path_argmax_differs_positions"] == 0
    assert (b["path_argmax_differs_positions"], b["path_positions"]) == (1, 284)


def test_tutorial_examples_are_project_independent():
    """知识库不依赖项目: 主干和专题里不出现小车; 只有第 15 节实践 (以及专题 C 引用一次它的重跑结果)。"""
    text = open(os.path.join(UNIT, "TUTORIAL.md"), encoding="utf-8").read()
    before, rest = text.split("## 15. 实践: 在项目里用", 1)
    after = rest.split("# 第二部分 · 延伸专题", 1)[1]
    assert "小车" not in before.split("## 0. 阅读指南", 1)[1].split("### 0.4", 1)[1]
    assert "小车" not in after.replace("第 15.3 节的小车评测", "")
