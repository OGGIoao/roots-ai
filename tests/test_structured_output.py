"""知识库 llm/structured-output (结构化输出) 的回归测试: 只用标准库和入库的实验结果, 不需要模型。"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIT = os.path.join(ROOT, "library", "llm", "structured-output")
sys.path.insert(0, UNIT)

from constrained import Trie, allowed_token_ids, car_tool_calls  # noqa: E402
import number_dfa  # noqa: E402


def test_car_calls_trie():
    calls = car_tool_calls()
    assert len(calls) == 4
    t = Trie(calls)
    assert t.allowed_next_chars('<tool_call>\n{"name": "') == {"g", "l"}
    assert all(t.is_complete(c) for c in calls)


def test_cross_boundary_token_allowed():
    t = Trie(car_tool_calls())
    text = '<tool_call>\n{"name": "go_to", "arguments": {"target": "red_cup'
    assert allowed_token_ids(t, text, ['"}}\n', "}", ""], 9) == {0}


def test_number_dfa_examples():
    assert number_dfa.is_complete("-0.25") and not number_dfa.is_complete("01")
    assert number_dfa.allowed_next_chars("-") == set("0123456789")


def test_recorded_numbers_used_in_tutorial():
    with open(os.path.join(UNIT, "experiments", "results", "structured.json"), encoding="utf-8") as f:
        r = json.load(f)
    assert r["toy"]["constrained_greedy"].startswith("toolcall\nname: look")
    q = {m["model"].split("/")[-1]: {row["setting"]: row for row in m["rows"]} for m in r["qwen"]}
    big = q["Qwen2.5-1.5B-Instruct"]
    assert big["不加约束, 重复惩罚 1.1"]["valid_calls"] == 3
    assert big["加约束, 重复惩罚 1.1"]["valid_calls"] == big["加约束, 重复惩罚 1.1"]["correct_target"] == 6


def test_exercise_tests_pass_on_reference():
    env = dict(os.environ, SO_SOLUTION="reference", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
                       cwd=os.path.join(UNIT, "exercises"), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
