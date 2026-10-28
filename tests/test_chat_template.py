"""知识库 llm/chat-template (对话模板) 的回归测试: 只用标准库和入库的标准答案, 不需要 transformers 和网络。"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIT = os.path.join(ROOT, "library", "llm", "chat-template")
sys.path.insert(0, UNIT)

from chat_template import IM_END, render, render_with_spans  # noqa: E402
from mistral_v03 import render_mistral  # noqa: E402


def golden(name):
    with open(os.path.join(UNIT, "experiments", "results", name), encoding="utf-8") as f:
        return json.load(f)


def test_qwen_matches_official_golden():
    g = golden("qwen25_golden.json")
    assert g["summary"]["random_same"] == g["summary"]["random_total"] == 500
    for c in g["cases"]:
        assert render(c["messages"], c["tools"], c["add_generation_prompt"]) == c["expected"], c["name"]


def test_spans_start_at_generation_prompt():
    for c in golden("qwen25_golden.json")["cases"]:
        msgs, tools = c["messages"], c["tools"]
        text, spans = render_with_spans(msgs, tools)
        idx = [k for k, m in enumerate(msgs) if m["role"] == "assistant"]
        assert len(spans) == len(idx)
        for k, (s, e) in zip(idx, spans):
            assert text[:s] == render(msgs[:k], tools, add_generation_prompt=True)
            assert text[s:e].endswith(IM_END)


def test_mistral_matches_official_golden():
    for c in golden("mistral_golden.json")["cases"]:
        if "error" in c:
            try:
                render_mistral(c["messages"])
            except ValueError:
                continue
            raise AssertionError(f"应该报错: {c['messages']}")
        assert render_mistral(c["messages"]) == c["expected"]


def test_measured_numbers_used_in_tutorial():
    with open(os.path.join(UNIT, "experiments", "results", "tokens.json"), encoding="utf-8") as f:
        t = json.load(f)
    assert t["budget"]["total_with_tools"] == 760 and t["budget"]["tools_block"] == 535
    assert t["injection"]["default_im_start_count"] == 4 and t["injection"]["safe_im_start_count"] == 3


def test_exercise_tests_pass_on_reference():
    env = dict(os.environ, CT_SOLUTION="reference", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
                       cwd=os.path.join(UNIT, "exercises"), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
