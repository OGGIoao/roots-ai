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


def results(name):
    with open(os.path.join(UNIT, "experiments", "results", name), encoding="utf-8") as f:
        return json.load(f)


def test_measured_numbers_used_in_tutorial():
    t = results("tokens.json")
    assert t["budget"]["total_with_tools"] == 377 and t["budget"]["tools_block"] == 337          # 第 8.1 节
    assert t["language"]["tools_json_zh_escaped"] == 511 and t["language"]["tools_json_zh"] == 257   # 第 8.3 节
    assert t["injection"]["default_im_start_count"] == 4 and t["injection"]["safe_im_start_count"] == 3
    assert t["injection"]["apply_chat_template_tokenize_count"] == 4                                 # 第 10.1 节
    assert results("practice_car.json")["budget"]["total_with_tools"] == 760                         # 第 16.1 节
    nt = results("no_template.json")["summary"]                                                      # 第 2.1 节
    assert (nt["raw"]["stopped"], nt["wrong"]["stopped"], nt["right"]["stopped"]) == (0, 0, 4)


def test_extended_topics_numbers():
    p = results("prefix_property.json")                                                               # 专题 B
    assert p["Qwen/Qwen3-8B"]["plain"]["prefix_ok"] == 0
    assert p["Qwen/Qwen2.5-1.5B-Instruct"]["plain"]["prefix_ok"] == p["Qwen/Qwen2.5-1.5B-Instruct"]["plain"]["total"]
    b = results("token_boundaries.json")                                                              # 专题 D、E
    assert b["hf_assistant_mask"]["ones"] == 0 and b["our_mask"]["ones"] == 3
    assert b["split_vs_whole"]["same"] == b["split_vs_whole"]["total"]
    assert b["plain_template_split_vs_whole"]["same"] == 7


def test_examples_are_project_independent():
    """知识库不依赖项目: 主干的例子里不出现小车。"""
    import examples
    text = json.dumps([examples.TOOLS, examples.CONVERSATION], ensure_ascii=False)
    assert "小车" not in text and "walk" not in text


def test_exercise_tests_pass_on_reference():
    env = dict(os.environ, CT_SOLUTION="reference", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
                       cwd=os.path.join(UNIT, "exercises"), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
