"""练习 2 的验收测试: 区间和参考实现一致, 并且区间里确实是 assistant 的话。"""

import pytest

from _target import golden, load
from chat_template import IM_END, render, render_with_spans

assistant_spans = load("ex2_assistant_spans", "assistant_spans", "chat_template")
CASES = [c for c in golden("qwen25_golden.json") if any(m["role"] == "assistant" for m in c["messages"])]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_spans(case):
    msgs, tools = case["messages"], case["tools"]
    got = [tuple(s) for s in assistant_spans(msgs, tools)]
    assert got == render_with_spans(msgs, tools)[1]


def test_span_content():
    msgs = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀"},
            {"role": "user", "content": "3 乘 9 是多少"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "calculator", "arguments": {"expression": "3*9"}}}]},
            {"role": "tool", "content": "27"}]
    text = render(msgs)
    spans = [tuple(s) for s in assistant_spans(msgs)]
    assert [text[s:e] for s, e in spans] == [
        "你好呀" + IM_END,
        '<tool_call>\n{"name": "calculator", "arguments": {"expression": "3*9"}}\n</tool_call>' + IM_END]
