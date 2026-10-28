"""练习 1 的验收测试: 和官方模板渲染结果逐字节比对。"""

import pytest

from _target import golden, load

render = load("ex1_render_fill", "render", "chat_template")
CASES = golden("qwen25_golden.json")


def where(a, b):
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"第一个不同在位置 {k}:\n  你的: {a[max(0, k - 40):k + 40]!r}\n  官方: {b[max(0, k - 40):k + 40]!r}"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_same_as_official(case):
    got = render(case["messages"], case["tools"], case["add_generation_prompt"])
    assert got == case["expected"], where(got, case["expected"])
