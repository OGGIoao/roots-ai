"""练习 3 的验收测试: 和 Mistral-7B-Instruct-v0.3 官方模板的渲染结果逐字节比对, 不合法的对话要报错。"""

import pytest

from _target import golden, load

render_mistral = load("ex3_mistral_blank", "render_mistral", "mistral_v03")
CASES = golden("mistral_golden.json")


@pytest.mark.parametrize("i", range(len(CASES)))
def test_case(i):
    case = CASES[i]
    if "error" in case:
        with pytest.raises(ValueError):
            render_mistral(case["messages"])
    else:
        assert render_mistral(case["messages"]) == case["expected"]
