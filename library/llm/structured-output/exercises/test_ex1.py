"""练习 1 的验收测试: 和参考实现的前缀树在随机字符串集合上逐个查询比对。"""

import random

import pytest

from _target import load  # 先导入它: 它把上一级目录加进搜索路径
from constrained import Trie as Ref  # noqa: E402

Trie = load("ex1_trie_fill", "Trie", "constrained")


def queries(strings, rng):
    qs = {"", "zzz"}
    for s in strings:
        for i in range(len(s) + 1):
            qs.add(s[:i])
            qs.add(s[:i] + rng.choice("abxy"))
    return sorted(qs)


@pytest.mark.parametrize("seed", range(8))
def test_same_as_reference(seed):
    rng = random.Random(seed)
    strings = ["".join(rng.choice("abc") for _ in range(rng.randint(1, 6))) for _ in range(rng.randint(1, 8))]
    mine, ref = Trie(strings), Ref(strings)
    for q in queries(strings, rng):
        assert mine.is_prefix(q) == ref.is_prefix(q), q
        assert mine.is_complete(q) == ref.is_complete(q), q
        assert mine.allowed_next_chars(q) == ref.allowed_next_chars(q), q
        assert (mine.walk(q) is None) == (ref.walk(q) is None), q


def test_tool_calls():
    from constrained import car_tool_calls
    calls = car_tool_calls()
    t = Trie(calls)
    assert all(t.is_complete(c) for c in calls)
    assert t.allowed_next_chars('<tool_call>\n{"name": "') == {"g", "l"}
    assert not t.is_prefix('<tool_call>\n{"name": "fly')
    assert not t.is_complete('<tool_call>\n{"name": "look"')
