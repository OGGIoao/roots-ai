"""练习 3 的验收测试: 用正则表达式穷举核对所有长度 ≤ 4 的字符串 (12^4 + ... ≈ 2.3 万个)。"""

import itertools
import re

from _target import load  # 先导入它: 它把上一级目录加进搜索路径

is_prefix = load("ex3_number_blank", "is_prefix", "number_dfa")
is_complete = load("ex3_number_blank", "is_complete", "number_dfa")
allowed_next_chars = load("ex3_number_blank", "allowed_next_chars", "number_dfa")

ALPHABET = "-0123456789."
NUMBER = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")


def complete_ref(s):
    return NUMBER.fullmatch(s) is not None


def prefix_ref(s):
    # 任何还能补成合法数字的前缀, 最多再补 2 个字符就够了 (比如 "-" 补 "0", "1." 补 "5")
    return any(complete_ref(s + "".join(t)) for n in range(3) for t in itertools.product(ALPHABET, repeat=n))


def all_strings(max_len=4):
    for n in range(max_len + 1):
        for t in itertools.product(ALPHABET, repeat=n):
            yield "".join(t)


def test_complete():
    for s in all_strings():
        assert is_complete(s) == complete_ref(s), s


def test_prefix_and_next():
    for s in all_strings(3):
        p = prefix_ref(s)
        assert is_prefix(s) == p, s
        want = {ch for ch in ALPHABET if prefix_ref(s + ch)} if p else set()
        assert allowed_next_chars(s) == want, s


def test_examples():
    for good in ["0", "7", "-3", "12", "0.5", "-0.25", "10.0"]:
        assert is_complete(good), good
    for bad in ["01", "1.", ".5", "-", "--1", "1.2.3", ""]:
        assert not is_complete(bad), bad
