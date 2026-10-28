"""练习 2 的验收测试: 跨边界的多字符 token、空 token、停止 token, 并在随机词表上和参考实现比对。"""

import random

import pytest

from _target import load  # 先导入它: 它把上一级目录加进搜索路径
from constrained import Trie, car_tool_calls  # noqa: E402
from constrained import allowed_token_ids as ref  # noqa: E402

allowed_token_ids = load("ex2_token_mask", "allowed_token_ids", "constrained")

CALLS = car_tool_calls()
TRIE = Trie(CALLS)
# 一个缩小的 "Qwen 式" 词表: 有跨边界的 token、有单字符、有空 token、有停止 token
VOCAB = ["<tool_call>", "\n", '{"', "name", '":', ' "', "go", "_to", '",', "arguments", ' {"', "target", "red",
         "_c", "up", '"}}\n', "</tool_call>", "look", '":', " {}}\n", "blue", "_ball", "green", "_box", "", "<|im_end|>",
         "x", '"', "}", "\n</", "tool_call>"]
EOS = VOCAB.index("<|im_end|>")


def ids_to_text(ids):
    return "".join(VOCAB[i] for i in ids)


def test_start():
    got = allowed_token_ids(TRIE, "", VOCAB, EOS)
    assert VOCAB.index("<tool_call>") in got
    assert VOCAB.index("x") not in got and VOCAB.index("") not in got and EOS not in got


def test_cross_boundary_token():
    text = '<tool_call>\n{"name": "go_to", "arguments": {"target": "red_cup'
    got = allowed_token_ids(TRIE, text, VOCAB, EOS)
    assert VOCAB.index('"}}\n') in got          # 一个 token 跨过引号、两个括号和换行, 仍然合法
    assert VOCAB.index('"') in got
    assert VOCAB.index("}") not in got


def test_eos_only_when_complete():
    for c in CALLS:
        assert EOS in allowed_token_ids(TRIE, c, VOCAB, EOS)
        assert EOS not in allowed_token_ids(TRIE, c[:-1], VOCAB, EOS)
    assert allowed_token_ids(TRIE, "<tool_call>\nhello", VOCAB, EOS) == set()


@pytest.mark.parametrize("seed", range(10))
def test_random_vocab_same_as_reference(seed):
    rng = random.Random(seed)
    alphabet = "abc"
    strings = ["".join(rng.choice(alphabet) for _ in range(rng.randint(2, 7))) for _ in range(5)]
    vocab = sorted({"".join(rng.choice(alphabet) for _ in range(rng.randint(0, 3))) for _ in range(40)})
    trie = Trie(strings)
    for s in strings:
        for i in range(len(s) + 1):
            for text in (s[:i], s[:i] + "c"):
                assert allowed_token_ids(trie, text, vocab, 999) == ref(trie, text, vocab, 999), (strings, text)


def random_walk(vocab, eos, rng):
    """每一步都从允许的集合里随便挑一个 (相当于模型打分完全随机), 返回写出的文字; 走进死胡同返回 None。"""
    ids = []
    while True:
        ok = allowed_token_ids(TRIE, "".join(vocab[i] for i in ids), vocab, eos)
        if not ok:
            return None
        t = rng.choice(sorted(ok))
        if t == eos:
            return "".join(vocab[i] for i in ids)
        ids.append(t)


def test_incomplete_vocab_can_dead_end():
    """讲义第 4.3 节: 前缀合法不等于写得完。这个小词表里没有单独的 ':', 写出 '{"name' + '"' 之后就接不下去了。"""
    rng = random.Random(0)
    outs = [random_walk(VOCAB, EOS, rng) for _ in range(50)]
    assert any(o is None for o in outs)
    assert all(o in CALLS for o in outs if o is not None)


def test_full_vocab_never_dead_ends():
    """把合法输出里出现的每个单字符都加进词表 (字节级 BPE 的词表就是这样), 随便怎么挑都能写完。"""
    chars = sorted({ch for c in CALLS for ch in c})
    vocab = VOCAB + chars
    rng = random.Random(1)
    for _ in range(50):
        assert random_walk(vocab, EOS, rng) in CALLS
