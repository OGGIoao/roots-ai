"""练习 1 的验收测试: 与参考实现 (../tokenizer.py) 逐条比对。"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from ex1_bpe_fill import MiniBPE, merge_pair  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402


def shakespeare(n):
    from data import get_corpus
    return get_corpus("shakespeare", verbose=False)[:n]


def test_merge_pair():
    assert merge_pair((97, 98, 97, 98), (97, 98), 256) == (256, 256)
    assert merge_pair((97, 97, 97, 98), (97, 97), 256) == (256, 97, 98)     # 重叠: 先到先得
    assert merge_pair((97, 97, 97, 97), (97, 97), 256) == (256, 256)
    assert merge_pair((1, 2, 3), (4, 5), 256) == (1, 2, 3)
    ids = (97, 98)
    merge_pair(ids, (97, 98), 256)
    assert ids == (97, 98)


def test_pair_freqs_weighted_and_overlapping():
    pf = MiniBPE.pair_freqs({(97, 97, 97, 97): 1, (108, 111, 119): 5})
    assert pf[(97, 97)] == 3            # 重叠的也要数 (和参考实现一致)
    assert pf[(108, 111)] == 5          # 乘上词的出现次数
    assert len(pf) == 3


TEXTS = [
    ("abababd", 270),
    ("low lower lowest", 270),
    ("low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3, 280),
    ("Hello hello hello", 300),                    # 会提前结束
    ("你好世界, 你好 世界! 2026 年", 300),
]


@pytest.mark.parametrize("text,vocab", TEXTS, ids=[t[:12] for t, _ in TEXTS])
def test_same_merges_small(text, vocab):
    ref, mine = BPETokenizer(), MiniBPE()
    ref.train(text, vocab, verbose=False)
    mine.train(text, vocab)
    assert list(mine.merges.items()) == list(ref.merges.items())


def test_same_merges_and_encoding_shakespeare():
    text = shakespeare(100_000)
    ref, mine = BPETokenizer(), MiniBPE()
    ref.train(text, 320, verbose=False)
    mine.train(text, 320)
    assert list(mine.merges.items()) == list(ref.merges.items())
    for s in ["To be, or not to be, that is the question.", " lowest", "Hello, world! 你好 🌍", ""]:
        assert mine.encode(s) == ref.encode(s)
        assert mine.decode(mine.encode(s)) == s
