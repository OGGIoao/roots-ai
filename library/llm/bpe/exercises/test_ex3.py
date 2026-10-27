"""练习 3 的验收测试: 结果与参考实现逐条相同, 且至少快 3 倍。"""

import os
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from ex3_fast_train import train_fast  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402


def ref_merges(text, vocab):
    t = BPETokenizer()
    t.train(text, vocab, verbose=False)
    return t.merges


SMALL = [
    ("abababd", 270),
    ("aaaa aaa aaaaa", 270),                       # 重叠 pair 的计数
    ("low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3, 280),
    ("Hello hello hello", 300),                    # 提前结束
    ("你好世界, 你好 世界! 2026 年 🌍🌍", 320),
]


@pytest.mark.parametrize("text,vocab", SMALL, ids=[t[:10] for t, _ in SMALL])
def test_identical_small(text, vocab):
    assert list(train_fast(text, vocab).items()) == list(ref_merges(text, vocab).items())


@pytest.fixture(scope="module")
def big():
    from data import get_corpus
    return get_corpus("shakespeare", verbose=False)[:300_000]


def test_identical_and_faster(big):
    t0 = time.perf_counter()
    ref = ref_merges(big, 600)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    mine = train_fast(big, 600)
    t_mine = time.perf_counter() - t0
    assert list(mine.items()) == list(ref.items())
    print(f"\n参考 {t_ref:.2f}s, 你的 {t_mine:.2f}s, 快 {t_ref / t_mine:.1f} 倍")
    assert t_ref / t_mine >= 3
