"""A1-bpe 的回归测试 (纯标准库路径, 不联网)。"""

import os
import sys

import pytest

_PROJ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "library", "llm", "bpe")
sys.path.insert(0, _PROJ)

import data  # noqa: E402
import train  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402


def test_shakespeare_source_does_not_silently_fall_back(monkeypatch):
    def fail():
        raise RuntimeError("offline")

    monkeypatch.setattr(data, "load_shakespeare", fail)
    with pytest.raises(RuntimeError):
        data.get_corpus(source="shakespeare", verbose=False)
    assert len(data.get_corpus(source="sample", verbose=False)) > 100  # 显式选择才用内置语料


class _Args:
    def __init__(self, source, vocab_size, output=None):
        self.source, self.vocab_size, self.output = source, vocab_size, output


def test_only_default_config_writes_tracked_tokenizer_json():
    tracked = os.path.join(_PROJ, "tokenizer.json")
    assert train._resolve_save_path(_Args("shakespeare", 500)) == tracked
    for args in (_Args("shakespeare", 1000), _Args("sample", 500), _Args("sample", 300)):
        assert train._resolve_save_path(args) != tracked
    assert train._resolve_save_path(_Args("sample", 300, "x.json")) == os.path.abspath("x.json")


def test_tracked_tokenizer_roundtrip():
    tok = BPETokenizer.load(os.path.join(_PROJ, "tokenizer.json"))
    s = "To be, or not to be, that is the question. 你好 🙂"
    assert tok.decode(tok.encode(s)) == s
