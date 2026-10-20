"""练习 2 的验收测试: 流式解码。"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from ex2_stream_decode import StreamDecoder  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

ZH = "分词器把文字切成片段。切得不好, 一个汉字可能拆成好几个编号, 模型更难学到词的意思。"


def tokenizers():
    en = BPETokenizer.load(os.path.join(ROOT, "tokenizer.json"))   # 英文词表: 汉字全是单字节
    zh = BPETokenizer()
    zh.train(ZH * 3, 400, verbose=False)                             # 中文词表: 很多 token 跨越汉字边界
    return {"en": en, "zh": zh}


TOKS = tokenizers()
TEXTS = ["To be, or not to be.", "你好世界 🌍!", ZH, "混合 mixed 文本 éñ 😀😀 end", ""]


def longest_complete_prefix(b: bytes) -> str:
    for cut in range(len(b), -1, -1):
        try:
            return b[:cut].decode("utf-8")
        except UnicodeDecodeError:
            continue
    return ""


@pytest.mark.parametrize("name", ["en", "zh"])
@pytest.mark.parametrize("text", TEXTS, ids=range(len(TEXTS)))
def test_stream_equals_whole_and_never_late(name, text):
    tok = TOKS[name]
    ids = tok.encode(text)
    dec = StreamDecoder(tok.vocab)
    got, seen = "", b""
    for i in ids:
        piece = dec.feed(i)
        assert isinstance(piece, str)
        assert "�" not in piece                      # 要求 2
        got += piece
        seen += tok.vocab[i]
        assert got == longest_complete_prefix(seen)       # 要求 3: 不拖延, 也不抢跑
    got += dec.flush()
    assert got == tok.decode(ids) == text                 # 要求 1


def test_crossing_tokens_exist():
    """确认中文词表里真的有跨越汉字边界的 token, 否则上面的测试太容易。"""
    bad = 0
    for i, b in TOKS["zh"].vocab.items():
        if i < 256:
            continue                                      # 只看训练学到的 token
        try:
            b.decode("utf-8")
        except UnicodeDecodeError:
            bad += 1
    assert bad >= 20


@pytest.mark.parametrize("tail", [b"\xe4", b"\xe4\xbd", b"\xf0\x9f\x98"])
def test_flush_incomplete_tail(tail):
    vocab = {i: bytes([i]) for i in range(256)}
    data = "ok".encode() + tail
    dec = StreamDecoder(vocab)
    got = "".join(dec.feed(b) for b in data)
    assert got == "ok"
    got += dec.flush()
    assert got == data.decode("utf-8", errors="replace")   # 要求 4
    assert dec.flush() == ""                                # flush 之后状态清空


def test_no_codecs():
    src = open(os.path.join(HERE, "ex2_stream_decode.py"), encoding="utf-8").read()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "getincrementaldecoder" not in code and "IncrementalDecoder" not in code   # 要求 5
