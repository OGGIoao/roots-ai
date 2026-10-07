"""common/ 的冒烟测试 (CPU 即可, 不联网)。运行: python -m pytest tests -q"""

import os

import pytest
import torch

from common.corpus import SHAKESPEARE_PATH, ensure_shakespeare, load_shakespeare
from common.text_data import CharTokenizer, TextDataset, make_loaders


def test_corpus_is_the_tracked_file():
    assert os.path.exists(SHAKESPEARE_PATH), "入库语料丢失"
    assert ensure_shakespeare() == SHAKESPEARE_PATH
    assert len(load_shakespeare()) > 1_000_000


def test_char_tokenizer_roundtrip_and_unknown():
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"
    assert tok.encode("hexz") == tok.encode("he")  # 词表外字符被丢弃
    assert tok.decode([9999]) == "?"               # 未知 ID 不抛 KeyError


@pytest.mark.parametrize("n,block,stride", [(100, 8, 1), (100, 8, 8), (97, 8, 8), (65, 8, 8), (64, 8, 8)])
def test_every_sample_is_full_length(n, block, stride):
    # 回归: n % block == 0 时, 旧 minigpt 实现的最后一块 x/y 少一个 token
    ds = TextDataset(torch.arange(n), block, stride)
    assert len(ds) > 0
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == y.shape == (block,)
        assert torch.equal(y[:-1], x[1:])
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_dataset_too_short_is_empty_and_loader_rejects():
    assert len(TextDataset(torch.arange(5), 8)) == 0
    with pytest.raises(ValueError):
        make_loaders(list(range(20)), block_size=32, batch_size=2)


def test_loaders_batch_without_collate_error():
    ids = list(range(64 * 40))  # 长度恰为 block 整数倍
    tr, va = make_loaders(ids, block_size=64, batch_size=4, stride=64)
    for loader in (tr, va):
        for x, y in loader:
            assert x.shape == y.shape and x.shape[1] == 64


def test_failed_download_raises_and_leaves_no_partial_file(tmp_path, monkeypatch):
    # 回归: 下载失败曾被静默降级成内置小语料; 中断的下载会留下残缺文件被当成有效语料
    import urllib.error
    import common.corpus as corpus

    def boom(url, dest):
        with open(dest, "w") as f:
            f.write("partial")
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(corpus.urllib.request, "urlretrieve", boom)
    target = tmp_path / "nested" / "shakespeare.txt"
    with pytest.raises(RuntimeError):
        corpus.ensure_shakespeare(str(target))
    assert not target.exists()
    assert not (tmp_path / "nested" / "shakespeare.txt.part").exists()
