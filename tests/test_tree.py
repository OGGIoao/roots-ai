"""llm-tree 的回归测试: 可插拔解码器与注意力线上的各个节点 (CPU, 极小模型)。"""

import math
import os
import sys

import pytest
import torch

_TREE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "library", "llm", "attention")
sys.path.insert(0, _TREE)   # base.py 与 variants.py 都在 A3-attention/

from base import Decoder, apply_rope, rope_table  # noqa: E402
from variants import GroupedQueryAttention as GQA, MultiHeadLatentAttention as MLA  # noqa: E402

DIM, HEADS = 64, 4
VARIANTS = {
    "MHA": lambda: GQA(DIM, HEADS, 4),
    "GQA": lambda: GQA(DIM, HEADS, 2),
    "MQA": lambda: GQA(DIM, HEADS, 1),
    "MLA": lambda: MLA(DIM, HEADS, 32, 16, 8, 16),
}


def test_rope_depends_only_on_relative_position():
    cos, sin = rope_table(16, 64)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def at(x, p):
        return apply_rope(x, cos[p:p + 1], sin[p:p + 1])

    assert torch.allclose((at(q, 5) * at(k, 3)).sum(), (at(q, 40) * at(k, 38)).sum(), atol=1e-5)


@pytest.mark.parametrize("name", VARIANTS)
def test_causal(name):
    torch.manual_seed(0)
    m = Decoder(50, 32, DIM, 2, VARIANTS[name]).eval()
    x = torch.randint(0, 50, (2, 20))
    x2 = x.clone()
    x2[:, 12:] = torch.randint(0, 50, (2, 8))           # 只改"未来"
    with torch.no_grad():
        assert torch.allclose(m(x)[0][:, :12], m(x2)[0][:, :12], atol=1e-5)


@pytest.mark.parametrize("name", VARIANTS)
def test_initial_loss_near_ln_vocab(name):
    # 注意 targets 必须与输入错开一位; 用 m(x, x) 会因为"答案就在输入里"而偏低
    torch.manual_seed(0)
    m = Decoder(50, 32, DIM, 2, VARIANTS[name]).eval()
    seq = torch.randint(0, 50, (8, 21))
    with torch.no_grad():
        loss = m(seq[:, :-1], seq[:, 1:])[1].item()
    assert abs(loss - math.log(50)) < 0.15


def test_kv_cache_sizes():
    layers = 3
    def kv(make):
        return Decoder(50, 32, DIM, layers, make).kv_cache_bytes_per_token(2)
    d = DIM // HEADS
    assert kv(VARIANTS["MHA"]) == layers * 2 * 4 * d * 2
    assert kv(VARIANTS["GQA"]) == layers * 2 * 2 * d * 2
    assert kv(VARIANTS["MQA"]) == layers * 2 * 1 * d * 2
    assert kv(VARIANTS["MLA"]) == layers * (32 + 8) * 2     # 潜向量 + 共享 RoPE 键


def test_mqa_with_one_group_equals_mha_with_tied_kv_heads():
    """GQA 的"复制 K、V 给组内各头"等价于一个 K、V 投影在各头之间完全相同的 MHA。"""
    torch.manual_seed(0)
    mqa, mha = GQA(DIM, HEADS, 1), GQA(DIM, HEADS, HEADS)
    with torch.no_grad():
        mha.wq.weight.copy_(mqa.wq.weight)
        mha.wo.weight.copy_(mqa.wo.weight)
        mha.wk.weight.copy_(mqa.wk.weight.repeat(HEADS, 1))
        mha.wv.weight.copy_(mqa.wv.weight.repeat(HEADS, 1))
        cos, sin = rope_table(DIM // HEADS, 16)
        x = torch.randn(2, 16, DIM)
        assert torch.allclose(mqa(x, cos, sin), mha(x, cos, sin), atol=1e-5)
