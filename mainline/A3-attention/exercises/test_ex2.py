"""练习 2 的验收测试。"""

import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from base import rope_table  # noqa: E402
from ex2_mha_to_gqa import mha_to_gqa  # noqa: E402
from variants import GroupedQueryAttention  # noqa: E402

DIM, HEADS, D = 64, 8, 8
COS, SIN = rope_table(D, 16)


def _x():
    torch.manual_seed(1)
    return torch.randn(2, 16, DIM)


def _mha():
    torch.manual_seed(0)
    return GroupedQueryAttention(DIM, HEADS, HEADS)


def test_type_and_groups():
    g = mha_to_gqa(_mha(), 2)
    assert isinstance(g, GroupedQueryAttention) and g.n_kv_heads == 2 and g.n_heads == HEADS


def test_identity_when_no_grouping():
    m = _mha()
    assert torch.allclose(mha_to_gqa(m, HEADS)(_x(), COS, SIN), m(_x(), COS, SIN), atol=1e-5)


def test_lossless_when_heads_in_group_are_identical():
    m = _mha()
    with torch.no_grad():                              # 让每组 (4 个头) 的 K、V 投影完全相同
        for w in (m.wk.weight, m.wv.weight):
            blocks = w.view(HEADS, D, DIM)
            for g0 in range(0, HEADS, 4):
                blocks[g0:g0 + 4] = blocks[g0].clone()
    assert torch.allclose(mha_to_gqa(m, 2)(_x(), COS, SIN), m(_x(), COS, SIN), atol=1e-5)


def test_group_convention_is_contiguous():
    m = _mha()
    with torch.no_grad():
        blocks = m.wk.weight.view(HEADS, D, DIM)
        for h in range(HEADS):
            blocks[h] = float(h)                       # 第 h 个头的 K 投影全部填 h
    g = mha_to_gqa(m, 2)
    k = g.wk.weight.view(2, D, DIM)
    assert torch.allclose(k[0], torch.full((D, DIM), 1.5))   # 头 0~3 的平均
    assert torch.allclose(k[1], torch.full((D, DIM), 5.5))   # 头 4~7 的平均


def test_does_not_modify_input():
    m = _mha()
    before = copy.deepcopy(m.state_dict())
    mha_to_gqa(m, 2)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
