"""练习 1 的验收测试: 与参考实现逐数值比对。"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # base.py
sys.path.insert(0, os.path.dirname(HERE))                    # variants.py
sys.path.insert(0, HERE)

from base import rope_table  # noqa: E402
from ex1_gqa_fill import GroupedQueryAttention as Mine  # noqa: E402
from variants import GroupedQueryAttention as Ref  # noqa: E402

DIM, HEADS = 64, 8


@pytest.mark.parametrize("n_kv", [8, 4, 2, 1], ids=["MHA", "GQA-4", "GQA-2", "MQA"])
def test_matches_reference(n_kv):
    torch.manual_seed(0)
    ref, mine = Ref(DIM, HEADS, n_kv), Mine(DIM, HEADS, n_kv)
    mine.load_state_dict(ref.state_dict())          # 参数名与形状必须和参考实现一致
    assert mine.kv_numbers_per_token == ref.kv_numbers_per_token
    cos, sin = rope_table(DIM // HEADS, 16)
    x = torch.randn(2, 16, DIM)
    assert torch.allclose(mine(x, cos, sin), ref(x, cos, sin), atol=1e-5)
