"""
练习 1 (① 填空): 补全 GQA

把下面三处 TODO 补全, 让 GroupedQueryAttention 在 MHA / GQA / MQA 三种配置下
都与参考实现 (../variants.py) 逐数值一致。

验收:
    cd mainline/A3-attention/exercises
    python -m pytest test_ex1.py -q

提示规则: 卡住了可以要提示, 按"方向 → 哪一行 → 代码"分级给, 每次只升一级。
写完再去对照 ../variants.py, 看看自己的写法和参考写法有什么不同。
"""

import os
import sys

import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # base.py
from base import apply_rope  # noqa: E402


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        assert dim % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # TODO 1: K、V 的投影。想清楚输出维度是多少 (提示: 不是 n_heads * head_dim)
        self.wk = None
        self.wv = None
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.rope_dim = self.head_dim
        # TODO 2: 推理时每个 token 在这一层要缓存多少个数
        self.kv_numbers_per_token = None

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)      # (B, H, S, d)
        # TODO 3: 算出 k、v (形状 (B, G, S, d)), 对 q、k 做 RoPE,
        #         再把每组 K、V 交给组内的查询头, 使 k、v 变成 (B, H, S, d)。
        #         组的约定: 头 0..(H/G - 1) 属于第 0 组, 接下来 H/G 个头属于第 1 组, 依此类推。
        raise NotImplementedError
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # noqa: F821
        return self.wo(out.transpose(1, 2).reshape(B, S, -1))
