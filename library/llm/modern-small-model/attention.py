"""
Attention Mechanisms — 从原始Scaled Dot-Product到现代GQA

架构演进:
  1. Scaled Dot-Product Attention (2017, 原始Transformer)
  2. Multi-Head Attention (2017, 原始Transformer)
  3. Grouped Query Attention (GQA, LLaMA-2+)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: torch.Tensor,      # (batch, seq_len, d_k)
    key: torch.Tensor,        # (batch, seq_len, d_k)
    value: torch.Tensor,      # (batch, seq_len, d_v)
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention (Vaswani et al., 2017)

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: (B, S, d_k)
        key:   (B, S, d_k)
        value: (B, S, d_v)
        mask:  (B, 1, S) 或 (B, S, S), True 表示需要 mask 掉的位置

    Returns:
        output: (B, S, d_v)
        attn_weights: (B, S, S) 注意力权重
    """
    d_k = query.size(-1)
    # QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (for causal attention or padding)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (2017, 原始Transformer)

    将 Q, K, V 投影到 h 个子空间,各自计算 attention 后拼接:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    维度分配:
        d_model = n_heads * d_k  (通常 d_k = d_v = d_model / n_heads)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 一个大矩阵包含所有头的 Q, K, V 投影 (更高效)
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                 # (B, S, d_model)
        mask: Optional[torch.Tensor] = None,  # (B, 1, S) 或 (B, S, S)
    ) -> torch.Tensor:
        B, S, _ = x.shape

        # 1. 投影到 Q, K, V
        qkv = self.W_qkv(x)              # (B, S, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)   # 各 (B, S, d_model)

        # 2. reshape: (B, S, d_model) -> (B, n_heads, S, d_k)
        q = q.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        self._attn_weights = attn_weights  # 保存用于可视化

        # 4. Concat heads: (B, n_heads, S, d_k) -> (B, S, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        # 5. 输出投影
        return self.W_o(attn_out)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    Reference: arxiv.org/abs/2305.13245 (Ainslie et al., 2023, LLaMA-2)

    核心思想:
        - 标准 MHA: n_heads 个 Q, K, V 头 — 每个头有独立的 KV
        - MQA: n_heads 个 Q 头, 但只有 1 个 KV 头 (所有 Q 头共享同一组 KV)
        - GQA: n_heads 个 Q 头, n_kv_heads 个 KV 头 (中间状态)
          每 n_heads // n_kv_heads 个 Q 头共享一组 KV

    优势:
        - 减少 KV-Cache 大小 (推理时内存节省显著)
        - 质量接近 MHA,速度接近 MQA

    LLaMA-2 70B:  n_heads=64, n_kv_heads=8  (GQA, 8组)
    LLaMA-3 8B:   n_heads=32, n_kv_heads=8  (GQA, 4组)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_heads_per_kv = n_heads // n_kv_heads  # 每个 KV 头对应几个 Q 头
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        # Q 投影到所有头
        q = self.W_q(x)  # (B, S, n_heads * d_k)
        # K, V 只投影到 n_kv_heads 个头
        k = self.W_k(x)  # (B, S, n_kv_heads * d_k)
        v = self.W_v(x)  # (B, S, n_kv_heads * d_k)

        # Reshape
        q = q.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 扩展 K, V 以匹配 Q 的头数 (repeat_interleave)
        # (B, n_kv_heads, S, d_k) -> (B, n_heads, S, d_k)
        if self.n_heads_per_kv > 1:
            k = k.repeat_interleave(self.n_heads_per_kv, dim=1)
            v = v.repeat_interleave(self.n_heads_per_kv, dim=1)

        # Scaled Dot-Product Attention
        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        self._attn_weights = attn_weights

        # Concat heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        return self.W_o(attn_out)


if __name__ == "__main__":
    # 简单验证
    B, S, D = 2, 16, 64
    x = torch.randn(B, S, D)

    print("=== Multi-Head Attention ===")
    mha = MultiHeadAttention(d_model=D, n_heads=4)
    out = mha(x)
    print(f"Input:  {x.shape}")    # (2, 16, 64)
    print(f"Output: {out.shape}")   # (2, 16, 64)

    print("\n=== Grouped Query Attention ===")
    gqa = GroupedQueryAttention(d_model=D, n_heads=4, n_kv_heads=2)
    out = gqa(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    # Causal mask 测试
    print("\n=== Causal Mask ===")
    causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0)  # (1, S, S)
    out = mha(x, mask=causal_mask)
    print(f"Causal output: {out.shape}")
