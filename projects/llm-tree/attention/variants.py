"""
variants.py — 注意力线上的节点

    MHA ──→ MQA ──→ GQA ──→ MLA
    (每个头一组 K、V)  (所有头共用一组)  (分组共用)  (把 K、V 压缩成一个低维潜向量再缓存)

这条线上每一步要解决的都是同一个问题: 推理时的 KV Cache 太大
(llm-minigpt 讲义专题 A 已经实测过: GPT-2 Small 一条 1024 长度的序列要缓存 72MB)。

两个类都满足 base.py 约定的接口: forward(x, cos, sin)、rope_dim、kv_numbers_per_token。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import RMSNorm, apply_rope


class GroupedQueryAttention(nn.Module):
    """n_kv_heads == n_heads 时为 MHA, == 1 时为 MQA, 介于两者之间为 GQA。

    n_heads 个查询头被分成 n_kv_heads 组, 同一组里的查询头共用一组 K、V。
    """

    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        assert dim % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)   # K、V 只有 n_kv_heads 组
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.rope_dim = self.head_dim
        self.kv_numbers_per_token = 2 * n_kv_heads * self.head_dim          # 每个 token 缓存 K 和 V

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)      # (B, H,   S, d)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, Hkv, S, d)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # 把每组 K、V 复制给组内的每个查询头: (B, Hkv, S, d) -> (B, H, S, d)
        # 复制只发生在计算时; 推理时缓存的仍然只是 Hkv 组, 这正是省显存的地方
        group = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(B, S, -1))


class MultiHeadLatentAttention(nn.Module):
    """MLA (DeepSeek-V2 提出, V3、Kimi-K2 沿用)。

    K、V 不再按头单独投影, 而是先把 x 压缩成一个 kv_rank 维的潜向量 c, 推理时只缓存 c;
    计算注意力时再由 c "解压" 出每个头的 K、V。

    难点在 RoPE: 位置旋转不能和"解压"矩阵交换次序, 所以每个头的 Q、K 被拆成两段:
      - nope 段 (d_nope 维): 不旋转, K 由潜向量 c 解压得到;
      - rope 段 (d_rope 维): 旋转, K 直接由 x 投影得到, 并且**所有头共用一份** (所以缓存很小)。
    推理时每个 token 每层缓存 kv_rank + d_rope 个数。

    字段名对应真实模型 (见 llm-anatomy 的张量名):
      w_dkv  ↔ self_attn.kv_a_proj_with_mqa   x → [潜向量 c, 共享 RoPE 键]
      kv_norm↔ self_attn.kv_a_layernorm       对潜向量做 RMSNorm
      w_ukv  ↔ self_attn.kv_b_proj            c → 每个头的 [K_nope, V]
    (DeepSeek 对 Q 也做了低秩压缩 q_a_proj / q_b_proj, 那是为了省训练时的激活, 与 KV Cache 无关, 这里省略。)
    """

    def __init__(self, dim, n_heads, kv_rank, d_nope, d_rope, d_v):
        super().__init__()
        self.n_heads, self.kv_rank = n_heads, kv_rank
        self.d_nope, self.d_rope, self.d_v = d_nope, d_rope, d_v
        self.wq = nn.Linear(dim, n_heads * (d_nope + d_rope), bias=False)
        self.w_dkv = nn.Linear(dim, kv_rank + d_rope, bias=False)
        self.kv_norm = RMSNorm(kv_rank)
        self.w_ukv = nn.Linear(kv_rank, n_heads * (d_nope + d_v), bias=False)
        self.wo = nn.Linear(n_heads * d_v, dim, bias=False)
        self.rope_dim = d_rope
        self.kv_numbers_per_token = kv_rank + d_rope                         # 只缓存 c 和共享的 RoPE 键

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        H = self.n_heads
        q = self.wq(x).view(B, S, H, self.d_nope + self.d_rope).transpose(1, 2)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)

        c, k_rope = self.w_dkv(x).split([self.kv_rank, self.d_rope], dim=-1)   # c: (B, S, r) 推理时缓存这两个
        k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin)                     # (B, 1, S, d_rope) 所有头共用
        kv = self.w_ukv(self.kv_norm(c)).view(B, S, H, self.d_nope + self.d_v).transpose(1, 2)
        k_nope, v = kv.split([self.d_nope, self.d_v], dim=-1)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope.expand(B, H, S, self.d_rope)], dim=-1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # 缩放默认为 1/√(d_nope+d_rope)
        return self.wo(out.transpose(1, 2).reshape(B, S, -1))
