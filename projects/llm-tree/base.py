"""
base.py — 树的"LLaMA 配方"节点: 一个组件可插拔的 decoder-only 模型

结构与 llm-transformer 的 LlamaModel 相同 (Pre-RMSNorm + RoPE + SwiGLU + 无 bias + 权重绑定),
区别是**注意力模块从外部传入**。树上的每个注意力节点 (MHA、GQA、MQA、MLA……) 都是一个
满足下面接口的模块, 替换进来后其余部分完全不变, 对比实验的差异就只来自注意力本身。

注意力模块的接口:
    attn(x, cos, sin) -> (B, S, C)        x: (B, S, C); cos/sin: RoPE 表, 形状 (S, d_rope/2)
    attn.rope_dim                          这个模块需要多宽的 RoPE (每个头里参与旋转的维度)
    attn.kv_numbers_per_token              推理时每个 token 在这一层要缓存多少个数
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rope_table(dim, max_len, theta=10000.0, device=None):
    """返回 cos、sin 表, 形状 (max_len, dim/2)。第 i 对维度以频率 theta^(-2i/dim) 旋转。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    angles = torch.outer(torch.arange(max_len, device=device).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, S, dim)。把最后一维看成 dim/2 对 (x1, x2), 每对按位置旋转一个角度。"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim, attn, ffn_hidden):
        super().__init__()
        self.norm1, self.attn = RMSNorm(dim), attn
        self.norm2, self.ffn = RMSNorm(dim), SwiGLU(dim, ffn_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))


class Decoder(nn.Module):
    """make_attn: 无参函数, 每调用一次返回一个新的注意力模块 (每层一个)。"""

    def __init__(self, vocab_size, block_size, dim, n_layers, make_attn, ffn_hidden=None, rope_theta=10000.0):
        super().__init__()
        ffn_hidden = ffn_hidden or 256 * math.ceil(8 * dim / 3 / 256)   # SwiGLU 惯例: 约 8/3 倍, 对齐到 256
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(Block(dim, make_attn(), ffn_hidden) for _ in range(n_layers))
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        cos, sin = rope_table(self.blocks[0].attn.rope_dim, block_size, rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        S = idx.size(1)
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, self.cos[:S], self.sin[:S])
        logits = self.lm_head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def kv_cache_bytes_per_token(self, dtype_bytes=2):
        return sum(b.attn.kv_numbers_per_token for b in self.blocks) * dtype_bytes
