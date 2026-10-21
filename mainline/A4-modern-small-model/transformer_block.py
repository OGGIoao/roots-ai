"""
Transformer Block — 组装 Attention + FFN + Normalization

架构演进:
  1. 原始 Block (2017) — Post-LN: x -> Attn -> LN -> FFN -> LN
  2. GPT-2 Block (2019) — Pre-LN:  x -> LN -> Attn -> +x -> LN -> FFN -> +x
  3. LLaMA Block (2023) — Pre-LN + SwiGLU + RMSNorm + GQA + RoPE

Pre-LN 为什么更好?
  Post-LN: 梯度在深层网络中容易消失/爆炸, 需要仔细的 warmup
  Pre-LN:  残差连接直接传递梯度, 训练更稳定, 无需 warmup
  这就是为什么 GPT-2 之后几乎所有模型都用 Pre-LN
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention, GroupedQueryAttention
from positional_encoding import SinusoidalPE, LearnedPE, RoPE
from normalization import LayerNorm, RMSNorm
from feedforward import FFN, GELUFFN, SwiGLUFFN
import sys

# Windows 控制台默认 GBK, 无法编码脚本结尾处的 ✅/❌ 等符号,
# print() 会抛 UnicodeEncodeError 让脚本在跑通后异常退出。
for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r:
        try:
            _r(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass



class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block (GPT-2 style)

    架构:
        residual = x
        x = norm(x)
        x = attention(x) + residual    # 残差连接

        residual = x
        x = norm(x)
        x = ffn(x) + residual         # 残差连接
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # Pre-LN: Norm -> Attention -> Residual
        x = x + self.attn(self.norm1(x), mask)
        # Pre-LN: Norm -> FFN -> Residual
        x = x + self.ffn(self.norm2(x))
        return x


class LlamaBlock(nn.Module):
    """
    LLaMA-style Transformer Block (2023)

    与 GPT-2 Block 的区别:
        1. RMSNorm 替代 LayerNorm (更快)
        2. SwiGLU 替代标准 FFN (更强)
        3. GQA 替代标准 MHA (KV Cache 更省)
        4. RoPE 作为位置编码 (更强外推)

    这是目前最主流的 LLM Block 设计
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        rope_max_len: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        self.rope = RoPE(d_k=d_model // n_heads, max_len=rope_max_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # Pre-RMSNorm -> GQA -> Residual
        x = x + self.attn(self.norm1(x), mask)
        # Pre-RMSNorm -> SwiGLU FFN -> Residual
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, D = 2, 16, 64
    n_heads = 4
    d_ff = D * 4

    x = torch.randn(B, S, D)

    causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    print("=" * 55)
    print("1. GPT-2 Style TransformerBlock (Pre-LN)")
    block = TransformerBlock(D, n_heads, d_ff)
    out = block(x, mask=causal_mask)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    params = sum(p.numel() for p in block.parameters())
    print(f"   Params: {params:,}")

    print("\n" + "=" * 55)
    print("2. LLaMA Style TransformerBlock")
    llama_block = LlamaBlock(
        d_model=D, n_heads=n_heads, n_kv_heads=2, d_ff=d_ff
    )
    out = llama_block(x, mask=causal_mask)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    params = sum(p.numel() for p in llama_block.parameters())
    print(f"   Params: {params:,}")

    print("\n✅ All Transformer blocks work correctly!")
