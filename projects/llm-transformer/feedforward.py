"""
Feed-Forward Network — 从标准 FFN 到 SwiGLU 的演进

架构演进:
  1. 标准 FFN (2017, 原始Transformer) — ReLU 激活的两层 MLP
  2. GELU FFN (2018, GPT-2) — 用 GELU 替代 ReLU, 更平滑
  3. SwiGLU FFN (2022, PaLM → LLaMA) — Swish + 门控线性单元, 现代标配

为什么 FFN 重要?
  Transformer Block = Attention + FFN
  Attention 负责 "token 间信息交换"
  FFN 负责 "每个 token 的非线性变换" (知识存储的主要载体)
  FFN 参数量占全模型 ~2/3, 是知识的主要存储位置

SwiGLU 为什么好?
  标准 FFN:  FFN(x) = W2 * ReLU(W1 * x)
  SwiGLU:    FFN(x) = W2 * (Swish(W_gate * x) ⊙ (W_up * x))

  门控机制让网络可以选择性地放大/抑制信息,
  表达力比标准 FFN 强, 训练效率更高。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FFN(nn.Module):
    """
    标准 Feed-Forward Network (2017, 原始 Transformer)

    FFN(x) = W2 * ReLU(W1 * x + b1) + b2

    维度: d_model -> d_ff -> d_model
    通常 d_ff = 4 * d_model
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.relu(self.w1(x))))


class GELUFFN(nn.Module):
    """
    GELU FFN (GPT-2, BERT)

    用 GELU 替代 ReLU: GELU(x) = x * Φ(x)
    更平滑的激活函数, 梯度更稳定

    FFN(x) = W2 * GELU(W1 * x)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (PaLM, LLaMA, Qwen)

    论文: "GLU Variants Improve Transformer" (Shazeer, 2020)
    使用: LLaMA, PaLM, Mistral, Gemma

    核心公式:
        FFN_SwiGLU(x) = W2 * (Swish(W_gate * x) ⊙ W_up * x)

    其中:
        Swish(x) = x * sigmoid(x)
        ⊙ = 逐元素相乘 (Hadamard product)

    三个权重矩阵:
        W_gate: 门控分支 (控制信息流)
        W_up:  上投影分支 (信息通道)
        W_down: 下投影 (回归 d_model)

    维度:
        W_gate: d_model -> d_ff
        W_up:   d_model -> d_ff
        W_down: d_ff -> d_model
        (参数量比标准 FFN 多 50%, 但 d_ff 通常设为 8/3 * d_model, 总参数量相近)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(W_gate * x) ⊙ (W_up * x)
        gate = F.silu(self.w_gate(x))  # silu = swish
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, D = 2, 16, 64
    d_ff = D * 4  # 标准配置

    x = torch.randn(B, S, D)

    print("=" * 55)
    print("1. Standard FFN (2017, ReLU)")
    ffn = FFN(D, d_ff)
    out = ffn(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    params = sum(p.numel() for p in ffn.parameters())
    print(f"   Params: {params:,}")

    print("\n" + "=" * 55)
    print("2. GELU FFN (GPT-2)")
    gelu_ffn = GELUFFN(D, d_ff)
    out = gelu_ffn(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    params = sum(p.numel() for p in gelu_ffn.parameters())
    print(f"   Params: {params:,}")

    print("\n" + "=" * 55)
    print("3. SwiGLU FFN (LLaMA, modern)")
    swiglu = SwiGLUFFN(D, d_ff)
    out = swiglu(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    params = sum(p.numel() for p in swiglu.parameters())
    print(f"   Params: {params:,}")
    print(f"   注意: SwiGLU 有3个矩阵(多50%), 但 d_ff 通常更小")

    print("\n✅ All FFN variants work correctly!")
