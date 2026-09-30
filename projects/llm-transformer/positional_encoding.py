"""
Positional Encoding — 从绝对位置到旋转位置编码的演进

架构演进:
  1. Sinusoidal PE (2017, 原始Transformer) — 固定正弦余弦
  2. Learned PE (2018, GPT-1/2) — 可学习的绝对位置编码
  3. ALiBi (2022, BLOOM) — 线性偏置, 无需训练即可外推
  4. RoPE (2021, Qwen) — 旋转位置编码, 现代LLM标配

为什么位置编码重要?
  Self-Attention 本身是置换不变的(permutation invariant),
  如果不加位置信息, 模型无法区分 "猫吃鱼" 和 "鱼吃猫"。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Sinusoidal Positional Encoding (2017)
# ============================================================
# 原始Transformer用的固定编码, 用不同频率的sin/cos生成:
#
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# 优点: 无需训练, 可外推到训练时未见过的长度 (理论上)
# 缺点: 实际外推能力有限, 且不编码相对位置关系

class SinusoidalPE(nn.Module):
    """Sinusoidal Positional Encoding (2017)"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # 预计算 PE 矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model) — 注册为 buffer, 不参与梯度更新
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d_model) -> (B, S, d_model)"""
        S = x.size(1)
        return x + self.pe[:, :S, :]


# ============================================================
# 2. Learned Positional Encoding (GPT-1/2)
# ============================================================
# 直接学习一个 (max_len, d_model) 的 embedding 矩阵
#
# GPT-1/2 用的就是这个
# 优点: 简单, 表达力强
# 缺点: 无法外推到训练时未见过的长度 (超过 max_len 就崩了)

class LearnedPE(nn.Module):
    """Learned Positional Embedding (GPT-1/2)"""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d_model) -> (B, S, d_model)"""
        S = x.size(1)
        positions = torch.arange(S, device=x.device)
        return x + self.pos_embedding(positions)


# ============================================================
# 3. ALiBi — Attention with Linear Biases (2022)
# ============================================================
# 论文: "Train Short, Test Long: Attention with Linear Biases
#        Enables Input Length Extrapolation" (Press et al., 2022)
#
# 核心思想: 不往输入里加位置编码, 而是在 attention score 上加偏置
#
#   attn_score[i][j] = q_i * k_j - m * |i - j|
#
#   其中 m 是每个头的固定斜率 (不同头用不同斜率)
#
# 优点:
#   - 无需额外参数 (偏置是固定计算的)
#   - 外推能力极强 (训练1024, 推理2048 效果不降)
#   - 实现简单
#
# 缺点:
#   - 线性偏置假设位置关系是线性的, 表达力有限
#   - 2023年后逐渐被 RoPE 取代

class ALiBi(nn.Module):
    """ALiBi — Attention with Linear Biases (2022)"""

    def __init__(self, n_heads: int, max_len: int = 2048):
        super().__init__()
        # 每个头的斜率: 2^(-8/n_heads), 2^(-16/n_heads), ...
        slopes = torch.tensor(
            [2 ** (-8 * i / n_heads) for i in range(1, n_heads + 1)]
        )
        self.register_buffer("slopes", slopes)

        # 预计算相对位置偏置矩阵
        # bias[i][j] = slopes * |i - j|
        positions = torch.arange(max_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # (S, S)
        alibi = relative_positions.abs().float()  # (S, S)
        # (1, n_heads, S, S) — 广播到所有batch
        alibi = -alibi.unsqueeze(0) * slopes.view(1, -1, 1, 1)
        self.register_buffer("alibi", alibi)

    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        在 attention score 上加上 ALiBi 偏置

        Args:
            attention_scores: (B, n_heads, S, S)
        Returns:
            attention_scores + alibi_bias
        """
        S = attention_scores.size(-1)
        return attention_scores + self.alibi[:, :, :S, :S]


# ============================================================
# 4. RoPE — Rotary Position Embedding (2021)
# ============================================================
# 论文: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
#        (Su et al., 2021, Qwen)
#
# 核心思想: 对 Q 和 K 做旋转变换, 让内积自然编码相对位置
#
#   q·k 的结果 = f(q, k, i-j)  — 只依赖相对位置差 (i-j)
#
# 实现: 将 (q_0, q_1) 视为复数, 乘以 e^(i*theta*pos)
#
# 优点:
#   - 天然编码相对位置 (注意力权重只依赖相对距离)
#   - 外推能力: 配合 NTK-aware scaling / YaRN 可以大幅外推
#   - 无额外参数
#
# 当前 LLM 标配: LLaMA, Qwen, Mistral, Gemma 等都用 RoPE

class RoPE(nn.Module):
    """Rotary Position Embedding (2021, Qwen)"""

    def __init__(self, d_k: int, max_len: int = 4096, base: float = 10000.0):
        """
        Args:
            d_k: 每个头的维度 (必须是偶数, 因为两两配对旋转)
            max_len: 最大序列长度
            base: 频率基数 (LLaMA 用 10000, 部分模型用 500000)
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        # 计算频率: theta_i = 1 / (base^(2i/d_k)), i = 0, 1, ..., d_k/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算 cos/sin
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (max_len, d_k/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, d_k)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))  # (1, 1, L, d_k)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,       # (B, n_heads, S, d_k)  — 通常是 Q 或 K
        offset: int = 0,        # KV-Cache 时的位置偏移
    ) -> torch.Tensor:
        """对 Q 或 K 施加旋转位置编码"""
        S = x.size(2)
        cos = self.cos_cached[:, :, offset:offset + S, :]  # (1, 1, S, d_k)
        sin = self.sin_cached[:, :, offset:offset + S, :]
        return self._apply_rotary(x, cos, sin)

    @staticmethod
    def _apply_rotary(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        旋转编码的核心操作

        对于 x 的每两个维度 (x_0, x_1):
            rotated = x_0 * cos - x_1 * sin
                     x_1 * cos + x_0 * sin

        等价于: x * cos + rotate_half(x) * sin
        """
        d2 = x.size(-1) // 2
        x1, x2 = x[..., :d2], x[..., d2:]
        # rotate_half: (a, b) -> (-b, a)
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, D = 2, 32, 64
    x = torch.randn(B, S, D)

    print("=" * 55)
    print("1. Sinusoidal PE (2017, 原始Transformer)")
    pe = SinusoidalPE(d_model=D, max_len=512)
    out = pe(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Params: 0 (fixed)")

    print("\n" + "=" * 55)
    print("2. Learned PE (GPT-1/2)")
    pe = LearnedPE(d_model=D, max_len=512)
    out = pe(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Params: {sum(p.numel() for p in pe.parameters()):,}")

    print("\n" + "=" * 55)
    print("3. ALiBi (2022, 无需参数即可外推)")
    alibi = ALiBi(n_heads=4, max_len=512)
    dummy_scores = torch.randn(B, 4, S, S)
    out = alibi(dummy_scores)
    print(f"   Input scores:  {dummy_scores.shape}")
    print(f"   Output scores: {out.shape}")
    print(f"   Params: 0 (fixed biases)")
    print(f"   训练长度: 128, 推理长度: 256 → 效果几乎不降!")

    print("\n" + "=" * 55)
    print("4. RoPE (2021, Qwen — 现代LLM标配)")
    rope = RoPE(d_k=D // 4, max_len=1024)  # d_k = 16
    q = torch.randn(B, 4, S, D // 4)
    out = rope(q)
    print(f"   Input Q:  {q.shape}")
    print(f"   Output:   {out.shape}")
    print(f"   Params: 0 (fixed rotations)")

    # 外推测试
    long_q = torch.randn(B, 4, 256, D // 4)  # 训练512, 推理256
    out_long = rope(long_q, offset=0)
    print(f"   外推: 训练max_len=1024, 推理seq_len=256 → OK!")

    print("\n✅ All positional encodings work correctly!")
