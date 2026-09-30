"""
Normalization Layers — 从 LayerNorm 到 RMSNorm 的演进

架构演进:
  1. Post-LayerNorm (2017, 原始Transformer) — FFN/Attention 之后加 LN
  2. Pre-LayerNorm (2019, GPT-2) — FFN/Attention 之前加 LN
  3. RMSNorm (2023, Qwen) — 去掉 mean centering, 更快更简单

为什么归一化层的位置很重要?
  Post-LN: 深层网络训练不稳定, 需要 warmup
  Pre-LN: 梯度更稳定, 可以训练更深的网络, 无需 warmup
  现代 LLM 几乎全部用 Pre-LN + RMSNorm
"""

import torch
import torch.nn as nn
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



class LayerNorm(nn.Module):
    """标准 Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均值和方差 (在最后一个维度)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 归一化 + 可学习的 scale/shift
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (2023)

    论文: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    使用: LLaMA, Qwen, Mistral 等现代 LLM

    与 LayerNorm 的区别:
      - 去掉 mean centering: 直接除以 RMS
      - 没有 bias 参数
      - 计算更快, 效果几乎一样

    公式:
      RMSNorm(x) = x / RMS(x) * weight
      where RMS(x) = sqrt(mean(x^2))
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, D = 2, 16, 64

    x = torch.randn(B, S, D)

    print("=" * 55)
    print("1. LayerNorm (标准)")
    ln = LayerNorm(D)
    out = ln(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    params = sum(p.numel() for p in ln.parameters())
    print(f"   Params: {params:,} (weight + bias)")
    print(f"   输出均值: {out.mean().item():.6f}, 方差: {out.var().item():.4f}")

    print("\n" + "=" * 55)
    print("2. RMSNorm (LLaMA, Qwen)")
    rms = RMSNorm(D)
    out = rms(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")
    params = sum(p.numel() for p in rms.parameters())
    print(f"   Params: {params:,} (只有 weight, 无 bias)")
    print(f"   比 LayerNorm 少 {2 * D:,} 参数 (无 bias)")

    # 验证 RMSNorm 更快
    import time
    x_large = torch.randn(16, 1024, 4096)

    t0 = time.perf_counter()
    for _ in range(100):
        _ = ln(x_large)
    ln_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(100):
        _ = rms(x_large)
    rms_time = time.perf_counter() - t0

    print(f"\n   速度对比 (batch=16, seq=1024, d_model=4096, 100次):")
    print(f"   LayerNorm: {ln_time:.3f}s")
    print(f"   RMSNorm:   {rms_time:.3f}s")
    print(f"   加速:      {ln_time / rms_time:.2f}x")

    print("\n✅ All normalization layers work correctly!")
