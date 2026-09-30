"""
config.py — miniGPT 模型配置

参考 Karpathy nanoGPT 的配置风格。
每个配置都是一个 dataclass, 方便实验不同规模的模型。

运行: python config.py  (查看所有配置的参数量)
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT 模型配置"""
    vocab_size: int = 50257     # 词表大小 (GPT-2 标准)
    block_size: int = 1024      # 最大序列长度 (context window)
    n_layer: int = 12           # Transformer 层数
    n_head: int = 12            # 注意力头数
    n_embd: int = 768           # 嵌入维度
    dropout: float = 0.0        # Dropout 概率
    bias: bool = True           # 是否使用 bias

    @property
    def d_k(self):
        return self.n_embd // self.n_head


# ── 预定义配置 ──────────────────────────────────────────────

# GPT-2 Small (124M) — 我们的目标
GPT2_SMALL = GPTConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
)

# Nano 级别 — 用于快速调试
NANO = GPTConfig(
    vocab_size=256,       # 字节级, 只有 256 个 token
    block_size=64,        # 短序列, 跑得快
    n_layer=2,            # 2 层
    n_head=2,             # 2 个头
    n_embd=64,            # 小维度
    dropout=0.1,
)

# Micro — 介于 nano 和 small 之间
MICRO = GPTConfig(
    vocab_size=50257,
    block_size=256,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.1,
)

# Medium (350M)
GPT2_MEDIUM = GPTConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=24,
    n_head=16,
    n_embd=1024,
    dropout=0.0,
)

# Large (774M)
GPT2_LARGE = GPTConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=36,
    n_head=20,
    n_embd=1280,
    dropout=0.0,
)


def count_parameters(config: GPTConfig) -> int:
    """估算模型参数量 (不含 Embedding 层)"""
    # 每层 Transformer Block:
    #   Attention: 4 * n_embd^2 (Q, K, V, O 投影)
    #   FFN: 8 * n_embd^2 (4* up, 4* down, 无 bias 时)
    #   LayerNorm: 4 * n_embd
    block_params = 4 * config.n_embd ** 2 + 8 * config.n_embd ** 2 + 4 * config.n_embd
    total = config.n_layer * block_params
    # Embedding 层: token embedding + position embedding
    total += config.vocab_size * config.n_embd + config.block_size * config.n_embd
    # Final layer norm
    total += 2 * config.n_embd
    return total


def format_params(n: int) -> str:
    """格式化参数量"""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


# ── 模型列表 ──────────────────────────────────────────────

MODELS = {
    "nano": NANO,
    "micro": MICRO,
    "gpt2-small": GPT2_SMALL,
    "gpt2-medium": GPT2_MEDIUM,
    "gpt2-large": GPT2_LARGE,
}


if __name__ == "__main__":
    print("=" * 60)
    print("miniGPT 模型配置")
    print("=" * 60)
    print(f"{'配置名':<15} {'层数':>4} {'头数':>4} {'维度':>6} {'参数量':>10}")
    print("-" * 60)
    for name, cfg in MODELS.items():
        params = count_parameters(cfg)
        print(f"{name:<15} {cfg.n_layer:>4} {cfg.n_head:>4} {cfg.n_embd:>6} {format_params(params):>10}")
    print("=" * 60)
    print(f"\nGPT-2 Small (目标) 参数量: {format_params(count_parameters(GPT2_SMALL))}")
    print(f"Nano (调试用) 参数量: {format_params(count_parameters(NANO))}")
