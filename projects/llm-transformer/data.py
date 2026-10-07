"""
Data — Tiny Shakespeare 数据集加载与 Tokenization

用于训练小型 GPT 模型的经典数据集。语料定位 / 字符级分词 / 数据集都来自
仓库根的 `common/` (与 llm-minigpt 共用), 本文件只保留本项目的组装逻辑。
BPE tokenizer 见 llm-bpe-tokenizer 项目。
"""

import os
import sys

# Windows 控制台默认 GBK, 无法编码本脚本结尾的 ✅, print() 会抛
# UnicodeEncodeError 让脚本在跑通后异常退出。切到 UTF-8 并降级替换。
for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r:
        try:
            _r(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

# 让 `import common` 在任意工作目录下可用 (append: 不会遮蔽项目内的同名模块)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from common.corpus import load_shakespeare  # noqa: E402
from common.text_data import CharTokenizer, TextDataset, make_loaders  # noqa: E402,F401


def load_tiny_shakespeare():
    """
    加载 Tiny Shakespeare 数据集

    获取顺序: 仓库内入库的语料 -> 在线下载 (见 common/corpus.py)

    Returns:
        text: 原始文本字符串
        tokenizer: CharTokenizer 实例
    """
    text = load_shakespeare()
    tokenizer = CharTokenizer(text)
    print(f"Dataset: {len(text):,} characters, {tokenizer.vocab_size} unique chars")
    return text, tokenizer


def prepare_data(
    text: str,
    tokenizer: CharTokenizer,
    block_size: int = 128,
    batch_size: int = 32,
    train_ratio: float = 0.9,
    device: str = "cpu",
):
    """
    准备训练/验证 DataLoader (滑动窗口: 每个起点一个样本)

    Args:
        text: 原始文本
        tokenizer: tokenizer 实例
        block_size: 上下文长度 (序列长度)
        batch_size: batch 大小
        train_ratio: 训练集比例
        device: 保留参数 (张量留在 CPU, 由训练循环负责搬运)

    Returns:
        train_loader, val_loader, vocab_size
    """
    train_loader, val_loader = make_loaders(
        tokenizer.encode(text),
        block_size=block_size,
        batch_size=batch_size,
        train_ratio=train_ratio,
        stride=1,
    )
    print(f"Train: {len(train_loader.dataset):,} samples, "
          f"Val: {len(val_loader.dataset):,} samples")
    return train_loader, val_loader, tokenizer.vocab_size


if __name__ == "__main__":
    text, tokenizer = load_tiny_shakespeare()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 测试 encode/decode
    sample = text[:100]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"Encode: '{sample[:50]}...' -> {encoded[:10]}...")
    print(f"Decode: '{decoded[:50]}...'")
    assert decoded == sample, "Encode/Decode mismatch!"

    # 测试 DataLoader
    train_loader, val_loader, vocab_size = prepare_data(
        text, tokenizer, block_size=128, batch_size=8
    )
    x, y = next(iter(train_loader))
    print(f"\nBatch shapes: x={x.shape}, y={y.shape}")
    print(f"Sample x: {tokenizer.decode(x[0].tolist())[:80]}...")
    print(f"Sample y: {tokenizer.decode(y[0].tolist())[:80]}...")

    print("\n✅ Data module works correctly!")
