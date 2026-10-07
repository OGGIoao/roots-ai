"""
data.py — miniGPT 数据加载与预处理

语料定位 / 字符级分词 / 数据集都来自仓库根的 `common/` (与 llm-transformer 共用,
不再各存一份)。本文件只保留 miniGPT 专属的组装逻辑。

核心功能:
  - 文本 -> token IDs (字符级分词, 简单直接)
  - 训练/验证集划分
  - Batching: 互不重叠的块 (stride = block_size)

运行: python data.py  (加载数据并打印统计信息)
"""

import os
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

# 让 `import common` 在任意工作目录下可用 (append: 不会遮蔽项目内的同名模块)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from common.corpus import load_shakespeare  # noqa: E402
from common.text_data import CharTokenizer, TextDataset, make_loaders  # noqa: E402,F401


def download_tiny_shakespeare() -> str:
    """加载 Tiny Shakespeare 语料 (本地入库文件优先, 缺失才下载)"""
    text = load_shakespeare()
    print(f"文本长度: {len(text):,} 字符")
    print(f"唯一字符数: {len(set(text))}")
    print(f"前 200 字符:\n{text[:200]}")
    return text


def create_dataloaders(
    text: str,
    block_size: int = 128,
    batch_size: int = 32,
    val_ratio: float = 0.1,
):
    """
    创建训练和验证 DataLoader

    Args:
        text: 原始文本
        block_size: 序列长度
        batch_size: 批大小
        val_ratio: 验证集比例

    Returns:
        (train_loader, val_loader, tokenizer)
    """
    tokenizer = CharTokenizer(text)
    print(f"字符级词表大小: {tokenizer.vocab_size}")

    train_loader, val_loader = make_loaders(
        tokenizer.encode(text),
        block_size=block_size,
        batch_size=batch_size,
        train_ratio=1 - val_ratio,
        stride=block_size,  # 互不重叠的块: 一个 epoch 恰好扫一遍语料
    )
    print(f"训练集: {len(train_loader.dataset)} blocks, "
          f"验证集: {len(val_loader.dataset)} blocks")
    return train_loader, val_loader, tokenizer


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("miniGPT 数据加载测试")
    print("=" * 60)

    text = download_tiny_shakespeare()
    train_loader, val_loader, tokenizer = create_dataloaders(
        text, block_size=64, batch_size=4
    )

    x, y = next(iter(train_loader))
    print(f"\nBatch shape: x={x.shape}, y={y.shape}")
    print(f"示例序列: {tokenizer.decode(x[0].tolist()[:20])}")
    print(f"目标序列: {tokenizer.decode(y[0].tolist()[:20])}")

    print("\n✅ 数据加载测试通过!")
