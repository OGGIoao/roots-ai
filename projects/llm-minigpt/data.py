"""
data.py — 数据加载与预处理

支持两种数据来源:
  1. Tiny Shakespeare (默认): 经典文本数据集, 用于快速训练验证
  2. 自定义文本文件: 读取任意 .txt 文件

核心功能:
  - 文本 -> token IDs (使用字符级分词, 简单直接)
  - 训练/验证集划分
  - Batching: 随机采样 + 块分割

运行: python data.py  (加载数据并打印统计信息)
"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 数据下载
# ============================================================

def download_tiny_shakespeare(data_dir: str = "data") -> str:
    """下载 Tiny Shakespeare 数据集"""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "tiny_shakespeare.txt")

    if not os.path.exists(filepath):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"正在下载 Tiny Shakespeare...")
        urllib.request.urlretrieve(url, filepath)
        print(f"下载完成: {filepath}")
    else:
        print(f"数据已存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"文本长度: {len(text):,} 字符")
    print(f"唯一字符数: {len(set(text))}")
    print(f"前 200 字符:\n{text[:200]}")
    return text


# ============================================================
# 2. 字符级 Tokenizer (简单但有效)
# ============================================================

class CharTokenizer:
    """
    字符级分词器
    
    为什么用字符级而不是 BPE?
      - Tiny Shakespeare 词表很小 (~65 字符), 字符级足够
      - 简单直接, 不需要预训练分词器
      - 训练速度更快
    
    缺点:
      - 英文单词会被拆成单个字符 (如 "hello" -> h,e,l,l,o)
      - 压缩率低 (每个字符 1 个 token)
      - 但对于小数据集训练验证来说足够
    """
    
    def __init__(self, text: str):
        # 构建词表: 字符 -> ID
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"字符级词表大小: {self.vocab_size}")
        print(f"字符集: {''.join(chars[:20])}{'...' if len(chars) > 20 else ''}")
    
    def encode(self, text: str) -> list:
        """文本 -> token IDs"""
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]
    
    def decode(self, ids: list) -> str:
        """token IDs -> 文本"""
        return "".join(self.id_to_char.get(i, "?") for i in ids)


# ============================================================
# 3. Dataset
# ============================================================

class TextDataset(Dataset):
    """
    文本数据集
    
    把长文本切成固定长度的块 (block), 每个块独立作为训练样本
    
    Example:
      text = "Hello world, how are you?"
      block_size = 8
      
      Block 0: "Hello wo" -> input: [H,e,l,l,o, ,w,o], target: [e,l,l,o, ,w,o,r]
      Block 1: "rld, how" -> input: [r,l,d,,,...], target: [l,d,,,...]
      ...
    """
    
    def __init__(self, token_ids: list, block_size: int):
        self.token_ids = token_ids
        self.block_size = block_size
        self.n_blocks = len(token_ids) // block_size
        
        print(f"数据集: {len(token_ids):,} tokens, {self.n_blocks} blocks")
    
    def __len__(self):
        return self.n_blocks
    
    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1  # +1 for target
        
        chunk = self.token_ids[start:end]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # input
        y = torch.tensor(chunk[1:], dtype=torch.long)    # target
        return x, y


# ============================================================
# 4. 数据加载器
# ============================================================

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
    """
    # 分词
    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)
    
    # 划分训练/验证集
    split = int(len(token_ids) * (1 - val_ratio))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]
    
    print(f"\n训练集: {len(train_ids):,} tokens ({len(train_ids)//block_size} blocks)")
    print(f"验证集: {len(val_ids):,} tokens ({len(val_ids)//block_size} blocks)")
    
    # 创建 Dataset
    train_dataset = TextDataset(train_ids, block_size)
    val_dataset = TextDataset(val_ids, block_size)
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    return train_loader, val_loader, tokenizer


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("miniGPT 数据加载测试")
    print("=" * 60)
    
    # 下载数据
    text = download_tiny_shakespeare()
    
    # 创建 DataLoader
    train_loader, val_loader, tokenizer = create_dataloaders(
        text, block_size=64, batch_size=4
    )
    
    # 取一个 batch 看看
    x, y = next(iter(train_loader))
    print(f"\nBatch shape: x={x.shape}, y={y.shape}")
    print(f"示例序列: {tokenizer.decode(x[0].tolist()[:20])}")
    print(f"目标序列: {tokenizer.decode(y[0].tolist()[:20])}")
    
    print("\n✅ 数据加载测试通过!")
