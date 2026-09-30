"""
Data — Tiny Shakespeare 数据集加载与 Tokenization

用于训练小型 GPT 模型的经典数据集。
提供字符级 tokenization (最简单, 不需要外部依赖) 和 BPE tokenizer 接口。
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

# Windows 控制台默认 GBK, 无法编码本脚本结尾的 ✅, print() 会抛
# UnicodeEncodeError 让脚本在跑通后异常退出。切到 UTF-8 并降级替换。
for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r:
        try:
            _r(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


class CharTokenizer:
    """
    字符级 Tokenizer — 最简单的 tokenization 方式

    优点: 无需外部依赖, 适合教学和小规模实验
    缺点: 词表小(通常~50), 无法处理未知字符

    实际 LLM 用 BPE/SentencePiece (见 llm-bpe-tokenizer 项目)
    """

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.idx_to_char[i] for i in ids])


class TextDataset(Dataset):
    """文本数据集 — 将文本切分为固定长度的 chunk"""

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# 本地已有的同一份语料 (llm-bpe-tokenizer 项目里那份), 避免重复下载。
# 本机代理 (127.0.0.1:7890) 没开时下载会失败, 有本地副本就能离线跑。
# 注意: 必须基于 __file__ 解析, 不能用相对 CWD 的 ".." —— 否则从仓库根目录
# 运行 (python projects/llm-transformer/data.py) 时会找不到, 白白触发一次下载。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_FALLBACKS = [
    os.path.join(_REPO_ROOT, "projects", "llm-bpe-tokenizer", "data", "tinyshakespeare.txt"),
]

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def _ensure_corpus(filepath: str) -> str:
    """
    确保语料存在: 本地副本 -> 下载 -> 明确报错

    顺序很重要: 先找本地, 再下载。这样断网/代理没开也能训练。
    """
    if os.path.exists(filepath):
        return filepath

    # 1) 复用 llm-bpe-tokenizer 里已经有的那份
    for cand in LOCAL_FALLBACKS:
        if os.path.exists(cand):
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(cand, "rb") as src, open(filepath, "wb") as dst:
                dst.write(src.read())
            print(f"Reused local corpus: {cand} -> {filepath}")
            return filepath

    # 2) 本地没有才下载, 且下载失败要给出可操作的报错
    print("Local corpus not found, downloading Tiny Shakespeare ...")
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, filepath)
        print(f"Downloaded to {filepath}")
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(
            f"无法获取 Tiny Shakespeare 语料: {e}\n"
            f"  已尝试的本地路径: {LOCAL_FALLBACKS}\n"
            f"  下载地址: {SHAKESPEARE_URL}\n"
            f"  解决方式(任选其一):\n"
            f"    1. 启动代理 (本机 WinINET 代理配置为 127.0.0.1:7890, 当前无进程监听)\n"
            f"    2. 手动下载后放到 {filepath}\n"
            f"    3. 复用 projects/llm-bpe-tokenizer/data/tinyshakespeare.txt"
        ) from e
    return filepath


def load_tiny_shakespeare(data_dir: str = None):
    """
    加载 Tiny Shakespeare 数据集

    获取顺序: 本地副本 -> 内置 fallback -> 在线下载
    在线来源:
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

    Returns:
        text: 原始文本字符串
        tokenizer: CharTokenizer 实例
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    filepath = os.path.join(data_dir, "tiny_shakespeare.txt")
    _ensure_corpus(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

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
    准备训练/验证 DataLoader

    Args:
        text: 原始文本
        tokenizer: tokenizer 实例
        block_size: 上下文长度 (序列长度)
        batch_size: batch 大小
        train_ratio: 训练集比例
        device: 设备

    Returns:
        train_loader, val_loader, vocab_size
    """
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Train: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")
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
