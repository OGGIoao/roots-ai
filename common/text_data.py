"""字符级分词器与定长块数据集 (llm-transformer / llm-minigpt 共用)。"""

import torch
from torch.utils.data import DataLoader, Dataset


class CharTokenizer:
    """字符级 Tokenizer: 词表 = 语料里出现过的全部字符 (Tiny Shakespeare 约 65 个)。

    优点: 无外部依赖, 适合教学和小规模实验。
    缺点: 压缩率低; 词表外字符在 encode 时被**静默丢弃**, 在 decode 时对未知 ID 输出 "?"。
    实际 LLM 用 BPE / SentencePiece (见 llm-bpe-tokenizer 项目)。
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list:
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, ids) -> str:
        return "".join(self.idx_to_char.get(int(i), "?") for i in ids)


class TextDataset(Dataset):
    """把一维 token 张量切成 (x, y) 样本, y 是 x 右移一位 (next-token 预测)。

    stride=1           滑动窗口: 每个起点一个样本 (样本数多, 样本间高度重叠)
    stride=block_size  互不重叠的块 (样本数少, 一个 epoch 恰好扫一遍语料)

    样本数按 `(len - block_size - 1) // stride + 1` 计算, 保证最后一个样本的
    x 和 y 都恰好有 block_size 个 token (不会出现长度不足导致 batch 拼接失败)。
    """

    def __init__(self, data: torch.Tensor, block_size: int, stride: int = 1):
        if block_size < 1 or stride < 1:
            raise ValueError("block_size 和 stride 必须 >= 1")
        self.data = data
        self.block_size = block_size
        self.stride = stride
        usable = len(data) - block_size - 1
        self.n_samples = usable // stride + 1 if usable >= 0 else 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if not 0 <= idx < self.n_samples:
            raise IndexError(idx)
        start = idx * self.stride
        x = self.data[start : start + self.block_size]
        y = self.data[start + 1 : start + self.block_size + 1]
        return x, y


def make_loaders(
    token_ids,
    block_size: int,
    batch_size: int,
    train_ratio: float = 0.9,
    stride: int = 1,
):
    """按顺序切分 (文本有先后, 不能 shuffle 后再切) 并返回 (train_loader, val_loader)。"""
    data = torch.tensor(token_ids, dtype=torch.long)
    n = int(train_ratio * len(data))
    train_ds = TextDataset(data[:n], block_size, stride)
    val_ds = TextDataset(data[n:], block_size, stride)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"语料太短 (train={n}, val={len(data) - n} tokens), "
            f"放不下 block_size={block_size} 的样本; 请减小 block_size"
        )
    pin = torch.cuda.is_available()  # CPU 上 pin_memory 只会告警
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)
    return train_loader, val_loader
