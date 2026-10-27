"""
练习 3 (③ 白纸): 增量更新的快速训练

../tokenizer.py 的 train() 每一轮都把所有不同的词完整扫一遍来数 pair (讲义 4.3 节), 每次合并约 46 毫秒。
工业实现只更新"包含被合并 pair 的那些词" (讲义第 12 节)。请从零写出这个加速版。

只给验收标准 (测试逐条检查):
  1. 函数签名: train_fast(text: str, vocab_size: int) -> dict[tuple[int, int], int]
     返回 merges, 格式与 BPETokenizer.merges 相同 (插入顺序 = 学到的顺序);
  2. 在若干小文本和莎士比亚前 30 万字符 (词表 600) 上, 与 BPETokenizer.train() 学到的 merges 逐条相同,
     包括并列时的选择、重叠 pair 的计数方式、提前结束;
  3. 在莎士比亚前 30 万字符、词表 600 上, 比参考实现至少快 3 倍。

可以直接用参考实现的预分词: from tokenizer import BPETokenizer; BPETokenizer()._pretokenize(text)

验收:
    cd library/llm/bpe/exercises
    python -m pytest test_ex3.py -q
"""


def train_fast(text: str, vocab_size: int) -> dict[tuple[int, int], int]:
    raise NotImplementedError
