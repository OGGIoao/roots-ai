"""
toy_lm.py — 一个不需要训练、不需要显卡的 "语言模型": 字符级三元模型 (只用标准库)

数一数莎士比亚语料里 "前两个字符是 ab 时, 下一个字符是 c" 出现了多少次, 就得到一个能预测下一个字符的模型:

    P(c | ab) = (count(abc) + α) / (count(ab·) + α·V)        α: 加法平滑, 没见过的组合也给一点点概率

它给出的 logits 就是 log P。真实的大模型用神经网络算 logits, 但解码 (从 logits 里挑下一个 token) 的做法完全一样,
所以本单元的解码实验都可以先在它上面做, 一秒内出结果。

    lm = ToyLM.from_corpus()          # 第一次约 1 秒
    logits = lm.logits("th")          # 长度为 V 的列表
    lm.decode([lm.stoi["e"]])         # 'e'
"""

import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)


class ToyLM:
    def __init__(self, text, alpha=0.01):
        self.vocab = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.alpha = alpha
        counts = defaultdict(Counter)
        for i in range(len(text) - 2):
            counts[text[i:i + 2]][text[i + 2]] += 1
        self.counts = counts
        self._cache = {}

    @classmethod
    def from_corpus(cls, alpha=0.01):
        from common.corpus import load_shakespeare
        return cls(load_shakespeare(), alpha)

    @property
    def V(self):
        return len(self.vocab)

    def logits(self, context):
        """context: 已经生成的文字 (字符串); 只看最后两个字符。返回长度 V 的 log 概率列表。"""
        key = context[-2:]
        if key not in self._cache:
            c = self.counts.get(key, Counter())
            total = sum(c.values()) + self.alpha * self.V
            self._cache[key] = [math.log((c.get(ch, 0) + self.alpha) / total) for ch in self.vocab]
        return self._cache[key]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)
