"""
练习 1 (① 填空): 补全 BPE 的四个核心步骤

下面的 MiniBPE 与 ../tokenizer.py 的 BPETokenizer 结构相同, 但挖掉了四处。
把四个 TODO 补全, 让它学到的 merges 与参考实现逐条相同, 编码结果也相同。

验收:
    cd mainline/A1-bpe/exercises
    python -m pytest test_ex1.py -q

提示规则: 卡住了可以要提示, 按"方向 → 哪一行 → 代码"分级给, 每次只升一级。
写完再去对照 ../tokenizer.py, 看看自己的写法和参考写法有什么不同。
"""

import re
from collections import defaultdict


def pretokenize(text: str) -> list[str]:
    """与参考实现相同的预分词 (已写好): 空白挂到后面的词上。"""
    result, pending = [], ""
    for t in re.findall(r"\S+|\s+", text):
        if t.isspace():
            pending = t
        else:
            result.append(pending + t)
            pending = ""
    if pending:
        result.append(pending)
    return result


def merge_pair(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """把 ids 里所有出现的 pair 换成 new_id, 从左到右, 重叠时先到先得。

    例: (97, 97, 97, 98), pair=(97, 97), new_id=256 → (256, 97, 98)
    """
    # TODO 1: 实现它 (不要修改输入, 返回新的元组)
    raise NotImplementedError


class MiniBPE:
    def __init__(self):
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: dict[tuple[int, int], int] = {}

    @staticmethod
    def pair_freqs(word_freqs: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
        """统计所有相邻 pair 的频率。"""
        # TODO 2: 每个词里的每一对相邻 ID 都要计数; 想清楚每次该加几
        raise NotImplementedError

    def train(self, text: str, vocab_size: int) -> None:
        word_freqs: dict[tuple[int, ...], int] = defaultdict(int)
        for w in pretokenize(text):
            word_freqs[tuple(w.encode("utf-8"))] += 1
        for merge_idx in range(vocab_size - 256):
            pf = self.pair_freqs(word_freqs)
            if not pf:
                break
            # TODO 3: 选出要合并的 pair。频率最高; 并列时选第一个 ID 更小的, 再并列选第二个 ID 更小的。
            best = None
            new_id = 256 + merge_idx
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            word_freqs = {merge_pair(w, best, new_id): f for w, f in word_freqs.items()}

    def encode_chunk(self, ids: list[int]) -> list[int]:
        """对一个预分词片段的字节 ID 反复合并, 直到不能再合并。"""
        # TODO 4: 每轮在相邻对里找出"在 merges 里且 rank 最小"的那一对, 用 merge_pair 合并; 没有可合并的就结束
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for w in pretokenize(text):
            out.extend(self.encode_chunk(list(w.encode("utf-8"))))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
