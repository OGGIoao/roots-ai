"""
tokenizer.py — 字节级 BPE Tokenizer 从零实现

教练注：
  这是本项目的核心文件，对应 CS336 Assignment 1 的主要编码任务。
  讲义: TUTORIAL.md 第 3~8 节 (逐函数拆解) 与图 3、4、7。

  阅读路径建议：
    1. __init__()       —— 了解数据结构
    2. train()          —— BPE 训练主循环
    3. _get_pair_freqs() —— 统计相邻 token 对频率（核心）
    4. _apply_merge_to_freqs() —— 把一次合并应用到词频表
    5. encode()         —— 编码流程
    6. _encode_chunk()  —— 贪心应用合并规则
    7. decode()         —— 解码（最简单）

  术语:
    字节 (byte)      0~255 的整数, 文本经 UTF-8 编码后的最小单位
    pre-token        预分词切出的片段 (一个"词"加上它前面的空白), BPE 合并不会跨越它的边界
    pair             相邻的两个 token ID
    rank             一条合并规则学到的先后次序; 本实现里直接用 new_id 表示 (越小越早学到)
"""

import re
import json
from collections import defaultdict


class BPETokenizer:
    """
    字节级 BPE (Byte Pair Encoding) Tokenizer。

    词表结构：
      - ID 0~255：初始词表，每个 ID 对应一个字节值（bytes([id])）
      - ID 256+：BPE 合并生成的新 token，按合并顺序分配 ID

    关键属性：
      self.vocab   : dict[int, bytes]              —— ID → 字节序列
      self.merges  : dict[tuple[int,int], int]     —— (id_a, id_b) → new_id
                     new_id 同时也是该合并的"优先级"：越小越早被应用
    """

    def __init__(self):
        # 初始词表：256 个字节，每个字节值就是它对应的 1 字节序列
        # 例: vocab[65] = b'A', vocab[10] = b'\n', vocab[228] = b'\xe4' (汉字"你"的第 1 个字节)
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # 合并规则表：按训练顺序记录，new_id 即为 rank（越小越优先）
        # Python 3.7+ 的 dict 保持插入顺序, 所以遍历 merges 的顺序就是学到的顺序
        self.merges: dict[tuple[int, int], int] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # 公开 API
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        """
        在 text 上训练 BPE，学习合并规则直到词表大小达到 vocab_size。

        Args:
            text:       训练语料（完整字符串）
            vocab_size: 目标词表大小，必须 > 256
            verbose:    是否打印训练进度

        注意: 如果语料里已经没有可合并的 pair (例如语料太短), 会提前结束,
              此时词表小于 vocab_size。
        """
        assert vocab_size > 256, "vocab_size 必须 > 256（初始词表已有 256 个字节 token）"
        num_merges = vocab_size - 256                      # 每次合并新增 1 个 token

        # ── Step 1：Pre-tokenize（预分词）────────────────────────────────────
        # 把整段文本拆成更小的"词"单元，BPE 合并不会跨越这些单元的边界。
        # 作用: 让学到的 token 都落在词的内部 (例如学到 " the", 而不会学到 "e t" 这种
        # 横跨两个词的片段), 同时把语料压缩成"不同的词 + 出现次数", 统计快得多。
        pretokens = self._pretokenize(text)

        # ── Step 2：构建初始词频表 ────────────────────────────────────────────
        # word_freqs: { 字节ID元组 : 出现次数 }
        # 例："low" (freq=5) → { (108, 111, 119): 5 }
        #      l=108, o=111, w=119
        # 同一个词只存一份, 后面统计 pair 时乘上它的次数 —— 这是训练能在几秒内完成的关键
        word_freqs: dict[tuple[int, ...], int] = defaultdict(int)
        for token in pretokens:
            byte_ids = tuple(token.encode("utf-8"))  # str → UTF-8 字节 → ID 元组 (元组才能当 dict 的键)
            word_freqs[byte_ids] += 1

        if verbose:
            total = sum(word_freqs.values())
            print(f"[训练] 语料 pre-token 统计：{total:,} 个（{len(word_freqs):,} 种）")
            print(f"[训练] 目标：{num_merges} 次合并，词表 256 → {vocab_size}")
            print("-" * 55)

        # ── Step 3：迭代合并 ──────────────────────────────────────────────────
        log_interval = max(1, num_merges // 10)  # 每完成 10% 打印一次

        for merge_idx in range(num_merges):
            # 3a. 统计当前词频表中所有相邻 token 对的频率
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:                             # 每个词都只剩 1 个 token 了
                if verbose:
                    print(f"[!] 语料已无可合并的 pair，在第 {merge_idx} 次合并时提前结束")
                break

            # 3b. 找最高频对
            #     频率相同时，取 ID 对字典序最小的——确保结果确定性（可复现）
            #     key 是一个三元组: 先比频率 (越大越好), 再比 -id_a、-id_b (id 越小越好)
            best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], -p[0], -p[1]))
            best_freq = pair_freqs[best_pair]

            # 3c. 分配新 token ID，并记录合并规则
            #     新 token 的字节内容 = 两个父 token 的字节内容拼接
            new_id = 256 + merge_idx
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # 3d. 把词频表里所有出现该 pair 的地方都替换成 new_id
            #     (被替换的词变短了, 相邻关系随之改变, 所以下一轮要重新统计)
            word_freqs = self._apply_merge_to_freqs(best_pair, new_id, word_freqs)

            if verbose and (merge_idx + 1) % log_interval == 0:
                token_repr = self._token_repr(new_id)
                print(
                    f"  合并 #{merge_idx + 1:4d}/{num_merges}："
                    f" ({best_pair[0]:3d}, {best_pair[1]:3d}) → {new_id:5d}"
                    f"  {token_repr!r}  (频率={best_freq})"     # !r: 换行显示成 \n, 不会把日志断成两行
                )

        if verbose:
            print("-" * 55)
            print(f"[训练完成] 词表大小：{len(self.vocab)}")

    def encode(self, text: str) -> list[int]:
        """
        把文本编码成 token ID 列表。

        流程：
          text → pretokenize → 每个 pre-token 转字节 → _encode_chunk → 拼接
        """
        pretokens = self._pretokenize(text)
        result: list[int] = []
        for token in pretokens:
            byte_ids = list(token.encode("utf-8"))     # 例: " the" → [32, 116, 104, 101]
            # 对每个 pre-token 独立应用合并规则 (与训练时一致, 合并不跨越 pre-token 边界)
            merged = self._encode_chunk(byte_ids)
            result.extend(merged)
        return result

    def decode(self, ids: list[int]) -> str:
        """
        把 token ID 列表解码成文本。

        流程：
          每个 ID → self.vocab 查字节 → 合并所有字节 → UTF-8 解码
        解码只需要 vocab, 不需要 merges: 每个 token 记着自己代表哪些字节, 拼起来就是原文。
        """
        # errors="replace"：遇到无效 UTF-8 序列时用 U+FFFD 替换，而不是报错
        # (例如单独解码汉字 3 个字节中的第 1 个, 或模型生成了不完整的多字节字符)
        raw_bytes = b"".join(self.vocab[i] for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        """当前词表大小。"""
        return len(self.vocab)

    def save(self, path: str) -> None:
        """
        把词表和合并规则保存到 JSON 文件。

        JSON 格式：
          {
            "vocab":  { "id": [byte, ...], ... },
            "merges": [ [id_a, id_b, new_id], ... ]
          }
        字节存成整数列表而不是字符串: 很多 token 不是合法的 UTF-8 (例如半个汉字), 无法存成文本。
        """
        data = {
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "merges": [[a, b, c] for (a, b), c in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[保存] Tokenizer 已保存到 {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """从 JSON 文件加载 tokenizer。"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}   # JSON 的键只能是字符串, 转回 int
        tok.merges = {(row[0], row[1]): row[2] for row in data["merges"]}
        return tok

    # ─────────────────────────────────────────────────────────────────────────
    # 内部辅助方法（供理解内部实现用）
    # ─────────────────────────────────────────────────────────────────────────

    def _pretokenize(self, text: str) -> list[str]:
        """
        预分词：把文本分割成不跨词边界的片段。

        这里使用简化版：把空白字符（空格、换行等）附加到其后面的词的开头，
        模仿 GPT-2 的 "Ġ"（▁）前缀设计。

        例 (实测)：
          "Hello world"   → ["Hello", " world"]
          " foo  bar"     → [" foo", "  bar"]
          "line1\\nline2"  → ["line1", "\\nline2"]          换行也附在后面的词上
          "line1\\nline2\\n" → ["line1", "\\nline2", "\\n"]  末尾的空白单独成一段

        注：完整的 GPT-2 pretokenization 使用复杂的 Unicode 正则
            （regex 库的 \\p{L} 等），这里为了不引入额外依赖而简化。
            两者的差别见讲义「正则表达式专题」。
        """
        if not text:
            return []

        # 找到所有连续的"非空白"片段和"空白"片段
        # 例: "Hello  world\\n" → ["Hello", "  ", "world", "\\n"]
        tokens = re.findall(r"\S+|\s+", text)

        result: list[str] = []
        pending_space = ""
        for t in tokens:
            if re.match(r"^\s+$", t):
                # 空白：暂存，等下一个词来"领走"
                pending_space = t
            else:
                # 非空白词：把前面积累的空白附在开头
                result.append(pending_space + t)
                pending_space = ""

        # 末尾可能剩下纯空白（如文本以换行结尾），也要保留
        if pending_space:
            result.append(pending_space)

        return result

    def _get_pair_freqs(
        self, word_freqs: dict[tuple[int, ...], int]
    ) -> dict[tuple[int, int], int]:
        """
        统计词频表中所有相邻 token 对的频率。

        教练注：
          这是 BPE 最核心的一步。每次合并后词频表会更新，
          所以每轮都要重新统计。这也是 BPE 训练 O(V * N) 复杂度的来源。

          一个容易忽略的细节: 重叠的 pair 会被重复计数。
          例如 "aaaa" 里 (a, a) 被数了 3 次, 但合并时从左到右只能替换 2 处
          ("aaa" 数 2 次, 只能替换 1 处)。所以这里的频率在少数情况下会偏高 (讲义「训练过程图解」的 abababd 例子)。

        Args:
            word_freqs: { token_id_tuple: count }

        Returns:
            pair_freqs: { (id_a, id_b): total_count }
        """
        pair_freqs: dict[tuple[int, int], int] = defaultdict(int)
        for word_ids, freq in word_freqs.items():
            for i in range(len(word_ids) - 1):
                pair = (word_ids[i], word_ids[i + 1])
                pair_freqs[pair] += freq                   # 加的是这个词出现的次数, 不是 1
        return pair_freqs

    def _apply_merge_to_freqs(
        self,
        pair: tuple[int, int],
        new_id: int,
        word_freqs: dict[tuple[int, ...], int],
    ) -> dict[tuple[int, ...], int]:
        """
        把一次合并应用到整个词频表：
        把所有词里出现的 pair 替换为 new_id。

        教练注：
          注意这里我们替换所有出现的 pair，而不只是第一个。
          这和 encode 时的处理方式一致。

        Args:
            pair:      要被合并的 (id_a, id_b) 对
            new_id:    合并后的新 token ID
            word_freqs: 当前词频表

        Returns:
            更新后的词频表
        """
        new_word_freqs: dict[tuple[int, ...], int] = {}
        for word_ids, freq in word_freqs.items():
            new_ids = _merge_pair_in_sequence(word_ids, pair, new_id)
            # 不同的词合并后不会变成同一个序列 (合并可逆), 所以这里直接赋值, 不需要累加
            new_word_freqs[new_ids] = freq
        return new_word_freqs

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        """
        对单个 pre-token 的字节 ID 序列贪心应用合并规则。

        算法：
          重复以下步骤直到没有可用合并：
            1. 找序列中所有存在合并规则的相邻对
            2. 选 rank（= new_id）最小的，即训练时最早学到的合并
            3. 把序列中所有该 pair 替换为 new_id

        注意：每次选"最小 rank"的 pair，保证编码确定性（等价于按训练顺序应用）。
        """
        while len(ids) >= 2:
            # 找当前序列中 rank 最小的可用合并
            best_pair = None
            best_rank = float("inf")
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                if pair in self.merges:
                    rank = self.merges[pair]  # new_id 即 rank
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break  # 没有可用合并了，结束

            new_id = self.merges[best_pair]
            # 替换所有出现的 best_pair (序列变短, 下一轮重新扫描)
            ids = list(_merge_pair_in_sequence(tuple(ids), best_pair, new_id))

        return ids

    def _token_repr(self, token_id: int) -> str:
        """把 token 的字节内容转成可读字符串（用于打印）。不是合法 UTF-8 的 (如半个汉字) 显示为 bytes。"""
        raw = self.vocab[token_id]
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return repr(raw)


# ─── 模块级辅助函数（不依赖 self，可单独测试）───────────────────────────────


def _merge_pair_in_sequence(
    ids: tuple[int, ...],
    pair: tuple[int, int],
    new_id: int,
) -> tuple[int, ...]:
    """
    把序列 ids 中所有出现的 pair 替换为 new_id，返回新序列。

    例：
      ids = (72, 101, 108, 108, 111)，pair = (108, 108)，new_id = 256
      → (72, 101, 256, 111)
      ids = (97, 97, 97)，pair = (97, 97)，new_id = 256
      → (256, 97)          重叠时从左到右, 先匹配的先替换

    教练注：
      这个函数设计成纯函数（不修改输入），方便测试和理解。
    """
    result: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2                                         # 跳过已被合并的两个位置
        else:
            result.append(ids[i])
            i += 1
    return tuple(result)
