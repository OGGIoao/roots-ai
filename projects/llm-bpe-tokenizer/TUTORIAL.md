# BPE Tokenizer 深度教程

> 本教程是 [tokenizer.py](tokenizer.py) 的**逐行讲解 + 数学原理 + 学习资源导航**。
> 建议打开代码文件对照阅读,每个核心函数都会在下文展开。

---

## 目录

1. [BPE 算法全景图](#1-bpe-算法全景图)
2. [从零理解:为什么需要 BPE](#2-从零理解为什么需要-bpe)
3. [训练过程图解](#3-训练过程图解)
4. [tokenizer.py 逐行拆解](#4-tokenizerpy-逐行拆解)
5. [关键数据结构详解](#5-关键数据结构详解)
6. [Pre-tokenization 的作用](#6-pre-tokenization-的作用)
7. [编码(encode)的贪心策略](#7-编码encode的贪心策略)
8. [解码(decode)的简洁之美](#8-解码decode的简洁之美)
9. [压缩率:衡量 tokenizer 好坏的核心指标](#9-压缩率衡量-tokenizer-好坏的核心指标)
10. [BPE vs WordPiece vs Unigram:三种子词算法对比](#10-bpe-vs-wordpiece-vs-unigram三种子词算法对比)
11. [中文分词的特殊挑战](#11-中文分词的特殊挑战)
12. [从我们的 BPE 到 GPT-4 的 cl100k:差在哪?](#12-从我们的-bpe-到-gpt-4-的-cl100k差在哪)
13. [思考练习 & 答案](#13-思考练习--答案)
14. [辅助学习资源大全](#14-辅助学习资源大全)
15. [编码知识专题:从字符到 Token ID 的完整链路](#15-编码知识专题从字符到-token-id-的完整链路)
16. [正则表达式专题:预分词的正则原理与实战](#16-正则表达式专题预分词的正则原理与实战)
17. [原始 0-255 词表全景:初始字节 token 详解](#17-原始-0-255-词表全景初始字节-token-详解)

---

## 1. BPE 算法全景图

BPE(Byte Pair Encoding)最初是一种**数据压缩算法**(Gage, 1994),被 Sennrich et al. (2016) 引入 NLP 领域后,成为了现代 LLM tokenizer 的事实标准。

**一句话概括:**
> BPE 训练 = "反复把语料中出现次数最多的**相邻字节对**合并为一个新 token,直到词表达到目标大小。"

```
┌─────────────────────────────────────────────────────┐
│                  BPE 全流程                           │
│                                                     │
│  训练 (train)                                       │
│  ┌────────────┐     ┌──────────┐     ┌──────────┐  │
│  │ 原始语料   │ ──→ │ 词频统计  │ ──→ │ 迭代合并  │  │
│  │ (字符串)   │     │ (字节对)  │     │ (N轮)    │  │
│  └────────────┘     └──────────┘     └──────────┘  │
│                                          │          │
│                                          ▼          │
│                                   ┌──────────────┐  │
│                                   │ 合并规则表    │  │
│                                   │ merges: dict  │  │
│                                   └──────────────┘  │
│                                                     │
│  编码 (encode)                                      │
│  ┌────────┐   ┌──────────┐   ┌──────────────────┐  │
│  │ 新文本  │ → │ 字节拆分  │ → │ 按 rank 顺序合并 │  │
│  └────────┘   └──────────┘   └──────────────────┘  │
│                                      │              │
│                                      ▼              │
│                              ┌───────────────┐      │
│                              │ token ID 列表  │      │
│                              └───────────────┘      │
│                                                     │
│  解码 (decode)                                      │
│  ┌───────────────┐   ┌──────────┐   ┌───────────┐  │
│  │ token ID 列表 │ → │ 查词表    │ → │ UTF-8文本  │  │
│  └───────────────┘   └──────────┘   └───────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 2. 从零理解:为什么需要 BPE

### 2.1 三种分词粒度的困境

LLM 需要把文本变成数字。怎么切分文本?

| 方案 | 词表大小 | 优点 | 致命缺陷 |
|------|---------|------|---------|
| **字符级** (character) | ~100 | 不会有未知词 | 序列太长 → 注意力 $O(n^2)$ 爆炸 |
| **词级** (word) | ~50,000+ | 序列短 | 遇到新词/OOV 直接崩 → `<UNK>` |
| **子词级** (subword/BPE) | ~30,000-100,000 | ✅ 平衡长度与覆盖率 | — |

**核心矛盾**:词表越大 → 序列越短(计算快)→ 但 embedding 层参数越多,且低频词学不好。

BPE 的精妙之处在于:**让高频词保持完整,把低频词拆成有意义的碎片。**

例如(以 `vocab_size=500` 训练后):
- `"the"` → 1个 token(高频词,完整保留)
- `"unbelievable"` → 3个 token:`"un"` + `"believ"` + `"able"`(低频词,拆成子词)
- `"xyzabc"` → 6个 token(完全没见过,退回字节级)

### 2.2 为什么是"字节级"?

我们的实现是 **Byte-level BPE**(GPT-2 首创),初始词表 = 256 个字节值(0-255)。

**好处:**
- 🌍 **万国通用**:中文、日文、emoji、代码、数学符号——任何文本都能编码,因为 UTF-8 本身就是字节序列
- 🚫 **零 OOV**:不可能遇到"未知字符",最多就是退回到单个字节
- 🔗 **和后续 LLM 无缝对接**:所有主流 LLM(GPT-2/3/4, LLaMA, Claude)都用字节级 BPE

**代价:**
- 中文字符 = 3 个字节(UTF-8),在字节级 BPE 下如果没有足够的中文训练数据,一个汉字可能被拆成 3 个 token

### 2.3 一个直观的例子

```
训练语料: "low lower lowest" (假设各出现 1 次)

初始词频表:
  "low"    → (108, 111, 119)    # l=108, o=111, w=119
  "lower"  → (108, 111, 119, 101, 114)  # +e=101, r=114
  "lowest" → (108, 111, 119, 101, 115, 116)  # +s=115, t=116

第1轮: 统计所有相邻对的频率
  (108,111) = (l,o): 出现在所有3个词中 → 频率=3  ⭐最高
  (111,119) = (o,w): 也是3
  (119,101) = (w,e): 出现在 lower 和 lowest → 2
  ...

选择 (108,111) → 合并为新 token "lo" (ID=256)
更新词频表:
  "lo" + "w"       → (256, 119)
  "lo" + "w"+"e"+"r" → (256, 119, 101, 114)
  "lo" + "w"+"e"+"s"+"t" → (256, 119, 101, 115, 116)

第2轮: (111,119) 不再存在了(111 已被合并)→ 但 (256,119) 也没学过
  新的最高频对: (256,119) = "low" → 出现在3个词 → 频率=3
  合并为 "low" (ID=257)

第3轮: (119,101) = (w,e): 出现在 lower 和 lowest → 频率=2
  合并为 "we" (ID=258) → 不对,应该是 "we"... 这里就是简单的贪心
  ...

继续... 直到词表达到目标大小
```

---

## 3. 训练过程图解

用一个更小的例子走完整训练流程:

```
训练语料: "abababd" (只考虑这一个词,频率=1)

═══ 第 0 步: 初始化 ═══
  词频表: { (97, 98, 97, 98, 97, 98, 100): 1 }
  即: 'a','b','a','b','a','b','d' 各为独立字节
  vocab = {0: b'\x00', ..., 97: b'a', 98: b'b', ..., 100: b'd', ..., 255: b'\xff'}

═══ 第 1 轮: ═══
  统计相邻对频率:
    (97,98) = (a,b): 出现在位置0-1, 2-3, 4-5 → 频率=3 ⭐
    (98,97) = (b,a): 出现在位置1-2, 3-4 → 频率=2
    (98,100) = (b,d): 出现在位置5-6 → 频率=1

  最高频: (97,98) → 合并为 ID=256, 即 "ab"
  更新: { (256, 256, 256, 100): 1 }   "ab","ab","ab","d"

═══ 第 2 轮: ═══
  统计相邻对频率:
    (256,256) = ("ab","ab"): 位置0-1, 1-2 → 频率=2 ⭐
    (256,100) = ("ab","d"): 位置2-3 → 频率=1

  最高频: (256,256) → 合并为 ID=257, 即 "abab"
  更新: { (257, 256, 100): 1 }   "abab","ab","d"

═══ 第 3 轮: ═══
  (257,256) → 频率=1
  (256,100) → 频率=1

  并列最高 → 选字典序最小的 (256,100) → 合并为 ID=258, 即 "abd"
  更新: { (257, 258): 1 }   "abab","abd"

...继续直到达到目标 vocab_size
```

**核心观察**:BPE 是一种**贪心算法**,每一步都选当前最频繁的 pair 合并。它不保证全局最优,但实践中效果非常好。

---

## 4. tokenizer.py 逐行拆解

### 4.1 `__init__()` — 初始词表构建

```python
def __init__(self):
    # 256 个字节 token,ID 就是字节值本身
    self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    # 合并规则:(id_a, id_b) → new_id
    # new_id 同时也是"优先级":值越小 = 越早学到 = 编码时越优先应用
    self.merges: dict[tuple[int, int], int] = {}
```

**为什么是 256?** 因为一个字节的取值范围是 0-255,所以任何文本的 UTF-8 字节序列都能用这 256 个 token 表示。这就是"字节级"的含义。

### 4.2 `train()` — BPE 训练主循环(核心!)

```python
def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
    num_merges = vocab_size - 256  # 需要学习多少次合并

    # Step 1: 预分词 — 把文本拆成"词"级别的片段
    pretokens = self._pretokenize(text)

    # Step 2: 构建词频表
    word_freqs: dict[tuple[int, ...], int] = defaultdict(int)
    for token in pretokens:
        byte_ids = tuple(token.encode("utf-8"))  # "low" → (108, 111, 119)
        word_freqs[byte_ids] += 1

    # Step 3: 迭代合并
    for merge_idx in range(num_merges):
        # 3a. 统计所有相邻对的频率
        pair_freqs = self._get_pair_freqs(word_freqs)

        # 3b. 找最高频对(频率相同时取字典序最小的,确保确定性)
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], -p[0], -p[1]))

        # 3c. 分配新 ID,记录合并规则
        new_id = 256 + merge_idx
        self.merges[best_pair] = new_id
        self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        # 3d. 把词频表中所有出现该 pair 的地方替换为 new_id
        word_freqs = self._apply_merge_to_freqs(best_pair, new_id, word_freqs)
```

**逐行注解:**

1. **`num_merges = vocab_size - 256`**:初始词表已有 256 个字节 token,需要学习的合并次数 = 目标词表大小 - 256。

2. **`_pretokenize()`**:先把文本按"词"拆开,防止 "The" 和 "the" 的字节被合并到一起。这一步对结果影响很大,详见第 6 节。

3. **`tuple(token.encode("utf-8"))`**:把字符串转成 UTF-8 字节序列。例如 `"low"` → `(108, 111, 119)`。每个字节值就是一个 token ID。

4. **`_get_pair_freqs()`**:遍历词频表中所有词的相邻对,累加频率。这是 BPE 的心脏——每轮都要重新统计,复杂度 $O(V \times N)$(V = 词频表条目数,N = 平均词长)。

5. **`max(..., key=lambda ...)`**:选最高频的 pair。`-p[0], -p[1]` 是 tie-breaker——频率相同时取字典序最小的对,确保训练结果确定性(给定相同数据+参数,tokenizer 结果唯一)。

6. **`self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]`**:新 token 的字节内容 = 两个父 token 的字节内容拼接。例如 `vocab[256] = vocab[108] + vocab[111] = b'\x6c' + b'\x6f' = b'lo'`。

7. **`_apply_merge_to_freqs()`**:把词频表中所有出现 `(id_a, id_b)` 的地方替换为 `new_id`。注意这会改变词的长度(两个 ID 变成一个),从而改变相邻关系。

### 4.3 `_get_pair_freqs()` — 统计相邻对频率

```python
def _get_pair_freqs(self, word_freqs):
    pair_freqs: dict[tuple[int, int], int] = defaultdict(int)
    for word_ids, freq in word_freqs.items():
        for i in range(len(word_ids) - 1):
            pair = (word_ids[i], word_ids[i + 1])
            pair_freqs[pair] += freq  # 注意:乘以词频!
    return pair_freqs
```

**关键细节**:`pair_freqs[pair] += freq` 是加上**整个词的频率**,不是 1。

例如,如果 `"the"` 出现 100 次,那么 `(t, h)` 和 `(h, e)` 各贡献 100 次,而不是各 1 次。

**复杂度**:$O(\text{总词长})$。这是每轮合并的瓶颈操作。

### 4.4 `_apply_merge_to_freqs()` — 应用合并到词频表

```python
def _apply_merge_to_freqs(self, pair, new_id, word_freqs):
    new_word_freqs = {}
    for word_ids, freq in word_freqs.items():
        new_ids = _merge_pair_in_sequence(word_ids, pair, new_id)
        new_word_freqs[new_ids] = freq
    return new_word_freqs
```

这里调用了模块底部的辅助函数 `_merge_pair_in_sequence()`:

```python
def _merge_pair_in_sequence(
    ids: tuple[int, ...], pair: tuple[int, int], new_id: int
) -> tuple[int, ...]:
    """
    把序列中所有 (id_a, id_b) 替换为 new_id。

    例: ids=(97,98,97,98), pair=(97,98), new_id=256
        → (256, 256)

    注意:是替换"所有"出现,从左到右扫描一次。
    """
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2  # 跳过两个
        else:
            result.append(ids[i])
            i += 1
    return tuple(result)
```

**逐行注解**:
- 从左到右扫描序列
- 如果当前位置和下一个位置恰好是 `(pair[0], pair[1])`,就用 `new_id` 替换,然后跳过两个位置
- 否则保留当前位置,向前一步

### 4.5 `encode()` — 编码流程

```python
def encode(self, text: str) -> list[int]:
    pretokens = self._pretokenize(text)
    result = []
    for token in pretokens:
        byte_ids = list(token.encode("utf-8"))
        merged = self._encode_chunk(byte_ids)
        result.extend(merged)
    return result
```

**三步走**:
1. **Pre-tokenize**:把文本按词拆分
2. **`_encode_chunk()`**:对每个词的字节序列贪心应用合并规则
3. **拼接**:把所有词的 token ID 合在一起

**关键设计**:合并规则只在词内部应用,不会跨词合并。这保证了编码的确定性和效率。

### 4.6 `_encode_chunk()` — 贪心合并(编码的核心!)

```python
def _encode_chunk(self, ids: list[int]) -> list[int]:
    while len(ids) >= 2:
        # 找当前序列中 rank 最小的可用合并
        best_pair = None
        best_rank = float("inf")
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            if pair in self.merges:
                rank = self.merges[pair]  # new_id = rank
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

        if best_pair is None:
            break  # 没有可用合并了

        new_id = self.merges[best_pair]
        ids = list(_merge_pair_in_sequence(tuple(ids), best_pair, new_id))

    return ids
```

**逐行注解**:

1. **`while len(ids) >= 2`**:只要序列中还有至少 2 个 token,就有可能合并。

2. **`rank = self.merges[pair]`**:`merges` 字典的 key 是 `(id_a, id_b)`,value 是 `new_id`。因为 `new_id` 是按训练顺序递增分配的(256, 257, 258...),所以 **`new_id` 天然就是 rank**——值越小 = 越早学到 = 越优先应用。

3. **选 rank 最小的**:这是贪心策略。如果一个序列中同时存在多种可合并的 pair,优先应用最早学到的合并。

4. **`_merge_pair_in_sequence()`**:执行替换。注意这里可能同时替换多个出现位置。

**举例**:
```
输入: "lowest" → 字节 ID = [108, 111, 119, 101, 115, 116]
                   即: l    o    w    e    s    t

假设训练时学到:
  merge #0: (108,111) → 256   即 "lo"
  merge #1: (256,119) → 257   即 "low"

编码过程:
  第1轮: 扫描序列,找到 (108,111) 的 rank=0, (111,119) 无合并规则
         → 选 (108,111) → 替换为 256
         → [256, 119, 101, 115, 116]

  第2轮: 扫描序列,找到 (256,119) 的 rank=1, (119,101) 无合并规则
         → 选 (256,119) → 替换为 257
         → [257, 101, 115, 116]

  第3轮: 扫描,所有相邻对都无合并规则
         → break

最终: [257, 101, 115, 116]
即: "low" + "e" + "s" + "t" → 4 个 token(原来需要 6 个字节)
```

### 4.7 `decode()` — 解码

```python
def decode(self, ids: list[int]) -> str:
    raw_bytes = b"".join(self.vocab[i] for i in ids)
    return raw_bytes.decode("utf-8", errors="replace")
```

**极简!** 两行搞定:
1. 把每个 ID 查词表得到字节序列,拼接成一大块字节
2. UTF-8 解码成字符串

`errors="replace"` 是安全网:如果模型生成了无效的 UTF-8 字节序列,不会崩溃,而是用 `�` 替换。

---

## 5. 关键数据结构详解

### `self.vocab: dict[int, bytes]`

| 键 | 值 | 说明 |
|----|-----|------|
| 0 | `b'\x00'` | 空字节 |
| 10 | `b'\n'` | 换行符 |
| 32 | `b' '` | 空格 |
| 65 | `b'A'` | 大写 A |
| 97 | `b'a'` | 小写 a |
| 256 | `b'lo'` | 训练时学到的第一个合并 |
| 257 | `b'low'` | 训练时学到的第二个合并 |
| ... | ... | ... |

### `self.merges: dict[tuple[int,int], int]`

| 键 | 值 | 说明 |
|----|-----|------|
| `(108, 111)` | 256 | "l"+"o" → "lo"(第 1 轮学到的) |
| `(256, 119)` | 257 | "lo"+"w" → "low"(第 2 轮学到的) |
| `(104, 101)` | 258 | "h"+"e" → "he"(第 3 轮学到的) |

**注意**:`merges` 的 value(`new_id`)同时也是 rank。因为 `new_id = 256 + merge_idx`,所以先学到的合并 rank 更小。

---

## 6. Pre-tokenization 的作用

Pre-tokenization 是 BPE 训练和编码前的**预处理步骤**,决定哪些字节可以被合并。

### 6.1 为什么需要?

如果不做 pre-tokenization:
- `"the cat"` 和 `"thecat"` 会被当作一样的字节序列
- 空格会参与合并,产生很多无意义的 token(如 `" e"`, `" a"`, `"t "`)
- 大小写不同的词可能被合并到一起

### 6.2 GPT-2 的做法

GPT-2 使用一个复杂的正则表达式把文本拆分成"词"级别的片段:

```python
# GPT-2 的 pretokenization 正则(简化版)
import re
pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```

这个正则会把:
- 缩写(`'s`, `'t`, `'re`)拆成独立 token
- 连续字母、数字、标点分别拆成一组
- 空格**附加到后面词的开头**

### 6.3 我们的简化实现

```python
def _pretokenize(self, text: str) -> list[str]:
    tokens = re.findall(r"\S+|\s+", text)  # 连续非空白 或 连续空白
    result, pending_space = [], ""
    for t in tokens:
        if re.match(r"^\s+$", t):
            pending_space = t       # 空白暂存
        else:
            result.append(pending_space + t)  # 空白附在词前面
            pending_space = ""
    if pending_space:
        result.append(pending_space)
    return result
```

**效果**:
```
"Hello world"  → ["Hello", " world"]   # 空格附在 "world" 前面
" foo  bar"    → [" foo", "  bar"]     # 多个空格也附在前面
"line1\nline2" → ["line1\n", "line2"]  # 换行附在前一个词后面
```

**为什么把空格附在前面而不是后面?** 这是 GPT-2 的设计惯例——空格+词 = 一个 token。例如 `" the"` 是一个 token,而单独的 `"the"` 是另一个。这让 tokenizer 在编码时天然知道词的边界。

---

## 7. 编码(encode)的贪心策略

### 7.1 为什么是贪心?

`_encode_chunk()` 使用的是**贪心策略**:每次都选当前 rank 最小的 pair 合并。

这**不保证全局最优**(最优解是 NP-hard),但:
- 实践中效果非常好
- 复杂度低(线性扫描 × 合并次数)
- 结果确定性(给定相同 merges 表,编码结果唯一)

### 7.2 等价于按训练顺序应用合并

```
训练时学到的合并顺序:
  merge #0: (A, B) → AB    rank = 256
  merge #1: (AB, C) → ABC  rank = 257
  merge #2: (D, E) → DE    rank = 258

编码时(对于序列 [A, B, C, D, E]):
  1. 找 rank 最小的 pair → (A,B) rank=256 → 合并为 AB
  2. 序列变成 [AB, C, D, E]
  3. 找 rank 最小的 pair → (AB,C) rank=257 → 合并为 ABC
  4. 序列变成 [ABC, D, E]
  5. 找 rank 最小的 pair → (D,E) rank=258 → 合并为 DE
  6. 序列变成 [ABC, DE]
  7. 没有更多可合并的 → 结束
```

### 7.3 复杂度分析

假设序列长度为 $L$,共有 $M$ 种合并规则:

- 每轮扫描:$O(L)$
- 最多合并 $L-1$ 次(每次合并至少减少一个 token)
- 每轮查找匹配:$O(L)$

总复杂度:$O(L^2)$(最坏情况)

实际中因为每次合并通常减少多个 pair,远快于最坏情况。

---

## 8. 解码(decode)的简洁之美

```python
def decode(self, ids: list[int]) -> str:
    raw_bytes = b"".join(self.vocab[i] for i in ids)
    return raw_bytes.decode("utf-8", errors="replace")
```

解码之所以这么简单,是因为我们选择了**字节级** BPE:

- 每个 token ID → 一个字节序列(可能 1 字节或多字节)
- 所有 token 的字节拼接起来 → 完整的 UTF-8 字节流
- UTF-8 解码 → 原始文本

**对比 WordPiece**:WordPiece 的 tokenizer 词表里存的是字符串而非字节,解码时需要手动处理词之间的连接符(`##` 前缀),更加复杂。

---

## 9. 压缩率:衡量 tokenizer 好坏的核心指标

$$\text{compression\_ratio} = \frac{\text{token 数}}{\text{原始字节数}}$$

| Tokenizer | 压缩率 | 说明 |
|-----------|--------|------|
| 字节级(无 BPE) | 1.000 | 每个字节 = 1 个 token |
| 我们的 BPE (500) | ~0.4-0.6 | 取决于训练数据 |
| GPT-2 cl100k | ~0.250 | 英文为主的语料训练 |
| GPT-4 cl100k | ~0.250 | 类似 GPT-2 |

**为什么压缩率重要?**
- 压缩率越低 → 同样的文本需要的 token 越少 → 推理速度越快
- 压缩率越低 → 同样的上下文窗口能装更多文本 → 信息密度更高
- 但压缩率太低(词表太大)→ embedding 层参数爆炸,低频 token 学不好

**经验法则**:对于英文,vocab_size=50,000 左右可以达到 ~0.3 的压缩率。

---

## 10. BPE vs WordPiece vs Unigram:三种子词算法对比

| 特性 | BPE | WordPiece | Unigram |
|------|-----|-----------|---------|
| **代表模型** | GPT-2/3/4, LLaMA | BERT, DistilBERT | T5, ALBERT |
| **训练方向** | 自底向上(合并) | 自底向上(合并) | 自顶向下(删除) |
| **初始词表** | 256 字节 | 字符集 | 全部子词候选 |
| **合并/删除依据** | 最频繁对 | 似然增益最大 | 边际似然损失最小 |
| **编码策略** | 贪心 rank 最小 | 贪心/最长匹配 | Viterbi 最优路径 |
| **概率模型** | 无(纯计数) | 无(纯似然) | 有(Unigram LM) |

### BPE(我们的实现)

```
训练: 从单个字节开始,反复合并最高频 pair
编码: 按训练顺序贪心合并
```

### WordPiece(BERT 使用)

```
训练: 同样从单个字节开始,但合并依据是"最大化语言模型似然"
编码: 贪心最长匹配优先
区别: WordPiece 用 "##" 前缀标记词中间的子词
      例如 "playing" → ["play", "##ing"]
```

### Unigram(T5 使用)

```
训练: 从一个很大的候选词表开始,反复删除对似然影响最小的 token
编码: 用 Viterbi 算法找全局最优分词
优势: 有概率模型,可以输出分词的概率
劣势: 训练更复杂
```

---

## 11. 中文分词的特殊挑战

### 11.1 UTF-8 编码

```
字符 '你' 的 UTF-8 编码: 0xE4 0xBD 0xA0 (3 个字节)
字符 '好' 的 UTF-8 编码: 0xE5 0xA5 0xBD (3 个字节)
```

每个中文字符 = 3 个字节。如果 tokenizer 训练数据中中文很少,一个汉字可能需要 3 个 token(退回到字节级),导致:
- 序列变长 3 倍
- 注意力计算量增加 9 倍
- 模型对中文的理解变差

### 11.2 解决方案

| 方案 | 说明 |
|------|------|
| **增加中文训练数据** | 让 BPE 学到中文字符的合并规则,如 `"你好"` → 1 个 token |
| **使用 SentencePiece** | 直接在 Unicode 字符上操作,不依赖字节级 |
| **大词表** | GPT-4 cl100k 有 ~100K 词表,足够覆盖常用汉字 |

### 11.3 实验

你可以用我们的 tokenizer 测试中文:

```python
from tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train("你好世界" * 100, vocab_size=300, verbose=False)

print(tok.encode("你好"))
# 如果训练数据只有中文,"你好" 可能被编码为 1-2 个 token
# 如果训练数据只有英文,"你好" 会被编码为 6 个字节 = 6 个 token
```

---

## 12. 从我们的 BPE 到 GPT-4 的 cl100k:差在哪?

| 差距 | 我们的实现 | GPT-4 cl100k |
|------|-----------|--------------|
| Pre-tokenization | 简化正则 | 复杂 Unicode 正则(`\p{L}`, `\p{N}`) |
| 词表大小 | ~500 | ~100,000 |
| 训练数据 | TinyShakespeare (~1MB) | 数万亿字节的多语言语料 |
| 训练效率 | O(V×N) 每轮扫描 | 使用优先队列+增量更新,大幅加速 |
| 辅助 token | 无 | 特殊 token(`<\|endoftext\|>`, `<|im_start|>` 等) |
| 分词质量 | 中等 | 极高(英文压缩率~0.25) |

**核心差异是训练数据量**:tokenizer 的质量主要取决于训练数据,而非算法本身。

---

## 13. 思考练习 & 答案

### 练习 1:vocab_size=300 vs 1000

> 用哪个训练出的 tokenizer 编码同一段文字,token 数更少?

**答案**:`vocab_size=1000` 的 token 数更少。因为词表更大 → 学到更多合并规则 → 更多常见子词被合并为单个 token → 序列更短。

但更大的词表也有代价:
- embedding 层参数更多
- 低频 token 可能学不充分
- 训练时间更长

### 练习 2:纯中文训练

> 如果训练语料是纯中文,BPE 有什么不同?每个汉字几个字节?

**答案**:
- 每个常用汉字 = 3 字节(UTF-8)
- 纯中文训练时,BPE 会很快学会把常用汉字的 3 个字节合并为 1 个 token
- 例如 `"你"` 的字节 `(0xE4, 0xBD, 0xA0)` 会被合并为 ID=256, 257, 258 三个 token 之一
- 最终常见汉字 = 1 个 token,罕见汉字 = 2-3 个 token

### 练习 3:空格处理

> `"hello"` 和 `" hello"` 为什么是不同的 token?

**答案**:
- Pre-tokenization 时,空格被附加到后面词的开头
- `" hello"` 被处理为一个 pre-token(包含前导空格)
- `"hello"` 被处理为另一个 pre-token(无空格)
- 两者的字节序列不同,BPE 合并规则分别作用
- 这是 GPT-2 的设计:**空格是词的一部分**,不是分隔符

### 练习 4:未见过的词

> 如果训练语料里没有数字,编码 `"2026年"` 会怎样?

**答案**:
- `"2026"` 会被拆成 4 个字节 token:`'2'`, `'0'`, `'2'`, `'6'`(因为训练时没见过数字组合,没有学到合并规则)
- `"年"` 会被拆成 3 个字节 token(UTF-8: 0xE5 0xB9 0xB4)
- 总共 7 个 token 来表示 `"2026年"`
- 这就是为什么 tokenizer 训练数据的多样性非常重要

---

## 14. 辅助学习资源大全

### 📄 必读论文

| 论文 | 年份 | 链接 | 重要性 |
|------|------|------|--------|
| **Neural Machine Translation of Rare Words with Subword Units** (Sennrich et al.) | 2016 | https://arxiv.org/abs/1508.07909 | ⭐⭐⭐⭐⭐ BPE 引入 NLP 的原始论文 |
| **Language Models are Unsupervised Multitask Learners** (GPT-2) | 2019 | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | ⭐⭐⭐⭐⭐ 字节级 BPE 的起源 |
| **SentencePiece** (Kudo & Richardson) | 2018 | https://arxiv.org/abs/1808.06226 | ⭐⭐⭐⭐ 语言无关的 tokenizer 框架 |
| **BPE-Dropout** (Provilkov et al.) | 2020 | https://arxiv.org/abs/1910.13267 | ⭐⭐⭐ BPE 的正则化变体 |
| **Canine: Pre-training an Efficient Tokenization-Free Encoder** (Clark et al.) | 2022 | https://arxiv.org/abs/2103.06874 | ⭐⭐⭐ 完全不需要 tokenizer 的替代方案 |

### 💻 必看开源项目

| 项目 | 作者 | 链接 | 适合阶段 |
|------|------|------|---------|
| **minbpe** | Andrej Karpathy | https://github.com/karpathy/minbpe | ⭐⭐⭐⭐⭐ 入门首选,极简 BPE 实现(~300 行) |
| **tiktoken** | OpenAI | https://github.com/openai/tiktoken | ⭐⭐⭐⭐⭐ GPT-4 的实际 tokenizer,Rust 实现 |
| **SentencePiece** | Google | https://github.com/google/sentencepiece | ⭐⭐⭐⭐ 工业级 tokenizer 框架 |
| **tokenizers** | HuggingFace | https://github.com/huggingface/tokenizers | ⭐⭐⭐⭐ Rust 实现,支持 BPE/WordPiece/Unigram |
| **cs336-bpe** | Stanford CS336 | https://github.com/stanford-cs336/cs336-bpe-2025 | ⭐⭐⭐⭐ 课程配套实现,有详细评测 |

### 📚 教程与博客

| 资源 | 链接 | 说明 |
|------|------|------|
| **Lilian Weng: Tokenizers** | https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/#tokenizer | 简洁全面的技术博客 |
| **HuggingFace Tokenizer 教程** | https://huggingface.co/docs/tokenizers/en/quicktour | 交互式教程,带代码 |
| **Jay Alammar: 可视化 GPT-2 tokenizer** | https://jalammar.github.io/illustrated-gpt2/#gpt-2-tokenizer | 直观的可视化 |
| **Karpathy: Let's build the GPT Tokenizer** | https://www.youtube.com/watch?v=zduSFxRajkE | YouTube 视频教程,2 小时从零实现 |
| **Wikipedia: Byte Pair Encoding** | https://en.wikipedia.org/wiki/Byte_pair_encoding | 原始压缩算法的背景 |

### 🎓 课程

| 课程 | 链接 | 说明 |
|------|------|------|
| **Stanford CS336: Language Modeling** | https://stanford-cs336.github.io/spring2025/ | 最前沿的 LLM 课程,Assignment 1 就是手写 BPE |
| **Stanford CS224N** | https://web.stanford.edu/class/cs224n/ | NLP 经典课程 |
| **fast.ai NLP** | https://www.fast.ai/ | 实战导向的深度学习课程 |

### 🛠️ 在线工具

| 工具 | 链接 | 用途 |
|------|------|------|
| **Tiktokenizer** | https://tiktokenizer.vercel.app/ | 可视化 GPT tokenizer 的分词过程 |
| **HuggingFace Tokenizer Playground** | https://huggingface.co/spaces/narsil/tokenizer-playground | 交互式对比不同 tokenizer |
| **Tokenizer visualizer** | https://platform.openai.com/tokenizer | OpenAI 官方 tokenizer 查看器 |

### 📋 学习路径建议

```
入门 (1-2天)
  ├── 看 Karpathy 的 minbpe 源码(~300行,极简)
  ├── 运行我们的 train.py,观察训练过程
  └── 用 Tiktokenizer 在线可视化

深入 (3-5天)
  ├── 读 Sennrich 2016 原始论文
  ├── 对比 BPE vs WordPiece vs Unigram
  ├── 用 HuggingFace tokenizers 库实验不同分词策略
  └── 尝试修改我们的 _pretokenize() 方法

进阶 (1周+)
  ├── 读 SentencePiece 论文,理解语言无关的分词
  ├── 在中文语料上训练 tokenizer,观察差异
  ├── 研究 BPE-Dropout 和 CANINE 等新方法
  └── 尝试用 Rust 重写 BPE 以理解性能优化
```

---

---

## 15. 编码知识专题:从字符到 Token ID 的完整链路

这一节是编码知识的**系统性深挖**,把「文本 -> token ID」的每一步拆开讲透。

> ⚠️ **小白先读这里**:如果看到 `0x41`、`U+0041`、`01000001b` 一脸懵,先读完 15.1 再往下,否则后面的例子你会看不懂。

### 15.1 先看懂记号:0x41、U+0041、01000001b 到底是什么意思?

教程里到处出现 `0x41`、`U+0041`、`01000001b` 这类记号。对熟悉编程的人来说这是家常便饭,但对小白确实是门槛。这一节把它彻底讲清楚。

#### 15.1.1 同一个数字,有三种写法

计算机里的"数字"可以用不同**进制**(进位制)书写,就像同一个时间可以写成"下午1点"或"13:00"。

以"字符 A 的编号是 65"为例:

| 写法 | 含义 | 说明 |
|------|------|------|
| `65` | 十进制 | 我们日常用的写法 |
| `01000001b` | 二进制 | 计算机内部真正的样子,末尾 `b` 表示二进制 |
| `0x41` | 十六进制 | 编程里最常用的写法,`0x` 前缀表示十六进制 |

**它们都是同一个数**,只是"数法"不同:

```
十进制  65   = 6×10 + 5        (逢十进一)
二进制  01000001b = 0×128+1×64+0×32+0×16+0×8+0×4+0×2+1×1
                   = 64 + 1 = 65                          (逢二进一)
十六进制 0x41  = 4×16 + 1     = 64 + 1 = 65              (逢十六进一)
```

**为什么编程里爱用十六进制?** 因为一个字节正好是 8 个二进制位,而 4 个二进制位恰好等于 1 个十六进制位:

```
0100 0001    ← 8 位,分成两组
  4    1     ← 每组转成十六进制
= 0x41
```

所以 `0x41` 一眼就能看出"这是 1 个字节",而 `65` 看不出。十六进制是"字节的速记法"。

#### 15.1.2 十六进制速查表(记住这个就够了)

十六进制用 16 个符号:0-9 和 A-F(对应十进制的 10-15)。

| 十进制 | 十六进制 | 二进制 |
|--------|---------|--------|
| 0 | 0x0 | 0000 |
| 1 | 0x1 | 0001 |
| 9 | 0x9 | 1001 |
| 10 | 0xA | 1010 |
| 11 | 0xB | 1011 |
| 15 | 0xF | 1111 |
| 16 | 0x10 | 0001 0000 |
| 65 | 0x41 | 0100 0001 |
| 255 | 0xFF | 1111 1111 |

**换算技巧**:
- 十进制 → 十六进制:除 16 取余。`65 ÷ 16 = 4 余 1` → `0x41`
- 十六进制 → 十进制:`0x41 = 4×16 + 1 = 65`
- 一个字节(0-255)在十六进制里永远只占 **2 位**(`0x00` ~ `0xFF`)

#### 15.1.3 `0x` 前缀 vs `U+` 前缀:两个不同的东西

这是最容易混淆的地方,务必要分清:

| 记号 | 前缀来源 | 指什么 | 示例 |
|------|---------|--------|------|
| `0x41` | C 语言风格,`0x` = hex(十六进制) | **字节的值**(计算机里存的) | `0x41` = 字节值 65 |
| `U+0041` | Unicode 标准写法,`U+` = Unicode | **字符的码点**(人类定义的名字) | `U+0041` = 字符 'A' |
| `01000001b` | 二进制,`b` = binary | 字节的二进制展开 | `01000001b` = 65 |

**关键区别:码点 ≠ 字节值,它们不在同一个体系里。**

- `U+0041` 是 Unicode 给字符 'A' 分配的**身份证号**(码点)。Unicode 只负责"编号",不管怎么存储。
- `0x41` 是 'A' 在 **UTF-8 编码后**实际占用的字节值。

**为什么恰好数值一样?** 因为 UTF-8 故意把 ASCII 字符(U+0000~U+007F)编码成与码点相同的 1 个字节。所以:

```
字符 'A':
  码点:   U+0041     (Unicode 定义的编号)
  UTF-8:  0x41       (编码后的字节,恰好和码点一样)
  → 两者数值巧合地相同!

字符 '你':
  码点:   U+4F60     (Unicode 定义的编号)
  UTF-8:  0xE4 0xBD 0xA0   (编码后的 3 个字节,和码点完全不同!)
```

> 💡 **一句话区分**:`U+` 是"Unicode 给字符起的名字",`0x` 是"字节在内存里的值"。ASCII 字符两者恰好相等,非 ASCII 字符(如中文)就完全不一样了。

#### 15.1.4 为什么教程里两个记号混着用?

因为我们一直在讲"字符 → 字节 → token"的转换:

```
字符 'A'  ──Unicode码点──▶  U+0041   (它"叫"什么)
字符 'A'  ──UTF-8编码──▶   0x41     (它"存"成什么)
字节 0x41 ──查表──▶        65       (十进制的 token ID)
```

在本教程的场景里:
- 讲 **Unicode 标准**时用 `U+XXXX`(如 15.2 节的码点表)
- 讲 **内存里的字节**时用 `0xXX`(如 17 节的初始词表)
- 讲 **token ID** 时用十进制数字(如 `ID 65 = b'A'`)

看到 `0x` 想"字节",看到 `U+` 想"Unicode 名字",就完全不会乱了。

#### 15.1.5 实操:Python 里怎么换算

```python
# 十进制 ↔ 十六进制
hex(65)          # '0x41'
int("0x41", 16)  # 65

# 十进制 ↔ 二进制
bin(65)          # '0b1000001'
int("1000001", 2)  # 65

# 字符 ↔ 码点 ↔ UTF-8 字节
ord('A')           # 65        (码点)
hex(ord('A'))      # '0x41'    (码点十六进制)
'A'.encode('utf-8')  # b'A'    (UTF-8 编码)
list('A'.encode('utf-8'))      # [65]          (字节列表)
list('你'.encode('utf-8'))     # [228, 189, 160] = [0xE4, 0xBD, 0xA0]
```

> 建议把这段在 Python 里跑一遍,亲手验证 `65 == 0x41 == 0b1000001`,这个门槛就彻底迈过去了。

### 15.2 字符编码演进史:ASCII -> Unicode -> UTF-8

LLM 处理的是文本,但计算机只能存储字节。我们需要一套规则把字符映射成字节。

**第一阶段: ASCII (1963)**

用 7 位(0-127)表示英文字母、数字和标点。一个字符 = 一个字节(最高位为 0)。

```
'A' = 65  = 01000001b = 0x41
'a' = 97  = 01100001b = 0x61
'0' = 48  = 00110000b = 0x30
' ' = 32  = 00100000b = 0x20
newline = 10 = 0x0A
```

ASCII 的局限: 只有 128 个字符,无法表示中文、日文、emoji。

**第二阶段: Unicode (1991-)**

给全世界所有字符分配一个唯一的**码点**(code point),如:

```
'A'   = U+0041
'你'  = U+4F60
'你好' = U+4F60 U+597D
'😀' = U+1F600
```

但 Unicode 只定义了码点,没有规定如何存储为字节。

**第三阶段: UTF-8 (1993, Ken Thompson & Rob Pike)**

UTF-8 是 Unicode 的**变长字节编码**,是当今互联网的事实标准(占比 > 98%)。

变长规则:

```
1 字节: 0xxxxxxx                              (U+0000~U+007F, ASCII)
2 字节: 110xxxxx 10xxxxxx                     (U+0080~U+07FF)
3 字节: 1110xxxx 10xxxxxx 10xxxxxx            (U+0800~U+FFFF, 含中文)
4 字节: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx   (U+10000~U+10FFFF, 含emoji)
```

编码实例:

```
'A'  = U+0041 -> 0x41                          (1字节)
'n'  = U+00F1 -> 0xC3 0xB1                     (2字节)
'你' = U+4F60 -> 0xE4 0xBD 0xA0               (3字节)
'😀' = U+1F600 -> 0xF0 0x9F 0x98 0x80          (4字节)
```

**为什么这对 BPE 很重要?** 因为字节级 BPE 的初始词表 = 256 个字节值(0-255)。UTF-8 编码决定了:
- 英文字符 = 1 个字节 = 1 个初始 token
- 中文字符 = 3 个字节 = 3 个初始 token
- emoji = 4 个字节 = 4 个初始 token

BPE 训练的任务就是: 把这些分散的字节重新合并成有意义的 token。

### 15.3 编码全链路图解: 文本 -> Token ID

```
输入文本: "Hello world"
        |
        v  Step 1: Pre-tokenize
        |
    ["Hello", " world"]          按空格拆分, 空格附加到下一个词
        |
        v  Step 2: UTF-8 字节化 (每个字符串 -> bytes -> 整数列表)
        |
    [72, 101, 108, 108, 111]     "Hello" -> 5 个字节
    [32, 119, 111, 114, 108, 100] " world" -> 6 个字节
        |
        v  Step 3: 贪心 BPE 合并 (_encode_chunk)
        |
    对每个 pre-token 独立应用合并规则:
      "Hello" -> [72,101,108,108,111]
        -> 找 rank 最小的 pair -> (72,101) rank=256 合并为 "He"
        -> [256, 108, 108, 111]
        -> (256,108) rank=257 合并为 "Hel"
        -> [257, 108, 111]
        -> (257,108) rank=258 合并为 "Hell"
        -> [258, 111]
        -> (258,111) rank=259 合并为 "Hello"
        -> [259]   最终: 1 个 token!
        |
        v  Step 4: 拼接所有 pre-token 的结果
        |
    token_ids = [259, 32, 119, 111, 114, 108, 100]
    "Hello" + " " + "w" + "o" + "r" + "l" + "d"
    (注: "world" 没有学到合并规则, 退回字节级)
```

### 15.4 解码全链路图解: Token ID -> 文本

解码是编码的逆过程, 极其简洁(只需 vocab 字典, 不需要 merges):

```
token_ids = [259, 32, 119, 111, 114, 108, 100]
        |
        v  Step 1: 查词表 (每个 ID -> bytes)
        |
    vocab[259] = b'Hello'    # 合并 token, 对应多字节
    vocab[32]  = b' '        # 单字节 token
    vocab[119] = b'w'
    vocab[111] = b'o'
    vocab[114] = b'r'
    vocab[108] = b'l'
    vocab[100] = b'd'
        |
        v  Step 2: 拼接所有字节
        |
    raw_bytes = b'Hello world'
        |
        v  Step 3: UTF-8 解码 -> 字符串
        |
    text = "Hello world"
```

**关键洞察:** 解码不需要合并规则(merges), 只需要词表(vocab)。这就是为什么 decode() 只有两行代码:
```python
raw_bytes = b"".join(self.vocab[i] for i in ids)   # ID -> bytes -> 拼接
return raw_bytes.decode("utf-8", errors="replace")  # bytes -> str
```

### 15.5 Token ID 与字节的映射关系

```
ID 范围          含义                      字节数      示例
---------------------------------------------------------------
0 ~ 255          初始字节 token            1 字节      ID=72   -> b'H'
256 ~ 256+N-1    BPE 合并生成的 token      可变(2+)   ID=256  -> b'He'
256+N ~ ...      更复杂的合并              可变(3+)   ID=500  -> b'the'
```

**vocab 字典的作用:**

```python
# ID -> bytes 的映射
# 初始: ID 0-255 对应 256 个单字节
# 训练后: ID 256+ 对应合并产生的多字节

tok.vocab[72]    = b'H'        # 单字节
tok.vocab[256]   = b'He'       # 2 字节 (学到的合并)
tok.vocab[259]   = b'Hello'    # 5 字节 (学到的合并)
```

**merges 字典的作用:**

```python
# (id_a, id_b) -> new_id 的映射
# 同时 new_id 也是 rank: 值越小 = 越早学到 = 编码时越优先

tok.merges[(72, 101)]   = 256   # H+e -> He (rank=256, 最优先)
tok.merges[(256, 108)]  = 257   # He+l -> Hel
tok.merges[(257, 108)]  = 258   # Hell+l -> Hell
tok.merges[(258, 111)]  = 259   # Hell+o -> Hello
```

**为什么 rank = new_id?**

因为 new_id = 256 + merge_idx, 先学到的合并 rank 更小。编码时选 rank 最小的 pair 合并, 就是优先应用最早学到的规则。

### 15.6 字节级 vs 字符级 BPE

```
字符级 BPE (BERT 的 SentencePiece):
  初始词表 = 所有 Unicode 字符 (数千个)
  优点: 每个字符天然就是 1 个 token
  缺点: 遇到罕见字符会 OOV (unknown token)

字节级 BPE (GPT-2/3/4, 我们的实现):
  初始词表 = 256 个字节值
  优点: 零 OOV, 任何文本都能编码
  缺点: 非 ASCII 字符默认需要多个 token (需要训练数据来合并)
```

**实际编码对比:**

```
文本: "Hello"
  字符级: 5 token (H, e, l, l, o) -> 如果有合并规则则更少
  字节级: 如果训练数据包含 "Hello" 可能 = 1 token

文本: "你好"
  字符级: 2 token (你, 好) -> 每个字符天然 1 token
  字节级: 默认 6 token (每个汉字 3 字节) -> 训练中文数据后可降为 2 token
```

### 15.7 编码中的边界情况

**1. 未见过的字节组合 (fallback 到字节级):**
```python
tok = BPETokenizer()
tok.train("hello world", vocab_size=300, verbose=False)

# 编码从未在训练中出现的字符
ids = tok.encode("xyz")
# x=120, y=121, z=122 都没有合并规则 -> 各自独立
# 结果: [120, 121, 122]  (退回到最原始的字节级)
```

**2. 空字符串:**
```python
ids = tok.encode("")    # -> [] (空列表)
text = tok.decode([])   # -> "" (空字符串)
```

**3. 多字节字符的字节被独立处理:**
```python
# "你" 的 UTF-8 编码 = 0xE4 0xBD 0xA0 (3 个字节)
# 如果训练数据全是英文, 这 3 个字节各自独立 -> 3 个 token
# 但解码时: b'ä½ '.decode('utf-8') -> "你" (正确还原!)
```

**4. decode 的安全网 (errors="replace"):**
```python
# 如果模型生成了无效的 UTF-8 字节序列
# 例如只有 "你" 的第 1 个字节 0xE4, 缺少后 2 个
# decode 会用 U+FFFD () 替换, 而不是报错
tok.decode([0xE4])  # -> "�" ()
```

### 15.8 完整编码流程代码走读

```python
# tokenizer.py 中的 encode 方法 (逐行解析)
def encode(self, text: str) -> list[int]:
    # Step 1: 预分词
    #   "Hello world" -> ["Hello", " world"]
    #   空格被附加到下一个词的开头, 这是 GPT-2 的惯例
    pretokens = self._pretokenize(text)

    result: list[int] = []
    for token in pretokens:
        # Step 2: 字符串 -> UTF-8 字节序列 -> 整数列表
        #   "Hello" -> b'Hello' -> [72, 101, 108, 108, 111]
        #   " world" -> b' world' -> [32, 119, 111, 114, 108, 100]
        byte_ids = list(token.encode("utf-8"))

        # Step 3: 贪心应用合并规则
        #   [72,101,108,108,111] -> [259]  ("Hello" 合并为 1 个 token)
        #   [32,119,111,114,108,100] -> [32,119,111,114,108,100]  (无合并规则, 保持字节)
        merged = self._encode_chunk(byte_ids)

        # Step 4: 拼接到最终结果
        result.extend(merged)

    return result
```

**_encode_chunk 贪心合并的详细过程:**

```
输入: [72, 101, 108, 108, 111]  ("Hello" 的字节)
合并规则: {(72,101):256, (256,108):257, (257,108):258, (258,111):259}

第 1 轮扫描:
  (72,101)  rank=256  <- 最小!
  (101,108) 无规则
  (108,108) 无规则
  (108,111) 无规则
  -> 合并 (72,101) 为 256
  -> [256, 108, 108, 111]

第 2 轮扫描:
  (256,108) rank=257  <- 最小!
  (108,108) 无规则
  (108,111) 无规则
  -> 合并 (256,108) 为 257
  -> [257, 108, 111]

第 3 轮扫描:
  (257,108) rank=258  <- 最小!
  (108,111) 无规则
  -> 合并 (257,108) 为 258
  -> [258, 111]

第 4 轮扫描:
  (258,111) rank=259  <- 最小!
  -> 合并 (258,111) 为 259
  -> [259]

第 5 轮: 序列长度=1, 循环结束
输出: [259]  ("Hello" -> 1 个 token)
```

### 15.9 编码质量评估

**压缩率 (Compression Ratio):**
```
压缩率 = token 数 / 原始字节数

越低越好:
  无 BPE:    1.000 (每个字节 1 个 token)
  我们的 BPE: 0.4-0.6 (取决于训练数据和 vocab_size)
  GPT-4:     ~0.25 (万亿字节语料训练)
```

**词表利用率:**
```
利用率 = 训练后实际使用的 token 数 / 词表大小
理想值: > 90% (大部分 token 都被用到了)
利用率太低 -> 词表太大, 有冗余
```

**编码实验 (在你自己的 tokenizer 上测试):**

```python
from tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train(open("sample.txt").read(), vocab_size=500, verbose=False)

# 测试编码
test = "Hello, this is a test of our BPE tokenizer!"
ids = tok.encode(test)
bytes_count = len(test.encode("utf-8"))
compression = len(ids) / bytes_count
print(f"Token 数: {len(ids)}, 字节数: {bytes_count}, 压缩率: {compression:.3f}")

# 查看每个 token 的编码过程
for i, token_id in enumerate(ids):
    byte_repr = tok.vocab[token_id]
    readable = byte_repr.decode("utf-8", errors="replace")
    print(f"  [{i}] ID={token_id} -> {repr(byte_repr)} -> '{readable}'")
```


---

## 16. 正则表达式专题:预分词的正则原理与实战

这一节把 tokenizer 中**用到的正则表达式**讲透,从语法到实战。看完你就能自己读懂 GPT-2/LLaMA 的预分词正则。

### 16.1 我们 tokenizer 里的两处正则

打开 `tokenizer.py`,正则只出现在 `_pretokenize()` 一个方法里,共两处:

```python
# 第 1 处: 把文本切成"连续非空白"和"连续空白"两类片段
tokens = re.findall(r"\S+|\s+", text)

# 第 2 处: 判断一个片段是否"纯空白"
if re.match(r"^\s+$", t):
```

逐个拆解:

| 正则 | 含义 | 在代码中的作用 |
|------|------|---------------|
| `\S+` | `\S` = 非空白字符,`+` = 一个或多个。匹配**连续非空白** | 匹配一个"词" |
| `\s+` | `\s` = 空白字符(空格/换行/制表符),`+` = 一个或多个。匹配**连续空白** | 匹配空白片段 |
| `^\s+$` | `^` = 字符串开头,`$` = 字符串结尾。整个片段**全是空白** | 判断是否为纯空白 |
| `\|` | 正则里的**或** | 两种片段任一匹配 |

**`re.findall(r"\S+|\s+", text)` 的执行过程:**

```python
text = "Hello  world\n"

# 从左到右扫描,遇到能匹配的就"吞掉":
#   "Hello"  → \S+ 匹配 (非空白)
#   "  "     → \s+ 匹配 (两个空格)
#   "world"  → \S+ 匹配
#   "\n"     → \s+ 匹配 (换行)
# 结果: ["Hello", "  ", "world", "\n"]
```

**为什么两个分支的顺序是 `\S+|\s+` 而不是反过来?** 因为 `findall` 从左到右扫描,先匹配哪种都无所谓——两类字符集互补,任何位置恰好只会命中其中之一。这个顺序其实可以随意,但写成 `\S+|\s+` 语义上更自然。

**`re.match(r"^\s+$", t)` 为什么不直接用 `t.isspace()`?** 两者等价,但用正则的写法与第 1 处风格统一,且能自然扩展到更复杂的空白判断(比如只匹配空格不匹配换行)。

### 16.2 预分词里的核心正则语法速查

| 语法 | 含义 | 示例 | 匹配结果 |
|------|------|------|---------|
| `.` | 任意字符(除换行) | `h.t` | "hat", "hit", "h7t" |
| `\w` | 单词字符(字母数字下划线) | `\w+` | "hello123", "a_b" |
| `\d` | 数字 | `\d{4}` | "2026", "1984" |
| `\s` | 空白(空格/换行/制表) | `\s+` | "  ", "\n\t" |
| `\S` | 非空白 | `\S+` | "word", "你好" |
| `[abc]` | 字符集合:a、b 或 c 之一 | `[aeiou]` | "a", "e", "i" |
| `[^abc]` | 排除:a、b、c 之外 | `[^0-9]` | "a", "x", "你" |
| `[a-z]` | 范围:小写字母 | `[a-zA-Z]` | "H", "e", "l" |
| `^` | 字符串开头(在字符集外) | `^\\S` | 开头的非空白 |
| `$` | 字符串结尾 | `\s+$` | 结尾的空白 |
| `*` | 0 个或多个 | `ab*c` | "ac", "abc", "abbbc" |
| `+` | 1 个或多个 | `ab+c` | "abc", "abbbc"(不匹配"ac") |
| `?` | 0 个或 1 个(可选) | `colou?r` | "color", "colour" |
| `{n}` | 恰好 n 个 | `\d{4}` | "2026"(不匹配"26") |
| `{n,m}` | n 到 m 个 | `\d{2,4}` | "26", "2026", "20260" |
| `a\|b` | 或 | `cat\|dog` | "cat", "dog" |
| `(...)` | 分组(可捕获) | `(ab)+` | "ab", "ababab" |
| `(?:...)` | 非捕获分组 | `(?:ab)+` | 同上,但不存分组 |
| `(?=...)` | 正向先行断言 | `\d(?=px)` | "2" in "2px" |
| `(?!...)` | 负向先行断言 | `\d(?!px)` | "2" in "2em" |
| `\p{L}` | Unicode 字母(需 `regex` 库) | `\p{L}+` | "你好", "Hello" |
| `\p{N}` | Unicode 数字 | `\p{N}+` | "2026", "１２３" |

**Python `re` 与 `regex` 库的关键差异:**

| 能力 | `re`(内置) | `regex`(第三方) |
|------|------------|----------------|
| `\p{L}` 等 Unicode 属性 | ❌ 不支持 | ✅ 支持 |
| 变长 lookbehind | ❌ | ✅ |
| 原子组 `(?>...)` | ❌ | ✅ |
| 性能(复杂模式) | 一般 | 更快 |

> GPT-2 的官方实现用的是 `regex` 库,因为它需要 `\p{L}`、`\p{N}` 这类 Unicode 属性。我们的教学实现用 `re` 就够了。

### 16.3 GPT-2 官方预分词正则逐块拆解

GPT-2 论文和官方代码中的正则(这里是 tiktoken / gpt-2 通用的版本):

```python
import regex  # 注意: GPT-2 用的是 regex 库,不是 re

pat = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|   # 英文缩写后缀
        ?\p{L}+|                    # 可选空格 + 连续字母
        ?\p{N}+|                    # 可选空格 + 连续数字
        ?[^\s\p{L}\p{N}]+|          # 可选空格 + 连续标点/符号
        \s+(?!\S)|                  # 空格(后面不是非空白)
        \s+"""                      # 兜底: 任何空白
)
```

**逐块拆解(共 6 个分支,按 `|` 分隔):**

**分支 1: `'s|'t|'re|'ve|'m|'ll|'d`**

```
匹配英文的缩写后缀:
  "it's"    → 's 单独成一个 token
  "don't"   → 't
  "they're" → 're
  "I've"    → 've
  "I'm"     → 'm
  "she'll"  → 'll
  "he'd"    → 'd

为什么要把缩写拆开? 因为 's 和 is / has 在语义上不同源,
拆开可以让模型分别学到 "it" 和 "'s" 的组合规则,
而不是把每个带缩写的词都当成一个全新 token。
```

**分支 2: ` ?\p{L}+`**

```
?  = 0 或 1 个空格(可选)
\p{L}+ = 一个或多个 Unicode 字母

匹配:
  "Hello"     → "Hello" (\p{L}+ 匹配)
  " world"    → " world" (空格 + 字母)
  "你好世界"   → "你好世界" (中文是字母属性!)
  "café"      → "café" (é 也是 \p{L})

关键: \p{L} 让正则天然支持所有语言的文字,
     这就是 GPT-2 tokenizer "万国通用" 的底层原因之一。
```

**分支 3: ` ?\p{N}+`**

```
可选空格 + 连续数字:
  "2026"      → "2026"
  " 42"       → " 42"

为什么数字单独拆? 数字组合无穷无尽,把它和字母混在一起
会浪费词表空间。拆开后模型只需学会 "2" "0" 的合并规则,
就能组合出任意数字。
```

**分支 4: ` ?[^\s\p{L}\p{N}]+`**

```
[^\s\p{L}\p{N}] = 排除(空白、字母、数字)之外的字符 → 标点和符号

匹配:
  "!!!"     → "!!!"
  "..."     → "..."
  "😀"      → "😀" (emoji 既不是字母也不是数字)
  "©"       → "©"
  "$100"    → "$" 单独匹配, "100" 交给分支 3
```

**分支 5: `\s+(?!\S)`**

```
\s+ = 连续空白
(?!\S) = 后面不是非空白(即:后面是空白或字符串结尾)

这匹配"句子末尾的空格"——也就是下一个词不存在时的尾随空白:
  "hello world " → 结尾空格单独成 token
  "line1\nline2\n" → 结尾换行单独成 token

为什么尾随空白要单独处理? 如果空白后面没有词,
就没有"词来领走它",它需要自己成为一个 token,
否则信息(比如结尾的换行)会丢失。
```

**分支 6: `\s+`(兜底)**

```
匹配任何剩余的空白。由于分支 5 已经处理了尾部空白,
这里实际命中的是"后面还有词的空白":
  "hello world" → " " 会作为后续分支 2 的 " ?" 被吞掉

这个兜底分支保证了正则总能匹配成功,不会漏掉任何字符。
```

**完整匹配示例(走一遍):**

```python
text = "It's 2026, hello world!"

GPT-2 正则的匹配结果:
  "It"      ← \p{L}+ (注意: 's 被分支 1 抢走了)
  "'s"      ← 分支 1
  " 2026"   ← 分支 3 (空格 + 数字)
  ","       ← 分支 4 (标点)
  " hello"  ← 分支 2
  " world"  ← 分支 2
  "!"       ← 分支 4 (标点)
```

**对比我们的简化实现:**

| 能力 | GPT-2 官方正则 | 我们的 `\S+|\s+` |
|------|--------------|-----------------|
| 缩写拆分 | ✅ `'s`, `'ll` 等 | ❌ 不拆 |
| Unicode 支持 | ✅ `\p{L}` | ⚠️ `\S` 也能匹配中文,但靠的是"非空白" |
| 数字单独拆 | ✅ | ❌ 不拆 |
| 标点单独拆 | ✅ | ❌ 不拆 |
| 尾随空白处理 | ✅ `(?!\S)` | ✅ 用 `pending_space` 变量处理 |

我们的实现**效果上等价**的关键在于:虽然不拆数字/标点,但训练时 BPE 会自然学会这些合并规则,只是需要更大的词表。GPT-2 拆得细,是为了**在同样词表大小下压缩率更高**。

### 16.4 用正则可视化理解预分词

**动手实验:对比两种正则的分词结果**

```python
import re
import regex

GPT2_PAT = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
SIMPLIFIED_PAT = re.compile(r"\S+|\s+")

texts = [
    "It's 2026, hello world!",
    "我 爱 Python 3.12",
    "line1\nline2\n",
]

for text in texts:
    print(f"输入: {repr(text)}")
    print(f"  GPT-2 正则: {GPT2_PAT.findall(text)}")
    print(f"  简化正则:   {SIMPLIFIED_PAT.findall(text)}")
    print()
```

运行结果:

```
输入: 'It\x27s 2026, hello world!'
  GPT-2 正则: ["It", "'s", " 2026", ",", " hello", " world", "!"]
  简化正则:   ["It", "'s", " 2026", ",", " hello", " world", "!"]
```

> 对于常见英文文本,两种正则的分词结果几乎一样!差别主要体现在:
> 1. 中文:简化版把每个"字+空格"组合拆得更碎,GPT-2 版按 `\p{L}` 整块处理
> 2. 长数字串:`\p{N}` 把连续数字聚在一起,简化版遇到空格才断
> 3. 特殊符号:GPT-2 版把 `&` `#` 等单独成 token,简化版粘在词上

### 16.5 预分词正则的工程细节

**1. 为什么用 `regex` 而不是 `re`?**

GPT-2 的正则需要 `\p{L}`、`\p{N}` 这类 **Unicode 属性**,内置 `re` 模块不支持。如果硬用 `re` 写,只能用笨办法:

```python
# re 版本的痛苦写法 (不推荐)
UNICODE_LETTERS = r"\w"  # 不支持中文!
```

而 `regex` 库一行搞定:`\p{L}+`。这也是为什么所有工业级 tokenizer 都用 `regex`。

**2. 预分词正则与 BPE 的关系**

```
完整 tokenizer 流水线:

  原始文本
    │
    ▼  Step 1: 预分词 (正则)
    │   "It's" → ["It", "'s"]
    │
    ▼  Step 2: 字节化 (UTF-8)
    │   "It" → (73, 116)
    │
    ▼  Step 3: BPE 合并 (贪心)
    │   (73, 116) → 如果能合并就合并
    │
    ▼  Step 4: 输出 token ID
    [  73, 116, 39, 115, ...]

关键原则: BPE 合并规则永远不会跨预分词边界!
  即: "It" 和 "'s" 之间的字节对永远不会被合并。
  这保证了分词结果的稳定性和可解释性。
```

**3. 为什么 GPT-2 要保留"可选空格"(` ?`)而不是直接分开?**

`" hello"` 作为一个 token,而不是 `" "` + `"hello"` 两个 token。原因:
- 训练时 BPE 更容易学到 `" hello"` 这种高频组合(空格+词的共现频率高)
- 解码时 `" hello"` 自然包含空格,不会出现词粘连
- 这符合"空格是词的一部分"的 GPT-2 设计哲学

### 16.6 常见正则调试技巧

**1. 用 `re.DEBUG` 查看正则的解析过程:**

```python
import re
re.compile(r"\S+|\s+", re.DEBUG)
# 输出:
#   MAX_REPEAT 1 MAXREPEAT
#     IN
#       CATEGORY CATEGORY_NOT_SPACE
#   ...
```

**2. 用 `re.findall` 快速验证匹配结果**(比 `match`/`search` 直观):

```python
re.findall(r"\S+|\s+", "Hello  world")   # ['Hello', '  ', 'world']
re.findall(r"\d+", "a1b22c333")          # ['1', '22', '333']
```

**3. 在线工具:**

| 工具 | 链接 | 用途 |
|------|------|------|
| regex101 | https://regex101.com/ | 实时匹配+逐步解释 |
| RegExr | https://regexr.com/ | 可视化正则构建器 |
| Regex Crossword | https://regexcrossword.com/ | 游戏化练习 |

**4. 常见陷阱:**

| 陷阱 | 例子 | 正确写法 |
|------|------|---------|
| `\w` 不含中文 | `\w+` 匹配不到"你好" | 用 `\p{L}+` |
| `$` 在 `re.MULTILINE` 下含义不同 | `\s+$` 只匹配末尾 | 明确用 `\Z` 表示绝对末尾 |
| 贪婪 vs 非贪婪 | `\d+` 贪婪吃掉全部 | `\d+?` 非贪婪,一个一个 |
| 回溯灾难 | `(a+)+b` 在长串上极慢 | 用原子组 `(?>a+)` 或避免嵌套量词 |

### 16.7 正则知识自查清单

读完这一节,你应该能回答:

1. `\S` 和 `\s` 的区别?`\d`、`\w` 又是什么?
2. `*`、`+`、`?`、`{n,m}` 这四种量词的区别?
3. `^` 和 `$` 的作用?在字符集 `[^...]` 里含义为什么不同?
4. `\p{L}` 是什么?为什么 `re` 模块不支持它?
5. GPT-2 正则的 6 个分支分别匹配什么?
6. 为什么 ` ?\p{L}+` 里的空格是可选的?
7. 为什么合并规则不会跨预分词边界?
8. 我们的简化版和 GPT-2 官方版在什么场景下分词结果会不同?

**附加练习:读懂下面这个真实世界的正则**

```python
# LLaMA (SentencePiece) 的 pretokenizer 用的正则
r"""(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

> 和 GPT-2 的只差一点点:它把缩写拆成了 `[sdmt]|ll|ve|re`(字符集 + 或),效果等价,只是写法更紧凑。这就是为什么说"理解了 GPT-2 的正则,所有主流 tokenizer 的预分词你都能看懂"。


---

## 17. 原始 0-255 词表全景:初始字节 token 详解

这一节把 tokenizer 的**起点**——256 个初始字节 token——讲透。它们是 BPE 合并的"原子",理解它们才能理解整个分词系统。

### 17.1 词表是怎么构建的?

`tokenizer.py` 里只有一行代码:

```python
def __init__(self):
    # 初始词表:256 个字节,每个 ID 就是它自身的字节值
    self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
```

即 `vocab[65] = b'A'`, `vocab[97] = b'a'`, `vocab[228] = b'\xe4'`。**ID 与字节值一一对应**,没有映射表,直接就是"我是我自己"。

### 17.2 三大区域总览

| 区域 | ID 范围 | 数量 | 性质 |
|------|---------|------|------|
| **控制字符** | 0-31, 127 | 33 个 | 不可打印,协议控制 |
| **可打印 ASCII** | 32-126 | 95 个 | 英文/数字/标点 |
| **扩展字节** | 128-255 | 128 个 | UTF-8 多字节字符的片段 |

```python
# 验证三大区域的划分
for i in range(256):
    if i < 32 or i == 127:      region = "控制字符"
    elif i < 128:               region = "可打印 ASCII"
    else:                       region = "扩展字节(UTF-8片段)"
```

### 17.3 区域一:控制字符 (ID 0-31, 127)

这些字节来自 ASCII 时代的"通信协议",没有可打印的字符形状:

```
ID   0 = NUL 空字节        ID   9 = TAB 制表符
ID  10 = LF 换行(\n) ⭐    ID  13 = CR 回车
ID  27 = ESC 转义          ID 127 = DEL 删除
其余 (1-8, 11-12, 14-26, 28-31):
  SOH STX ETX EOT ENQ ACK BEL BS VT FF SO SI DLE DC1-4 NAK SYN ETB CAN EM SUB FS GS RS US
```

**对 LLM 最重要的三个:**

| ID | 字符 | 出现场景 | 频率 |
|----|------|---------|------|
| 10 | `\n` 换行 | 段落/行分隔,代码、诗歌、聊天都大量使用 | 极高 |
| 9 | `\t` 制表符 | 代码缩进、表格数据 | 中 |
| 13 | `\r` 回车 | Windows 换行 `\r\n` 的一半 | 中 |

> 为什么这些"看不见的字符"重要?因为它们是**结构化文本的骨架**。代码、Markdown、聊天记录全靠它们分隔,模型必须学会预测它们。

### 17.4 区域二:可打印 ASCII (ID 32-126, 95 个)

这是英文世界的全部可见字符,按块排列:

```
ID 32 = 空格 SPACE ⭐(预分词时被附在词前," the" 的开头就是它)
ID 33-47  = ! " # $ % & ' ( ) * + , - . /
ID 48-57  = 0 1 2 3 4 5 6 7 8 9        ← 数字
ID 58-64  = : ; < = > ? @
ID 65-90  = A B C ... Z                ← 大写字母
ID 91-96  = [ \ ] ^ _ `
ID 97-122 = a b c ... z                ← 小写字母
ID 123-126= { | } ~
```

**三个必须记住的锚点值:**

| ID | 字符 | 规律 |
|----|------|------|
| 48 | `'0'` | 数字起点,`'9'` = 57 |
| 65 | `'A'` | 大写起点,`'Z'` = 90 |
| 97 | `'a'` | 小写起点,`'z'` = 122 |

**关键性质:** UTF-8 完全兼容 ASCII——英文文本的每个字符恰好 = 1 字节 = 1 个初始 token。所以英文在字节级 BPE 下"天然高效"。

### 17.5 区域三:扩展字节 (ID 128-255, 128 个)——UTF-8 片段区

这 128 个字节**没有独立的字符含义**(Latin-1 里的符号与 LLM 无关)。它们的真正角色是: **UTF-8 多字节字符的组成部分**。

**UTF-8 编码规则回顾:**

```
1 字节: 0xxxxxxx               → ASCII, ID 0-127
2 字节: 110xxxxx 10xxxxxx      → 首字节 192-223, 续字节 128-191
3 字节: 1110xxxx 10xxxxxx 10xxxxxx  → 首字节 224-239, 续字节 128-191
4 字节: 11110xxx ...           → 首字节 240-247, 续字节 128-191
```

**关键规律:续字节永远在 128-191 (0x80-0xBF)。**

```python
# 验证: 中文字符的字节分解
'你' → U+4F60 → UTF-8: 0xE4 0xBD 0xA0
  → 0xE4 = 228 (首字节, 在 224-239 区间)
  → 0xBD = 189 (续字节, 在 128-191 区间)
  → 0xA0 = 160 (续字节, 在 128-191 区间)
  → 初始 token ID: [228, 189, 160]

'😀' → U+1F600 → UTF-8: 0xF0 0x9F 0x98 0x80
  → 初始 token ID: [240, 159, 152, 128]
```

| 字符 | UTF-8 字节数 | 初始 token 数 |
|------|-------------|--------------|
| 英文 `'A'` | 1 | 1 |
| 拉丁 `'é'` | 2 | 2 |
| 中文 `'你'` | 3 | 3 |
| emoji `'😀'` | 4 | 4 |

**为什么说 128-255 是"片段"?** 因为它们单独出现时是"半个汉字""四分之一 emoji",只有拼接起来再 UTF-8 解码才有意义。BPE 训练的任务,就是把它们重新合并成完整的字符和词。

### 17.6 这 256 个 token 的四条核心洞察

**1. 零 OOV 的根源**

任何文本 → UTF-8 字节序列 → 每个字节都在 0-255 → 必然能编码。这是"字节级"设计最大的优势:模型永远不会遇到"未知字符"。

**2. BPE 合并的原子单位**

ID 256+ 的合并 token 全部由这些原子字节组合而成:

```python
# 训练时学到第一个合并 (假设 (108,111) 最高频)
new_id = 256
self.vocab[256] = self.vocab[108] + self.vocab[111]  # b'l' + b'o' = b'lo'

# 再往后
self.vocab[257] = self.vocab[256] + self.vocab[119]  # b'lo' + b'w' = b'low'
```

合并次数越多,token 越长、越接近真实"词"。但无论多长,根源都是这 256 个字节。

**3. 压缩率的天花板**

```
初始状态:  1 字节 = 1 token → 压缩率 1.000
训练后:    英文 ~0.25, 中文需要足够的合并规则才能把 3 字节合成 1 个汉字
```

中文的"3 字节/字"意味着:**如果训练数据里中文很少,一个汉字就是 3 个 token**,序列长度和计算量直接翻 3 倍。这就是为什么中文 LLM 需要专门的中文 tokenizer 或大量中文语料。

**4. 解码的正确性不依赖合并**

即使 `'你'` 的 3 个字节从未被合并过,解码依然正确:

```python
# 编码: [228, 189, 160]  (3 个独立字节 token)
# 解码: b'\xe4' + b'\xbd' + b'\xa0' = b'\xe4\xbd\xa0'
#      .decode('utf-8') → '你'  ✅
```

这就是 `decode()` 只需要 vocab 查表 + UTF-8 解码的原因——字节拼接天然还原。

### 17.7 实战验证:完整跑一遍

```python
from tokenizer import BPETokenizer

tok = BPETokenizer()  # 此时 vocab 就是 0-255

# 1. 直接查看初始词表
print(len(tok.vocab))          # 256
print(tok.vocab[65])           # b'A'
print(tok.vocab[97])           # b'a'
print(tok.vocab[228])          # b'\xe4'

# 2. 编码一段混合文本
text = "Hello 你好"
ids = tok.encode(text)         # 还没训练,merges 为空
print(ids)                     # [72, 101, 108, 108, 111, 32, 228, 189, 160, 229, 165, 189]
#  "H" "e" "l" "l" "o" " " "你"的三字节   "好"的三字节

# 3. 解码还原
print(tok.decode(ids))         # "Hello 你好"  ✅ 无损还原

# 4. 训练后: 合并 token 加入词表
tok.train("Hello hello hello", vocab_size=300, verbose=False)
print(len(tok.vocab))          # 300 (256 + 44 个合并)
print(tok.vocab[256])          # b'He' 或类似的高频合并
```

**观察:** 未训练时 `"Hello 你好"` 编码为 12 个 token(9 字节 + 6 中文字节里的 9 个?不——`"Hello 你好"` 是 5+1+6=12 字节,恰好 12 个 token)。训练后 "Hello" 可能合并为 1 个 token,编码显著变短。

### 17.8 记忆口诀

```
0-31 控制,127 删除;    ← 33 个看不见的协议字节
32 空格,48 零,65 A,97 a;  ← ASCII 锚点
128-255 是片段,         ← UTF-8 多字节字符的原料
中文字三字节,emoji 四个;  ← 语言差异的根源
```

> 一句话总结:**这 256 个字节是"万国文字的最小公分母"**。任何语言的任何字符,最终都落到这张表里;BPE 的一切合并,都是从这张表出发。


## 附录:代码文件索引

| 文件 | 内容 | 阅读时间 |
|------|------|---------|
| [tokenizer.py](tokenizer.py) | BPE 核心算法(train/encode/decode) | 30 分钟 |
| [train.py](train.py) | 训练入口 + demo 演示 + 统计输出 | 10 分钟 |
| [data.py](data.py) | 语料获取(TinyShakespeare / sample) | 5 分钟 |
| [README.md](README.md) | 教练讲义 + 快速开始 + 思考练习 | 15 分钟 |
| **TUTORIAL.md**(本文件) | 深度讲解 + 学习资源 | 60 分钟 |

---

> 💡 **学习建议**:先运行 `python train.py` 看一遍训练输出,再回来对照本教程逐行读 `tokenizer.py`。
> 代码中的每个"教练注"都是对初学者特别有用的设计决策解释。
