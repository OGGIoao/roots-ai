# 模块 L1：BPE Tokenizer 从零实现

> **教练讲义** | CS336 Assignment 1 风格 | 作者：Claw  
> **目标**：理解 LLM 的"文字眼睛"——tokenizer 是如何把文本变成数字的

---

## 🎯 本节你要解决的问题

> *GPT 看到 "Hello" 时，它看到的是什么？*

答：不是字符 H-e-l-l-o，而是一个或几个**整数 ID**。这个从文本到整数列表的映射，就是 **Tokenizer** 做的事。

---

## 📖 核心概念

### 1. 为什么不直接用字符？

| 方案 | 词表大小 | 问题 |
|---|---|---|
| 字符级（char-level） | ~100 | 序列太长，注意力计算量爆炸 |
| 词级（word-level） | ~50,000+ | 遇到新词/拼写错误就 OOV（out-of-vocab） |
| **子词级（subword，BPE）** | ~30,000-100,000 | ✅ 平衡序列长度和词表覆盖率 |

### 2. 为什么从字节开始？

GPT-2 开始流行**字节级 BPE**：

- 初始词表 = 256 个字节（0-255），**永远不会有未知字符**
- 中文、日文、emoji、代码……全都能处理
- 同一个 tokenizer 可以处理任何语言的文本

### 3. BPE 算法（Byte Pair Encoding）

**训练阶段**（给定语料，学习合并规则）：

```
初始状态：把每个词拆成字节序列
  "low"  → [l, o, w]
  "lower" → [l, o, w, e, r]
  "newest" → [n, e, w, e, s, t]

第 1 步：统计所有相邻字节对的频率
  (l, o) 出现 2 次
  (o, w) 出现 2 次
  (n, e) 出现 1 次
  ...

第 2 步：找最高频对 (l, o)，合并为新 token "lo"（ID=256）
  "low"   → [lo, w]
  "lower" → [lo, w, e, r]

第 3 步：继续找最高频对 (lo, w)，合并为 "low"（ID=257）
  ...

重复直到词表大小达到目标（如 500 或 50000）
```

**编码阶段**（把新文本转成 token IDs）：

```
输入："lowest"
字节序列：[l, o, w, e, s, t]
应用 merge #1 (l,o)→256：[256, w, e, s, t]
应用 merge #2 (256,w)→257：[257, e, s, t]
无更多可用 merge：[257, e, s, t]
最终 IDs：[257, 101, 115, 116]
```

**解码阶段**（token IDs 还原成文本）：

```
每个 ID → 字节序列 → 合并所有字节 → UTF-8 解码 → 字符串
```

---

## 📐 核心公式

**压缩率**（越低越好，GPT-4 的 cl100k 大约是 0.25）：

$$\text{compression\_ratio} = \frac{\text{编码后 token 数}}{\text{原始字节数}}$$

**词表增长**：

$$|\text{vocab}| = 256 + \text{num\_merges}$$

---

## 🗂️ 文件结构

```
llm-bpe-tokenizer/
├── README.md          ← 你正在看的教练讲义
├── data.py            ← 语料下载 + 预处理
├── tokenizer.py       ← BPE 核心算法（重点阅读）
└── train.py           ← 训练 + Demo 演示（运行入口）
```

> 依赖统一由仓库根目录 `requirements-mlc.txt` 声明（conda 环境 `mlc`）。
> 核心 BPE（`tokenizer.py` / `train.py`）只用标准库；`modern_tokenizers.py` 的进阶演示按需用
> `tokenizers` / `sentencepiece` / `regex`（已在清单中）和可选的 `tiktoken`，缺失时该演示自动 SKIP。

---

## 🚀 快速开始

```bash
cd projects/llm-bpe-tokenizer

# 运行默认演示（vocab_size=500，在 TinyShakespeare 上训练）
python train.py

# 指定词表大小（更大的词表，更好的压缩率，训练更慢）
python train.py --vocab_size 1000

# 使用内置 sample 文本（无网络环境）
python train.py --source sample --vocab_size 300
```

---

## 🔍 代码导读（建议阅读顺序）

### Step 1：先看 `tokenizer.py` 的核心三个方法

```python
# 1. train() — 从语料学习合并规则
tokenizer.train(text, vocab_size=500)

# 2. encode() — 把文本转成整数列表
ids = tokenizer.encode("Hello, world!")  # → [72, 101, 108, ...]

# 3. decode() — 把整数列表还原成文本
text = tokenizer.decode(ids)  # → "Hello, world!"
```

### Step 2：理解 `_get_pair_freqs()`

这是 BPE 的心脏：统计当前词频表中所有相邻 token 对的出现频率。

```python
# 如果词频表是 { (104, 101, 108): 5, (119, 111): 3 }
# 那么 pair (104, 101) 出现了 5 次，(101, 108) 出现了 5 次……
```

### Step 3：理解 `_encode_chunk()`

编码时，对每个 pre-token 的字节序列反复应用合并规则，直到没有可用的合并为止。

---

## 🧪 思考练习

完成代码后，试着回答这些问题（不用写代码，思考就行）：

1. **vocab_size=300 和 vocab_size=1000** 训练出的 tokenizer，对同一段文字编码后，哪个的 token 数更少？为什么？

2. 如果训练语料是**纯中文**，BPE 会有什么不同表现？每个汉字是几个字节？

3. `"hello"` 和 `" hello"`（前面带空格）在 GPT-2 中是**不同的 token**——为什么要这样设计？

4. 如果训练语料里没有数字，但你想编码 `"2026年"`，tokenizer 会怎么处理？

---

## 📎 参考资料

- Andrej Karpathy 的 [minbpe](https://github.com/karpathy/minbpe)（本节代码的精神参考）
- [CS336 Assignment 1 讲义](https://stanford-cs336.github.io/spring2025/)
- [Wikipedia: Byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)
- GPT-2 论文：[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## 🗡️ 手撕训练板块

> 面试目标：不看任何参考，20 分钟内写出，并能边写边讲。
>
> **3 步法**：① 口述伪代码（~30秒）→ ② 写函数骨架（~2分钟）→ ③ 填实现 + 说复杂度（~15分钟）

---

### 题目 1：手写 BPE 训练主循环

**面试频率**：⭐⭐⭐⭐⭐ LLM 岗位/NLP 岗位高频

**面试官说**：*"解释 BPE 算法并手写训练过程的核心逻辑，不用 HuggingFace。"*

**伪代码**：
```
初始化词表 vocab = {0: b'\x00', ..., 255: b'\xff'}

把语料转成字节ID序列，构建词频表：
  word_freqs = { (72, 101, 108, 108, 111): 5, ... }

重复 (vocab_size - 256) 次：
  1. 统计所有相邻 (id_a, id_b) 的出现次数 → pair_freqs
  2. 找最高频对 best_pair
  3. new_id = 256 + 当前合并轮次
  4. vocab[new_id] = vocab[id_a] + vocab[id_b]
  5. 把 word_freqs 里所有 best_pair 替换为 new_id
```

**代码实现**（核心部分）：
```python
from collections import defaultdict

def bpe_train(word_freqs: dict, vocab_size: int) -> dict:
    """
    Args:
        word_freqs: { token_id_tuple: count }，如 { (72,101,108): 5 }
        vocab_size: 目标词表大小（必须 > 256）
    Returns:
        merges: { (id_a, id_b): new_id }
    """
    merges = {}
    for merge_idx in range(vocab_size - 256):
        # Step 1: 统计相邻对频率
        pair_freqs = defaultdict(int)
        for ids, freq in word_freqs.items():
            for i in range(len(ids) - 1):
                pair_freqs[(ids[i], ids[i+1])] += freq

        if not pair_freqs:
            break

        # Step 2: 找最高频对
        best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
        new_id = 256 + merge_idx
        merges[best_pair] = new_id

        # Step 3: 更新词频表（替换所有出现的 best_pair）
        new_word_freqs = {}
        for ids, freq in word_freqs.items():
            new_ids = merge_pair(ids, best_pair, new_id)
            new_word_freqs[new_ids] = freq
        word_freqs = new_word_freqs

    return merges


def merge_pair(ids: tuple, pair: tuple, new_id: int) -> tuple:
    """把序列中所有出现的 pair 替换为 new_id。"""
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return tuple(result)
```

**复杂度**：时间 O(V·N)，V 为合并次数，N 为语料总 token 数；空间 O(词表大小)

**追问清单**：
- BPE 的初始词表为什么是 256 个字节而不是字符？（答：字节级词表永远不会有 OOV；不同语言/encoding 共用同一套）
- vocab_size=50000 意味着做了多少次合并？（答：50000 - 256 = 49744 次）
- 为什么编码时要按"训练时的合并顺序"应用规则？（答：保证确定性；同一对字节在不同位置都被同样处理）

---

### 题目 2：手写 BPE encode（贪心策略）

**面试频率**：⭐⭐⭐⭐ 通常紧接题目 1 追问

**面试官说**：*"现在你有了 merges 规则，如何把一段新文本编码成 token ID 列表？"*

**伪代码**：
```
# 输入：文本字符串，已有的 merges 规则（按 new_id 排序，越小越早）
文本 → UTF-8 字节 → 整数列表 ids

while ids 长度 >= 2:
    找 ids 中所有满足 merges 规则的相邻对
    选 new_id 最小的那个（= 训练时最早合并的）
    把 ids 中所有该对替换为 new_id

return ids
```

**代码实现**：
```python
def bpe_encode(text: str, merges: dict) -> list:
    """
    Args:
        text:   待编码文本
        merges: { (id_a, id_b): new_id }
    Returns:
        token ID 列表
    """
    ids = list(text.encode("utf-8"))  # str → 字节 → int 列表

    while len(ids) >= 2:
        # 找当前序列中 rank(=new_id) 最小的可用合并
        best_pair, best_rank = None, float("inf")
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            if pair in merges and merges[pair] < best_rank:
                best_pair = pair
                best_rank = merges[pair]

        if best_pair is None:
            break  # 没有可用合并

        ids = list(merge_pair(tuple(ids), best_pair, merges[best_pair]))

    return ids
```

**追问清单**：
- 为什么选"rank 最小"而不是"频率最高"的合并？（答：encode 时没有频率信息；按训练顺序应用保证结果唯一确定）
- encode 的时间复杂度？（答：最坏 O(L²)，L 为字节序列长度；实践中远小于此）
- decode 怎么写？（答：每个 ID → vocab 查字节 → bytes 拼接 → UTF-8 解码，3 行搞定）

---

### 📊 手撕准备度自测

完成本项目后，用以下标准测试自己的掌握程度：

| 测试方式 | 达标标准 | 说明 |
|---|---|---|
| 看代码讲逻辑 | 能解释每一行 | 理解层（Demo 结束后立刻做） |
| 有注释手写 | 30 分钟内写完 | 记忆层（1 周后练） |
| 无提示手写 | 20 分钟内写完 | 面试层（2-3 周后练） |
| 边写边讲解 | 全程解释思路 | 实战层（模拟面试练） |

**建议练习节奏**：学完当天 → 1天后默写 → 1周后再默写 → 3周后模拟面试。

---

*下一模块：L2 — Transformer 组件拆解（Attention、RoPE、RMSNorm）*
