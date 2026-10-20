"""
information_theory.py -- Information Theory for Tokenization

Covers the mathematical foundations behind tokenization efficiency:
  1. Shannon Entropy -- the theoretical minimum bits per character
  2. Cross Entropy -- measuring prediction quality
  3. KL Divergence -- measuring distribution divergence
  4. Tokenization Efficiency -- why BPE works
  5. Huffman Coding -- the ancestor of BPE
  6. Arithmetic Coding -- the theoretical optimum

Run: python information_theory.py
"""

import math
from collections import Counter
from typing import List, Dict, Tuple


# ============================================================
# 1. Shannon Entropy -- H(X) = -sum(p(x) * log2(p(x)))
# ============================================================
# 信息论之父 Shannon (1948) 提出:
#   - 信息量 = -log2(p)  (概率越小, 信息量越大)
#   - 熵 H(X) = 期望信息量 = -sum(p(x) * log2(p(x)))
#
# 为什么和 Tokenizer 有关?
#   - 这里算的是"一元字符熵": 把每个字符当作独立抽取, 每个字符平均最少需要多少 bit
#   - 它只是"逐字符、互相独立地编码"这种做法的下界 (Huffman 能逼近它)
#   - 它不是 tokenizer 的下界: tokenizer 把常见的多字符组合当成一个符号, 利用了上下文,
#     完全可以低于它 (第 4 节实测会看到效率 > 1)。真正的下界是文本的熵率, 要用语言模型去估

def shannon_entropy(text: str) -> float:
    """
    计算文本的 Shannon 熵 (bits per character).

    H(X) = -sum(p(x) * log2(p(x)))

    Args:
        text: 输入文本

    Returns:
        熵值 (bits per character)
    """
    if not text:
        return 0.0

    # 统计每个字符的频率
    freq = Counter(text)
    total = len(text)

    # 计算熵
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def demo_entropy():
    """演示 Shannon Entropy"""
    print("=" * 60)
    print("1. Shannon Entropy")
    print("=" * 60)
    print("   H(X) = -sum(p(x) * log2(p(x)))")
    print("   单位: bits per character (bpc)")
    print()

    examples = [
        ("a", "单字符 (零熵)"),
        ("aaabbb", "低熵 (低随机性)"),
        ("abcdefghij", "高熵 (均匀分布)"),
        ("the quick brown fox jumps over the lazy dog", "英文句子"),
        ("Hello, how are you? 你好世界!", "中英混合"),
    ]

    for text, desc in examples:
        h = shannon_entropy(text)
        n_chars = len(text)
        n_unique = len(set(text))
        n_bytes = len(text.encode("utf-8"))
        print(f"  {desc}:")
        print(f"    Text: {repr(text)}")
        print(f"    Chars: {n_chars}, Unique: {n_unique}")
        print(f"    Bytes (UTF-8): {n_bytes}")
        print(f"    Entropy: {h:.3f} bpc")
        print(f"    Theoretical min bits: {h * n_chars:.1f}")
        print()


# ============================================================
# 2. Cross Entropy -- H(p, q) = -sum(p(x) * log2(q(x)))
# ============================================================
# 衡量模型预测分布 q 与真实分布 p 的差距:
#   - H(p, q) >= H(p)  (等号成立当且仅当 p == q)
#   - H(p, q) - H(p) = KL(p || q)  (KL 散度)
#
# 为什么和 LLM 有关?
#   - LLM 训练目标 = 最小化 next-token prediction 的交叉熵
#   - 因此: 训练损失 (loss) = 交叉熵 = 模型的 "困惑度" 指标

def cross_entropy(p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
    """
    计算交叉熵 H(p, q) = -sum(p(x) * log2(q(x))).

    Args:
        p_dist: 真实分布 {symbol: probability}
        q_dist: 模型预测分布 {symbol: probability}

    Returns:
        交叉熵值 (bits)
    """
    h = 0.0
    for symbol, p in p_dist.items():
        q = q_dist.get(symbol, 1e-10)  # 避免 log(0)
        if p > 0:
            h -= p * math.log2(q)
    return h


def perplexity(cross_ent: float) -> float:
    """
    困惑度 = 2^(交叉熵).

    直觉: 模型在每一步预测时平均有多少个等可能的选择.
    越低越好: 1.0 = 完美预测, 无穷大 = 完全随机.

    Args:
        cross_ent: 交叉熵值

    Returns:
        困惑度
    """
    return 2 ** cross_ent


def demo_cross_entropy():
    """演示 Cross Entropy 和 Perplexity"""
    print("=" * 60)
    print("2. Cross Entropy & Perplexity")
    print("=" * 60)
    print("   H(p, q) = -sum(p(x) * log2(q(x)))")
    print("   Perplexity = 2^H(p,q)")
    print()

    # 场景: 英文文本的 next-character prediction
    # 真实分布: e 出现频率最高, z 最低
    p_dist = {"e": 0.13, "t": 0.09, "a": 0.08, "o": 0.07, "i": 0.07,
              "n": 0.07, "s": 0.06, "h": 0.06, "r": 0.06, "_other": 0.31}

    # 模型 A: 很好的预测 (接近真实分布)
    q_good = {"e": 0.12, "t": 0.10, "a": 0.09, "o": 0.08, "i": 0.07,
              "n": 0.07, "s": 0.06, "h": 0.06, "r": 0.05, "_other": 0.30}

    # 模型 B: 很差的预测 (均匀分布, 没学到规律)
    q_bad = {k: 0.1 for k in p_dist.keys()}

    h_p = shannon_entropy("the quick brown fox")
    h_good = cross_entropy(p_dist, q_good)
    h_bad = cross_entropy(p_dist, q_bad)

    print(f"  True entropy H(p):     {h_p:.3f} bits")
    print(f"  Good model H(p,q):     {h_good:.3f} bits, perplexity={perplexity(h_good):.2f}")
    print(f"  Bad model H(p,q):      {h_bad:.3f} bits, perplexity={perplexity(h_bad):.2f}")
    print(f"  KL divergence (good):  {h_good - h_p:.3f} bits")
    print(f"  KL divergence (bad):   {h_bad - h_p:.3f} bits")
    print(f"\n  Key insight: Good model is close to theoretical minimum!")


# ============================================================
# 3. KL Divergence -- D_KL(p || q) = sum(p(x) * log(p(x)/q(x)))
# ============================================================
# KL 散度衡量两个分布的 "距离":
#   - D_KL(p || q) >= 0  (等号成立当且仅当 p == q)
#   - 不对称: D_KL(p || q) != D_KL(q || p)
#   - 无界: 可以是无穷大
#
# 在 Tokenizer 中的应用:
#   - 衡量 tokenizer 编码后的分布 vs 理想分布
#   - 不同 tokenizer 产生不同分布, KL 散度可比较

def kl_divergence(p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
    """
    计算 KL 散度 D_KL(p || q).

    Args:
        p_dist: 分布 p
        q_dist: 分布 q

    Returns:
        KL 散度值 (bits)
    """
    dkl = 0.0
    for symbol, p in p_dist.items():
        q = q_dist.get(symbol, 1e-10)
        if p > 0:
            dkl += p * math.log2(p / q)
    return dkl


def demo_kl_divergence():
    """演示 KL Divergence"""
    print("\n" + "=" * 60)
    print("3. KL Divergence")
    print("=" * 60)
    print("   D_KL(p || q) = sum(p(x) * log(p(x)/q(x)))")
    print()

    # 分布 p: 英文字母频率
    p = {"a": 0.08, "b": 0.01, "c": 0.03, "d": 0.04, "e": 0.13,
         "f": 0.02, "g": 0.02, "h": 0.06, "other": 0.61}

    # 分布 q1: 接近 p
    q1 = {"a": 0.09, "b": 0.02, "c": 0.03, "d": 0.05, "e": 0.12,
          "f": 0.02, "g": 0.03, "h": 0.05, "other": 0.59}

    # 分布 q2: 均匀分布
    q2 = {k: 1/len(p) for k in p}

    dkl_1 = kl_divergence(p, q1)
    dkl_2 = kl_divergence(p, q2)

    print(f"  D_KL(p || q1) = {dkl_1:.4f} bits  (close distributions)")
    print(f"  D_KL(p || q2) = {dkl_2:.4f} bits  (uniform distribution)")
    print(f"\n  D_KL is asymmetric: D_KL(p||q) != D_KL(q||p)")
    print(f"  D_KL(p||q) = 0 iff p == q")


# ============================================================
# 4. Tokenization Efficiency -- why BPE works
# ============================================================
# 核心思想: 自然语言的字符分布是高度不均匀的
#   - 高频字符/字节对 -> 合并为一个 token -> 编码更短
#   - 低频字符/字节对 -> 保持独立 -> 不浪费编码空间
#
# 这和 Huffman 编码的思想完全一致:
#   - 高频符号用短码
#   - 低频符号用长码
#   - 期望码长 = 熵 (最优)
#
# BPE 的 "贪心" 为什么不差?
#   - Huffman 编码对单个符号是最优的
#   - BPE 对 "pair" 做合并, 是 Huffman 编码的推广
#   - 合并高频 pair 等价于: 给高频组合分配更短的码

def tokenization_efficiency(text: str, token_counts: Dict[str, int]) -> float:
    """
    计算 tokenizer 的编码效率.

    效率 = 逐字符编码的下界 / 实际比特数
    逐字符编码的下界 = 一元字符熵 (bit/字符) * 字符数   (注意单位: 乘字符数, 不是字节数)

    Args:
        text: 原始文本
        token_counts: {token: count} tokenizer 输出的 token 统计

    Returns:
        效率; 可以大于 1 (说明 tokenizer 利用了字符之间的相关性, 见 demo_tokenization_efficiency)
    """
    entropy = shannon_entropy(text)
    theoretical_bits = entropy * len(text)

    # 实际比特数: 每个 token 至少需要 log2(vocab_size) bit
    n_tokens = sum(token_counts.values())
    vocab_size = len(token_counts)
    bits_per_token = math.log2(vocab_size) if vocab_size > 1 else 1
    actual_bits = n_tokens * bits_per_token

    return theoretical_bits / actual_bits if actual_bits > 0 else 0


def demo_tokenization_efficiency():
    """演示 Tokenization Efficiency (全部用真实分词器实测, 不用估算系数)"""
    print("\n" + "=" * 60)
    print("4. Tokenization Efficiency")
    print("=" * 60)
    print("   Efficiency = (一元字符熵 x 字符数) / (token 数 x log2(词表大小))")
    print("   分母: 每个 token 用定长 ID 存储时需要的比特数")
    print()

    import os
    from data import get_corpus, get_train_val_split
    from tokenizer import BPETokenizer

    _, val = get_train_val_split(get_corpus("shakespeare", verbose=False), 0.1)
    text = val[:10000]                      # 与 train.py 的压缩率统计用同一段
    h = shannon_entropy(text)
    theoretical_min = h * len(text)

    rows = [("Char-level", len(text), len(set(text)))]
    words = text.split()
    # 词级的词表只算了这段文本里出现过的词, 偏小; 真实的词级分词器还要面对 OOV
    rows.append(("Word-level*", len(words), len(set(words))))

    here = os.path.dirname(os.path.abspath(__file__))
    ours = BPETokenizer.load(os.path.join(here, "tokenizer.json"))
    rows.append(("Our BPE (500)", len(ours.encode(text)), len(ours.vocab)))
    try:
        from tokenizers import Tokenizer
        gpt2 = Tokenizer.from_pretrained("openai-community/gpt2")
        rows.append(("GPT-2 (50257)", len(gpt2.encode(text).ids), gpt2.get_vocab_size()))
    except Exception as e:  # 离线时跳过, 不让脚本崩掉
        print(f"  [SKIP] GPT-2 tokenizer 无法加载: {str(e)[:60]}")

    print(f"  Text: 莎士比亚验证集前 {len(text)} 字符")
    print(f"  一元字符熵: {h:.3f} bit/字符  ->  逐字符编码下界 {theoretical_min:.0f} bits")
    print()
    print(f"  {'Method':<15} {'Tokens':<10} {'Bits/token':<12} {'Total bits':<12} {'Efficiency':<10}")
    print(f"  {'-'*60}")
    for name, n_tok, vsize in rows:
        bpt = math.log2(vsize) if vsize > 1 else 1
        total = n_tok * bpt
        eff = theoretical_min / total if total > 0 else 0
        print(f"  {name:<15} {n_tok:<10} {bpt:<12.1f} {total:<12.0f} {eff:<10.3f}")
    print()
    print("  效率 > 1 并不违反信息论: 一元字符熵只是'逐字符独立编码'的下界,")
    print("  而 BPE / 词级把常见组合当成一个符号, 利用了字符之间的相关性。")


# ============================================================
# 5. Huffman Coding -- the ancestor of BPE
# ============================================================
# Huffman (1952): 构建最优前缀码
#   - 高频符号 -> 短码 (few bits)
#   - 低频符号 -> 长码 (many bits)
#   - 平均码长 = 熵 (信息论下界)
#
# BPE 和 Huffman 的关系:
#   - Huffman: 优化单个符号的编码长度
#   - BPE: 优化 pair 的编码 (把高频 pair 合并)
#   - BPE = Huffman on pairs

def huffman_codes(freq: Dict[str, int]) -> Dict[str, str]:
    """
    构建 Huffman 编码表.

    Args:
        freq: {symbol: frequency}

    Returns:
        {symbol: binary_code_string}
    """
    # 简化版实现 (用 list 模拟优先队列)
    nodes = [[weight, sym, None, None] for sym, weight in freq.items()]
    nodes.sort()

    while len(nodes) > 1:
        left = nodes.pop(0)
        right = nodes.pop(0)
        merged = [left[0] + right[0], None, left, right]
        nodes.append(merged)
        nodes.sort()

    # 生成编码
    codes = {}

    def traverse(node, code=""):
        if node[1] is not None:  # leaf
            codes[node[1]] = code if code else "0"
            return
        if node[2]:
            traverse(node[2], code + "0")
        if node[3]:
            traverse(node[3], code + "1")

    if nodes:
        traverse(nodes[0])

    return codes


def demo_huffman():
    """演示 Huffman Coding"""
    print("\n" + "=" * 60)
    print("5. Huffman Coding (ancestor of BPE)")
    print("=" * 60)
    print("   High freq -> short code, Low freq -> long code")
    print()

    freq = {"a": 45, "b": 13, "c": 12, "d": 16, "e": 9, "f": 5}
    codes = huffman_codes(freq)

    total_bits = sum(freq[s] * len(codes[s]) for s in freq)
    total_freq = sum(freq.values())
    avg_bits = total_bits / total_freq

    print(f"  Symbol frequencies: {freq}")
    print(f"\n  Huffman codes:")
    for sym in sorted(codes, key=lambda s: freq[s], reverse=True):
        print(f"    '{sym}' (freq={freq[sym]:2d}): {codes[sym]} ({len(codes[sym])} bits)")

    print(f"\n  Average bits per symbol: {avg_bits:.2f}")
    print(f"  Shannon entropy:        {shannon_entropy_from_freq(freq):.2f}")
    print(f"  Gap (Huffman - Entropy): {avg_bits - shannon_entropy_from_freq(freq):.4f}")
    print(f"  Key insight: Huffman is nearly optimal!")


def shannon_entropy_from_freq(freq: Dict[str, int]) -> float:
    """计算基于频率分布的熵"""
    total = sum(freq.values())
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ============================================================
# 6. Arithmetic Coding -- the theoretical optimum
# ============================================================
# 算术编码: 把整条消息映射到 [0,1) 区间的一个数
#   - 理论最优: 编码长度 = 熵
#   - 没有 "每个符号一个码字" 的限制
#   - 但工程复杂, 需要高精度算术
#
# 为什么 tokenizer 不用算术编码?
#   - 算术编码是 "无损压缩", 目标是还原原始字节
#   - Tokenizer 的目标不同: 保留语义, 允许部分信息丢失
#   - Tokenizer 输出的是 "有意义的子词", 不是任意比特流

def arithmetic_coding_demo():
    """演示 Arithmetic Coding 的概念"""
    print("\n" + "=" * 60)
    print("6. Arithmetic Coding (theoretical optimum)")
    print("=" * 60)
    print("   Maps entire message to a single number in [0,1)")
    print()

    # 简化演示: 编码 "ab"
    freq = {"a": 0.6, "b": 0.4}
    message = "aab"

    low, high = 0.0, 1.0
    print(f"  Encoding: {repr(message)}")
    print(f"  Freq: {freq}")
    print()

    for i, symbol in enumerate(message):
        range_size = high - low
        if symbol == "a":
            high = low + range_size * freq["a"]
        else:
            low = low + range_size * freq["a"]
        print(f"  Step {i+1}: symbol='{symbol}', interval=[{low:.6f}, {high:.6f})")

    code = (low + high) / 2
    print(f"\n  Encoded value: {code:.10f}")
    print(f"  Bits needed: {-math.log2(high - low):.2f}")
    print(f"  Entropy of message: {sum(-math.log2(freq[s]) for s in message) / len(message):.2f} bpc")
    print(f"\n  Key insight: Arithmetic coding approaches the entropy limit!")


# ============================================================
# Summary: Connection between Information Theory and Tokenizer
# ============================================================

def print_summary():
    """输出信息论与 Tokenizer 的关系总结"""
    print("\n" + "=" * 60)
    print("SUMMARY: Information Theory x Tokenizer")
    print("=" * 60)
    print("""
  1. Shannon Entropy -> Theoretical compression limit
     - 一元字符熵只是"逐字符独立编码"的下界, 不是 tokenizer 的下界
     - 真正的下界是文本的熵率 (考虑上下文), 需要用语言模型估计

  2. Cross Entropy -> LLM training objective
     - LLM loss = cross entropy of next-token prediction
     - 注意: 不同分词器的 loss 不能直接比, 要换算成 bit/字节 (讲义「压缩率」一节)

  3. KL Divergence -> Comparing tokenizers
     - D_KL(tokenizer_A || tokenizer_B) measures distribution difference
     - Helps choose tokenizer for specific languages/domains

  4. Tokenization Efficiency = Entropy / Actual Bits
     - 实测 (莎士比亚 1 万字符): 字符级 0.82, 我们的 BPE-500 1.07, GPT-2 0.95, 词级 2.7 (词表偏小, 且有 OOV)
     - 大于 1 说明分词利用了字符间的相关性; 词表越大, 每个 ID 的定长比特数也越高

  5. Huffman Coding -> Ancestor of BPE
     - Same principle: high freq = short code
     - BPE extends Huffman to pairs of symbols

  6. Arithmetic Coding -> Theoretical optimum
     - Shows the ultimate limit of compression
     - Tokenizer sacrifices some efficiency for meaningful tokens
""")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    print("Information Theory for Tokenization")
    print("=" * 60)

    demo_entropy()
    demo_cross_entropy()
    demo_kl_divergence()
    demo_tokenization_efficiency()
    demo_huffman()
    arithmetic_coding_demo()
    print_summary()
