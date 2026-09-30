"""
modern_tokenizers.py -- 主流大模型 Tokenizer 对比实现

涵盖 6 种主流分词方案:
  1. Byte-level BPE (GPT-2/3/4, 我们的实现)
  2. Tiktoken (gpt-5.6-sol/3.5/4, Rust 加速版 BPE)
  3. SentencePiece (LLaMA/Mistral/Qwen, 语言无关)
  4. HuggingFace Tokenizers (多语言, Rust 后端)
  5. WordPiece (BERT, 最早的子词方法)
  6. Byte-level fallback (现代混合方案)

运行: python modern_tokenizers.py
"""

import os
import re
from typing import Optional, List, Dict, Tuple


# ============================================================
# 1. Tiktoken -- gpt-5.6-sol/3.5/4 的分词器
# ============================================================
# 底层是 BPE, 但用 Rust 实现, 速度比纯 Python 快 3-6 倍。
# cl100k_base: gpt-5.6-sol/3.5-turbo 的词表 (100,277 tokens)
# o200k_base:  gpt-4o 的词表 (200,019 tokens)
#
# 与我们 BPE 实现的区别:
#   - 合并规则存储为 rank dict (ID -> rank), 而非 list
#   - 使用正则预分词(比空格分割更复杂)
#   - 编码时从最长匹配开始, 而非贪心合并

def demo_tiktoken():
    """演示 Tiktoken 分词器"""
    try:
        import tiktoken
    except ImportError:
        print("  [SKIP] tiktoken 未安装, 运行: pip install tiktoken")
        print("  GitHub: https://github.com/openai/tiktoken")
        return

    print("=" * 60)
    print("1. Tiktoken (gpt-5.6-sol/3.5/4)")
    print("=" * 60)

    test_text = "Hello, how are you? 你好世界! Hi"
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(test_text)
    print(f"  Text:     {test_text}")
    print(f"  cl100k_base (gpt-5.6-sol/3.5-turbo):")
    print(f"  Token IDs: {ids}")
    print(f"  Token count: {len(ids)}")
    for i, tid in enumerate(ids):
        tok = enc.decode([tid])
        print(f"    [{i:2d}] ID={tid:6d} -> {repr(tok)}")

    # gpt-4o
    try:
        enc4o = tiktoken.get_encoding("o200k_base")
        ids4o = enc4o.encode(test_text)
        print(f"\n  o200k_base (gpt-4o):")
        print(f"  Token IDs: {ids4o}")
        print(f"  Token count: {len(ids4o)}")
    except Exception:
        print("  [SKIP] o200k_base needs newer tiktoken")

    n_bytes = len(test_text.encode("utf-8"))
    print(f"\n  Compression (text = {n_bytes} bytes):")
    print(f"    cl100k_base: {len(ids)} tokens, ratio={len(ids)/n_bytes:.3f}")


# ============================================================
# 2. SentencePiece -- LLaMA/Mistral/Qwen 的分词器
# ============================================================
# Google 开发的语言无关分词器:
#   - 直接在 raw text 上训练, 不做预分词
#   - 空格编码为 special Unicode char
#   - 支持 BPE 和 Unigram 两种算法
#   - 词表含特殊标记 ([BOS], [EOS], [PAD])

def demo_sentencepiece():
    """演示 SentencePiece 分词器"""
    try:
        import sentencepiece as spm
    except ImportError:
        print("  [SKIP] sentencepiece 未安装, 运行: pip install sentencepiece")
        print("  GitHub: https://github.com/google/sentencepiece")
        return

    print("\n" + "=" * 60)
    print("2. SentencePiece (LLaMA/Mistral/Qwen)")
    print("=" * 60)

    # 准备训练数据
    # 语料太薄会直接卡住 SentencePiece: 字符集占 ~292, 但能产出的 merge
    # 只有 ~56, 词表无论开大还是开小都会报错(下限 292 / 上限 56)。
    # 这里的 demo 不需要真实分布, 但需要足够行数把 merge 数量堆上去。
    data_path = "data.txt"
    if not os.path.exists(data_path):
        base_lines = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world this is a test.",
            "Transformer models are amazing.",
            "BPE tokenization is fascinating.",
            "The cat sat on the mat.",
        ]
        with open(data_path, "w", encoding="utf-8") as f:
            for line in base_lines:
                f.write(line + "\n")
                # 滑窗扩写: 重复拼接相邻句, 造出可组合的共享子串
                f.write(line + " " + line + "\n")
                f.write(line.replace(".", ", and more.") + "\n")
            f.write("The cat sat on the mat.\n" * 40)

    print("  Training SentencePiece BPE model ...")
    # byte_fallback 会把 256 个字节也算进必需字符, 加上字母/标点共 ~292,
    # 所以词表不能低于这个数; 反过来又受限于语料能产出的 merge 数量。
    spm.SentencePieceTrainer.train(
        input=data_path,
        model_prefix="spm_bpe",
        vocab_size=300,
        model_type="bpe",
        character_coverage=0.9995,
        num_threads=1,
        split_digits=True,
        byte_fallback=True,
    )

    sp = spm.SentencePieceProcessor(model_file="spm_bpe.model")
    test_texts = [
        "Hello world",
        "The quick brown fox jumps",
        "Transformer models are amazing",
    ]

    print(f"  Vocab size: {sp.get_piece_size()}")
    print(f"  Special: BOS={sp.bos_id()}, EOS={sp.eos_id()}, UNK={sp.unk_id()}")

    for text in test_texts:
        ids = sp.encode_as_ids(text)
        pieces = sp.encode_as_pieces(text)
        print(f"\n  Text: {text}")
        print(f"  IDs:    {ids}")
        print(f"  Pieces: {pieces}")

    # Unigram 对比
    print("\n  --- Unigram mode comparison ---")
    spm.SentencePieceTrainer.train(
        input=data_path,
        model_prefix="spm_unigram",
        vocab_size=300,
        model_type="unigram",
        character_coverage=0.9995,
        num_threads=1,
        byte_fallback=True,
    )
    spu = spm.SentencePieceProcessor(model_file="spm_unigram.model")
    text = "The transformer architecture changed everything"
    print(f"  Text: {text}")
    print(f"  BPE:     {sp.encode_as_pieces(text)}")
    print(f"  Unigram: {spu.encode_as_pieces(text)}")

    # cleanup
    for f in ["spm_bpe.model", "spm_bpe.vocab",
              "spm_unigram.model", "spm_unigram.vocab", data_path]:
        if os.path.exists(f):
            os.remove(f)


# ============================================================
# 3. HuggingFace Tokenizers -- 多语言分词框架
# ============================================================
# 最灵活的分词框架:
#   - 支持 BPE / WordPiece / Unigram 三种算法
#   - Rust 后端, 速度极快
#   - 预训练模型: bert, gpt2, llama, qwen, etc.

def demo_huggingface():
    """演示 HuggingFace Tokenizers"""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  [SKIP] transformers 未安装, 运行: pip install transformers")
        print("  GitHub: https://github.com/huggingface/tokenizers")
        return

    print("\n" + "=" * 60)
    print("3. HuggingFace Tokenizers (multi-lingual)")
    print("=" * 60)

    test_text = "Hello, how are you? 你好世界!"
    models = [
        ("bert-base-uncased", "BERT (WordPiece)"),
        ("bert-base-chinese", "BERT-Chinese (WordPiece)"),
        ("gpt2", "GPT-2 (Byte-level BPE)"),
    ]

    for model_name, display_name in models:
        try:
            tok = AutoTokenizer.from_pretrained(model_name)
            ids = tok.encode(test_text)
            tokens = tok.convert_ids_to_tokens(ids)
            print(f"\n  {display_name}:")
            print(f"    Vocab: {tok.vocab_size}")
            print(f"    Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            print(f"    IDs:    {ids[:20]}{'...' if len(ids) > 20 else ''}")
        except Exception as e:
            print(f"\n  {display_name}: [SKIP] {str(e)[:60]}")


# ============================================================
# 4. WordPiece -- BERT 的分词算法
# ============================================================
# 与 BPE 类似但合并标准不同:
#   - BPE: 合并频率最高的 pair
#   - WordPiece: 合并使语言模型概率提升最大的 pair
#   - 非首子词用 ## 前缀 (如 "playing" -> "play" + "##ing")

def demo_wordpiece():
    """演示 WordPiece"""
    try:
        from transformers import BertTokenizer
    except ImportError:
        print("  [SKIP] transformers 未安装")
        return

    print("\n" + "=" * 60)
    print("4. WordPiece (BERT)")
    print("=" * 60)

    try:
        tok = BertTokenizer.from_pretrained("bert-base-uncased")
    except Exception as e:
        # 离线或代理异常时不该让整个脚本崩掉
        print(f"  [SKIP] 无法加载 bert-base-uncased: {str(e)[:60]}")
        return

    test_texts = ["Hello world", "unbelievable", "tokenization is fascinating"]

    for text in test_texts:
        ids = tok.encode(text, add_special_tokens=True)
        tokens = tok.convert_ids_to_tokens(ids)
        subword = [t for t in tokens if t.startswith("##")]
        print(f"\n  Text: {text}")
        print(f"  Tokens: {tokens}")
        if subword:
            print(f"  Subword fragments: {subword}")


# ============================================================
# 5. GPT-2 Byte-level BPE 核心概念
# ============================================================

def demo_bytelevel_bpe():
    """演示字节级 BPE 核心概念"""
    print("\n" + "=" * 60)
    print("5. Byte-level BPE (GPT-2 innovation)")
    print("=" * 60)

    test_text = "Hello 你好"
    bytes_repr = list(test_text.encode("utf-8"))
    chars_repr = list(test_text)

    print(f"  Text: {test_text}")
    print(f"  Char-level ({len(chars_repr)} items): {chars_repr}")
    print(f"  Byte-level ({len(bytes_repr)} items): {bytes_repr}")
    print(f"\n  GPT-2 innovation: map bytes to printable chars")
    print(f"  Chinese 'you' = 0xE4 0xBD 0xA0 (3 bytes)")
    print(f"  GPT-2 maps to printable chars to avoid control chars")


# ============================================================
# 6. 主流 LLM Tokenizer 速查表
# ============================================================

TOKENIZER_TABLE = """
+------------------+-------------------+-----------+------------------+
| Model            | Algorithm         | Vocab     | Pre-tokenization |
+------------------+-------------------+-----------+------------------+
| GPT-2            | Byte-level BPE    | 50,257    | Regex split      |
| GPT-3.5/4        | Byte-level BPE    | 100,277   | cl100k_base      |
| GPT-4o           | Byte-level BPE    | 200,019   | o200k_base       |
| BERT             | WordPiece         | 30,522    | Whitespace+Punc  |
| LLaMA-1/2/3      | SentencePiece BPE | 32,000    | Raw text         |
| Qwen-2           | Tiktoken BPE      | 151,936   | Regex split      |
| Mistral/Mixtral  | SentencePiece BPE | 32,000    | Raw text         |
| Gemma            | SentencePiece BPE | 256,000   | Raw text         |
| Claude (est.)    | Byte-level BPE    | ~100,000  | Regex split      |
| Phi-2/3          | Tiktoken BPE      | 32,064    | Regex split      |
+------------------+-------------------+-----------+------------------+
"""


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    print("Modern LLM Tokenizer Comparison")
    print("=" * 60)
    print(TOKENIZER_TABLE)

    demo_tiktoken()
    demo_sentencepiece()
    demo_huggingface()
    demo_wordpiece()
    demo_bytelevel_bpe()

    print("\n" + "=" * 60)
    print("Summary:")
    print("  - Byte-level BPE: GPT-2/3/4 family")
    print("  - SentencePiece BPE: LLaMA/Mistral/Qwen/Gemma")
    print("  - WordPiece: BERT family")
    print("  - Tiktoken: OpenAI fast BPE (Rust)")
    print("  - All modern LLMs use subword tokenization")
    print("=" * 60)
