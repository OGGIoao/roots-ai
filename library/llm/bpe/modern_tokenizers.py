"""
modern_tokenizers.py -- 主流分词方案的动手演示 (讲义「主流分词器解剖」一节)

涵盖:
  1. Tiktoken (OpenAI 的 BPE 实现, cl100k_base / o200k_base)      需要 pip install tiktoken (可选)
  2. SentencePiece (BPE 与 Unigram 两种模式, 现场训练一个小模型)
  3. HuggingFace 上的真实分词器: BERT (WordPiece) 与 GPT-2 (字节级 BPE)
  4. WordPiece 的 ## 子词
  5. 字节级 BPE 的核心概念 (与本项目 tokenizer.py 相同的思路)

"哪个模型用了哪种分词器"不在这里手写: 那张表由 experiments/tokenizer_anatomy.py 从各模型官方仓库的
tokenizer.json 读出, 本脚本只负责打印它 (旧版本手写的表里有多处错误, 例如把 Qwen 归为 SentencePiece)。

运行: python modern_tokenizers.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 0. 主流模型的分词器 (实测数据, 不手写)
# ============================================================

def print_anatomy_table():
    path = os.path.join(HERE, "experiments", "results", "tokenizer_anatomy.json")
    if not os.path.exists(path):
        print("  [SKIP] 还没有实测数据, 先运行: python experiments/tokenizer_anatomy.py")
        return
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    print(f"  {'模型':<34} {'算法':<10} {'词表':>8}  {'字节回退':<6} 预分词")
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['repo']:<34} ({r['status']})")
            continue
        pre = "+".join(r["pre_tokenizer"] or ["无"])
        print(f"  {r['repo']:<34} {r['algorithm']:<10} {r['vocab_size']:>8,}  {str(r['byte_fallback']):<6} {pre}")
    print("  (数据来自各模型官方仓库的 tokenizer.json; 完整内容见 experiments/results/tokenizer_anatomy.json)")


# ============================================================
# 1. Tiktoken -- OpenAI 的分词器
# ============================================================
# 底层就是字节级 BPE (与本项目同一个算法), 核心用 Rust 实现。
# 官方 README 称它比同类开源分词器快 3-6 倍。
#   cl100k_base: GPT-3.5-turbo / GPT-4 使用 (n_vocab = 100,277, 含特殊 token)
#   o200k_base:  GPT-4o 使用
#
# 与本项目 tokenizer.py 的异同:
#   - 同: 编码时同样按合并规则的 rank (学到的先后次序) 反复合并相邻字节对
#   - 异: 词表存成 {字节序列: rank}, 而本项目存成 {(id_a, id_b): new_id}; 预分词用更复杂的正则 (讲义「正则表达式专题」)

def demo_tiktoken():
    """演示 Tiktoken 分词器"""
    try:
        import tiktoken
    except ImportError:
        print("  [SKIP] tiktoken 未安装, 运行: pip install tiktoken")
        print("  GitHub: https://github.com/openai/tiktoken")
        return

    print("=" * 60)
    print("1. Tiktoken (OpenAI)")
    print("=" * 60)

    test_text = "Hello, how are you? 你好世界! Hi"
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(test_text)
    print(f"  Text:     {test_text}")
    print(f"  cl100k_base (GPT-3.5-turbo / GPT-4):")
    print(f"  Token IDs: {ids}")
    print(f"  Token count: {len(ids)}")
    for i, tid in enumerate(ids):
        tok = enc.decode([tid])
        print(f"    [{i:2d}] ID={tid:6d} -> {repr(tok)}")

    try:
        enc4o = tiktoken.get_encoding("o200k_base")
        ids4o = enc4o.encode(test_text)
        print(f"\n  o200k_base (GPT-4o):")
        print(f"  Token IDs: {ids4o}")
        print(f"  Token count: {len(ids4o)}")
    except Exception:
        print("  [SKIP] o200k_base needs newer tiktoken")

    n_bytes = len(test_text.encode("utf-8"))
    print(f"\n  Compression (text = {n_bytes} bytes):")
    print(f"    cl100k_base: {len(ids)} tokens, ratio={len(ids)/n_bytes:.3f}")


# ============================================================
# 2. SentencePiece -- Google 的分词框架
# ============================================================
# 实测 (tokenizer_anatomy.json) 用 SentencePiece 风格分词器的有: Mistral-7B、Phi-3-mini 等;
# LLaMA-1/2 也是 (官方仓库受限, 未能实测)。
#   - 不需要外部预分词; 内部默认仍按空白、文字类别、数字切分 (见训练日志里的
#     split_by_whitespace / split_by_unicode_script / split_digits)
#   - 空格编码为特殊字符 "▁" (U+2581), 所以解码时能精确还原空格
#   - 支持 BPE 和 Unigram 两种算法
#   - byte_fallback=True 时, 词表里没有的字符退回到 <0xE4> 这样的字节 token, 不会产生 <unk>
#   - 默认特殊 token: <unk>=0, <s>=1, </s>=2; PAD 默认关闭 (pad_id = -1)

def demo_sentencepiece():
    """演示 SentencePiece 分词器"""
    try:
        import sentencepiece as spm
    except ImportError:
        print("  [SKIP] sentencepiece 未安装, 运行: pip install sentencepiece")
        print("  GitHub: https://github.com/google/sentencepiece")
        return

    print("\n" + "=" * 60)
    print("2. SentencePiece (BPE / Unigram)")
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
        minloglevel=2,                       # 只打印错误, 不刷屏训练日志
    )

    sp = spm.SentencePieceProcessor(model_file="spm_bpe.model")
    test_texts = [
        "Hello world",
        "The quick brown fox jumps",
        "Transformer models are amazing",
    ]

    print(f"  Vocab size: {sp.get_piece_size()}")
    print(f"  Special: <s>={sp.bos_id()}, </s>={sp.eos_id()}, <unk>={sp.unk_id()}, pad={sp.pad_id()} (-1 表示关闭)")

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
        minloglevel=2,
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
# 3. HuggingFace 上的真实分词器
# ============================================================
# transformers / tokenizers 库能直接加载各模型官方发布的分词器 (Rust 后端, 支持 BPE / WordPiece / Unigram)。

def demo_huggingface():
    """演示 HuggingFace Tokenizers"""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  [SKIP] transformers 未安装, 运行: pip install transformers")
        print("  GitHub: https://github.com/huggingface/tokenizers")
        return

    print("\n" + "=" * 60)
    print("3. HuggingFace Tokenizers")
    print("=" * 60)

    test_text = "Hello, how are you? 你好世界!"
    models = [
        ("bert-base-uncased", "BERT (WordPiece, 英文词表: 多数汉字会变成 [UNK])"),
        ("bert-base-chinese", "BERT-Chinese (WordPiece)"),
        ("gpt2", "GPT-2 (字节级 BPE: 汉字被拆成字节, 显示成 'ä½' 这类乱码样的字符)"),
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
#   - WordPiece: 合并使训练语料似然提升最大的 pair (频率除以两个部分各自的频率)
#   - 初始词表是字符 (不是字节), 词表外的字符会变成 [UNK]
#   - 非首子词用 ## 前缀 (如 "tokenization" -> "token" + "##ization")

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
# 5. 字节级 BPE 的核心概念
# ============================================================

def demo_bytelevel_bpe():
    """演示字节级 BPE 核心概念"""
    print("\n" + "=" * 60)
    print("5. Byte-level BPE (GPT-2 起使用)")
    print("=" * 60)

    test_text = "Hello 你好"
    bytes_repr = list(test_text.encode("utf-8"))
    chars_repr = list(test_text)

    print(f"  Text: {test_text}")
    print(f"  Char-level ({len(chars_repr)} items): {chars_repr}")
    print(f"  Byte-level ({len(bytes_repr)} items): {bytes_repr}")
    print(f"\n  '你' = 0xE4 0xBD 0xA0 (3 bytes)")
    print(f"  GPT-2 把 256 个字节一一映射成可打印字符 (例如空格 -> 'Ġ'), 再在这些字符上做 BPE,")
    print(f"  所以在 HuggingFace 里看到的 GPT-2 token 是 'Ġhow'、'ä½' 这种样子; 本项目直接在字节上做, 不需要这层映射。")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    print("主流分词方案演示")
    print("=" * 60)
    print_anatomy_table()
    print()

    demo_tiktoken()
    demo_sentencepiece()
    demo_huggingface()
    demo_wordpiece()
    demo_bytelevel_bpe()

    print("\n" + "=" * 60)
    print("小结 (以实测的 tokenizer_anatomy.json 为准):")
    print("  - 字节级 BPE + 正则预分词: GPT-2、phi-4、Qwen2.5、DeepSeek-V3、GLM-4.5、gpt-oss")
    print("  - SentencePiece 风格 BPE + 字节回退: Mistral-7B、Phi-3-mini (LLaMA-1/2 同类, 未能实测)")
    print("  - WordPiece: BERT 系列")
    print("  - Unigram: T5")
    print("=" * 60)
