"""
train.py — BPE Tokenizer 训练 + Demo 演示

这是本项目的运行入口。它做以下几件事：
  1. 加载语料（TinyShakespeare 或 sample）
  2. 训练 BPE tokenizer
  3. 展示编码/解码效果
  4. 打印词表样例 & 压缩率统计
  5. 保存 tokenizer 到 tokenizer.json

用法：
  python train.py                          # 默认：莎士比亚, vocab_size=500
  python train.py --vocab_size 1000        # 更大词表
  python train.py --source sample          # 内置语料（无网络）
  python train.py --vocab_size 300 --no_save  # 不保存文件
"""

import argparse
import os
import sys
import time

from data import get_corpus, get_train_val_split
from tokenizer import BPETokenizer

# Windows 控制台默认 GBK, 无法编码脚本结尾处的 ✅/❌ 等符号,
# print() 会抛 UnicodeEncodeError 让脚本在跑通后异常退出。
for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r:
        try:
            _r(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


# ─── 演示用的测试句子 ─────────────────────────────────────────────────────────
_DEMO_SENTENCES = [
    "To be, or not to be, that is the question.",
    "Hello, world! 你好世界 🌍",
    "The quick brown fox jumps over the lazy dog.",
    "OpenAI trained GPT-4 using 1.8 trillion tokens.",
    "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
]


def _resolve_save_path(args) -> str:
    """tokenizer.json 是入库的默认产物 (shakespeare, vocab=500), 其它配置不得覆盖它。"""
    here = os.path.dirname(os.path.abspath(__file__))
    if args.output:
        return os.path.abspath(args.output)
    if args.source == "shakespeare" and args.vocab_size == 500:
        return os.path.join(here, "tokenizer.json")
    return os.path.join(here, f"tokenizer_{args.source}_{args.vocab_size}.json")


def main():
    parser = argparse.ArgumentParser(description="训练 BPE Tokenizer")
    parser.add_argument(
        "--vocab_size", type=int, default=500,
        help="目标词表大小（默认 500，必须 > 256）"
    )
    parser.add_argument(
        "--source", type=str, default="shakespeare",
        choices=["shakespeare", "sample"],
        help="语料来源（默认 shakespeare；离线演示请显式用 sample）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="tokenizer 保存路径。默认: 仅 shakespeare+词表500(默认配置)才写入入库的 "
             "tokenizer.json, 其余写到 tokenizer_<来源>_<词表>.json, 避免覆盖共享产物"
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="不保存 tokenizer.json"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="训练时不打印每次合并的详情"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  BPE Tokenizer 训练 Demo")
    print(f"  vocab_size = {args.vocab_size}，语料 = {args.source}")
    print("=" * 60)
    print()

    # ── 1. 加载语料 ──────────────────────────────────────────────────────────
    text = get_corpus(source=args.source, verbose=True)
    train_text, val_text = get_train_val_split(text, val_ratio=0.1)
    print(f"[数据] 训练集：{len(train_text):,} 字符 / 验证集：{len(val_text):,} 字符")
    print()

    # ── 2. 训练 tokenizer ────────────────────────────────────────────────────
    tokenizer = BPETokenizer()
    t0 = time.time()
    tokenizer.train(
        text=train_text,
        vocab_size=args.vocab_size,
        verbose=not args.quiet,
    )
    elapsed = time.time() - t0
    print(f"\n[耗时] 训练用时 {elapsed:.1f} 秒")
    print()

    # ── 3. 编码/解码演示 ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  📝 编码 / 解码 演示")
    print("=" * 60)
    for sentence in _DEMO_SENTENCES:
        _demo_encode_decode(tokenizer, sentence)

    # ── 4. 压缩率统计 ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  📊 压缩率统计（在验证集上）")
    print("=" * 60)
    _print_compression_stats(tokenizer, val_text)

    # ── 5. 词表样例 ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  📖 词表样例（ID 256 以后的合并 token）")
    print("=" * 60)
    _print_vocab_samples(tokenizer, n=20)

    # ── 6. 保存 tokenizer ────────────────────────────────────────────────────
    if not args.no_save:
        print()
        save_path = _resolve_save_path(args)
        print(f"[保存] {save_path}")
        tokenizer.save(save_path)
        size_kb = os.path.getsize(save_path) // 1024
        print(f"[文件大小] {size_kb} KB")

        # 验证 save/load 正确性
        print()
        print("[验证] 测试 save/load 的正确性...")
        tok2 = BPETokenizer.load(save_path)
        test_text = "To be, or not to be, that is the question."
        assert tokenizer.encode(test_text) == tok2.encode(test_text), \
            "❌ save/load 后编码结果不一致！"
        print("[验证] ✅ save/load 正常，编码结果一致")

    print()
    print("=" * 60)
    print("  🎉 全部完成！接下来你可以：")
    print("  1. 修改 --vocab_size 观察压缩率变化")
    print("  2. 阅读 tokenizer.py 理解 BPE 核心算法")
    print("  3. 做 TUTORIAL.md 第 14 节和 exercises/ 里的练习")
    print("  4. 下一步: library/llm/minigpt 从零训练一个 GPT")
    print("=" * 60)


# ─── 辅助打印函数 ─────────────────────────────────────────────────────────────

def _demo_encode_decode(tokenizer: BPETokenizer, text: str) -> None:
    """展示一个句子的编码/解码过程。"""
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    match = "✅" if decoded == text else "⚠️"

    print(f"\n  输入:    {repr(text)}")
    print(f"  Token IDs ({len(ids)} 个): {ids[:15]}{'...' if len(ids) > 15 else ''}")
    # 展示每个 token 对应的字符串
    token_strs = []
    for i in ids[:10]:
        raw = tokenizer.vocab[i]
        try:
            token_strs.append(repr(raw.decode("utf-8")))
        except UnicodeDecodeError:
            token_strs.append(repr(raw))
    print(f"  Token 内容 (前10): {token_strs}")
    print(f"  解码结果: {repr(decoded[:80])}{'...' if len(decoded) > 80 else ''}  {match}")


def _print_compression_stats(tokenizer: BPETokenizer, text: str) -> None:
    """打印在给定文本上的压缩率统计。"""
    # 取前 10000 字符，避免统计太慢
    sample = text[:10000]
    ids = tokenizer.encode(sample)
    n_bytes = len(sample.encode("utf-8"))
    n_tokens = len(ids)
    ratio = n_tokens / n_bytes

    print(f"  原始字节数：{n_bytes:,}")
    print(f"  Token 数：  {n_tokens:,}")
    print(f"  压缩率：    {ratio:.3f}  （token数 / 字节数）")
    print(f"  平均每个 token：{n_bytes / n_tokens:.2f} 字节")
    print()
    print("  参考值（越低越好）：")
    print("  - 字节级（无压缩）          = 1.000")
    print(f"  - 你的 tokenizer           = {ratio:.3f}")
    # 2026-10-20 实测: GPT-2 官方分词器 (词表 50257) 在同一段验证集前 10000 字节上的结果
    # (测量代码见 experiments/examples.py; 讲义「压缩率」一节)
    print("  - GPT-2 (词表 50257, 实测) = 0.323")


def _print_vocab_samples(tokenizer: BPETokenizer, n: int = 20) -> None:
    """打印词表中 ID 256 以后的合并 token 样例。"""
    merge_ids = sorted(
        [k for k in tokenizer.vocab if k >= 256],
        key=lambda x: x
    )
    show_ids = merge_ids[:n]

    print(f"  （共 {len(merge_ids)} 个合并 token，展示前 {len(show_ids)} 个）")
    print()
    for id_ in show_ids:
        raw = tokenizer.vocab[id_]
        try:
            s = raw.decode("utf-8")
        except UnicodeDecodeError:
            s = repr(raw)
        # 找出是哪两个 token 合并而来
        parent_pair = next(
            (pair for pair, nid in tokenizer.merges.items() if nid == id_), None
        )
        if parent_pair:
            a_raw = tokenizer.vocab[parent_pair[0]]
            b_raw = tokenizer.vocab[parent_pair[1]]
            try:
                a_str = a_raw.decode("utf-8")
            except UnicodeDecodeError:
                a_str = repr(a_raw)
            try:
                b_str = b_raw.decode("utf-8")
            except UnicodeDecodeError:
                b_str = repr(b_raw)
            print(f"  ID {id_:5d}:  {repr(s):20s}  ←  {repr(a_str)} + {repr(b_str)}")
        else:
            print(f"  ID {id_:5d}:  {repr(s)}")


if __name__ == "__main__":
    main()
