"""
examples.py — 讲义里每一个示例输出的来源 (都是实际运行得到的, 不手写)

用法:
  python experiments/examples.py fast    # 玩具例子、编码轨迹、预分词对比、中文实验等 (约 1 分钟)
  python experiments/examples.py sweep   # 一次训练到词表 4096, 记录每次合并的频率和耗时 (约几分钟)

结果:
  results/examples.json      fast 的全部输出
  results/vocab_sweep.json   sweep 的输出 (压缩率 vs 词表大小、合并频率曲线、训练耗时)

为什么 sweep 只训练一次: BPE 每一步只依赖前面的合并结果, 所以"训练到 4096 后只保留前 k 条合并规则"
与"直接训练到 256+k"得到的分词器完全相同。脚本会把前 244 条与 train(vocab_size=500) 的结果逐条比对来验证这一点。
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict

import regex

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from data import get_corpus, get_train_val_split  # noqa: E402
from tokenizer import BPETokenizer, _merge_pair_in_sequence  # noqa: E402

RESULTS = os.path.join(HERE, "results")

# GPT-2 的预分词正则 (与 openai/gpt-2 的 encoder.py 相同; 下面会和 HuggingFace 版 GPT-2 分词器逐条核对)
GPT2_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 与 tokenizer_anatomy.py 共用同一段自写中文 (避免引用有版权的文本)
with open(os.path.join(RESULTS, "tokenizer_anatomy.json"), encoding="utf-8") as _f:
    ANATOMY = json.load(_f)
ZH = ANATOMY["zh_text"]


def show(b: bytes) -> str:
    """token 的可读形式: 合法 UTF-8 就显示文字, 否则显示字节 (例如半个汉字)。"""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return repr(b)


def corpus():
    text = get_corpus("shakespeare", verbose=False)
    return get_train_val_split(text, 0.1)


def hf(repo):
    """从本地缓存 (或网络) 加载官方 tokenizer.json; 拿不到就返回 None, 不让整个脚本失败。"""
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        return Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] {repo}: {type(e).__name__}")
        return None


# ─── 训练轨迹: 用 tokenizer.py 自己的内部函数, 逐轮记下发生了什么 ─────────────────

def trace_train(word_freqs: dict, n_merges: int):
    """
    在给定的词频表上跑 n_merges 轮合并, 每轮记录: 所有 pair 的频率、选中的 pair、合并后的词频表。
    用的是 BPETokenizer 的 _get_pair_freqs / _apply_merge_to_freqs 和与 train() 相同的 tie-break 规则,
    所以轨迹就是 train() 实际走过的路径。
    """
    tok = BPETokenizer()
    wf = {tuple(k.encode("utf-8")): v for k, v in word_freqs.items()}
    rounds = []
    for i in range(n_merges):
        pf = tok._get_pair_freqs(wf)
        if not pf:
            rounds.append({"round": i + 1, "stop": "没有可合并的 pair 了"})
            break
        best = max(pf, key=lambda p: (pf[p], -p[0], -p[1]))
        new_id = 256 + i
        tok.merges[best] = new_id
        tok.vocab[new_id] = tok.vocab[best[0]] + tok.vocab[best[1]]
        before = dict(wf)
        wf = tok._apply_merge_to_freqs(best, new_id, wf)
        # "实际替换次数" 与 "统计到的频率" 在重叠时不同 (例如 aaaa)
        replaced = sum(f * (len(k) - len(nk)) for (k, f), nk in zip(before.items(), wf.keys()))
        rounds.append({
            "round": i + 1,
            "pairs": sorted(([show(tok.vocab[a]), show(tok.vocab[b]), [a, b], f] for (a, b), f in pf.items()),
                            key=lambda r: (-r[3], r[2])),
            "chosen": [best[0], best[1]], "new_id": new_id, "new_token": show(tok.vocab[new_id]),
            "freq": pf[best], "replaced": replaced,
            "words_after": [[[show(tok.vocab[t]) for t in k], list(k), f] for k, f in wf.items()],
        })
    return rounds


def trace_encode(tok: BPETokenizer, piece: str):
    """把 _encode_chunk 的每一轮记下来: 当前序列、所有可用合并的 rank、选中的那个。"""
    ids = list(piece.encode("utf-8"))
    steps = []
    while len(ids) >= 2:
        cands = [[ids[i], ids[i + 1], tok.merges[(ids[i], ids[i + 1])]]
                 for i in range(len(ids) - 1) if (ids[i], ids[i + 1]) in tok.merges]
        steps.append({"seq": [show(tok.vocab[t]) for t in ids], "ids": list(ids),
                      "candidates": [[show(tok.vocab[a]), show(tok.vocab[b]), r] for a, b, r in cands]})
        if not cands:
            break
        a, b, r = min(cands, key=lambda c: c[2])
        steps[-1]["chosen"] = [show(tok.vocab[a]), show(tok.vocab[b]), r]
        ids = list(_merge_pair_in_sequence(tuple(ids), (a, b), r))
    if len(ids) < 2:
        steps.append({"seq": [show(tok.vocab[t]) for t in ids], "ids": list(ids), "candidates": []})
    assert ids == tok._encode_chunk(list(piece.encode("utf-8"))), "轨迹与 _encode_chunk 不一致"
    return steps


def merge_tree(tok: BPETokenizer, tid: int):
    """一个 token 是由哪两个 token 合成的, 递归展开到单字节。"""
    parent = {v: k for k, v in tok.merges.items()}
    def node(t):
        if t not in parent:
            return {"id": t, "tok": show(tok.vocab[t])}
        a, b = parent[t]
        return {"id": t, "tok": show(tok.vocab[t]), "kids": [node(a), node(b)]}
    return node(tid)


def segments(text, tok: BPETokenizer):
    """把编码结果还原成原文里的一段段 (用于"分词可视化"图)。"""
    return [show(tok.vocab[i]) for i in tok.encode(text)]


def _byte_decoder():
    """GPT-2 把 256 个字节一一映射成可打印字符 (空格 -> 'Ġ' 等) 再做 BPE; 这里是它的逆映射。"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DEC = _byte_decoder()


def hf_segments(text, t):
    """字节级 BPE 分词器 (GPT-2 等) 的每个 token 还原成真实字节再显示; 半个汉字显示成 b'..'。"""
    enc = t.encode(text, add_special_tokens=False)
    return [show(bytes(BYTE_DEC[c] for c in tok)) for tok in enc.tokens]


# ─── fast ────────────────────────────────────────────────────────────────────

def fast():
    out = {}
    train_text, val_text = corpus()
    val = val_text[:10000]
    ours = BPETokenizer.load(os.path.join(ROOT, "tokenizer.json"))   # 入库的默认产物: 莎士比亚, 词表 500
    gpt2 = hf("openai-community/gpt2")

    # 1. 玩具训练轨迹 -------------------------------------------------------
    out["toy_low"] = trace_train({"low": 1, "lower": 1, "lowest": 1}, 4)
    out["toy_sennrich"] = trace_train({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 10)
    out["toy_abababd"] = trace_train({"abababd": 1}, 4)
    out["toy_aaaa"] = trace_train({"aaaa": 1}, 2)
    # 直接调 train() 的样子 (含预分词): 空格会附在后面的词上
    t = BPETokenizer()
    t.train("low lower lowest", 260, verbose=False)
    out["toy_low_via_train"] = {"pretokens": t._pretokenize("low lower lowest"),
                                "merges": [[show(t.vocab[a]), show(t.vocab[b]), n] for (a, b), n in t.merges.items()]}

    # 2. 提前结束 -------------------------------------------------------------
    import contextlib
    import io
    buf = io.StringIO()
    t = BPETokenizer()
    with contextlib.redirect_stdout(buf):
        t.train("Hello hello hello", 300, verbose=True)
    out["early_stop"] = {"vocab": len(t.vocab), "merges": [show(t.vocab[n]) for n in t.merges.values()],
                         "log": [ln for ln in buf.getvalue().splitlines() if "提前" in ln or "完成" in ln]}

    # 3. 真实 tokenizer.json (莎士比亚, 500) 的样子 -----------------------------
    merges = list(ours.merges.items())
    out["real_first_merges"] = [[n, show(ours.vocab[a]), show(ours.vocab[b]), show(ours.vocab[n])]
                                for (a, b), n in merges[:30]]
    out["real_newline_tokens"] = [[n, show(ours.vocab[n])] for n in sorted(ours.vocab)
                                  if n >= 256 and ours.vocab[n].startswith(b"\n")]
    lens = Counter(len(v) for k, v in ours.vocab.items() if k >= 256)
    out["real_token_len_hist"] = sorted(lens.items())
    longest = max((k for k in ours.vocab if k >= 256), key=lambda k: (len(ours.vocab[k]), -k))
    out["real_longest"] = [longest, show(ours.vocab[longest])]
    out["merge_tree_longest"] = merge_tree(ours, longest)
    the_id = ours.encode(" the")
    out["merge_tree_the"] = merge_tree(ours, the_id[0]) if len(the_id) == 1 else {"ids": the_id}

    # 4. 编码轨迹 ---------------------------------------------------------------
    out["encode_trace"] = {w: trace_encode(ours, w) for w in ["Hello", " lowest", " question.", " the"]}

    # 5. 单词在我们的分词器 / GPT-2 上的切法 (讲义第 2 节的说法以此为准) ---------------
    words = ["the", " the", "The", "Hello", " hello", "hello", "unbelievable", " unbelievable",
             "xyzabc", "2026年", " question.", "你好"]
    out["words"] = {w: {"ours": segments(w, ours), "gpt2": hf_segments(w, gpt2) if gpt2 else None} for w in words}

    # 6. 句子分词可视化 + 压缩率 --------------------------------------------------
    sents = ["To be, or not to be, that is the question.", "Hello, world! 你好世界 🌍",
             "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"]
    out["sentences"] = [{"text": s, "ours": segments(s, ours), "gpt2": hf_segments(s, gpt2) if gpt2 else None}
                        for s in sents]
    nb = len(val.encode("utf-8"))
    comp = {"val_bytes": nb, "val_chars": len(val), "ours_tokens": len(ours.encode(val))}
    if gpt2:
        comp["gpt2_tokens"] = len(gpt2.encode(val, add_special_tokens=False).ids)
    out["compression"] = comp

    # 词表利用率: 编码验证集时用到了词表里多少个不同的 token
    used = set(ours.encode(val_text))
    out["vocab_usage"] = {"ours_used": len(used), "ours_vocab": len(ours.vocab),
                          "ours_bytes_used": len([i for i in used if i < 256])}
    if gpt2:
        g = set(gpt2.encode(val_text, add_special_tokens=False).ids)
        out["vocab_usage"].update({"gpt2_used": len(g), "gpt2_vocab": gpt2.get_vocab_size()})

    # 编码速度 (同一段 ~11 万字符的验证集)
    t0 = time.perf_counter(); ours.encode(val_text); t_ours = time.perf_counter() - t0
    sp = {"chars": len(val_text), "ours_s": t_ours}
    if gpt2:
        t0 = time.perf_counter(); gpt2.encode(val_text); sp["gpt2_hf_s"] = time.perf_counter() - t0
    out["encode_speed"] = sp

    # BPE 的贪心编码 vs 同一张词表下"token 最少"的切法 (动态规划)
    toks = set(ours.vocab.values())
    maxlen = max(len(b) for b in toks)
    def min_split(b: bytes) -> int:
        best = [0] + [10 ** 9] * len(b)
        for i in range(1, len(b) + 1):
            for j in range(max(0, i - maxlen), i):
                if b[j:i] in toks and best[j] + 1 < best[i]:
                    best[i] = best[j] + 1
        return best[-1]
    n_bpe = n_dp = 0
    diff = []
    for p in ours._pretokenize(val):
        a, b = len(ours._encode_chunk(list(p.encode("utf-8")))), min_split(p.encode("utf-8"))
        n_bpe, n_dp = n_bpe + a, n_dp + b
        if a > b and len(diff) < 8 and p not in [d[0] for d in diff]:
            diff.append([p, segments(p, ours), a, b])
    out["greedy_vs_optimal"] = {"bpe_tokens": n_bpe, "min_tokens": n_dp, "examples": diff}

    # "每轮选 rank 最小的" 与 "按训练顺序把每条规则依次应用一遍" 结果是否相同 (在整个验证集上逐词比对)
    def in_order(ids):
        for pair, new_id in ours.merges.items():
            ids = list(_merge_pair_in_sequence(tuple(ids), pair, new_id))
        return ids
    pts = ours._pretokenize(val_text)
    same = sum(in_order(list(p.encode("utf-8"))) == ours._encode_chunk(list(p.encode("utf-8"))) for p in pts)
    out["min_rank_vs_in_order"] = {"pretokens": len(pts), "identical": same}

    # 7. 边界情况 -------------------------------------------------------------
    nihao = list("你".encode("utf-8"))
    out["edge"] = {
        "encode_empty": ours.encode(""), "decode_empty": ours.decode([]),
        "decode_E4": ours.decode([0xE4]),
        "decode_ni_bytes": ours.decode(nihao),
        "decode_one_by_one": [ours.decode([b]) for b in nihao],    # 流式逐个解码会出乱码
        "ni_ids": nihao,
        "hello_nihao_untrained": BPETokenizer().encode("Hello 你好"),
        "roundtrip_val": ours.decode(ours.encode(val_text)) == val_text,
    }

    # 8. 预分词对比 -------------------------------------------------------------
    texts = ["It's 2026, hello world!", "我 爱 Python 3.12", "line1\nline2\n", "Hello  world  ",
             "价格是 1234567 元", "I'M HERE, they'RE", "x = foo(a,b)\n    return x"]
    qwen = hf("Qwen/Qwen2.5-7B")
    qwen_pat = next(r for r in ANATOMY["rows"] if r["repo"] == "Qwen/Qwen2.5-7B")["regex"][0]
    phi4_pat = next(r for r in ANATOMY["rows"] if r["repo"] == "microsoft/phi-4")["regex"][0]
    ds = hf("deepseek-ai/DeepSeek-V3")
    pre = []
    for s in texts:
        row = {"text": s, "ours": BPETokenizer()._pretokenize(s), "simple_re": regex.findall(r"\S+|\s+", s),
               "gpt2_regex": regex.findall(GPT2_PAT, s), "qwen_regex": regex.findall(qwen_pat, s),
               "phi4_regex": regex.findall(phi4_pat, s)}
        if gpt2:   # HuggingFace 的 GPT-2 预分词: 用 offsets 映射回原文, 应与上面的正则完全一致
            row["gpt2_hf"] = [s[a:b] for _, (a, b) in gpt2.pre_tokenizer.pre_tokenize_str(s)]
        if qwen:
            row["qwen_hf"] = [s[a:b] for _, (a, b) in qwen.pre_tokenizer.pre_tokenize_str(s)]
        if ds:
            row["deepseek_hf"] = [s[a:b] for _, (a, b) in ds.pre_tokenizer.pre_tokenize_str(s)]
        pre.append(row)
    out["pretokenize"] = pre
    out["pretokenize_check"] = {
        "gpt2_regex_equals_hf": all(r["gpt2_regex"] == r.get("gpt2_hf") for r in pre),
        "qwen_regex_equals_hf": all(r["qwen_regex"] == r.get("qwen_hf") for r in pre),
    }

    # 9. 正则写法的坑: 带注释的多行写法 ------------------------------------------------
    commented = ("'s|'t|'re|'ve|'m|'ll|'d|   # 英文缩写后缀\n"
                 "        ?\\p{L}+|                    # 可选空格 + 连续字母\n"
                 "        ?\\p{N}+|\n        ?[^\\s\\p{L}\\p{N}]+|\n        \\s+(?!\\S)|\n        \\s+")
    probe = "It's 2026, hello world!"
    pit = {}
    for name, flags in [("no_flag", 0), ("verbose", regex.VERBOSE)]:
        try:
            pit[name] = regex.findall(commented, probe, flags=flags)
        except regex.error as e:
            pit[name] = f"regex.error: {e}"
    pit["correct"] = regex.findall(GPT2_PAT, probe)
    out["regex_pitfall"] = pit

    # 10. 中文实验 --------------------------------------------------------------
    zh = {}
    ids = ours.encode(ZH)
    zh["en_tok_on_zh"] = {"chars": len(ZH), "bytes": len(ZH.encode("utf-8")), "tokens": len(ids),
                          "pretokens": len(ours._pretokenize(ZH))}
    t = BPETokenizer()
    t.train("你好世界" * 100, 300, verbose=False)          # 旧讲义 11.3 的原样实验
    zh["repeat_nihao"] = {"pretokens": len(t._pretokenize("你好世界" * 100)), "merges": len(t.merges),
                          "encode_nihao": segments("你好", t), "longest_bytes": max(len(v) for v in t.vocab.values())}
    t = BPETokenizer()
    t.train(ZH, 400, verbose=False)
    learned = [t.vocab[n] for n in t.merges.values()]
    def is_utf8(b):
        try:
            b.decode("utf-8"); return True
        except UnicodeDecodeError:
            return False
    zh["train_on_zh"] = {
        "merges": len(t.merges), "pretokens": len(t._pretokenize(ZH)),
        "first": [show(b) for b in learned[:20]],
        "invalid_utf8": sum(not is_utf8(b) for b in learned),
        "tokens_on_zh": len(t.encode(ZH)),
    }
    zh["models"] = [{"repo": r["repo"], "zh_tokens": r.get("zh_tokens"), "zh_chars": r.get("zh_chars"),
                     "zh_unk": r.get("zh_unk"), "lossless": r.get("zh_lossless"), "pieces": r.get("zh_pieces")}
                    for r in ANATOMY["rows"] if r["status"] == "ok"]
    out["chinese"] = zh

    # 11. UTF-8 与 0-255 字节表 -------------------------------------------------
    out["utf8"] = [{"ch": c, "cp": f"U+{ord(c):04X}", "bytes": [f"0x{b:02X}" for b in c.encode("utf-8")],
                    "bin": [f"{b:08b}" for b in c.encode("utf-8")]} for c in "Aéñ你😀"]
    def byte_class(b):
        if b < 32 or b == 127: return "control"
        if b < 128: return "ascii"
        if b < 192: return "cont"
        if b in (192, 193): return "never"
        if b < 224: return "lead2"
        if b < 240: return "lead3"
        if b < 245: return "lead4"
        return "never"
    tb = Counter(train_text.encode("utf-8"))
    zb = Counter(ZH.encode("utf-8"))
    # 验证 "从不出现" 的字节: 把所有合法码点编码一遍, 统计出现过的字节
    seen = set()
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        seen.update(chr(cp).encode("utf-8"))
    out["byte_map"] = {"class": [byte_class(b) for b in range(256)],
                       "never_seen_in_valid_utf8": sorted(set(range(256)) - seen),
                       "shakespeare_counts": [tb.get(b, 0) for b in range(256)],
                       "zh_counts": [zb.get(b, 0) for b in range(256)]}
    assert [b for b in range(256) if byte_class(b) == "never"] == out["byte_map"]["never_seen_in_valid_utf8"]

    path = os.path.join(RESULTS, "examples.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写入", path)


# ─── sweep ───────────────────────────────────────────────────────────────────

def sweep(max_vocab=4096):
    train_text, val_text = corpus()
    val = val_text[:10000]
    tok = BPETokenizer()
    t0 = time.perf_counter()
    pretokens = tok._pretokenize(train_text)
    wf = defaultdict(int)
    for p in pretokens:
        wf[tuple(p.encode("utf-8"))] += 1
    t_prep = time.perf_counter() - t0
    freqs, cum_t = [], []
    for i in range(max_vocab - 256):
        pf = tok._get_pair_freqs(wf)
        if not pf:
            break
        best = max(pf, key=lambda p: (pf[p], -p[0], -p[1]))   # 与 train() 相同的规则
        new_id = 256 + i
        tok.merges[best] = new_id
        tok.vocab[new_id] = tok.vocab[best[0]] + tok.vocab[best[1]]
        wf = tok._apply_merge_to_freqs(best, new_id, wf)
        freqs.append(pf[best])
        cum_t.append(time.perf_counter() - t0)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} merges, {cum_t[-1]:.0f}s", flush=True)

    # 验证: 前 244 条与 train(vocab_size=500) 完全一致
    ref = BPETokenizer()
    ref.train(train_text, 500, verbose=False)
    assert list(ref.merges.items()) == list(tok.merges.items())[:244], "前缀与直接训练的结果不一致"

    all_merges = list(tok.merges.items())
    rows = []
    for v in [256, 300, 400, 500, 750, 1000, 1500, 2000, 3000, max_vocab]:
        sub = BPETokenizer()
        sub.merges = dict(all_merges[: v - 256])
        sub.vocab = {k: tok.vocab[k] for k in range(v)}
        n = len(sub.encode(val))
        rows.append({"vocab": v, "tokens": n, "ratio": n / len(val.encode("utf-8")),
                     "train_s": cum_t[v - 257] if v > 256 else t_prep})
        print(f"  vocab {v:5d}: ratio {rows[-1]['ratio']:.3f}")
    out = {"val_bytes": len(val.encode("utf-8")), "pretokens": len(pretokens), "distinct": len(wf),
           "prep_s": t_prep, "merge_freqs": freqs, "cum_time_s": cum_t, "rows": rows,
           "verified_prefix_equals_train_500": True,
           "tokens_4096_sample": [show(tok.vocab[n]) for n in range(4086, max_vocab)]}
    path = os.path.join(RESULTS, "vocab_sweep.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写入", path)


if __name__ == "__main__":
    {"fast": fast, "sweep": sweep}[sys.argv[1] if len(sys.argv) > 1 else "fast"]()
