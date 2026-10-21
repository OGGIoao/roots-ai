"""
tokenizer_anatomy.py — 读取真实模型的 tokenizer.json, 核对它们用了什么分词方案, 并在同一段文本上实测

与 electives/llm-anatomy (解剖台) 同一个思路: 不凭印象写"某模型用某算法", 而是读官方仓库里的分词器文件。
读出: 算法 (BPE / WordPiece / Unigram)、词表大小、预分词方式 (含正则)、是否有字节回退 (byte_fallback)。
实测: 同一段英文 (莎士比亚验证集) 和同一段中文 (本文件里自写的一段) 各需要多少 token。

结果写入 results/tokenizer_anatomy.json; 讲义「主流分词器解剖」一节与图引用它。
运行: python experiments/tokenizer_anatomy.py   (需要能访问 huggingface.co; 只下载分词器文件, 不下载权重)
"""

import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import hf_hub_download  # noqa: E402
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from data import get_corpus, get_train_val_split  # noqa: E402

MODELS = [
    "openai-community/gpt2",
    "google-bert/bert-base-uncased",
    "google-bert/bert-base-chinese",
    "google-t5/t5-small",
    "mistralai/Mistral-7B-v0.1",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/phi-4",
    "Qwen/Qwen2.5-7B",
    "deepseek-ai/DeepSeek-V3",
    "zai-org/GLM-4.5",
    "openai/gpt-oss-20b",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-3.1-8B",
]

# 一段自写的中文 (避免引用有版权的文本), 约 200 字
ZH = ("分词器是大语言模型的第一道工序。它把一段文字切成一个个小片段, 再给每个片段分配一个编号, "
      "模型真正读到的只是这一串编号。切得好, 同样一句话用到的编号就少, 模型一次能看到的内容就多, "
      "训练和推理也更快; 切得不好, 一个汉字可能要拆成好几个编号, 既浪费算力, 也让模型更难学到词的意思。"
      "所以不同语言在同一个分词器下的效率可能相差好几倍, 这正是很多中文模型要重新训练分词器的原因。")


def algorithm_of(m):
    """新格式的 tokenizer.json 在 model 里写明 type; 较老的文件 (GPT-2、BERT、T5) 没写, 按结构推断。"""
    if "type" in m:
        return m["type"]
    if "merges" in m:
        return "BPE"
    if "continuing_subword_prefix" in m or "max_input_chars_per_word" in m:
        return "WordPiece"
    if isinstance(m.get("vocab"), list):          # Unigram 的词表是 [片段, 对数概率] 的列表
        return "Unigram"
    return "?"


def describe(tok_json):
    m = tok_json["model"]
    info = {"algorithm": algorithm_of(m), "vocab_in_model": len(m.get("vocab") or {}),
            "added_tokens": len(tok_json.get("added_tokens") or []),
            "byte_fallback": m.get("byte_fallback"), "pre_tokenizer": None, "regex": None,
            "normalizer": (tok_json.get("normalizer") or {}).get("type")}
    pre = tok_json.get("pre_tokenizer")

    def walk(p):
        if not p:
            return []
        if p["type"] == "Sequence":
            return [x for q in p["pretokenizers"] for x in walk(q)]
        return [p]

    steps = walk(pre)
    info["pre_tokenizer"] = [s["type"] for s in steps] or None
    regexes = []                                   # 可能串联多个 Split (例如 DeepSeek-V3 有 3 个)
    for s in steps:
        if s["type"] == "Split" and isinstance(s.get("pattern"), dict) and "Regex" in s["pattern"]:
            regexes.append(s["pattern"]["Regex"])
        if s["type"] == "ByteLevel" and s.get("use_regex"):
            regexes.append("(ByteLevel 内置的 GPT-2 正则)")
    info["regex"] = regexes or None
    return info


def main():
    text = get_corpus("shakespeare", verbose=False)
    _, val = get_train_val_split(text, 0.1)
    en = val[:10000]
    rows = []
    for repo in MODELS:
        row = {"repo": repo}
        try:
            path = hf_hub_download(repo, "tokenizer.json")
            with open(path, encoding="utf-8") as f:
                row.update(describe(json.load(f)))
            tok = Tokenizer.from_file(path)
            row["vocab_size"] = tok.get_vocab_size(with_added_tokens=True)
            # 未知词按 ID 统计: T5 会把一整段不认识的中文合并成一个 <unk>, 按 token 文字统计会漏掉
            unk_ids = {tok.token_to_id(u) for u in ("[UNK]", "<unk>")} - {None}
            for name, sample in (("en", en), ("zh", ZH)):
                enc = tok.encode(sample, add_special_tokens=False)
                nb = len(sample.encode("utf-8"))
                row[f"{name}_tokens"] = len(enc.ids)
                row[f"{name}_bytes"] = nb
                row[f"{name}_chars"] = len(sample)
                row[f"{name}_unk"] = sum(1 for i in enc.ids if i in unk_ids)
                # 能否无损还原: 解码后去掉空白差异再比较 (有的分词器会规范化空格)
                back = tok.decode(enc.ids, skip_special_tokens=False)
                row[f"{name}_lossless"] = "".join(back.split()) == "".join(sample.split())
            row["zh_pieces"] = tok.encode(ZH[:12], add_special_tokens=False).tokens
            row["status"] = "ok"
        except GatedRepoError:
            row["status"] = "gated (需要 HF_TOKEN)"
        except EntryNotFoundError:
            row["status"] = "仓库里没有 tokenizer.json"
        except Exception as e:  # noqa: BLE001
            row["status"] = f"error: {type(e).__name__}: {str(e)[:100]}"
        rows.append(row)
        print(f"  {row['status'][:8]:<8} {repo}")
    out = os.path.join(HERE, "results", "tokenizer_anatomy.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"zh_text": ZH, "en_note": "莎士比亚验证集前 10000 字符", "rows": rows}, f, ensure_ascii=False, indent=1)
    print("写入", out)


if __name__ == "__main__":
    main()
