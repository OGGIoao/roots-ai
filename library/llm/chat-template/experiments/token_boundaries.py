"""
token_boundaries.py — 延伸专题 D、E 的数字: 模板拼好的文字, 再切成 token 时会出什么问题

  A. 分开分词 vs 整段分词: tokenize(提示) + tokenize(回复) 和 tokenize(提示 + 回复) 一样吗?
     (推理时提示和回复是分开得到的; 训练时常常整段分词, 两边不一样就是 "训练和推理看到的 token 不同")
  B. 字符区间对得上 token 边界吗: render_with_spans 给出的 assistant 区间, 起点和终点是不是正好落在 token 的边界上
  C. 开头标记重复: DeepSeek-V3 的模板自己写了 <｜begin▁of▁sentence｜>, 再用默认设置分词会不会多一个
  D. transformers 自带的 return_assistant_tokens_mask 在 Qwen2.5 的模板上能不能用

只加载分词器 (CPU)。结果写进 results/token_boundaries.json。
    python experiments/token_boundaries.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
sys.path.insert(0, UNIT)
sys.path.insert(0, HERE)

from chat_template import render, render_with_spans  # noqa: E402

QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = os.path.join(HERE, "results", "token_boundaries.json")


def golden_cases():
    with open(os.path.join(HERE, "results", "qwen25_golden.json"), encoding="utf-8") as f:
        return json.load(f)["cases"]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    res = {}

    # A + B: 用标准答案里所有带 assistant 回复的对话
    a_total = a_same = 0
    a_example = None
    b_total = b_aligned = 0
    b_example = None
    for case in golden_cases():
        msgs, tools = case["messages"], case["tools"]
        text, spans = render_with_spans(msgs, tools)
        enc = tok(text, return_offsets_mapping=True)
        starts = {s for s, e in enc["offset_mapping"]}
        ends = {e for s, e in enc["offset_mapping"]}
        for s, e in spans:
            a_total += 1
            whole = tok.encode(text[:e])
            parts = tok.encode(text[:s]) + tok.encode(text[s:e])
            if whole == parts:
                a_same += 1
            elif a_example is None:
                a_example = {"prompt_tail": text[max(0, s - 20):s], "reply_head": text[s:s + 20],
                             "whole": [tok.decode([i]) for i in whole[-8:]], "parts": [tok.decode([i]) for i in parts[-8:]]}
            for pos in (s, e):
                b_total += 1
                if (pos in starts) or (pos in ends):
                    b_aligned += 1
                elif b_example is None:
                    b_example = {"around": text[max(0, pos - 10):pos + 10], "pos": pos}
    res["split_vs_whole"] = {"same": a_same, "total": a_total, "first_difference": a_example}

    # A'. 对照: 没有特殊标记当 "墙" 的纯文字模板, 生成提示末尾带一个空格 ("Assistant: ")
    plain_total = plain_same = 0
    plain_example = None
    replies = ["你好", "Hello!", "台北现在 26 度。", "27", "OK, sure."]
    for r in replies:
        for prompt in ("User: 你好\nAssistant:", "User: 你好\nAssistant: "):
            reply = (" " + r) if prompt.endswith(":") else r
            whole = tok.encode(prompt + reply)
            parts = tok.encode(prompt) + tok.encode(reply)
            plain_total += 1
            if whole == parts:
                plain_same += 1
            elif plain_example is None:
                plain_example = {"prompt": prompt, "reply": reply, "whole": [tok.decode([i]) for i in whole[-4:]],
                                 "parts": [tok.decode([i]) for i in parts[-5:]]}
    res["plain_template_split_vs_whole"] = {"same": plain_same, "total": plain_total, "first_difference": plain_example}
    print("A'. 纯文字模板 (末尾冒号 / 末尾冒号加空格):", plain_same, "/", plain_total, plain_example or "")
    res["span_alignment"] = {"aligned": b_aligned, "total": b_total, "first_misaligned": b_example}
    print("A. 分开分词 = 整段分词:", a_same, "/", a_total, a_example or "")
    print("B. 区间端点落在 token 边界上:", b_aligned, "/", b_total, b_example or "")

    # C: DeepSeek-V3 的开头标记
    try:
        from template_zoo import load_template
        ds = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
        tpl, cfg = load_template("deepseek-ai/DeepSeek-V3")
        bos = ds.convert_tokens_to_ids(cfg["bos_token"]["content"] if isinstance(cfg["bos_token"], dict) else cfg["bos_token"])
        msgs = [{"role": "user", "content": "你好"}]
        text = ds.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, chat_template=tpl)
        res["double_bos"] = {
            "rendered_text_head": text[:30],
            "encode_default": ds.encode(text)[:3].count(bos),
            "encode_no_special": ds.encode(text, add_special_tokens=False)[:3].count(bos),
            "apply_chat_template_tokenize": ds.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                                  chat_template=tpl)["input_ids"][:3].count(bos),
        }
        print("C. 开头标记个数 (默认分词 / 不加特殊 token / apply_chat_template 直接分词):", res["double_bos"])
    except Exception as e:                           # 没有缓存 DeepSeek-V3 的分词器时跳过
        res["double_bos"] = {"skipped": f"{type(e).__name__}: {str(e)[:100]}"}
        print("C. 跳过:", res["double_bos"]["skipped"])

    # D: transformers 自带的 assistant 掩码
    msgs = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀"}]
    tool_msgs = [{"role": "user", "content": "3 乘 9 是多少"},
                 {"role": "assistant", "content": "", "tool_calls": [
                     {"type": "function", "function": {"name": "calculator", "arguments": {"expression": "3*9"}}}]},
                 {"role": "tool", "content": "27"}, {"role": "assistant", "content": "等于 27。"}]
    t2, sp2 = render_with_spans(tool_msgs)
    e2 = tok(t2, return_offsets_mapping=True)
    res["mask_example"] = [[tok.decode([i]), int(any(s <= a and b <= e for s, e in sp2))]
                           for i, (a, b) in zip(e2["input_ids"], e2["offset_mapping"])]
    try:
        out = tok.apply_chat_template(msgs, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
        mask = out.get("assistant_masks")
        res["hf_assistant_mask"] = {"ones": int(sum(mask)) if mask is not None else None, "length": len(out["input_ids"]),
                                    "template_has_generation_tag": "{% generation %}" in tok.chat_template}
    except Exception as e:
        res["hf_assistant_mask"] = {"error": f"{type(e).__name__}: {str(e)[:160]}",
                                    "template_has_generation_tag": "{% generation %}" in tok.chat_template}
    text, spans = render_with_spans(msgs)
    enc = tok(text, return_offsets_mapping=True)
    ours = [int(any(s <= a and b <= e for s, e in spans)) for a, b in enc["offset_mapping"]]
    res["our_mask"] = {"ones": sum(ours), "length": len(ours),
                       "tokens": [tok.decode([i]) for i, m in zip(enc["input_ids"], ours) if m]}
    print("D. transformers 自带的掩码:", res["hf_assistant_mask"])
    print("   本单元的区间换成 token:", res["our_mask"])

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(OUT, UNIT))


if __name__ == "__main__":
    main()
