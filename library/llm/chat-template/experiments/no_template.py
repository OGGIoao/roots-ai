"""
no_template.py — 讲义第 2 节: 不套模板、套错模板、套对模板, 同一个对话模型分别会写出什么

用 Qwen2.5-0.5B-Instruct 在 CPU 上 (约 2 GB 内存, 不用显卡), 对 5 个问题各试三种喂法:
  raw      直接把问题文字喂进去, 不加任何标记
  wrong    套一个别的家族的格式 (Mistral 的 [INST] ... [/INST])
  right    套模型自己的 ChatML 模板, 末尾加生成提示
贪心解码, 不加重复惩罚, 最多写 60 个 token; 记录写出了什么、有没有自己停下 (写出 <|im_end|>)。

    python experiments/no_template.py      # 约 1~2 分钟
结果写进 results/no_template.json。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
sys.path.insert(0, UNIT)

from chat_template import render  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = os.path.join(HERE, "results", "no_template.json")
QUESTIONS = ["台北现在几度?", "法国的首都是哪里?", "用一句话解释什么是对话模板。", "3 乘 9 等于多少?", "写一个 Python 函数, 返回两数之和。"]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    end_of_text = tok.convert_tokens_to_ids("<|endoftext|>")

    def run(text):
        ids = tok(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=60, do_sample=False, repetition_penalty=1.0,
                                 eos_token_id=[im_end, end_of_text], pad_token_id=end_of_text)
        new = out[0, ids.shape[1]:].tolist()
        return {"output": tok.decode(new, skip_special_tokens=False), "n_tokens": len(new),
                "stopped": bool(new) and new[-1] in (im_end, end_of_text),
                "stop_token": tok.decode([new[-1]]) if new and new[-1] in (im_end, end_of_text) else None}

    rows = []
    for q in QUESTIONS:
        feeds = {
            "raw": q,
            "wrong": f"[INST] {q} [/INST]",
            "right": render([{"role": "user", "content": q}], add_generation_prompt=True),
        }
        row = {"question": q}
        for name, text in feeds.items():
            row[name] = run(text)
            print(f"{q} [{name}] 停了={row[name]['stopped']} ({row[name]['n_tokens']} token): {row[name]['output'][:70]!r}",
                  flush=True)
        rows.append(row)

    summary = {name: {"stopped": sum(r[name]["stopped"] for r in rows), "n": len(rows)} for name in ("raw", "wrong", "right")}
    print("自己停下的次数:", summary)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(OUT, UNIT))


if __name__ == "__main__":
    main()
