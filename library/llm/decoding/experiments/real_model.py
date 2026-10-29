"""
real_model.py — 讲义专题 B、C: 把前面在 ToyLM 上看到的现象, 放到真实的小模型上量一量 (只用 CPU)

例子是 chat-template 单元的通用小助手 (查天气、算数、查资料三个工具), 12 个问题: 6 个该调工具, 6 个不该。
对每个问题:
  1. 贪心写出回复两次: 关掉重复惩罚 (repetition_penalty=1.0), 和模型自带的 1.1。看回复变没变、第一个 token 变没变、
     还是不是工具调用;
  2. 在 "关掉惩罚" 的那条回复上逐位置算:
       p_greedy       原始分布下, 贪心选中的 token 的概率
       p_keep_sample  用 Qwen2.5 自带的采样设置 (重复惩罚 1.1 → 温度 0.7 → top-k 20 → top-p 0.8) 处理后, 仍然抽中它的概率
     整条回复在采样下不变的概率 = 各位置 p_keep_sample 的乘积 (精确计算, 不是抽样统计);
  3. 两条计算路径的分数: generate 逐步算 (带 KV 缓存) 和把整段一次算完, 排第一的 token 有几个位置不同, 分数最多差多少。
加 --precision 时, 另外用 bf16 再贪心写一遍, 和默认精度的回复比 (专题 C)。

    python experiments/real_model.py                                         # 0.5B, fp32, 约 2 GB 内存, 几分钟
    python experiments/real_model.py --precision                             # 再加 0.5B bf16
    python experiments/real_model.py --model Qwen/Qwen2.5-1.5B-Instruct      # 1.5B 用 bf16, 约 3 GB 内存, 十几分钟
结果写进 results/real_model_<模型>.json。
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
sys.path.insert(0, UNIT)
sys.path.insert(0, os.path.join(os.path.dirname(UNIT), "chat-template"))

import sampling as S  # noqa: E402
from examples import SYSTEM, TOOLS  # noqa: E402

QWEN_SAMPLING = dict(rep=1.1, temp=0.7, k=20, p=0.8)
QUESTIONS = [
    ("台北现在几度?", True),
    ("东京今天下雨吗?", True),
    ("帮我算一下 37 乘以 49。", True),
    ("(128 + 256) / 3 等于多少?", True),
    ("在资料库里查一下年假怎么请。", True),
    ("搜一下报销流程的相关段落。", True),
    ("你好!", False),
    ("用一句话介绍你自己。", False),
    ("把 '谢谢' 翻译成英文。", False),
    ("写一句关于秋天的诗。", False),
    ("Python 里怎么把列表倒过来?", False),
    ("1 加 1 等于几?", False),
]


def greedy(model, tok, pids, rep, n):
    import torch
    with torch.no_grad():
        out = model.generate(pids, max_new_tokens=n, do_sample=False, repetition_penalty=rep,
                             temperature=None, top_p=None, top_k=None,
                             output_logits=True, return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
    gen = out.sequences[0, pids.shape[1]:].tolist()
    inc = [x[0].double().tolist() for x in out.logits]          # 逐步算出来的原始 logits
    return out.sequences, gen, inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--precision", action="store_true")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float32 if "0.5B" in args.model else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).eval()
    low = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval() if args.precision else None
    call_id = tok.convert_tokens_to_ids("<tool_call>")

    rows = []
    for q, should_call in QUESTIONS:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False)
        pids = tok(prompt, return_tensors="pt")["input_ids"]
        seq, gen, inc = greedy(model, tok, pids, 1.0, args.max_new)
        _, gen_pen, _ = greedy(model, tok, pids, 1.1, args.max_new)
        with torch.no_grad():
            full = model(seq).logits[0, pids.shape[1] - 1:-1].double().tolist()     # 整段一次算

        prev = pids[0].tolist()
        x0 = inc[0]
        pen0 = S.repetition_penalty(x0, prev, 1.1)
        top = lambda x: [(tok.decode([i]), round(S.softmax(x)[i], 4)) for i in sorted(range(len(x)), key=lambda i: -x[i])[:3]]
        steps, keep_all, flips = [], 1.0, 0
        for t, x in zip(gen, inc):
            keep = S.softmax(S.process(x, prev, do_sample=True, **QWEN_SAMPLING))[t]
            flips += S.greedy(S.repetition_penalty(x, prev, 1.1)) != t
            steps.append({"token": tok.decode([t]), "p_greedy": round(S.softmax(x)[t], 4), "p_keep_sample": round(keep, 4)})
            keep_all *= keep
            prev.append(t)
        path_flips = sum(S.greedy(a) != S.greedy(b) for a, b in zip(inc, full))
        path_maxdiff = max(max(abs(u - v) for u, v in zip(a, b)) for a, b in zip(inc, full))
        row = {"question": q, "should_call": should_call,
               "reply": tok.decode(gen), "reply_with_penalty": tok.decode(gen_pen),
               "calls": gen[0] == call_id, "calls_with_penalty": gen_pen[0] == call_id,
               "same_reply_with_penalty": gen == gen_pen, "first_token_changed_by_penalty": gen[0] != gen_pen[0],
               "first_top3": top(x0), "first_top3_with_penalty": top(pen0),
               "n_tokens": len(gen), "positions_flipped_by_penalty": flips,
               "p_whole_reply_same_under_sampling": round(keep_all, 4),
               "weakest": sorted(steps, key=lambda s: s["p_keep_sample"])[:3],
               "incremental_vs_full_argmax_differs": path_flips, "incremental_vs_full_max_abs_diff": round(path_maxdiff, 4),
               "steps": steps}
        if low is not None:
            _, gen_low, _ = greedy(low, tok, pids, 1.0, args.max_new)
            diverge = next((i for i, (a, b) in enumerate(zip(gen, gen_low)) if a != b), None)
            if diverge is None and len(gen) != len(gen_low):
                diverge = min(len(gen), len(gen_low))
            row.update({"reply_bf16": tok.decode(gen_low), "same_reply_bf16": gen == gen_low, "bf16_diverges_at": diverge})
        rows.append(row)
        print(f"{q}\n  关掉惩罚: {row['reply'][:70]!r}\n  惩罚 1.1: {row['reply_with_penalty'][:70]!r}\n"
              f"  第一个 token 变了: {row['first_token_changed_by_penalty']}; 惩罚换掉第一名的位置 {flips}/{len(gen)}; "
              f"整条回复在采样下不变的概率 {keep_all:.3f}; 两条计算路径第一名不同 {path_flips}, 分数最多差 {path_maxdiff:.4f}"
              + (f"\n  bf16 一样: {row['same_reply_bf16']} (从第 {row['bf16_diverges_at']} 个 token 开始不同)" if low is not None else ""),
              flush=True)

    summary = {
        "n": len(rows),
        "calls_correct": sum(r["calls"] == r["should_call"] for r in rows),
        "calls_correct_with_penalty": sum(r["calls_with_penalty"] == r["should_call"] for r in rows),
        "reply_changed_by_penalty": sum(not r["same_reply_with_penalty"] for r in rows),
        "first_token_changed_by_penalty": sum(r["first_token_changed_by_penalty"] for r in rows),
        "sample_same_ge_0.9": sum(r["p_whole_reply_same_under_sampling"] >= 0.9 for r in rows),
        "sample_same_lt_0.1": sum(r["p_whole_reply_same_under_sampling"] < 0.1 for r in rows),
        "path_argmax_differs_positions": sum(r["incremental_vs_full_argmax_differs"] for r in rows),
        "path_positions": sum(r["n_tokens"] for r in rows),
        "path_max_abs_diff": max(r["incremental_vs_full_max_abs_diff"] for r in rows),
    }
    if low is not None:
        summary["bf16_reply_differs"] = sum(not r["same_reply_bf16"] for r in rows)
    print(json.dumps(summary, ensure_ascii=False))
    out = os.path.join(HERE, "results", "real_model_" + args.model.split("/")[-1] + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "dtype": str(dtype), "sampling": QWEN_SAMPLING, "summary": summary, "rows": rows},
                  f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(out, UNIT))


if __name__ == "__main__":
    main()
