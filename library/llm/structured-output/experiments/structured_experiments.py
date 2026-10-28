"""
structured_experiments.py — 结构化输出讲义的数字从这里来

  A. ToyLM (只用标准库, 1 秒): 一个只读过莎士比亚的字符级模型, 加上约束后能不能写出合法的工具调用? 写哪一个?
  B. Qwen2.5 (CPU, 不用显卡): 小车 v0.4 go_to 模式的 6 条指令, 在 "重复惩罚 1.1" (decoding 讲义第 8.3 节,
     1.5B 会写成普通文字) 下, 不加约束 / 加约束 / 加约束且关掉惩罚, 各写出几个合法的工具调用、目标对不对、
     约束每一步多花多少时间。

    python experiments/structured_experiments.py           # A + B (0.5B 约 2 分钟; 1.5B 用 bf16, 约 10 分钟)
    python experiments/structured_experiments.py --only-toy

结果写进 results/structured.json。
"""

import argparse
import json

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(UNIT)))
sys.path.insert(0, UNIT)
sys.path.insert(0, os.path.join(ROOT, "library", "llm", "decoding"))
sys.path.insert(0, os.path.join(ROOT, "library", "agent", "agent-loop"))
sys.path.insert(0, os.path.join(ROOT, "projects", "car", "v0.4-text-commands"))
sys.path.insert(0, os.path.join(ROOT, "sim"))

from constrained import Trie, allowed_token_ids, car_tool_calls, mask_logits  # noqa: E402

OUT = os.path.join(HERE, "results", "structured.json")


def toy_part():
    """ToyLM 是字符级的, 每个字符就是一个 token, 约束最简单。"""
    import sampling as S
    from toy_lm import ToyLM
    lm = ToyLM.from_corpus()
    calls = car_tool_calls()
    # ToyLM 的词表里没有 '<' '>' '_' '{' '}' 这几个字符吗? 先查, 没有的话它根本写不出来
    missing = sorted({ch for c in calls for ch in c} - set(lm.vocab))
    res = {"missing_chars": missing}
    if missing:
        print("A. ToyLM 的词表 (莎士比亚里出现过的 65 个字符) 缺这些字符:", missing)
    # 只保留 ToyLM 能写出来的部分: 把缺的字符换掉之后的 "工具调用"
    table = {ch: "" for ch in missing}
    calls_toy = sorted({c.translate(str.maketrans(table)) for c in calls})
    trie = Trie(calls_toy)
    vocab = lm.vocab
    text, prompt = "", "ROMEO:\n"
    while not trie.is_complete(text):
        x = lm.logits(prompt + text)
        x = mask_logits(x, allowed_token_ids(trie, text, vocab))
        text += vocab[S.greedy(x)]
    # 每个合法输出在模型 (不加约束) 下的总概率, 以及加约束后贪心选了哪个
    logp = {}
    for c in calls_toy:
        logp[c] = sum(lm.logits(prompt + c[:i])[lm.stoi[c[i]]] for i in range(len(c)))
    free, _ = None, None
    from generate import generate
    out, _ = generate(lambda ids: lm.logits(lm.decode(ids[-2:])), lm.encode(prompt), 60, do_sample=False)
    free = lm.decode(out)
    res.update({"calls": calls_toy, "constrained_greedy": text, "unconstrained_greedy_head": free,
                "logprob_of_each_call": {c: round(v, 2) for c, v in logp.items()}})
    print("A. 不加约束:", repr(free[:40]))
    print("   加约束 (贪心):", repr(text))
    for c, v in sorted(logp.items(), key=lambda kv: -kv[1]):
        print(f"   {v:8.2f}  {c!r}")
    return res


def qwen_part(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agent import make_tasks
    from llm import parse_qwen_output
    from skills import SKILL_SCHEMAS, SKILL_SYSTEM_PROMPT

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float32 if "0.5B" in model_id else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
    vocab = [tok.decode([i]) for i in range(len(tok))]
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    trie = Trie(car_tool_calls())

    settings = [("不加约束, 重复惩罚 1.1", False, 1.1), ("加约束, 重复惩罚 1.1", True, 1.1), ("加约束, 重复惩罚 1.0", True, 1.0)]
    rows = []
    for name, constrained, rep in settings:
        mask_times, results = [], []
        for task in make_tasks(6):
            msgs = [{"role": "system", "content": SKILL_SYSTEM_PROMPT}, {"role": "user", "content": task["instruction"]}]
            prompt = tok.apply_chat_template(msgs, tools=SKILL_SCHEMAS, add_generation_prompt=True, tokenize=False)
            pids = tok(prompt, return_tensors="pt")["input_ids"]
            n0 = pids.shape[1]

            def allowed(batch_id, ids):
                t0 = time.perf_counter()
                text = tok.decode(ids[n0:], skip_special_tokens=False)
                ok = allowed_token_ids(trie, text, vocab, eos)
                mask_times.append(time.perf_counter() - t0)
                return sorted(ok) if ok else [eos]

            with torch.no_grad():
                out = model.generate(pids, max_new_tokens=40, do_sample=False, repetition_penalty=rep,
                                     pad_token_id=tok.eos_token_id,
                                     prefix_allowed_tokens_fn=allowed if constrained else None)
            text = tok.decode(out[0, n0:], skip_special_tokens=True)
            content, calls = parse_qwen_output(text)
            ok_call = [c for c in calls if c["name"] in ("go_to", "look")]
            target = None
            if ok_call and ok_call[0]["name"] == "go_to":
                try:
                    target = json.loads(ok_call[0]["arguments"]).get("target")
                except json.JSONDecodeError:
                    pass
            results.append({"instruction": task["instruction"], "want": task["target"], "output": text,
                            "valid_call": bool(ok_call), "tool": ok_call[0]["name"] if ok_call else None,
                            "target": target, "target_correct": target == task["target"]})
        row = {"setting": name, "constrained": constrained, "repetition_penalty": rep,
               "valid_calls": sum(r["valid_call"] for r in results), "correct_target": sum(r["target_correct"] for r in results),
               "n": len(results), "results": results}
        if mask_times:
            row["mask_ms_per_step"] = round(1000 * sum(mask_times) / len(mask_times), 1)
        rows.append(row)
        print(f"  {model_id.split('/')[-1]:<24} {name:<18} 合法调用 {row['valid_calls']}/6  目标正确 {row['correct_target']}/6"
              + (f"  约束每步 {row['mask_ms_per_step']} ms" if mask_times else ""), flush=True)
        for r in results:
            print(f"      {r['instruction']:<16} {r['output'][:70]!r}", flush=True)
    return {"model": model_id, "vocab_size": len(vocab), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-toy", action="store_true")
    ap.add_argument("--models", nargs="*", default=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"])
    args = ap.parse_args()
    res = {"toy": toy_part()}
    if not args.only_toy:
        res["qwen"] = [qwen_part(m) for m in args.models]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(OUT, UNIT))



if __name__ == "__main__":
    main()
