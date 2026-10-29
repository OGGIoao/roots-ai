"""
practice_car_confidence.py — 实践: 小车 v0.4 的提示下, 模型写工具调用时每个 token 有多大把握 (讲义第 16.2 节)

通用版本 (不依赖任何项目) 是 real_model.py; 这个脚本只为讲义的实践一节服务。

默认用 Qwen2.5-0.5B-Instruct 在 CPU 上 (约 2 GB 内存, 不用显卡; --model 换成 1.5B 时用 bf16, 约 3 GB 内存), 对 v0.4 skill 模式的 6 条指令:
  1. 贪心 (不加重复惩罚) 写出第一轮回复, 一般是一个工具调用;
  2. 把 "提示 + 回复" 一次前向, 拿到回复里每个位置的原始 logits;
  3. 每个位置算三个数 (都是精确计算, 不是抽样统计):
       p_greedy       原始分布下, 贪心选中的那个 token 的概率
       p_keep_sample  用 Qwen2.5 自带的采样设置 (重复惩罚 1.1 → 温度 0.7 → top-k 20 → top-p 0.8) 处理后,
                      这一步仍然抽中同一个 token 的概率; 1 - 它 = 这一步被抽歪的概率
       flip_by_penalty  只加重复惩罚 1.1 (看提示词和前文) 时, 排第一的 token 会不会换掉
  4. 整条回复都和贪心一样的概率 = 各位置 p_keep_sample 的乘积。

处理 logits 用本单元的 sampling.py (已和 transformers 逐步核对过)。
    python experiments/practice_car_confidence.py
    python experiments/practice_car_confidence.py --model Qwen/Qwen2.5-1.5B-Instruct
结果写进 results/practice_car_confidence_<模型>.json; 另外记下第一个位置排前三的候选, 加重复惩罚前后各一份。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(UNIT)))
sys.path.insert(0, UNIT)
sys.path.insert(0, os.path.join(ROOT, "library", "agent", "agent-loop"))
sys.path.insert(0, os.path.join(ROOT, "projects", "car", "v0.4-text-commands"))
sys.path.insert(0, os.path.join(ROOT, "sim"))

import sampling as S  # noqa: E402

QWEN_SAMPLING = dict(rep=1.1, temp=0.7, k=20, p=0.8)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()
    MODEL = args.model
    OUT = os.path.join(HERE, "results", "practice_car_confidence_" + MODEL.split("/")[-1] + ".json")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agent import make_tasks
    from skills import SKILL_SCHEMAS, SKILL_SYSTEM_PROMPT

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    tok = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.float32 if "0.5B" in MODEL else torch.bfloat16      # 1.5B 用 bf16, 省一半内存
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).eval()

    rows = []
    for task in make_tasks(6):
        msgs = [{"role": "system", "content": SKILL_SYSTEM_PROMPT}, {"role": "user", "content": task["instruction"]}]
        prompt = tok.apply_chat_template(msgs, tools=SKILL_SCHEMAS, add_generation_prompt=True, tokenize=False)
        pids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model.generate(pids, max_new_tokens=60, do_sample=False, repetition_penalty=1.0,
                                 pad_token_id=tok.eos_token_id)
            gen = out[0, pids.shape[1]:].tolist()
            logits = model(out).logits[0, pids.shape[1] - 1:-1].double().tolist()     # 每个回复位置的原始 logits
        prev = pids[0].tolist()
        x0 = logits[0]
        top_raw = sorted(range(len(x0)), key=lambda i: -x0[i])[:3]
        pen0 = S.repetition_penalty(x0, prev, 1.1)
        top_pen = sorted(range(len(pen0)), key=lambda i: -pen0[i])[:3]
        first_step = {"raw": [(tok.decode([i]), round(S.softmax(x0)[i], 4)) for i in top_raw],
                      "with_penalty_1.1": [(tok.decode([i]), round(S.softmax(pen0)[i], 4)) for i in top_pen]}
        steps, keep_all = [], 1.0
        for t, x in zip(gen, logits):
            probs = S.softmax(x)
            processed = S.process(x, prev, do_sample=True, **QWEN_SAMPLING)
            keep = S.softmax(processed)[t]
            flip = S.greedy(S.repetition_penalty(x, prev, 1.1)) != t
            steps.append({"token": tok.decode([t]), "p_greedy": round(probs[t], 4), "p_keep_sample": round(keep, 4),
                          "flip_by_penalty": flip})
            keep_all *= keep
            prev.append(t)
        text = tok.decode(gen, skip_special_tokens=False)
        weakest = sorted(steps, key=lambda s: s["p_keep_sample"])[:3]
        rows.append({"instruction": task["instruction"], "target": task["target"], "output": text,
                     "n_tokens": len(gen), "p_whole_reply_same_under_sampling": round(keep_all, 4),
                     "n_flips_by_penalty": sum(s["flip_by_penalty"] for s in steps),
                     "weakest": weakest, "first_step_top3": first_step, "steps": steps})
        print(f"{task['instruction']}\n  回复: {text!r}\n  整条回复在采样下不变的概率 {keep_all:.3f}; "
              f"重复惩罚换掉第一名的位置 {rows[-1]['n_flips_by_penalty']} 个; 最没把握的: "
              + ", ".join(f"{w['token']!r} {w['p_keep_sample']:.2f}" for w in weakest)
              + f"\n  第一个位置前三: 原始 {first_step['raw']}; 加惩罚 {first_step['with_penalty_1.1']}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "sampling": QWEN_SAMPLING, "rows": rows}, f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(OUT, UNIT))


if __name__ == "__main__":
    main()
