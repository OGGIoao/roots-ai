"""
toy_experiments.py — 讲义第 3~6、8 节的数字从这里来 (只用标准库, 全部在 ToyLM 上, CPU 约 1 分钟)

  A. 贪心为什么会转圈: 找出贪心生成里的循环 (周期和起点)
  B. 多样性 vs 像不像: 各种解码设置, 每种 20 个种子, 量 "不重复的 4 字片段占比" 和 "模型自己给的平均对数概率"
  C. 重复惩罚和 logits 的整体平移: softmax 不在乎所有分数加同一个常数, 重复惩罚在乎
  D. 截断留下几个候选: 在 2000 个真实上下文上, top-k / top-p / min-p 各留下多少个字符, 和分布的熵什么关系

    python experiments/toy_experiments.py
结果写进 results/toy_experiments.json; 图由 figures/make_charts.py 画。
"""

import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
sys.path.insert(0, UNIT)

import sampling as S  # noqa: E402
from generate import generate  # noqa: E402
from toy_lm import ToyLM  # noqa: E402

OUT = os.path.join(HERE, "results", "toy_experiments.json")
PROMPTS = ["ROMEO:\n", "KING HENRY:\n", "First Citizen:\n", "JULIET:\nO ", "MENENIUS:\nI "]


def find_cycle(s, min_reps=3):
    """s 的结尾是否在重复同一段: 返回 (周期, 开始重复的位置) 或 None。"""
    n = len(s)
    for period in range(1, n // min_reps + 1):
        tail = s[n - period:]
        if s[n - period * min_reps:] != tail * min_reps:
            continue
        start = n - period * min_reps
        while start - period >= 0 and s[start - period:start] == tail:
            start -= period
        while start > 0 and s[start - 1] == s[start - 1 + period]:
            start -= 1
        return period, start
    return None


def distinct(s, n=4):
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def main():
    lm = ToyLM.from_corpus()
    step = lambda ids: lm.logits(lm.decode(ids[-2:]))      # noqa: E731
    res = {"V": lm.V}

    # A. 贪心转圈
    loops = []
    for p in PROMPTS:
        out, _ = generate(step, lm.encode(p), 300, do_sample=False)
        text = lm.decode(out)
        cyc = find_cycle(text)
        loops.append({"prompt": p, "text_head": text[:80], "cycle": text[cyc[1]:cyc[1] + cyc[0]] if cyc else None,
                      "period": cyc[0] if cyc else None, "loop_starts_at": cyc[1] if cyc else None})
    res["greedy_loops"] = loops
    print("A. 贪心转圈:", [(d["prompt"].strip(), d["cycle"], d["loop_starts_at"]) for d in loops])

    # B. 多样性 vs 像不像
    settings = [
        ("贪心", dict(do_sample=False)),
        ("贪心 + 重复惩罚 1.1", dict(do_sample=False, rep=1.1)),
        ("贪心 + 重复惩罚 1.3", dict(do_sample=False, rep=1.3)),
        ("贪心 + 重复惩罚 2.0", dict(do_sample=False, rep=2.0)),
        ("温度 0.5", dict(temp=0.5)),
        ("温度 0.7", dict(temp=0.7)),
        ("温度 1.0", dict(temp=1.0)),
        ("温度 1.5", dict(temp=1.5)),
        ("温度 1.0 + top-k 5", dict(temp=1.0, k=5)),
        ("温度 1.0 + top-p 0.9", dict(temp=1.0, p=0.9)),
        ("温度 1.0 + min-p 0.1", dict(temp=1.0, minp=0.1)),
        ("温度 1.5 + min-p 0.1", dict(temp=1.5, minp=0.1)),
        ("Qwen2.5 自带设置 (温度 0.7, top-k 20, top-p 0.8, 重复惩罚 1.1)", dict(temp=0.7, k=20, p=0.8, rep=1.1)),
    ]
    rows = []
    for name, cfg in settings:
        cfg = {"do_sample": True, **cfg}
        divs, lps, looped = [], [], 0
        seeds = range(1) if not cfg["do_sample"] else range(20)
        for seed in seeds:
            for p in PROMPTS:
                ctx = lm.encode(p)
                out, _ = generate(step, ctx, 300, rng=random.Random(seed), **cfg)
                text = lm.decode(out)
                divs.append(distinct(text))
                full = p + text
                lps.append(sum(lm.logits(full[:i])[lm.stoi[full[i]]] for i in range(len(p), len(full))) / len(text))
                looped += find_cycle(text) is not None
        rows.append({"setting": name, "cfg": cfg, "runs": len(divs), "distinct4": round(sum(divs) / len(divs), 4),
                     "avg_logprob": round(sum(lps) / len(lps), 4), "loop_rate": round(looped / len(divs), 3)})
        print(f"  {name:<58} 不重复4字片段 {rows[-1]['distinct4']:.3f}  平均对数概率 {rows[-1]['avg_logprob']:.3f}  "
              f"转圈 {rows[-1]['loop_rate']:.0%}")
    res["tradeoff"] = rows

    # 参照: 语料本身 (同样长度的真实片段) 的两个指标
    from common.corpus import load_shakespeare
    text = load_shakespeare()
    rng = random.Random(0)
    divs, lps = [], []
    for _ in range(100):
        i = rng.randrange(2, len(text) - 300)
        seg = text[i:i + 300]
        divs.append(distinct(seg))
        lps.append(sum(lm.logits(text[:i + j])[lm.stoi[text[i + j]]] for j in range(300)) / 300)
    res["corpus_reference"] = {"distinct4": round(sum(divs) / len(divs), 4), "avg_logprob": round(sum(lps) / len(lps), 4)}
    print("  语料本身:", res["corpus_reference"])

    # C. 平移
    base = lm.logits("th")
    seen = [lm.stoi["e"]]
    shift_rows = []
    for c in (-20.0, 0.0, 20.0):
        x = [z + c for z in base]
        before = S.softmax(x)[seen[0]]
        after = S.softmax(S.repetition_penalty(x, seen, 1.3))[seen[0]]
        shift_rows.append({"shift": c, "p_e_before": round(before, 4), "p_e_after": round(after, 4)})
    res["penalty_shift"] = {"context": "th", "token": "e", "penalty": 1.3, "rows": shift_rows}
    print("C. 平移:", shift_rows)

    # D. 截断留下几个
    rng = random.Random(1)
    pts = []
    for _ in range(2000):
        i = rng.randrange(2, len(text))
        x = lm.logits(text[:i])
        probs = S.softmax(x)
        ent = -sum(p * math.log2(p) for p in probs if p > 0)
        kept = lambda y: sum(z != S.NEG_INF for z in y)       # noqa: E731
        pts.append({"entropy_bits": round(ent, 3), "top_k5": kept(S.top_k(x, 5)), "top_p0.9": kept(S.top_p(x, 0.9)),
                    "min_p0.1": kept(S.min_p(x, 0.1))})
    res["truncation"] = pts
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 6)):
        sel = [p for p in pts if lo <= p["entropy_bits"] < hi]
        if sel:
            avg = lambda k: sum(p[k] for p in sel) / len(sel)  # noqa: E731
            print(f"D. 熵 {lo}-{hi} bit ({len(sel)} 个上下文): top-k5 留 {avg('top_k5'):.1f}, "
                  f"top-p0.9 留 {avg('top_p0.9'):.1f}, min-p0.1 留 {avg('min_p0.1'):.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(OUT, UNIT))


if __name__ == "__main__":
    main()
