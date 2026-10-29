"""
toy_more.py — 讲义第 4、6、7、9 节和专题 A 的数字 (只用标准库, 全部在 ToyLM 上, CPU 几秒)

  E. 哪种惩罚能打破贪心的转圈: 全历史重复惩罚 / 只看最近 64 个 (llama.cpp) / 加法的频率和出现惩罚 / 禁止重复 n 元组
  F. 加法惩罚和整体平移: 和 toy_experiments.py 的 C 部分对照
  G. 三个框架的执行顺序: 同样的设置, 在 2000 个真实上下文上, 留下的候选集合有多常不一样, 抽签分布差多少
  H. 温度和熵: 温度从 0.25 到 3, 下一个字符分布的平均熵怎么变
  I. 束宽和总分: 束宽 1 到 32, 总分升高了多少, 还转不转圈

    python experiments/toy_more.py
结果写进 results/toy_more.json。
"""

import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
sys.path.insert(0, UNIT)
sys.path.insert(0, HERE)

import sampling as S  # noqa: E402
from toy_experiments import PROMPTS, distinct, find_cycle  # noqa: E402
from toy_lm import ToyLM  # noqa: E402

OUT = os.path.join(HERE, "results", "toy_more.json")


def entropy_bits(logits):
    probs = S.softmax(logits)
    return -sum(p * math.log2(p) for p in probs if p > 0)


def greedy_with(lm, prompt, n, penalize):
    """贪心生成 n 个字符; penalize(logits, 提示+已写的 ids, 已写的 ids) 返回处理后的分数。"""
    ids = lm.encode(prompt)
    new = []
    for _ in range(n):
        x = penalize(lm.logits(lm.decode(ids[-2:])), ids, new)
        t = S.greedy(x)
        ids.append(t)
        new.append(t)
    return lm.decode(new)


def main():
    lm = ToyLM.from_corpus()
    res = {}

    # E. 惩罚的几种写法 vs 贪心转圈
    variants = {
        "不加惩罚": lambda x, ids, new: x,
        "重复惩罚 1.3 (看全部历史, transformers)": lambda x, ids, new: S.repetition_penalty(x, ids, 1.3),
        "重复惩罚 1.3 (只看最近 64 个, llama.cpp)": lambda x, ids, new: S.windowed_repetition_penalty(x, ids, 1.3, 64),
        "重复惩罚 1.3 (只看最近 8 个)": lambda x, ids, new: S.windowed_repetition_penalty(x, ids, 1.3, 8),
        "出现惩罚 0.5 (加法)": lambda x, ids, new: S.presence_frequency_penalty(x, new, presence=0.5),
        "频率惩罚 0.3 (加法, 按次数)": lambda x, ids, new: S.presence_frequency_penalty(x, new, frequency=0.3),
        "禁止重复 4 元组": lambda x, ids, new: S.no_repeat_ngram(x, ids, 4),
        "禁止重复 6 元组": lambda x, ids, new: S.no_repeat_ngram(x, ids, 6),
    }
    rows = []
    for name, f in variants.items():
        texts = [greedy_with(lm, p, 300, f) for p in PROMPTS]
        lps = []
        for p, t in zip(PROMPTS, texts):
            full = p + t
            lps.append(sum(lm.logits(full[:i])[lm.stoi[full[i]]] for i in range(len(p), len(full))) / len(t))
        rows.append({"setting": name, "loop_rate": sum(find_cycle(t) is not None for t in texts) / len(texts),
                     "distinct4": round(sum(distinct(t) for t in texts) / len(texts), 4),
                     "avg_logprob": round(sum(lps) / len(lps), 4), "sample": texts[0][:90]})
        print(f"E. {name:<36} 转圈 {rows[-1]['loop_rate']:.0%}  不重复4字片段 {rows[-1]['distinct4']:.3f}  "
              f"平均对数概率 {rows[-1]['avg_logprob']:.3f}  {texts[0][:50]!r}")
    res["penalties"] = rows

    # F. 加法惩罚不受平移影响
    base = lm.logits("th")
    e = lm.stoi["e"]
    shift_rows = []
    for c in [x * 2.5 for x in range(-8, 9)]:          # -20 到 +20
        x = [z + c for z in base]
        shift_rows.append({"shift": c,
                           "repetition_1.3": round(S.softmax(S.repetition_penalty(x, [e], 1.3))[e], 4),
                           "presence_0.5": round(S.softmax(S.presence_frequency_penalty(x, [e], presence=0.5))[e], 4)})
    res["shift"] = {"context": "th", "token": "e", "p_before": round(S.softmax(base)[e], 4), "rows": shift_rows}
    print("F.", res["shift"])

    # G. 三个框架的顺序
    from common.corpus import load_shakespeare
    text = load_shakespeare()
    rng = random.Random(2)
    settings = {"温度 0.7, top-k 20, top-p 0.8 (Qwen2.5 的设置, 不含惩罚)": dict(temp=0.7, k=20, p=0.8),
                "温度 0.7, top-p 0.9, min-p 0.05": dict(temp=0.7, p=0.9, minp=0.05),
                "温度 1.5, top-p 0.9, min-p 0.1": dict(temp=1.5, p=0.9, minp=0.1),
                "温度 1.0, top-k 20, top-p 0.8, min-p 0.05": dict(temp=1.0, k=20, p=0.8, minp=0.05)}
    positions = [rng.randrange(2, len(text)) for _ in range(2000)]
    order_rows = []
    for name, cfg in settings.items():
        diff_sets = {"vllm": 0, "llama.cpp": 0}
        tvd = {"vllm": [], "llama.cpp": []}
        kept = {o: [] for o in S.ORDERS}
        for pos in positions:
            x = lm.logits(text[:pos])
            dists = {o: S.softmax(S.process_order(x, (), o, **cfg)) for o in S.ORDERS}
            for o, d in dists.items():
                kept[o].append(sum(v > 0 for v in d))
            ref = dists["transformers"]
            for o in ("vllm", "llama.cpp"):
                d = dists[o]
                if [v > 0 for v in d] != [v > 0 for v in ref]:
                    diff_sets[o] += 1
                tvd[o].append(0.5 * sum(abs(a - b) for a, b in zip(d, ref)))
        row = {"setting": name, "cfg": cfg, "n_contexts": len(positions),
               "kept_mean": {o: round(sum(v) / len(v), 2) for o, v in kept.items()},
               "kept_set_differs_from_transformers": diff_sets,
               "mean_tvd_vs_transformers": {o: round(sum(v) / len(v), 4) for o, v in tvd.items()},
               "max_tvd_vs_transformers": {o: round(max(v), 4) for o, v in tvd.items()}}
        order_rows.append(row)
        print(f"G. {name}: 平均留下 {row['kept_mean']}; 候选集合和 transformers 不同的上下文 {diff_sets}; "
              f"平均总变差 {row['mean_tvd_vs_transformers']}")
    res["orders"] = order_rows

    # H. 温度与熵
    temps = [0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    ctx = [lm.logits(text[:p]) for p in positions[:500]]
    res["temperature_entropy"] = [{"temp": t, "mean_entropy_bits": round(sum(entropy_bits(S.temperature(x, t)) for x in ctx) / len(ctx), 3)}
                                  for t in temps]
    res["uniform_entropy_bits"] = round(math.log2(lm.V), 3)
    print("H.", [(r["temp"], r["mean_entropy_bits"]) for r in res["temperature_entropy"]], "均匀分布", res["uniform_entropy_bits"])

    # I. 束宽
    from beam import beam_search
    step = lambda ids: lm.logits(lm.decode(ids[-2:]))
    pids = lm.encode("ROMEO:\n")
    res["beam"] = []
    for b in (1, 2, 4, 8, 16, 32):
        seq, score = beam_search(step, pids, b, 40)
        t = lm.decode(seq)
        res["beam"].append({"width": b, "score": round(score, 2), "text": t, "loops": find_cycle(t) is not None})
        print(f"I. 束宽 {b:>2}: 总分 {score:.2f}  {t!r}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(OUT, UNIT))


if __name__ == "__main__":
    main()
