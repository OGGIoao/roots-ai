"""
check_against_hf.py — 核对 sampling.py 的每一步和 transformers 的 LogitsProcessor 结果相同

不需要模型: 随机生成 N 组 logits (有平的、有尖的、有大量并列的), 分别交给手写函数和 transformers 的
RepetitionPenalty / Temperature / TopK / TopP / MinP 处理器, 比较:
  - 被去掉的位置 (-inf) 是否完全一样;
  - 留下来的分数是否相同 (float64, 误差 < 1e-9)。
最后再按 generate 的顺序串起来比一次。CPU, 几秒。

    python experiments/check_against_hf.py
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

OUT = os.path.join(HERE, "results", "check_against_hf.json")


def random_case(rng, vocab=50):
    kind = rng.choice(["flat", "peaked", "ties"])
    if kind == "flat":
        logits = [rng.gauss(0, 1) for _ in range(vocab)]
    elif kind == "peaked":
        logits = [rng.gauss(0, 1) for _ in range(vocab)]
        logits[rng.randrange(vocab)] += rng.uniform(3, 10)
    else:                                            # 很多并列: 分数只取几个整数
        logits = [float(rng.randint(-2, 2)) for _ in range(vocab)]
    prev = [rng.randrange(vocab) for _ in range(rng.randint(0, 20))]
    cfg = {"rep": rng.choice([1.0, 1.05, 1.1, 1.3, 2.0]), "temp": rng.choice([0.3, 0.7, 1.0, 1.5]),
           "k": rng.choice([0, 1, 5, 20]), "p": rng.choice([1.0, 0.95, 0.8, 0.5, 0.1]),
           "minp": rng.choice([0.0, 0.05, 0.1, 0.5])}
    return kind, logits, prev, cfg


def same(a, b):
    for x, y in zip(a, b):
        if (x == S.NEG_INF) != (y == S.NEG_INF):
            return False
        if x != S.NEG_INF and abs(x - y) > 1e-9:
            return False
    return True


def same_up_to_ties(a, b):
    """只在并列 token 的取舍上不同: 留下的个数一样, 留下的分数 (排序后) 一样。"""
    ka = sorted(x for x in a if x != S.NEG_INF)
    kb = sorted(y for y in b if y != S.NEG_INF)
    return len(ka) == len(kb) and all(abs(x - y) < 1e-9 for x, y in zip(ka, kb))


def main():
    import torch
    from transformers.generation.logits_process import (MinPLogitsWarper, RepetitionPenaltyLogitsProcessor,
                                                        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)

    def hf(proc, logits, prev):
        ids = torch.tensor([prev or [0]], dtype=torch.long)
        scores = torch.tensor([logits], dtype=torch.float64)
        if not prev:                                 # 没有历史时, 重复惩罚什么都不做; 别的处理器不看 ids
            ids = ids[:, :0]
        return proc(ids, scores)[0].tolist()

    rng = random.Random(0)
    n = 2000
    names = ["repetition_penalty", "temperature", "top_k", "top_p", "min_p", "chain"]
    tally = {name: 0 for name in names}
    ties_only = {name: 0 for name in names}      # 不完全相同, 但只差在并列 token 留哪个
    kinds = {}
    first_bad = {}
    for _ in range(n):
        kind, logits, prev, c = random_case(rng)
        kinds[kind] = kinds.get(kind, 0) + 1
        checks = {
            "repetition_penalty": (S.repetition_penalty(logits, prev, c["rep"]),
                                   hf(RepetitionPenaltyLogitsProcessor(c["rep"]), logits, prev)),
            "temperature": (S.temperature(logits, c["temp"]), hf(TemperatureLogitsWarper(c["temp"]), logits, prev)),
            "top_k": (S.top_k(logits, c["k"]) if c["k"] else logits,
                      hf(TopKLogitsWarper(c["k"]), logits, prev) if c["k"] else logits),
            "top_p": (S.top_p(logits, c["p"]), hf(TopPLogitsWarper(c["p"]), logits, prev) if c["p"] < 1 else logits),
            "min_p": (S.min_p(logits, c["minp"]) if c["minp"] else logits,
                      hf(MinPLogitsWarper(c["minp"]), logits, prev) if c["minp"] else logits),
        }
        # 串起来: 按 generate 的顺序
        x = hf(RepetitionPenaltyLogitsProcessor(c["rep"]), logits, prev)
        x = hf(TemperatureLogitsWarper(c["temp"]), x, prev)
        if c["k"]:
            x = hf(TopKLogitsWarper(c["k"]), x, prev)
        if c["p"] < 1:
            x = hf(TopPLogitsWarper(c["p"]), x, prev)
        if c["minp"]:
            x = hf(MinPLogitsWarper(c["minp"]), x, prev)
        checks["chain"] = (S.process(logits, prev, **c), x)
        for name, (mine, theirs) in checks.items():
            if same(mine, theirs):
                tally[name] += 1
                continue
            if same_up_to_ties(mine, theirs):
                ties_only[name] += 1
            elif name not in first_bad:
                first_bad[name] = {"kind": kind, "cfg": c, "logits": logits, "prev": prev}

    for name, ok in tally.items():
        print(f"  {name:<20} {ok}/{n} 完全相同, {ties_only[name]} 只差在并列 token 留哪个, "
              f"{n - ok - ties_only[name]} 真不同")
    res = {"n": n, "kinds": kinds, "same": tally, "ties_only": ties_only, "first_mismatch": first_bad}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda v: None if v == -math.inf else v)
    print("写入", os.path.relpath(OUT, UNIT))


if __name__ == "__main__":
    main()
