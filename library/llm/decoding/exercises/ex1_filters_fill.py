"""
练习 1 (① 填空): 补全三个 logits 处理步骤

softmax、温度、top-k 已经写好, 当作例子。把 top_p、min_p、repetition_penalty 三处 TODO 补全,
让它们在 600 组随机打分上和参考实现 (../sampling.py) 的结果相同: 被去掉的位置 (-inf) 完全一样,
留下来的分数误差 < 1e-9。参考实现又和 transformers 的处理器逐一核对过 (讲义第 8.2 节)。

验收:
    cd library/llm/decoding/exercises
    python -m pytest test_ex1.py -q

提示规则: 卡住了可以要提示, 按"方向 → 哪一行 → 代码"分级给, 每次只升一级。写完再去对照 ../sampling.py。
"""

import math

NEG_INF = float("-inf")


def softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) if z != NEG_INF else 0.0 for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


def temperature(logits, t):
    return [z / t for z in logits]


def top_k(logits, k):
    if k <= 0 or k >= len(logits):
        return list(logits)
    kth = sorted(logits, reverse=True)[k - 1]
    return [z if z >= kth else NEG_INF for z in logits]          # 和第 k 名同分的也留下


def top_p(logits, p, min_keep=1):
    """
    留下 "按概率从高到低累加, 刚好凑够 p" 的最少几个 token, 其余设成 -inf。
    为了和 transformers 逐位一致, 请按它的写法: 按概率**从低到高**排序, 依次累加,
    累计概率 <= 1 - p 的那些去掉; 但无论如何, 概率最高的 min_keep 个不去掉。
    p >= 1 时原样返回。
    """
    # TODO 1
    raise NotImplementedError


def min_p(logits, p, min_keep=1):
    """
    概率低于 "最高概率 × p" 的设成 -inf。如果留下的不足 min_keep 个, 把概率最高的 min_keep 个补回来。
    """
    # TODO 2
    raise NotImplementedError


def repetition_penalty(logits, prev_ids, penalty):
    """
    prev_ids 里出现过的每个 token (出现几次都只罚一次): 分数 > 0 的除以 penalty, 否则乘以 penalty。
    其余 token 不变。返回新列表, 不要改动传进来的 logits。
    """
    # TODO 3
    raise NotImplementedError
