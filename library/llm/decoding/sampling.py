"""
sampling.py — 手写"从模型的打分里挑下一个 token" 的全部步骤 (只用标准库)

模型每走一步, 给词表里每个 token 打一个分 (logits, 一串实数)。怎么从这串分数里挑出下一个 token,
就是解码 (decoding)。本文件按 transformers 的 generate 实际执行的顺序实现每一步 (另两家的顺序见文件末尾的 ORDERS):

    重复惩罚 → 温度 → top-k → top-p → min-p → (贪心: 取最大 | 采样: 按概率抽)

每一步都是"改 logits": 把不想要的 token 的分数设成 -inf (softmax 之后概率为 0)。
experiments/check_against_hf.py 在随机 logits 上核对过, 每一步和 transformers 对应的 LogitsProcessor 结果相同。

用列表表示 logits, 方便逐行看; 真实系统用张量一次处理一批。
"""

import math
import random

NEG_INF = float("-inf")


def softmax(logits):
    """把分数变成概率: 先减去最大值再取指数 (防止溢出), 再除以总和。-inf 的概率是 0。"""
    m = max(logits)
    exps = [math.exp(z - m) if z != NEG_INF else 0.0 for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


def repetition_penalty(logits, prev_ids, penalty):
    """
    重复惩罚: 已经出现过的 token (prompt 里的也算), 分数往"不像"的方向推。
    正分除以 penalty, 负分乘以 penalty, 所以 penalty > 1 时总是变小。同一个 token 出现几次都只罚一次。
    """
    if penalty == 1.0:
        return list(logits)
    out = list(logits)
    for i in set(prev_ids):
        z = out[i]
        out[i] = z / penalty if z > 0 else z * penalty
    return out


def temperature(logits, t):
    """温度: 所有分数除以 t。t < 1 拉大差距 (更确定), t > 1 缩小差距 (更随机)。t 必须 > 0。"""
    if t <= 0:
        raise ValueError("temperature 必须 > 0; 想要'永远选最大', 用 greedy")
    return [z / t for z in logits]


def top_k(logits, k):
    """只留分数最高的 k 个; 和第 k 名同分的也留下 (transformers 的做法)。"""
    if k <= 0 or k >= len(logits):
        return list(logits)
    kth = sorted(logits, reverse=True)[k - 1]
    return [z if z >= kth else NEG_INF for z in logits]


def top_p(logits, p, min_keep=1):
    """
    核采样 (nucleus): 按概率从高到低累加, 留下"刚好凑够 p"的最少几个 token。
    transformers 的写法是反过来从低往高累加, 把累计概率 <= 1-p 的那些去掉; 两种说法等价 (除了恰好相等的边界)。
    """
    if p >= 1.0:
        return list(logits)
    probs = softmax(logits)
    order = sorted(range(len(logits)), key=lambda i: probs[i])        # 从低到高
    out, cum = list(logits), 0.0
    for rank, i in enumerate(order):
        cum += probs[i]
        if cum <= 1 - p and rank < len(order) - min_keep:
            out[i] = NEG_INF
    return out


def min_p(logits, p, min_keep=1):
    """min-p: 概率低于 "最高概率 × p" 的去掉。模型很有把握时只留很少几个, 没把握时留很多。"""
    probs = softmax(logits)
    bar = max(probs) * p
    keep = [pr >= bar for pr in probs]
    if sum(keep) < min_keep:                       # 保底: 至少留 min_keep 个最高的
        for i in sorted(range(len(probs)), key=lambda i: -probs[i])[:min_keep]:
            keep[i] = True
    return [z if k else NEG_INF for z, k in zip(logits, keep)]


def process(logits, prev_ids=(), rep=1.0, temp=1.0, k=0, p=1.0, minp=0.0, do_sample=True):
    """
    按 transformers 的顺序依次处理。注意 do_sample=False (贪心) 时, transformers 只跳过温度、top-k、top-p、min-p,
    重复惩罚**照样执行**: Qwen2.5 自带的 generation_config 里 repetition_penalty=1.1 (讲义第 8.3 节)。
    """
    x = repetition_penalty(logits, prev_ids, rep)
    if do_sample:
        if temp != 1.0:
            x = temperature(x, temp)
        if k:
            x = top_k(x, k)
        if p < 1.0:
            x = top_p(x, p)
        if minp > 0:
            x = min_p(x, minp)
    return x


# ── 别的惩罚写法 (讲义第 7 节) ──────────────────────────────────────────────

def windowed_repetition_penalty(logits, prev_ids, penalty, last_n=64):
    """只看最近 last_n 个 token 的重复惩罚 (llama.cpp 的做法, 默认窗口 64); 公式和 repetition_penalty 相同。"""
    return repetition_penalty(logits, list(prev_ids)[-last_n:] if last_n else [], penalty)


def presence_frequency_penalty(logits, generated_ids, presence=0.0, frequency=0.0):
    """
    加法惩罚 (OpenAI 接口、vLLM、llama.cpp 都有): 分数直接减去
        frequency × 出现次数 + presence × (出现过就是 1)
    减法不受 logits 整体平移的影响 (讲义第 6.3 节); 只看模型自己生成的 token, 不看提示词 (vLLM 的说明如此)。
    """
    counts = {}
    for t in generated_ids:
        counts[t] = counts.get(t, 0) + 1
    out = list(logits)
    for t, c in counts.items():
        out[t] -= frequency * c + presence
    return out


def no_repeat_ngram(logits, ids, n):
    """禁止重复 n 元组: 如果接上某个 token 会凑出一个前面出现过的 n 元组, 就把它设成 -inf (transformers 的 no_repeat_ngram_size)。"""
    if n <= 0 or len(ids) < n - 1:
        return list(logits)
    prefix = tuple(ids[len(ids) - (n - 1):]) if n > 1 else ()
    banned = {ids[i + n - 1] for i in range(len(ids) - n + 1) if tuple(ids[i:i + n - 1]) == prefix}
    return [NEG_INF if i in banned else z for i, z in enumerate(logits)]


# ── 三个框架的执行顺序 (讲义专题 A; 都读的是源码) ─────────────────────────

ORDERS = {
    "transformers": ["rep", "temp", "k", "p", "minp"],
    "vllm": ["rep", "temp", "minp", "k", "p"],
    "llama.cpp": ["rep", "k", "p", "minp", "temp"],
}


def process_order(logits, prev_ids=(), order="transformers", rep=1.0, temp=1.0, k=0, p=1.0, minp=0.0):
    """按某个框架的顺序做同样几步。只差顺序, 每一步本身的写法相同。"""
    x = list(logits)
    for step in ORDERS[order]:
        if step == "rep":
            x = repetition_penalty(x, prev_ids, rep)
        elif step == "temp" and temp != 1.0:
            x = temperature(x, temp)
        elif step == "k" and k:
            x = top_k(x, k)
        elif step == "p" and p < 1.0:
            x = top_p(x, p)
        elif step == "minp" and minp > 0:
            x = min_p(x, minp)
    return x


def greedy(logits):
    """贪心: 取分数最高的; 同分取下标最小的 (和 torch.argmax 一致)。"""
    return max(range(len(logits)), key=lambda i: (logits[i], -i))


def sample(logits, rng=random):
    """按 softmax 后的概率抽一个。rng 用 random.Random(种子) 可以复现。"""
    probs = softmax(logits)
    r, cum = rng.random(), 0.0
    for i, pr in enumerate(probs):
        cum += pr
        if r < cum:
            return i
    return max(i for i, pr in enumerate(probs) if pr > 0)   # 浮点误差兜底


def next_token(logits, prev_ids=(), do_sample=True, rng=random, **cfg):
    x = process(logits, prev_ids, do_sample=do_sample, **cfg)
    return sample(x, rng) if do_sample else greedy(x)


if __name__ == "__main__":
    # "今天天气很" 后面的下一个字, 一个 6 个 token 的小词表; 讲义第 2 节用这组数手算
    vocab = ["好", "热", "冷", "晴", "差", "怪"]
    logits = [2.0, 1.5, 1.0, 0.5, -1.0, -2.0]
    show = lambda x: "  ".join(f"{v}:{pr:.2f}" for v, pr in zip(vocab, softmax(x)))
    print("原始        ", show(logits))
    print("重复惩罚 1.5 (之前写过'好')", show(repetition_penalty(logits, [0], 1.5)))
    print("温度 0.5    ", show(temperature(logits, 0.5)))
    print("温度 2.0    ", show(temperature(logits, 2.0)))
    print("top-k 3     ", show(top_k(logits, 3)))
    print("top-p 0.8   ", show(top_p(logits, 0.8)))
    print("min-p 0.3   ", show(min_p(logits, 0.3)))
    print("贪心选中     ", vocab[greedy(logits)])
