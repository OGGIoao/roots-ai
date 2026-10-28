"""
beam.py — 束搜索 (beam search), 练习 3 的参考答案

贪心每一步只留 1 条路; 束搜索每一步留 B 条 "到目前为止总分最高" 的路, 最后挑总分最高的一条。
总分 = 每一步 log softmax 概率之和 (概率连乘取对数)。B = 1 时就是贪心; B 足够大时就是穷举。

    beam_search(step, prompt_ids, beam_width, max_new_tokens, stop_ids=())
        -> (新生成的 ids, 总分)

规则 (练习 3 的测试按这个判):
  1. 每一步, 把所有没结束的路各自往下展开一个 token (词表里每个 token 都试), 得到候选;
  2. 已经以停止 token 结尾的路算 "结束了", 不再展开, 但留在候选里和别人比总分;
  3. 候选按 (总分从高到低, 再按 token 序列从小到大) 排序, 留前 B 条;
  4. 全部结束或到达 max_new_tokens 时, 返回排第一的那条。
  不做长度归一化 (真实系统常做, 讲义第 7 节说为什么)。
"""

import math


def log_softmax(logits):
    m = max(logits)
    lse = m + math.log(sum(math.exp(z - m) for z in logits))
    return [z - lse for z in logits]


def beam_search(step, prompt_ids, beam_width, max_new_tokens, stop_ids=()):
    beams = [((), 0.0)]                                  # (新 token 序列, 总分)
    for _ in range(max_new_tokens):
        cands = []
        for seq, score in beams:
            if seq and seq[-1] in stop_ids:
                cands.append((seq, score))               # 结束了: 原样参与比较
                continue
            lp = log_softmax(step(list(prompt_ids) + list(seq)))
            for tok, l in enumerate(lp):
                cands.append((seq + (tok,), score + l))
        cands.sort(key=lambda c: (-c[1], c[0]))
        beams = cands[:beam_width]
        if all(seq and seq[-1] in stop_ids for seq, _ in beams):
            break
    best = beams[0]
    return list(best[0]), best[1]
