"""
generate.py — 自回归生成的完整循环: 问模型 → 处理 logits → 挑一个 token → 接上去 → 判断要不要停

    generate(step, prompt_ids, max_new_tokens, stop_ids=(), stop_strings=(), decode=None, rng=None, **cfg)
        step(ids) -> logits                 模型: 给定目前所有 token, 返回下一个 token 的打分
        cfg                                 解码设置, 原样传给 sampling.next_token (do_sample、temp、k、p、minp、rep)
    返回 (新生成的 ids, 停止原因)

停止原因 (和 transformers / vLLM 的说法一致):
    "stop_token"    生成了停止 token (比如 <|im_end|>); 停止 token 本身在输出里
    "stop_string"   解码后的新文字里出现了停止字符串 (比如 "\\n\\n"); 和 transformers 一样, 输出里保留它,
                    要去掉就在解码后的文字上截 (vLLM 默认替你截掉, 两家行为不同, 讲义第 6 节)
    "length"        到了 max_new_tokens 还没停

重复惩罚看的是 "目前所有 token", **包括提示词里的** (transformers 就是这样做的, 讲义第 5 节)。
"""

import random

from sampling import next_token


def generate(step, prompt_ids, max_new_tokens, stop_ids=(), stop_strings=(), decode=None, rng=None, **cfg):
    rng = rng or random.Random(0)
    ids = list(prompt_ids)
    new = []
    for _ in range(max_new_tokens):
        tok = next_token(step(ids), prev_ids=ids, rng=rng, **cfg)
        ids.append(tok)
        new.append(tok)
        if tok in stop_ids:
            return new, "stop_token"
        if stop_strings and any(s in decode(new) for s in stop_strings):
            return new, "stop_string"
    return new, "length"
