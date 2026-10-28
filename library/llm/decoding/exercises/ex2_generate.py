"""
练习 2 (② 接口): 写出完整的生成循环

    generate(step, prompt_ids, max_new_tokens, stop_ids=(), stop_strings=(), decode=None, rng=None, **cfg)
        -> (新生成的 ids, 停止原因)

要求 (和 ../generate.py 一致, 讲义第 6 节):
  - 每一步: logits = step(目前所有 ids) → 用 sampling.next_token 挑一个 token (cfg 原样传过去);
    重复惩罚要看 "目前所有 token", 包括提示词里的;
  - rng 没给时用 random.Random(0), 这样同样的设置结果可复现;
  - 三种停止原因:
      "stop_token"   刚生成的 token 在 stop_ids 里 (这个 token 留在输出里)
      "stop_string"  decode(新生成的 ids) 里出现了 stop_strings 中的任意一个 (输出保留它)
      "length"       生成满 max_new_tokens 个还没停
    同一步里两个条件都满足时, 报 "stop_token"。

可以直接用参考实现里的 sampling.next_token, 不要用 ../generate.py。

验收:
    cd library/llm/decoding/exercises
    python -m pytest test_ex2.py -q
"""

import os
import random  # noqa: F401
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sampling import next_token  # noqa: E402, F401


def generate(step, prompt_ids, max_new_tokens, stop_ids=(), stop_strings=(), decode=None, rng=None, **cfg):
    raise NotImplementedError
