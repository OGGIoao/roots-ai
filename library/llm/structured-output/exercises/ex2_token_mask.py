"""
练习 2 (② 接口): 从字符级走到 token 级

真实模型一次写一个 token, 而一个 token 常常是好几个字符, 还可能跨过格式边界
(Qwen 的 '"}}\\n' 就是一个 token: 一个引号、两个右括号、一个换行)。写一个函数:

    allowed_token_ids(trie, text, vocab, eos_id=None) -> set

  - trie: 合法输出的前缀树 (可以用 ../constrained.py 里的 Trie, 它的 walk 支持从任意节点出发);
  - text: 目前已经生成的文字; vocab: 列表, vocab[i] 是 token i 解码出来的字符串;
  - 返回所有 "接到 text 后面以后, 仍然是某个合法字符串前缀" 的 token id;
  - text 本身已经是完整的合法字符串时, 再加上 eos_id (如果给了的话), 让模型可以停;
  - text 已经走不通时, 返回空集合;
  - 解码出来是空字符串的 token 不允许 (否则模型可以原地不动、无限循环)。

可以 import ../constrained.py 的 Trie, 不要用它的 allowed_token_ids。

验收:
    cd library/llm/structured-output/exercises
    python -m pytest test_ex2.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constrained import Trie  # noqa: E402, F401


def allowed_token_ids(trie, text, vocab, eos_id=None):
    raise NotImplementedError
