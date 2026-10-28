"""
constrained.py — 约束解码: 让模型只能写出合法的输出 (只用标准库)

思路只有一句话: **每一步, 把 "接上去之后就不可能再合法" 的 token 的分数设成 -inf**, 剩下的照常挑。
模型仍然自己决定写什么, 只是不合法的路被堵死了。

本文件实现最直接的一种: 合法输出是一个**有限的字符串集合** (比如小车的所有合法工具调用), 用前缀树装起来。

    trie = Trie(["<tool_call>\\n{...red_cup...}", ...])
    trie.allowed_next_chars("<tool_call>\\n{")        # 字符级: 下一个字符可以是什么
    allowed_token_ids(trie, text, vocab, eos_id)     # token 级: 词表里哪些 token 接上去还合法

token 级比字符级难, 因为一个 token 可能跨过好几个格式边界 (Qwen 的 '"}}\\n' 是一个 token)。
所以不能一个字符一个字符判, 要把整个 token 接上去, 看结果是不是某个合法字符串的前缀。

更一般的约束 (正则、JSON Schema、文法) 原理相同, 只是把 "合法字符串集合" 换成一台自动机, 见讲义第 5 节。
"""


class Trie:
    """前缀树: 把一组合法字符串按字符装起来, 用来回答 "这段文字还能不能接成合法输出"。"""

    def __init__(self, strings):
        self.root = {}
        for s in strings:
            node = self.root
            for ch in s:
                node = node.setdefault(ch, {})
            node[""] = True                      # 空键表示 "到这里是一个完整的合法字符串"

    def walk(self, text, node=None):
        """从 node (默认根) 沿 text 往下走; 走不通返回 None。"""
        node = self.root if node is None else node
        for ch in text:
            node = node.get(ch)
            if node is None:
                return None
        return node

    def is_prefix(self, text):
        return self.walk(text) is not None

    def is_complete(self, text):
        node = self.walk(text)
        return node is not None and "" in node

    def allowed_next_chars(self, text):
        node = self.walk(text)
        return set() if node is None else {ch for ch in node if ch != ""}


def allowed_token_ids(trie, text, vocab, eos_id=None):
    """
    text: 目前已经生成的文字; vocab: 每个 token id 对应的字符串 (列表)。
    返回允许的 token id 集合: 接上去之后仍是某个合法字符串的前缀; 如果 text 已经完整, 再加上 eos_id。
    空字符串的 token (解码出来什么都没有) 一律不允许, 否则模型可以原地不动。
    """
    node = trie.walk(text)
    if node is None:
        return set()
    ok = {i for i, s in enumerate(vocab) if s and trie.walk(s, node) is not None}
    if eos_id is not None and "" in node:
        ok.add(eos_id)
    return ok


def mask_logits(logits, allowed):
    """不允许的 token 分数设成 -inf。"""
    neg = float("-inf")
    return [z if i in allowed else neg for i, z in enumerate(logits)]


# ── 小车的合法工具调用 (Qwen2.5 的格式, 见 chat-template 讲义第 4.2 节) ──────────────

TARGETS = ("red_cup", "blue_ball", "green_box")


def car_tool_calls(targets=TARGETS):
    """go_to 的三个目标 + look, 共 4 个合法的第一轮回复。"""
    calls = ['{"name": "go_to", "arguments": {"target": "' + t + '"}}' for t in targets]
    calls.append('{"name": "look", "arguments": {}}')
    return ["<tool_call>\n" + c + "\n</tool_call>" for c in calls]
