"""
练习 1 (① 填空): 补全前缀树

前缀树把一组合法字符串按字符装起来。每个节点是一个 dict: 键是下一个字符, 值是子节点;
特殊的键 "" (空字符串) 表示 "走到这里正好是一个完整的合法字符串"。__init__ 已经写好。

把四处 TODO 补全, 让它在随机生成的字符串集合上和参考实现 (../constrained.py 的 Trie) 回答一致。

验收:
    cd library/llm/structured-output/exercises
    python -m pytest test_ex1.py -q

提示规则: 卡住了可以要提示, 按"方向 → 哪一行 → 代码"分级给, 每次只升一级。
"""


class Trie:
    def __init__(self, strings):
        self.root = {}
        for s in strings:
            node = self.root
            for ch in s:
                node = node.setdefault(ch, {})
            node[""] = True

    def walk(self, text, node=None):
        """从 node (默认根) 沿 text 一个字符一个字符往下走, 返回最后到达的节点; 中途走不通返回 None。"""
        # TODO 1
        raise NotImplementedError

    def is_prefix(self, text):
        """text 是不是某个合法字符串的前缀 (包括它本身就是完整的合法字符串, 以及空字符串)。"""
        # TODO 2
        raise NotImplementedError

    def is_complete(self, text):
        """text 本身是不是一个完整的合法字符串。"""
        # TODO 3
        raise NotImplementedError

    def allowed_next_chars(self, text):
        """text 后面可以接哪些字符 (集合); text 走不通时返回空集合。注意别把表示 "完整" 的空键算进去。"""
        # TODO 4
        raise NotImplementedError
