"""
number_dfa.py — 用一台有限自动机约束 "数字" 的格式, 练习 3 的参考答案

合法的数字 (和 JSON 的数字差不多, 不含指数): -?(0|[1-9][0-9]*)(\\.[0-9]+)?
    合法: 0  7  -3  12  0.5  -0.25  10.0
    不合法: 01 (前导零)  1.  .5  -  --1  1.2.3

合法字符串有无穷多个, 没法像工具调用那样全部列出来装进前缀树。办法是画一台自动机: 几个 "状态", 读到一个字符就
按规则跳到下一个状态; 跳不过去就说明这个前缀已经不可能变成合法数字。约束解码只需要问它两件事:
    is_prefix(s)          s 还能不能接成合法数字
    is_complete(s)        s 本身是不是一个完整的合法数字
    allowed_next_chars(s) 下一个字符可以是什么

状态 (见讲义第 5.1 节的图):
    start   什么都没读          可接: '-' → sign; '0' → zero; '1'-'9' → int
    sign    读了负号            可接: '0' → zero; '1'-'9' → int
    zero    整数部分是 0        可接: '.' → dot                    (完整)
    int     整数部分非 0 开头    可接: 数字 → int; '.' → dot          (完整)
    dot     读了小数点          可接: 数字 → frac
    frac    小数部分            可接: 数字 → frac                    (完整)
"""

ALPHABET = "-0123456789."
DIGITS = "0123456789"
ACCEPT = {"zero", "int", "frac"}


def step(state, ch):
    if state == "start":
        if ch == "-":
            return "sign"
    if state in ("start", "sign"):
        if ch == "0":
            return "zero"
        if ch in "123456789":
            return "int"
        return None
    if state == "zero":
        return "dot" if ch == "." else None
    if state == "int":
        if ch in DIGITS:
            return "int"
        return "dot" if ch == "." else None
    if state in ("dot", "frac"):
        return "frac" if ch in DIGITS else None
    return None


def run(s):
    state = "start"
    for ch in s:
        state = step(state, ch)
        if state is None:
            return None
    return state


def is_prefix(s):
    return run(s) is not None


def is_complete(s):
    return run(s) in ACCEPT


def allowed_next_chars(s):
    state = run(s)
    if state is None:
        return set()
    return {ch for ch in ALPHABET if step(state, ch) is not None}
