"""
练习 3 (③ 白纸): 用自动机约束数字格式

合法字符串有无穷多个时, 没法全部列出来装进前缀树。从零写出一台有限自动机, 约束 "数字" 的格式:

    合法的数字:  -?(0|[1-9][0-9]*)(\\.[0-9]+)?
    合法: 0  7  -3  12  0.5  -0.25  10.0
    不合法: 01 (前导零)  1.  .5  -  --1  1.2.3

只考虑这 12 个字符: "-0123456789."。写三个函数:

    is_prefix(s)          -> bool   s 还能不能接成合法数字 (空字符串算 True)
    is_complete(s)        -> bool   s 本身是不是完整的合法数字
    allowed_next_chars(s) -> set    下一个字符可以是哪些 (s 已经走不通时返回空集合)

不许用 re 模块 (测试会用 re 来核对你), 自己画状态、写跳转。讲义第 5.1 节讲了思路, 先别看图, 卡住了再看。

验收 (所有长度不超过 4 的字符串逐个核对):
    cd library/llm/structured-output/exercises
    python -m pytest test_ex3.py -q
"""


def is_prefix(s):
    raise NotImplementedError


def is_complete(s):
    raise NotImplementedError


def allowed_next_chars(s):
    raise NotImplementedError
