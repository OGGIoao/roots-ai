"""
练习 2 (② 接口): 把训练好的 MHA 改造成 GQA

GQA 论文 (Ainslie et al. 2023) 的做法: 不必从头训练 GQA, 而是拿一个训练好的 MHA,
把同一组里各个头的 K、V 投影取平均, 作为这一组的投影, 再用少量数据继续训练。

实现下面这个函数。只给签名, 内部怎么做自己设计。

验收 (测试会检查):
    1. 返回值是 variants.GroupedQueryAttention, n_kv_heads 正确;
    2. n_kv_heads == n_heads 时, 输出与原 MHA 完全一致;
    3. 如果原 MHA 同一组里各头的 K、V 投影本来就相同, 转换是无损的 (输出一致);
    4. 组的约定与 variants.py 一致: 头 0..(H/G - 1) 是第 0 组, 依此类推;
    5. 不修改传入的 mha。

    python -m pytest test_ex2.py -q

进阶 (不在测试里, 自己动手做实验):
    用 compare.py 的设置训练一个 MHA 到最佳点, 转换成 GQA-2, 分别测
    (a) 转换后不训练的 val_loss; (b) 再训练 200 步后的 val_loss; (c) 从头训练的 GQA-2。
    记录三者的差别, 想一想它说明了什么。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from variants import GroupedQueryAttention  # noqa: E402,F401


def mha_to_gqa(mha, n_kv_heads):
    """mha: n_kv_heads == n_heads 的 GroupedQueryAttention。返回一个新的 GroupedQueryAttention。"""
    raise NotImplementedError
