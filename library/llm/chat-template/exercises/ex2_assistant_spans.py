"""
练习 2 (② 接口): 找出"模型自己说的话"

微调对话模型时, 只在模型该说的那部分算 loss: 系统提示、用户的话、工具结果都是"题目", 不是"答案"
(讲义第 6 节)。写一个函数, 给出每条 assistant 回复在渲染文字里的区间:

    assistant_spans(messages, tools=None) -> [(start, end), ...]

要求 (和 ../chat_template.py 的 render_with_spans 一致):
  - 按 assistant 消息出现的顺序, 每条一个区间, 下标是 render(messages, tools) 这串文字里的字符下标;
  - start: 紧跟在这条回复的 "<|im_start|>assistant\\n" 之后;
  - end: 这条回复的 "<|im_end|>" 结束的位置 (区间包含 <|im_end|>, 模型要学会自己停下来)。

可以直接调用参考实现里的 render (../chat_template.py), 不要用 render_with_spans。
思路不止一种。如果卡住, 讲义第 6.2 节的最后一段有一个只用 render 的办法。

验收:
    cd library/llm/chat-template/exercises
    python -m pytest test_ex2.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chat_template import IM_END, render  # noqa: E402, F401


def assistant_spans(messages, tools=None):
    raise NotImplementedError
