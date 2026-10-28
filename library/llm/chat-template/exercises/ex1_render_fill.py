"""
练习 1 (① 填空): 补全 Qwen2.5 对话模板

把下面四处 TODO 补全, 让 render 在 39 段对话上 (9 个边界用例 + 30 段随机对话) 的输出和官方模板
**逐字节相同**。标准答案在 ../experiments/results/qwen25_golden.json, 是用 transformers 的
apply_chat_template 渲染出来的; 这道题本身不需要装 transformers。

验收:
    cd library/llm/chat-template/exercises
    python -m pytest test_ex1.py -q

测试失败时会打印第一个不同的位置, 前后各 40 个字符。差一个换行也算错: 模型训练时见过的就是那个格式。
提示规则: 卡住了可以要提示, 按"方向 → 哪一行 → 代码"分级给, 每次只升一级。
写完再去对照 ../chat_template.py。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chat_template import DEFAULT_SYSTEM, IM_END, IM_START, TOOLS_HEAD, TOOLS_TAIL  # noqa: E402, F401  固定文字直接给你用


def tojson(x):
    return json.dumps(x, ensure_ascii=False)


def render(messages, tools=None, add_generation_prompt=False):
    out = []

    # TODO 1: 开头的 system 块。
    #   - messages[0] 是 system 就用它的 content, 否则用 DEFAULT_SYSTEM;
    #   - tools 非空时, 在 system 文字后面接上 TOOLS_HEAD、每个工具 ("\n" + tojson(工具))、TOOLS_TAIL;
    #   - 整块是 IM_START + "system\n" + 文字 + IM_END + "\n"

    for i, m in enumerate(messages):
        role = m["role"]
        if role == "system" and i == 0:
            continue
        if role in ("user", "system") or (role == "assistant" and not m.get("tool_calls")):
            out.append(IM_START + role + "\n" + m["content"] + IM_END + "\n")
        elif role == "assistant":
            # TODO 2: 带工具调用的回复。先写 IM_START + "assistant";
            #   content 非空时接 "\n" + content; 然后每个调用接一段
            #   '\n<tool_call>\n{"name": "<名字>", "arguments": <tojson(参数)>}\n</tool_call>'
            #   (调用可能外面包了一层 "function", 也可能没包); 最后 IM_END + "\n"
            pass
        elif role == "tool":
            # TODO 3: 工具结果。连续几条 tool 消息合成一个 user 块:
            #   这一串的第一条前面写 IM_START + "user", 每条写 "\n<tool_response>\n" + content + "\n</tool_response>",
            #   这一串的最后一条后面写 IM_END + "\n"
            pass
        # 其它 role 什么都不输出 (官方模板就是这样)

    # TODO 4: 生成提示
    return "".join(out)
