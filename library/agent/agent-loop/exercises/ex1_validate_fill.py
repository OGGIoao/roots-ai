"""
练习 1 (① 填空): 补全安全层 validate_call

大模型说要做的事, 不等于应该直接去做。把下面四处 TODO 补全, 让 validate_call 在测试给的
二十几个调用上 (有正常的, 也有讲义第 4 节实测到的各种错法) 和参考实现 (../tools.py) 判断一致:
该放行的放行, 而且清洗后的参数相同; 该拦的拦下, 并给出一句非空的错误说明 (措辞随你)。

验收:
    cd library/agent/agent-loop/exercises
    python -m pytest test_ex1.py -q

提示规则: 卡住了可以要提示, 按"方向 → 哪一行 → 代码"分级给, 每次只升一级。
写完再去对照 ../tools.py。
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import MAX_TURN_DEG, MAX_WALK_M, TOOL_NAMES, TOOL_SCHEMAS  # noqa: E402, F401  工具说明书直接给你用


def to_number(v):
    """
    把一个参数值转成 float; 转不了就抛 ValueError (带一句说明)。
    要求:
      - int / float 直接转;  字符串 "0.5" 可以转 (云端模型偶尔这么写);
      - "0.5m" 这种带单位的不猜, 抛 ValueError;
      - True / False 不算数字 (注意: Python 里 bool 是 int 的子类);
      - 转出来是 nan 或 inf 的, 也抛 ValueError。
    """
    # TODO 1
    raise NotImplementedError


def validate_call(name, args):
    """
    返回 (ok, 结果): ok=True 时结果是清洗后的参数 dict (值都是 float); ok=False 时结果是一句错误说明。
    args 可以是 dict, 也可以是 JSON 字符串 (空字符串当作没有参数)。
    """
    if name not in TOOL_NAMES:
        return False, f"没有叫 {name!r} 的工具, 可用的工具: {', '.join(TOOL_NAMES)}"

    # TODO 2: 如果 args 是字符串, 用 json.loads 解析; 解析失败、或者解析出来不是 dict, 都返回 (False, 说明)

    schema = next(s["function"]["parameters"] for s in TOOL_SCHEMAS if s["function"]["name"] == name)
    # TODO 3: 对照 schema 检查参数名: 有 schema["properties"] 里没有的参数 → 拦; 缺 schema["required"] 里的参数 → 拦。
    #         然后用 to_number 把每个值转成 float, 转不了 → 拦

    # TODO 4: 范围检查: walk 的 |distance_m| ≤ MAX_WALK_M, speed (如果给了) 必须 > 0; turn 的 |angle_deg| ≤ MAX_TURN_DEG
    raise NotImplementedError
