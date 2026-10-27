"""
练习 2 (② 接口): 从签名写出"一步一停"的 Agent 循环

只给接口和测试, 循环自己写。规则见讲义第 3 节:

    messages = [系统提示, 用户指令]
    重复最多 max_turns 轮:
        reply = llm.chat(messages, tools=TOOL_SCHEMAS)
        没有工具调用 → 结束, ended_by = "answered"
        有工具调用   → 把 assistant 消息 (带全部 tool_calls) 追加进 messages;
                       只执行第一个调用 (过安全层: execute(car, name, arguments)),
                       其余的不执行, 回一条错误观测, 告诉模型"每轮只执行第一个";
                       每个调用都要追加一条 role="tool" 的消息, tool_call_id 对上 (否则云端 API 会报错)
    用完 max_turns 轮 → ended_by = "max_turns"

验收:
    cd mainline/C1-agent-loop/exercises
    python -m pytest test_ex2.py -q

测试用剧本模型 (ScriptedLLM) 驱动你的循环, 检查:
  1. 返回的记录: tool_calls (真正执行了几次)、skipped_calls (退回了几次)、ended_by;
  2. 模型第 k 次被调用时看到的 messages 和参考实现一样 (角色、顺序、tool_call_id、观测内容;
     退回调用的错误措辞随你, 只要 "ok" 为 False 且有 "error");
  3. 小车真的只执行了第一个调用。
写完再去对照 ../agent.py 的 run_episode。
"""

import json  # noqa: F401
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import SYSTEM_PROMPT  # noqa: E402, F401  系统提示直接用
from tools import TOOL_SCHEMAS, execute  # noqa: E402, F401  工具说明书和安全层直接用


def run_step_episode(llm, car, task, max_turns=30):
    """
    task: {"instruction": 中文指令, ...}
    llm.chat(messages, tools) 返回的 reply 有 .content (str) 和 .tool_calls ([{"id", "name", "arguments"}];
    arguments 是 JSON 字符串)。

    assistant 消息的格式 (OpenAI 格式):
        {"role": "assistant", "content": reply.content,
         "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}, ...]}
    工具结果消息:
        {"role": "tool", "tool_call_id": c["id"], "content": json.dumps(观测, ensure_ascii=False)}

    返回 dict, 至少包含: "tool_calls" (int), "skipped_calls" (int), "ended_by" ("answered" | "max_turns")。
    """
    raise NotImplementedError
