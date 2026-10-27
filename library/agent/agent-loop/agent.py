"""
agent.py — 手写的 Agent 循环 (不用 LangChain 等框架, 每一步都看得见)

一个 Agent 就是这样一个循环:
    1. 把 [系统提示, 任务, 之前所有的动作和观测] 交给大模型;
    2. 大模型要么调用工具 (walk / turn / look ...), 要么说"完成了";
    3. 调用工具 → 先过安全层 → 交给小车执行 → 把结果作为新的观测追加到对话里 → 回到 1。

三种模式 (讲义的案例 ①②③):
    chat  不提供工具, 只问一次: 模型只能回一段话, 小车不会动;
    plan  先 look 一次, 把结果告诉模型, 要求它一次性给出全部动作; 按顺序执行, 中途不再问模型 (开环);
    loop  标准的 Agent 循环 (闭环): 每执行完一轮工具调用, 都把结果交回模型再决定下一步;
          但模型可以在一轮里发出多个调用, 它们会全部执行, 中间不看结果 (本仓库实测: 小模型一轮能发 13 个);
    step  一步一停: 和 loop 一样, 但每轮只执行第一个调用, 其余的退回, 逼模型看完结果再决定。
    skill 换一种分工: 不给五个底层动作, 只给更高层的工具 (例如 go_to(target)); 循环和 loop 一样,
          只是工具集换了。工具集由调用方通过 toolset 传入 (小车上的 go_to 在主线 v0.4 的 skills.py)。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # library/agent/agent-loop → 仓库根
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from tools import TOOL_SCHEMAS, execute  # noqa: E402

NAMES_ZH = {"red_cup": "红杯子", "blue_ball": "蓝色的球", "green_box": "绿盒子"}

SYSTEM_PROMPT = """你在控制一台两轮小车, 它在一个 2 米 × 2 米、四周有墙的场地里。
场地里有三个目标物: red_cup (红杯子)、blue_ball (蓝色的球)、green_box (绿盒子), 还有几个障碍物 (obstacle_0、obstacle_1……)。
你看不到整个场地, 只能用 look 看车头正前方左右各 30 度、2 米以内的东西。
小车执行动作有误差: 走的距离和转的角度都不完全准, 所以走一段之后最好再 look 确认。
撞到墙、障碍物或目标物都算失败。
任务完成的标准: 车停在目标跟前, 车身离目标不超过 0.25 米, 全程没有碰撞。完成后调用 stop, 然后用一句话报告结果。"""


def run_episode(llm, car, task, mode="loop", max_turns=30, safety=True, force_first_tool=False, toolset=None):
    """
    跑一次任务。task: {"instruction": 中文指令, "target": 目标物名}。
    返回一条记录: 每一轮的模型输出与工具调用、token 与耗时统计、是否成功由调用方用 sim.car2d.reached 判断。
    force_first_tool: 第一轮强制模型调用工具 (本地模型靠预填 <tool_call>, API 靠 tool_choice)。
    toolset: 换一套工具时传入 {"schemas": [...], "system": 系统提示, "execute": f(car, name, args) -> 观测}。
             skill 模式必须传; loop / step 也可以传 (例如想给模型加一个新工具)。
    """
    tools, run_tool = TOOL_SCHEMAS, lambda name, args: execute(car, name, args, safety=safety)
    system = SYSTEM_PROMPT
    if mode == "skill" and toolset is None:
        raise ValueError("skill 模式需要传入 toolset (小车的 go_to 见 mainline/v0.4-text-commands/skills.py 的 SKILL_TOOLSET)")
    if toolset is not None:
        tools, system = toolset["schemas"], toolset["system"]
        run_tool = lambda name, args: toolset["execute"](car, name, args)   # noqa: E731
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task["instruction"]}]
    rec = {"mode": mode, "turns": [], "tool_calls": 0, "invalid_calls": 0, "bare_calls": 0, "skipped_calls": 0, "prompt_tokens": 0,
           "completion_tokens": 0, "latency_s": 0.0, "ended_by": None}

    def account(reply):
        rec["prompt_tokens"] += reply.usage["prompt_tokens"]
        rec["completion_tokens"] += reply.usage["completion_tokens"]
        rec["latency_s"] += reply.latency_s

    if mode == "chat":
        reply = llm.chat(messages, tools=None)
        account(reply)
        rec["turns"].append({"content": reply.content, "calls": []})
        rec["ended_by"] = "answered"
        return rec

    if mode == "plan":
        first = car.look()                         # 给模型一次观测, 之后就不再给了
        messages[1]["content"] += ("\n\n小车刚刚看了一眼, 结果是: " + json.dumps(first, ensure_ascii=False) +
                                   "\n请一次性给出完成任务需要的全部工具调用 (可以多个), 之后不会再告诉你执行结果。")
        reply = llm.chat(messages, tools=TOOL_SCHEMAS)
        account(reply)
        turn = {"content": reply.content, "calls": []}
        for c in reply.tool_calls:
            obs = execute(car, c["name"], c["arguments"], safety=safety)
            rec["tool_calls"] += 1
            rec["invalid_calls"] += "error" in obs
            rec["bare_calls"] += bool(c.get("bare"))
            turn["calls"].append({"name": c["name"], "arguments": c["arguments"], "result": obs})
        rec["turns"].append(turn)
        rec["ended_by"] = "plan_done"
        return rec

    # mode == "loop" / "step" / "skill"
    for k_turn in range(max_turns):
        if force_first_tool and k_turn == 0:
            reply = llm.chat(messages, tools=tools, require_tool=True)
        else:
            reply = llm.chat(messages, tools=tools)
        account(reply)
        turn = {"content": reply.content, "calls": []}
        if not reply.tool_calls:                   # 模型不再调用工具: 它认为做完了 (或放弃了)
            rec["turns"].append(turn)
            rec["ended_by"] = "answered"
            return rec
        messages.append({"role": "assistant", "content": reply.content, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in reply.tool_calls]})
        for k, c in enumerate(reply.tool_calls):
            if mode == "step" and k > 0:
                # 一步一停: 这一轮只执行第一个调用。其余的退回 (每个 tool_call_id 都必须有一条回复, 否则 API 会报错)
                obs = {"ok": False, "error": "未执行: 每轮只执行第一个工具调用。请先看上一个工具的结果, 再决定下一步"}
                rec["skipped_calls"] += 1
            elif c["name"] == "__invalid__":       # 本地模型把工具调用的 JSON 写坏了
                obs = {"ok": False, "error": "工具调用不是合法的 JSON, 请按格式重写"}
            else:
                obs = run_tool(c["name"], c["arguments"])
            if not (mode == "step" and k > 0):
                rec["tool_calls"] += 1
                rec["invalid_calls"] += "error" in obs
                rec["bare_calls"] += bool(c.get("bare"))
            turn["calls"].append({"name": c["name"], "arguments": c["arguments"], "result": obs})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(obs, ensure_ascii=False)})
        rec["turns"].append(turn)
    rec["ended_by"] = "max_turns"
    return rec


def make_tasks(n, seed0=0):
    """
    任务集: 第 i 个任务用场景 seed0 + i, 目标物轮流取三种, 指令句式也轮换 (避免只会一种说法)。
    同一个 (n, seed0) 永远生成同样的任务。
    """
    templates = ["去{zh}那里, 停在它跟前", "找到{zh}, 开到它旁边停下", "帮我走到{zh}前面"]
    targets = list(NAMES_ZH)
    tasks = []
    for i in range(n):
        t = targets[i % len(targets)]
        tasks.append({"seed": seed0 + i, "target": t,
                      "instruction": templates[(i // len(targets)) % len(templates)].format(zh=NAMES_ZH[t])})
    return tasks
