"""
demo_minimal.py — 不需要显卡、不需要下载模型, 十秒内看懂 "Agent 循环" 到底在传什么

做的事: 用一个**假的大模型** (ToyBrain, 十几行 if-else) 去开仿真小车, 但走的是和真模型完全一样的循环
(agent.py 的 run_episode)。跑完会把整段对话打印出来: 你能亲眼看到每一轮里

    assistant 说了什么 / 发出了什么工具调用  →  小车执行  →  tool 消息里的观测  →  下一轮

真的大模型 (本地 Qwen 或云端 API) 做的事就是把 ToyBrain 换掉: 同样的 messages 进去, 同样格式的工具调用出来。

用法 (在 mainline/C1-agent-loop 目录里):
    python demo_minimal.py              # 场景 seed 0
    python demo_minimal.py --seed 3     # 换一个随机场景
    python demo_minimal.py --quiet      # 只打印结果, 不打印对话
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from agent import NAMES_ZH, run_episode  # noqa: E402
from car2d import Car2D, make_world, reached  # noqa: E402
from llm import Reply  # noqa: E402


class ToyBrain:
    """
    一个假的 "大模型": 接口和真模型一样 (chat(messages, tools) → Reply), 但内部只是几条规则。
    它只看对话里**最近一条观测** (tool 消息) 决定下一步 —— 和真模型一样, 看不到仿真器内部, 只靠 look 的结果。
    """

    def __init__(self, target):
        self.target = target
        self.searching = 0                       # 连续转了几次还没看到目标

    @staticmethod
    def _last_call_and_result(messages):
        """取出最近一次 (工具名, 观测)。没有工具消息时返回 (None, None)。"""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "tool":
                for j in range(i - 1, -1, -1):
                    if messages[j]["role"] == "assistant" and messages[j].get("tool_calls"):
                        name = messages[j]["tool_calls"][0]["function"]["name"]
                        return name, json.loads(messages[i]["content"])
        return None, None

    def chat(self, messages, tools=None):
        name, obs = self._last_call_and_result(messages)
        if name is None or name in ("turn", "walk"):
            return self._call("look", {})                       # 开头, 或动作之后: 先看一眼
        if name == "stop":
            return Reply(content="已经停在目标跟前了。")          # 只说话、不再调工具 = 循环结束
        seen = next((o for o in obs["objects"] if o["name"] == self.target), None)   # name == "look"
        if seen is None:
            self.searching += 1
            if self.searching > 8:
                return Reply(content="转了一圈没找到目标。")
            return self._call("turn", {"angle_deg": 45})         # 没看到: 原地左转 45° 再看
        self.searching = 0
        if abs(seen["bearing_deg"]) > 3:
            return self._call("turn", {"angle_deg": seen["bearing_deg"]})          # 先对准
        if seen["distance_m"] > 0.2:
            return self._call("walk", {"distance_m": round(min(0.5, seen["distance_m"] - 0.12), 2)})   # 再走一段
        return self._call("stop", {})

    def _call(self, name, args):
        return Reply(content="", tool_calls=[{"id": f"toy_{name}", "name": name, "arguments": json.dumps(args)}])


def show(messages):
    """把对话打印成人能看的样子。"""
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            calls = ", ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in m["tool_calls"])
            print(f"  [assistant]  发出工具调用: {calls}")
        elif role == "tool":
            print(f"  [tool]       小车返回: {m['content']}")
        elif role == "assistant":
            print(f"  [assistant]  {m['content']}")
        else:
            text = m["content"].replace("\n", " ")
            print(f"  [{role}]".ljust(15) + (text[:70] + "…" if len(text) > 70 else text))


class Recorder:
    """包一层: 记下每次交给模型的完整对话, 最后一次的就是整段对话。"""

    def __init__(self, brain):
        self.brain, self.last_messages = brain, []

    def chat(self, messages, tools=None):
        reply = self.brain.chat(messages, tools)
        self.last_messages = [dict(m) for m in messages]
        if reply.tool_calls or reply.content:             # 把这一轮模型的回复也补进去, 方便打印
            entry = {"role": "assistant", "content": reply.content}
            if reply.tool_calls:
                entry["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"]}} for c in reply.tool_calls]
            self.last_messages.append(entry)
        return reply


def run_demo(seed=0, target="red_cup", quiet=False):
    world = make_world(seed)
    car = Car2D(world, seed=seed)
    task = {"instruction": f"去{NAMES_ZH[target]}那里, 停在它跟前", "target": target}
    rec = Recorder(ToyBrain(target))
    result = run_episode(rec, car, task, mode="loop")
    ok = reached(car, target)
    if not quiet:
        print(f"场景 seed={seed}: 任务 \"{task['instruction']}\"")
        show(rec.last_messages)
        print()
    print(f"结果: {'成功' if ok else '失败'}  (离目标 {car.distance_to(target):.2f} m, 碰撞 {car.collisions} 次, "
          f"工具调用 {result['tool_calls']} 次, 问了模型 {len(result['turns'])} 轮)")
    return ok, result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", choices=list(NAMES_ZH), default="red_cup")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_demo(args.seed, args.target, args.quiet)
