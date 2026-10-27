"""
skills.py — 给大模型一个"高层技能"而不是五个底层动作 (讲义第 6 节)

前五节里, 大模型要自己一步一步 look / turn / walk 把车开过去。小模型在这件事上很差 (见讲义的实测)。
这里换一种分工:
    大模型   只做它擅长的: 听懂"帮我走到红杯子前面", 决定要去哪个目标, 最后汇报;
    go_to   怎么开过去, 交给一段不用大模型的控制代码 (baseline.py 的规则控制器)。
在主线 E 里, 这段规则会换成强化学习训练出来的策略; 对大模型来说接口不变, 还是 go_to(target)。

工具只有两个: go_to(target) 和 look()。go_to 只会说它自己"认为"到没到 (根据最后一次 look),
不会偷看仿真器的真实位置——和真车一样。
"""

import json

from baseline import run_rules

TARGETS = ["red_cup", "blue_ball", "green_box"]
GO_TO_BUDGET = 30          # 一次 go_to 内部最多调用几次底层工具 (和 step 模式的轮数上限相同)

SKILL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "go_to",
        "description": "自动把小车开到指定目标物跟前 (会自己找、绕开障碍物, 到了自动停车)。返回是否到达。",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "enum": TARGETS, "description": "目标物名称"}},
            "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "look",
        "description": "用摄像头看一眼正前方 (视场左右各 30 度, 最远 2 米)。返回看到的物体: 名称、距离 (米)、方位角 (度, 左正右负)。",
        "parameters": {"type": "object", "properties": {}}}},
]

SKILL_SYSTEM_PROMPT = """你在控制一台两轮小车, 它在一个 2 米 × 2 米的场地里。
场地里有三个目标物: red_cup (红杯子)、blue_ball (蓝色的球)、green_box (绿盒子), 还有几个障碍物。
你有一个 go_to 工具, 它会自动把车开到指定目标跟前并停下。
听懂用户要去哪个目标, 调用 go_to, 然后用一句话报告结果。"""


def execute_skill(car, name, args):
    """执行 go_to / look, 返回给大模型的观测。参数检查同样在这里做 (安全层不因为换了工具就消失)。"""
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"参数不是合法的 JSON: {e.msg}"}
    if not isinstance(args, dict):
        return {"ok": False, "error": "参数应为 JSON 对象"}
    if name == "look":
        if args:
            return {"ok": False, "error": "look 不需要参数"}
        return car.look()
    if name == "go_to":
        target = args.get("target")
        if set(args) != {"target"} or target not in TARGETS:
            return {"ok": False, "error": f"go_to 需要一个参数 target, 取值为 {TARGETS} 之一, 收到 {args}"}
        res = run_rules(car, target, max_calls=GO_TO_BUDGET)
        out = {"ok": res["arrived"], "low_level_calls": res["calls"]}
        if res["arrived"]:
            out["message"] = f"已到达 {target} 跟前, 距离约 {res['last_seen_m']} 米"
        elif res["last_seen_m"] is None:
            out["message"] = f"没有找到 {target}"
        else:
            out["message"] = f"没能到达 {target}, 最后一次看到时距离 {res['last_seen_m']} 米"
        return out
    return {"ok": False, "error": f"没有叫 {name!r} 的工具, 可用的是 go_to 和 look"}
