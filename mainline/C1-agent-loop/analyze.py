"""
analyze.py — 把 results/*.json 里每一局归到一类结局, 汇总成讲义第 5 节的表 (失败分类)

一局的结局 (按顺序判断, 先命中的算):
    success        成功: 车身离目标 ≤ 0.25 m, 全程无碰撞
    no_action      一个工具都没执行 (只回了一段话)
    wrong_target   去了别的目标 (skill 模式: go_to 的目标不对; 其他模式: 停在了别的目标物跟前)
    collision      撞了 (墙、障碍物或目标物)
    false_report   没撞、没到, 但最后一句话声称到了 / 完成了   ← 最危险: 用户会信
    out_of_turns   轮数用完 (step 模式下也就是工具调用次数用完)
    stopped_short  没撞, 自己停了, 也没声称成功 (plan 模式动作做完了也算这一类)

另外统计两个行为指标:
    walk_before_look   第一次 walk 之前没有 look 过的局的比例 (plan 模式有一次注入的 look, 不计)
    max_calls_in_turn  一轮里最多发了几个工具调用

    python analyze.py              # 打印汇总表, 写 results/analysis.json
"""

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "sim"))

from car2d import Car2D, make_world  # noqa: E402

OUTCOMES = ["success", "no_action", "wrong_target", "collision", "false_report", "out_of_turns", "stopped_short"]
OUTCOME_ZH = {"success": "成功", "no_action": "没动", "wrong_target": "去错目标", "collision": "撞了",
              "false_report": "没到却说到了", "out_of_turns": "轮数用完", "stopped_short": "没到, 停了"}
CLAIM_RE = re.compile(r"到达|(?<!找)到了|已经.{0,6}(跟前|旁边|前面|附近)|已.{0,4}停|完成|成功|抵达")
# 否定: "没能(直接)到达"、"没有找到"、"未能"、"无法"、"失败"、"找不到"、"抱歉" 等。
# 只在否定词后面 6 个字以内跟着"到达/完成/找到"时才算否定, 这样 "到达了, 没有碰撞" 仍算声称成功。
NEG_RE = re.compile(r"(没|未|无法|不能)[^，。,.;；]{0,6}(到达|到了|完成|成功|找到|抵达|停)|未能|没能|失败|找不到|抱歉|对不起")
TARGETS = ("red_cup", "blue_ball", "green_box")


def near_other_target(ep):
    """真实终点是否停在了别的目标物跟前 (用终点坐标重算, 不需要原来的小车对象)。"""
    world = make_world(ep["seed"])
    car = Car2D(world, seed=ep["seed"])
    car._x, car._y = ep["trace"][-1]
    return any(car.distance_to(t) <= 0.25 for t in TARGETS if t != ep["target"])


def classify(ep, mode):
    calls = [c for t in ep["turns"] for c in t["calls"]]
    executed = [c for c in calls if not str(c["result"].get("error", "")).startswith("未执行")]
    final_text = ep["turns"][-1]["content"] if ep["turns"] else ""
    if ep["success"]:
        return "success"
    if not executed:
        return "no_action"
    gotos = [c for c in executed if c["name"] == "go_to"]
    if mode == "skill" and gotos and all(json.loads(c["arguments"]).get("target") != ep["target"]
                                         for c in gotos if c["arguments"].strip().startswith("{")):
        return "wrong_target"
    if mode != "skill" and near_other_target(ep):
        return "wrong_target"
    if ep["collisions"]:
        return "collision"
    if ep["ended_by"] == "max_turns":
        return "out_of_turns"
    if ep["ended_by"] == "answered" and CLAIM_RE.search(final_text) and not NEG_RE.search(final_text):
        return "false_report"
    return "stopped_short"


def behaviour(ep, mode):
    first_walk = first_look = None
    k = 0
    for t in ep["turns"]:
        for c in t["calls"]:
            if c["name"] == "walk" and first_walk is None:
                first_walk = k
            if c["name"] == "look" and first_look is None:
                first_look = k
            k += 1
    blind = first_walk is not None and (first_look is None or first_look > first_walk) and mode != "plan"
    return blind, max((len(t["calls"]) for t in ep["turns"]), default=0)


def analyze_file(path):
    d = json.load(open(path, encoding="utf-8"))
    mode = d["mode"]
    if mode == "rules":
        counts = {o: 0 for o in OUTCOMES}
        for ep in d["episodes"]:
            counts["success" if ep["success"] else "collision" if ep["collisions"] else "stopped_short"] += 1
        return {"name": os.path.basename(path)[:-5], "backend": "rules", "mode": "rules", "safety": True,
                "summary": d["summary"], "outcomes": counts, "walk_before_look": 0.0, "max_calls_in_turn": 1, "examples": {}}
    counts = {o: 0 for o in OUTCOMES}
    examples = {}
    blind = 0
    max_burst = 0
    for ep in d["episodes"]:
        o = classify(ep, mode)
        counts[o] += 1
        examples.setdefault(o, ep["seed"])
        b, m = behaviour(ep, mode)
        blind += b
        max_burst = max(max_burst, m)
    tag = mode + ("-forced" if d.get("force_first_tool") else "")
    return {"name": os.path.basename(path)[:-5], "backend": d["backend"], "mode": tag, "safety": d["safety"],
            "summary": d["summary"], "outcomes": counts, "walk_before_look": blind / len(d["episodes"]),
            "max_calls_in_turn": max_burst, "examples": examples}


def main():
    rows = [analyze_file(p) for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json")))
            if not p.endswith(("analysis.json", "baseline_budget.json"))]
    with open(os.path.join(HERE, "results", "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    head = f"{'组合':<52}" + "".join(f"{OUTCOME_ZH[o]:>8}" for o in OUTCOMES) + "   先走后看  一轮最多"
    print(head)
    for r in rows:
        print(f"{r['name'][:52]:<52}" + "".join(f"{r['outcomes'][o]:>10}" for o in OUTCOMES) +
              f"   {r['walk_before_look']:>6.0%}  {r['max_calls_in_turn']:>6}")
    print("写入 results/analysis.json")


if __name__ == "__main__":
    main()
