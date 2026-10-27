"""
baseline.py — 不用大模型的规则控制器 (讲义第 5 节的对照组)

同样的五个工具、同样的安全层、同样的任务, 只是"大脑"换成几十行 if-else:
    1. look; 看不到目标就原地转一圈找 (每次左转 45° 再看);
       转满一圈还找不到 (被挡住了), 就往这一圈里最空的方向挪一段, 换个位置再找;
    2. 看到目标: 先转向它, 再走 "剩余距离 - 0.12 m" (一次最多 0.5 m), 走完再 look;
    3. 路线上有东西挡着: 往旁边转 50° 绕一段;
    4. 每次 walk 之前: 刚转过身就重新 look 看正前方, 再用 get_pose 估计离墙多远 (look 看不到墙), 把这段路截短;
    5. 剩余距离 ≤ 0.2 m 就 stop。

它回答的问题: 这套工具本身够不够完成任务、大概要多少次调用? 规则能做到多少, 大模型 Agent 至少应该接近多少。
它做不到的: 听懂自然语言。目标名是直接告诉它的 (task["target"]), 不是从"帮我走到红杯子前面"里读出来的。

    python baseline.py      # 30 个任务 (预算 30 次调用, 和 step 模式的上限相同) → results/baseline_rules_n30.json
                            # 外加 300 个任务在不同调用预算下的成功率 → results/baseline_budget.json
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from agent import make_tasks  # noqa: E402
from car2d import Car2D, make_world, reached  # noqa: E402
from tools import execute  # noqa: E402

STOP_WITHIN_M = 0.20      # 比成功线 0.25 m 留一点余量 (look 的距离有 5% 噪声)
APPROACH_GAP_M = 0.12     # 每次走到 "目标前 0.12 m" 处
MAX_LEG_M = 0.5           # 一段最多走 0.5 m, 走完重新 look 修正方向


WALL_MARGIN_M = 0.15     # 车身中心离墙至少这么远 (look 看不到墙, 只能靠 get_pose 的估计位置躲)
CORRIDOR_M = 0.25        # 走的方向两侧这么宽之内有东西, 就算挡路


class _OutOfBudget(Exception):
    pass


def run_rules(car, target, max_calls=40, field=(2.0, 2.0)):
    """在 car 上跑规则控制器 (工具调用都经过安全层)。"""
    return drive(lambda name, **args: execute(car, name, args), target, max_calls, field)


def drive(tool, target, max_calls=40, field=(2.0, 2.0)):
    """
    规则控制器本体。它碰不到小车对象, 只能通过 tool(name, **args) 调用五个工具 (和大模型的处境一样)。
    返回 {"calls": 用了几次工具, "arrived": 是否自己判断到了并 stop, "last_seen_m": 最后一次看到目标时的距离或 None}。
    """
    calls = 0

    def call(name, **args):
        nonlocal calls
        if calls >= max_calls:
            raise _OutOfBudget
        calls += 1
        return tool(name, **args)

    def ahead_clearance(objs):
        """正前方 (路线两侧 CORRIDOR_M 以内) 最近的东西有多远; 返回 (距离, 那个东西的名字)。"""
        best = (9.0, None)
        for o in objs:
            rel = math.radians(o["bearing_deg"])
            if math.cos(rel) > 0 and abs((o["distance_m"] + 0.1) * math.sin(rel)) < CORRIDOR_M:
                best = min(best, (o["distance_m"] * math.cos(rel), o["name"]))
        return best

    def wall_room(pose, heading_deg):
        """按 get_pose 估计的位置, 朝 heading_deg 走多远会到离墙 WALL_MARGIN_M 的地方。"""
        h = math.radians(heading_deg)
        room = 9.0
        for pos, u, size in ((pose["x"], math.cos(h), field[0]), (pose["y"], math.sin(h), field[1])):
            if u > 1e-6:
                room = min(room, (size - WALL_MARGIN_M - pos) / u)
            elif u < -1e-6:
                room = min(room, (WALL_MARGIN_M - pos) / u)
        return room

    def safe_walk(d, objs=None):
        """
        走之前先看正前方 (刚转过身就重新 look): 有东西挡着就只走到它前面; 再截短到不撞墙。太短就不走。
        (实测: 不加这一步, 失败里大部分是"转完身直接走, 撞上刚才视野外的东西"。)
        """
        if objs is None:
            objs = call("look")["objects"]
        dist, name = ahead_clearance(objs)
        d = min(d, dist - (APPROACH_GAP_M if name == target else 0.1))
        p = call("get_pose")
        d = min(d, wall_room(p, p["heading_deg"]))
        if d < 0.05:
            return False
        call("walk", distance_m=round(d, 3))
        return True

    def search():
        """原地转一圈找目标; 找不到 (被挡住了) 就往一圈里最空的方向挪一段, 换个位置再找。返回是否已经看到目标。"""
        p = call("get_pose")
        options = []
        for k in range(8):
            objs = call("look")["objects"]
            if any(o["name"] == target for o in objs):
                return True
            room = min(ahead_clearance(objs)[0] - 0.1, wall_room(p, p["heading_deg"] + 45 * k))
            options.append((room, k))
            call("turn", angle_deg=45)
        room, k = max(options)
        if k:
            call("turn", angle_deg=45 * k if k <= 4 else 45 * (k - 8))
        safe_walk(min(0.5, room))
        return False

    last_seen = None
    try:
        while calls < max_calls:
            objs = call("look")["objects"]
            tgt = next((o for o in objs if o["name"] == target), None)
            if tgt is not None:
                last_seen = tgt["distance_m"]
            if tgt is None:
                search()
                continue
            if tgt["distance_m"] <= STOP_WITHIN_M:
                call("stop")
                return {"calls": calls, "arrived": True, "last_seen_m": tgt["distance_m"]}
            # 朝目标的路线两侧 CORRIDOR_M 以内、比目标近的东西都算挡路
            blocker = None
            for o in objs:
                if o["name"] == target or o["distance_m"] >= tgt["distance_m"]:
                    continue
                rel = math.radians(o["bearing_deg"] - tgt["bearing_deg"])
                if abs((o["distance_m"] + 0.1) * math.sin(rel)) < CORRIDOR_M and math.cos(rel) > 0:
                    blocker = o
                    break
            if blocker:
                side = -50 if blocker["bearing_deg"] >= tgt["bearing_deg"] else 50
                call("turn", angle_deg=tgt["bearing_deg"] + side)
                if not safe_walk(0.3):
                    call("turn", angle_deg=-2 * side)      # 那一侧走不了, 换另一侧绕
                    safe_walk(0.3)
                continue
            leg = min(MAX_LEG_M, tgt["distance_m"] - APPROACH_GAP_M)
            if abs(tgt["bearing_deg"]) > 3:
                call("turn", angle_deg=tgt["bearing_deg"])
                moved = safe_walk(leg)
            else:
                moved = safe_walk(leg, objs)                # 没转身, 刚才那一眼还有效
            if not moved:
                call("walk", distance_m=round(max(0.0, tgt["distance_m"] - APPROACH_GAP_M), 3))   # 目标贴着墙: 信 look
    except _OutOfBudget:                              # 预算用完: 不再调用任何工具
        pass
    return {"calls": calls, "arrived": False, "last_seen_m": last_seen}


def run_set(n, max_calls):
    rows = []
    for task in make_tasks(n):
        car = Car2D(make_world(task["seed"]), seed=task["seed"])
        calls = run_rules(car, task["target"], max_calls=max_calls)["calls"]
        rows.append({**task, "success": reached(car, task["target"]), "collisions": car.collisions,
                     "final_distance_m": round(car.distance_to(task["target"]), 3), "tool_calls": calls,
                     "trace": [(round(x, 3), round(y, 3)) for x, y in car.trace[::5]] + [tuple(round(v, 3) for v in car.trace[-1])]})
    ok = sum(r["success"] for r in rows)
    summary = {"n": n, "max_calls": max_calls, "success": ok, "success_rate": ok / n,
               "collision_episodes": sum(r["collisions"] > 0 for r in rows),
               "mean_tool_calls": sum(r["tool_calls"] for r in rows) / n}
    return summary, rows


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    summary, rows = run_set(30, max_calls=30)
    with open(os.path.join(HERE, "results", "baseline_rules_n30.json"), "w", encoding="utf-8") as f:
        json.dump({"backend": "rules", "mode": "rules", "summary": summary, "episodes": rows}, f, ensure_ascii=False, indent=1)
    print("30 个任务:", json.dumps(summary, ensure_ascii=False))
    budget = []
    for b in (10, 15, 20, 30, 40, 60, 100):
        s, _ = run_set(300, max_calls=b)
        budget.append(s)
        print(f"预算 {b:>3} 次调用: 成功率 {s['success_rate']:.2f}  有碰撞的局 {s['collision_episodes']}")
    with open(os.path.join(HERE, "results", "baseline_budget.json"), "w", encoding="utf-8") as f:
        json.dump(budget, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
