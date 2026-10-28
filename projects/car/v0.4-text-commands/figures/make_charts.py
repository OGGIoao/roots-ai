"""
make_charts.py — 由评测结果生成 C1 讲义里的数据图 (SVG)

数据来源 (都在 ../results/):
  analysis.json        (python analyze.py 生成)  → fig_success.svg   各模型 × 各模式的成功率
                                                 → fig_outcomes.svg  每个组合 30 局的结局分类
  baseline_budget.json (python baseline.py 生成) → fig_budget.svg    规则控制器: 调用预算 vs 成功率
  各组合的 *.json 里的真实轨迹                    → fig_traj.svg      同一个任务, 不同模式下车走过的路

运行: python figures/make_charts.py   (评测重跑后, 先 python analyze.py, 再运行这个)

画法约定同其他章节 (dataviz 规范): 同一个模式在所有图里颜色相同; 只有一个 y 轴; 讲义里每张图旁边配数据表。
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C1 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(C1)))   # projects/car/<站> → 仓库根
RES = os.path.join(C1, "results")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from car2d import make_world  # noqa: E402

from common.svgfig import AXIS, BG, GRID, INK, INK2, MUTED, SERIES, Chart, Svg  # noqa: E402

MODES = ["plan", "loop", "step", "skill", "skill-forced"]
MODE_ZH = {"chat": "只聊天", "plan": "开环 plan", "loop": "闭环 loop", "step": "一步一停 step", "skill": "技能 go_to", "skill-forced": "go_to + 强制调用", "rules": "规则 (无大模型)"}
COLOR = {"plan": SERIES[1], "loop": SERIES[0], "step": SERIES[2], "skill": SERIES[4], "skill-forced": SERIES[5], "rules": INK2, "chat": SERIES[3]}
MODELS = [("Qwen/Qwen2.5-0.5B-Instruct", "0.5B"), ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"), ("Qwen/Qwen2.5-3B-Instruct", "3B")]
OUTCOME_COLOR = {"success": "#1baf7a", "no_action": "#c9c8c2", "wrong_target": "#e87ba4", "collision": "#eb6834",
                 "false_report": "#b3261e", "out_of_turns": "#eda100", "stopped_short": "#8fb8e8"}


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def out(name):
    return os.path.join(HERE, name)


def legend_boxes(c, items, x, y, step=110):
    for i, (name, color) in enumerate(items):
        lx = x + i * step
        c.add(f'<rect x="{lx}" y="{y - 6}" width="12" height="12" rx="2" fill="{color}"/>')
        c.text(lx + 18, y, name, 11, INK, "start")


def find(rows, model, mode, safety=True):
    """只取默认解码设置的结果 (贪心 + 模型自带的重复惩罚); 解码对照 (-rep、-sample、_rerun) 只在讲义第 6.1 节的表里出现。"""
    for r in rows:
        if any(tag in r["name"] for tag in ("-rep", "-sample", "_rerun")):
            continue
        if r["backend"] == f"local:{model}" and r["mode"] == mode and r["safety"] == safety:
            return r
    return None


# ── 第 5 / 6 节: 成功率 ─────────────────────────────────────────────

def fig_success(rows):
    groups = [short for _, short in MODELS]
    vals = []
    MISSING = 0.001                                  # 没跑过的组合: 画一根看不见的柱子, 标 "待测" (不能画成 0%)
    for model, _ in MODELS:
        vals.append([(find(rows, model, m)["summary"]["success_rate"] * 100 if find(rows, model, m) else MISSING) for m in MODES])
    c = Chart(820, 380, right=40, top=78)
    c.scales(-0.5, len(groups) - 0.5, 0, 100)
    c.axes([], [0, 25, 50, 75, 100], "", "成功率 (30 个任务)", yfmt=lambda v: f"{v:g}%")
    for i, g in enumerate(groups):
        c.text(c.X(i), c.y0 + 16, f"Qwen2.5-{g}", 12, INK2)
    c.bars(groups, MODES, vals, [COLOR[m] for m in MODES], fmt=lambda v: "待测" if v < 0.01 and v > 0 else f"{v:.0f}%")
    rules = next((r for r in rows if r["mode"] == "rules"), None)
    if rules:
        y = c.Y(rules["summary"]["success_rate"] * 100)
        c.add(f'<line x1="{c.x0}" y1="{y:.1f}" x2="{c.x1}" y2="{y:.1f}" stroke="{INK2}" stroke-dasharray="5 4"/>')
        c.text(c.x1 - 4, y - 10, f"规则控制器 (不用大模型) {rules['summary']['success_rate'] * 100:.0f}%", 12, INK, "end")
    legend_boxes(c, [(MODE_ZH[m], COLOR[m]) for m in MODES], c.x0 + 10, 62, 130)
    c.save(out("fig_success.svg"), "小模型自己开车几乎全失败; 只让它选目标 (go_to), 成功率接近规则控制器",
           "每个组合 30 个任务 (场景 seed 0~29), 本地贪心解码; \"待测\" = 还没跑 (显存限制, 见讲义第 5 节)")


# ── 第 5 节: 结局分类 ───────────────────────────────────────────────

def fig_outcomes(rows):
    from analyze import OUTCOME_ZH, OUTCOMES
    order = []
    for model, short in MODELS:
        for m in ["chat"] + MODES:
            r = find(rows, model, m)
            if r:
                order.append((f"{short} {MODE_ZH[m]}", r))
    for model, short in MODELS:
        r = find(rows, model, "step", safety=False)
        if r:
            order.append((f"{short} step 无安全层", r))
    rules = next((r for r in rows if r["mode"] == "rules"), None)
    if rules:
        order.append(("规则控制器", rules))
    row_h, left, top, width = 24, 190, 92, 520
    s = Svg(left + width + 40, top + row_h * len(order) + 40)
    s.text(20, 22, "失败的样子各不相同: 30 局里每一局的结局", 15, INK, "start", "bold")
    s.text(20, 42, "按顺序判断: 成功 → 没动 → 去错目标 → 撞了 → 没到却说到了 → 轮数用完 → 没到, 停了 (analyze.py)", 12, INK2, "start")
    for i, o in enumerate(OUTCOMES):                            # 图例 (两行)
        lx, ly = 20 + (i % 4) * 150, 62 + (i // 4) * 18
        s.rect(lx, ly - 6, 12, 12, OUTCOME_COLOR[o], OUTCOME_COLOR[o], rx=2, sw=0)
        s.text(lx + 18, ly, OUTCOME_ZH[o], 11, INK, "start")
    for j, (label, r) in enumerate(order):
        y = top + j * row_h
        s.text(left - 10, y + row_h / 2, label, 12, INK, "end")
        n = sum(r["outcomes"].values())
        x = left
        for o in OUTCOMES:
            k = r["outcomes"][o]
            if not k:
                continue
            w = width * k / n
            s.items.append(f'<rect x="{x:.1f}" y="{y + 3}" width="{max(w - 1.5, 0.5):.1f}" height="{row_h - 6}" fill="{OUTCOME_COLOR[o]}"/>')
            if w > 16:
                ink = BG if o in ("false_report", "collision", "success") else INK
                s.text(x + w / 2, y + row_h / 2, str(k), 10, ink)
            x += w
    s.save(out("fig_outcomes.svg"))


# ── 第 5 节: 预算 ──────────────────────────────────────────────────

def fig_budget():
    d = load("baseline_budget.json")
    pts = [(r["max_calls"], r["success_rate"] * 100) for r in d]
    c = Chart(720, 340, right=60, top=62)
    c.scales(0, 105, 0, 100)
    c.axes([10, 15, 20, 30, 40, 60, 100], [0, 25, 50, 75, 100], "每局最多调用几次工具", "成功率 (300 个任务)", yfmt=lambda v: f"{v:g}%")
    c.series(pts, INK2, "circle")
    for x, y in pts:
        c.text(c.X(x), c.Y(y) - 14, f"{y:.0f}%", 11, INK)
    x30 = c.X(30)
    c.add(f'<line x1="{x30:.1f}" y1="{c.y1}" x2="{x30:.1f}" y2="{c.y0}" stroke="{MUTED}" stroke-dasharray="4 3"/>')
    c.text(x30 + 6, c.y1 + 10, "大模型的上限: 30 轮", 11, INK2, "start")
    c.save(out("fig_budget.svg"), "连规则控制器都要 20~30 次调用: 找目标、绕障碍、每段路之前看一眼",
           "baseline.py, 300 个任务 (seed 0~299); 预算 30 时失败的局里大多数没有碰撞, 只是预算内没走到")


# ── 第 2 / 3 / 6 节: 轨迹 ──────────────────────────────────────────

def draw_field(s, x0, y0, size, world, trace, target, title, sub):
    k = size / world["width"]
    X = lambda v: x0 + v * k                       # noqa: E731
    Y = lambda v: y0 + size - v * k                # noqa: E731  y 轴朝上
    s.rect(x0, y0, size, size, "#ffffff", AXIS, rx=0, sw=1)
    for t in world["things"]:
        if t.kind == "obstacle":
            fill, stroke = "#d9d8d2", "#9a9993"
        else:
            fill = {"red_cup": "#d64545", "blue_ball": "#2a78d6", "green_box": "#1baf7a"}[t.name]
            stroke = INK if t.name == target else fill
        sw = 2.5 if t.name == target else 1
        s.items.append(f'<circle cx="{X(t.x):.1f}" cy="{Y(t.y):.1f}" r="{max(t.r * k, 3):.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        if t.name == target:
            s.items.append(f'<circle cx="{X(t.x):.1f}" cy="{Y(t.y):.1f}" r="{(t.r + 0.08 + 0.25) * k:.1f}" fill="none" stroke="{MUTED}" stroke-dasharray="3 3"/>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(px):.1f},{Y(py):.1f}" for i, (px, py) in enumerate(trace))
    s.items.append(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="1.8" stroke-linejoin="round"/>')
    sx, sy, sh = world["start"]
    hx, hy = sx + 0.12 * math.cos(math.radians(sh)), sy + 0.12 * math.sin(math.radians(sh))
    s.items.append(f'<line x1="{X(sx):.1f}" y1="{Y(sy):.1f}" x2="{X(hx):.1f}" y2="{Y(hy):.1f}" stroke="{INK}" stroke-width="1.5" marker-end="url(#arrow)"/>')
    s.items.append(f'<circle cx="{X(sx):.1f}" cy="{Y(sy):.1f}" r="{0.08 * k:.1f}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    ex, ey = trace[-1]
    s.items.append(f'<circle cx="{X(ex):.1f}" cy="{Y(ey):.1f}" r="{0.08 * k:.1f}" fill="{INK}" fill-opacity="0.35" stroke="{INK}"/>')
    s.text(x0 + size / 2, y0 + size + 16, title, 12, INK, weight="bold")
    s.text(x0 + size / 2, y0 + size + 33, sub, 11, INK2)


def fig_traj(seed=0, model="Qwen/Qwen2.5-1.5B-Instruct"):
    from analyze import OUTCOME_ZH, classify
    panels = []
    for m in ["plan", "loop", "step", "skill"]:
        name = f"local-{model.replace('/', '-')}_{m}_safe_n30.json"
        if os.path.exists(os.path.join(RES, name)):
            d = load(name)
            ep = next(e for e in d["episodes"] if e["seed"] == seed)
            panels.append((MODE_ZH[m], ep, f"{OUTCOME_ZH[classify(ep, m)]}, 工具 {ep['tool_calls']} 次"))
    if os.path.exists(os.path.join(RES, "baseline_rules_n30.json")):
        ep = next(e for e in load("baseline_rules_n30.json")["episodes"] if e["seed"] == seed)
        panels.append((MODE_ZH["rules"], ep, f"{'成功' if ep['success'] else '失败'}, 工具 {ep['tool_calls']} 次"))
    size, gap = 170, 26
    s = Svg(max(20 + len(panels) * (size + gap), 760), size + 148)
    ep0 = panels[0][1]
    s.text(20, 22, f"同一个任务 \"{ep0['instruction']}\" (场景 seed {seed}), 不同模式下车实际走过的路", 15, INK, "start", "bold")
    s.text(20, 42, f"Qwen2.5-{model.split('-')[1]}; 空心圆 = 起点 (箭头是车头), 灰色实心圆 = 终点; 场地 2 m × 2 m", 12, INK2, "start")
    s.text(20, 59, "黑边圆 = 这次的目标, 虚线圈 = 成功范围 (车身离目标 ≤ 0.25 m); 大灰圆 = 障碍物", 12, INK2, "start")
    world = make_world(seed)
    for i, (title, ep, sub) in enumerate(panels):
        draw_field(s, 20 + i * (size + gap), 76, size, world, [tuple(p) for p in ep["trace"]], ep["target"], title, sub)
    s.save(out("fig_traj.svg"))


if __name__ == "__main__":
    sys.path.insert(0, C1)
    rows = load("analysis.json")
    fig_success(rows)
    fig_outcomes(rows)
    fig_budget()
    fig_traj()
