"""
make_charts.py — 对话模板讲义的数据图 (SVG, 只用标准库; 数字来自 ../experiments/results/)

  chart_budget.svg         通用小助手一次提示的 token 账本: 工具说明占大头                  (第 8 节)
  chart_tools_scaling.svg  工具从 0 个加到 12 个, 提示怎么变长; 中文 / 英文 / 转义的对照       (第 8 节)
  chart_prefix.svg         8 个模型家族: 这一轮的文字是不是下一轮提示的前缀                   (专题 B)
  chart_car_budget.svg     实践: 小车项目 v0.4 的提示账本                                   (第 16 节)

运行: python figures/make_charts.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(UNIT)))
sys.path.insert(0, ROOT)

from common.svgfig import INK, INK2, KV_FILL, KV_LINE, SERIES, Chart, Svg  # noqa: E402

RES = os.path.join(UNIT, "experiments", "results")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def out(name):
    return os.path.join(HERE, name)


def budget_bars(b, title, subtitle, name, tool_label):
    rows = [(tool_label, b["tools_block"], SERIES[1]), ("系统提示", b["system_prompt"], SERIES[0]),
            ("模板标记", b["template_markers"], SERIES[2]), ("用户的话", b["user"], SERIES[3])]
    total = b["total_with_tools"]
    s = Svg(840, 350)
    s.text(20, 22, title.format(total=total, share=b["tools_block"] / total), 15, INK, "start", "bold")
    s.text(20, 42, subtitle, 12, INK2, "start")
    x0, w = 190, 520
    for i, (lab, v, c) in enumerate(rows):
        y = 80 + i * 44
        s.text(x0 - 12, y + 13, lab, 13, INK, "end")
        bw = max(w * v / total, 2)
        s.items.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="26" rx="3" fill="{c}"/>')
        s.text(x0 + bw + 8, y + 13, f"{v}  ({v / total:.0%})", 12, INK2, "start")
    y = 80 + 4 * 44 + 18
    s.text(20, y, f"工具说明里每个工具的 JSON (不含模板固定的英文说明 {b['tools_block'] - b['tools_json_only']} 个 token):", 12, INK2, "start")
    xs, tw = x0, w + 80
    for nm, v in sorted(b["per_tool"].items(), key=lambda kv: -kv[1]):
        bw = tw * v / b["tools_json_only"]
        s.items.append(f'<rect x="{xs:.1f}" y="{y + 16}" width="{bw - 3:.1f}" height="36" rx="2" fill="{KV_FILL}" '
                       f'stroke="{KV_LINE}" stroke-width="0.8"/>')
        s.text(xs + bw / 2 - 1.5, y + 27, nm, 11, INK)
        s.text(xs + bw / 2 - 1.5, y + 42, str(v), 11, INK2)
        xs += bw
    s.save(out(name))


def chart_budget():
    b = load("tokens.json")["budget"]
    budget_bars(b, "一个带三个工具的小助手, 一次提示 {total} 个 token, 工具说明占 {share:.0%}",
                "Qwen2.5 分词器实测 (experiments/measure_tokens.py); 指令 \"台北现在几度?\"; 每一轮都要把这些重新喂给模型",
                "chart_budget.svg", "工具说明 (3 个工具)")


def chart_tools_scaling():
    t = load("tokens.json")
    pts = [(r["n_tools"], r["tokens"]) for r in t["scaling"]]
    c = Chart(860, 420, left=80, right=200, top=66, bottom=64)
    c.scales(0, 12, 0, 1400)
    c.axes(range(0, 13, 2), [0, 200, 400, 600, 800, 1000, 1200, 1400], "工具个数", "一次提示的 token 数",
           xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    c.series(pts, SERIES[0], "circle")
    first = pts[1][1] - pts[0][1]
    step = (pts[-1][1] - pts[1][1]) / (len(pts) - 2)
    c.text(c.X(1) + 10, c.Y(pts[1][1]) + 18, f"第 1 个工具 +{first} (含模板固定的说明文字)", 11, INK, "start")
    c.text(c.X(3.2), c.Y(pts[7][1]) - 10, f"之后每个 +{step:.0f}, 直线增长", 11, INK, "start")
    lang = t["language"]
    c.text(c.x1 + 14, c.Y(1000), "3 个工具时:", 11, INK, "start", "bold")
    c.text(c.x1 + 14, c.Y(1000) + 20, f"中文说明  {lang['zh_description_total']}", 11, INK2, "start")
    c.text(c.x1 + 14, c.Y(1000) + 38, f"英文说明  {lang['en_description_total']}", 11, INK2, "start")
    c.text(c.x1 + 14, c.Y(1000) + 62, "其中工具 JSON:", 11, INK, "start", "bold")
    c.text(c.x1 + 14, c.Y(1000) + 80, f"中文原样  {lang['tools_json_zh']}", 11, INK2, "start")
    c.text(c.x1 + 14, c.Y(1000) + 98, f"中文转义 \\uXXXX  {lang['tools_json_zh_escaped']}", 11, INK2, "start")
    c.save(out("chart_tools_scaling.svg"), "工具说明是直线增长的成本: 每多一个工具, 每一轮都多约 100 个 token",
           "同样结构的工具复制 0~12 个, Qwen2.5 分词器实测 (experiments/measure_tokens.py)")


def chart_prefix():
    p = load("prefix_property.json")
    names = [("Qwen2.5", "Qwen/Qwen2.5-1.5B-Instruct"), ("Qwen3", "Qwen/Qwen3-8B"), ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
             ("Phi-4", "microsoft/phi-4"), ("SmolLM2", "HuggingFaceTB/SmolLM2-1.7B-Instruct"), ("GLM-4.5", "zai-org/GLM-4.5"),
             ("gpt-oss", "openai/gpt-oss-20b"), ("Mistral-v0.3", "mistralai/Mistral-7B-Instruct-v0.3")]
    vals = []
    for _, repo in names:
        r = p[repo]
        vals.append([100 * r[v]["prefix_ok"] / r[v]["total"] for v in ("plain", "reasoning")])
    c = Chart(900, 400, left=80, right=40, top=78, bottom=64)
    c.scales(-0.5, len(names) - 0.5, 0, 100)
    c.axes([], [0, 25, 50, 75, 100], "", "下一轮能整段复用的比例", yfmt=lambda v: f"{v:g}%")
    for i, (lab, _) in enumerate(names):
        c.text(c.X(i), c.y0 + 16, lab, 12, INK2)
    c.bars([n for n, _ in names], ["普通对话", "带思考过程"], vals, [SERIES[0], SERIES[1]],
           fmt=lambda v: f"{v:.0f}%")
    c.text(c.x1 - 4, c.y1 - 26, "蓝: 普通对话   橙: 回复带思考过程", 11, INK2, "end")
    c.save(out("chart_prefix.svg"), "这一轮的文字是不是下一轮提示的开头? 三个家族不是",
           "每家 200 段随机多轮对话, 逐轮检查 (experiments/prefix_property.py); 不是开头 = 下一轮不能整段复用这一轮的缓存")


def chart_car_budget():
    b = load("practice_car.json")["budget"]
    budget_bars(b, "实践: 小车 v0.4 一次提示 {total} 个 token, 工具说明占 {share:.0%}",
                "小车项目的 5 个底层工具 (experiments/practice_car.py); 量法和第 8 节的通用小助手完全一样",
                "chart_car_budget.svg", "工具说明 (5 个工具)")


if __name__ == "__main__":
    for f in (chart_budget, chart_tools_scaling, chart_prefix, chart_car_budget):
        f()

