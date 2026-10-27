"""
make_diagrams.py — 对话模板讲义的配图 (SVG)

  fig_render.svg     一段对话 (左) 被模板拼成一整段文字 (右); 蓝底是模型自己要写的部分 (讲义第 1、6 节)
  chart_budget.svg   v0.4 小车的一次提示, 各部分各占多少 token (讲义第 5 节, 数字来自 results/tokens.json)

运行: python figures/make_diagrams.py   (只用标准库)
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(UNIT)))
sys.path.insert(0, ROOT)
sys.path.insert(0, UNIT)

from chat_template import render_with_spans  # noqa: E402
from common.svgfig import (INK, INK2, KV_FILL, KV_LINE, MUTED, OP_FILL, Q_FILL, Q_LINE, SERIES, Svg)  # noqa: E402

GREEN_FILL, GREEN_LINE = "#dcf3e9", "#1baf7a"
MONO = "Consolas,'Courier New',monospace"
SPECIAL = re.compile(r"(<\|im_start\|>|<\|im_end\|>|<tool_call>|</tool_call>)")   # 词表里的新增 token, 各占 1 个


def out(name):
    return os.path.join(HERE, name)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


CONV = [
    {"role": "system", "content": "你是小车。"},
    {"role": "user", "content": "往前走 0.5 米"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "walk", "arguments": {"distance_m": 0.5}}}]},
    {"role": "tool", "content": '{"ok": true}'},
    {"role": "assistant", "content": "走好了。"},
]


def fig_render():
    text, spans = render_with_spans(CONV, add_generation_prompt=False)
    lines, pos = [], 0
    for ln in text.split("\n"):
        mid = pos + len(ln) // 2 if ln else pos
        trainable = any(s <= mid < e for s, e in spans) or any(s <= pos < e for s, e in spans)
        lines.append((ln, trainable))
        pos += len(ln) + 1
    if lines and lines[-1][0] == "":
        lines.pop()

    H = 130 + 26 * len(lines)
    s = Svg(900, H)
    s.text(20, 22, "对话模板: 把一条条消息拼成一整段文字, 模型只会接着这段文字往下写", 15, INK, "start", "bold")
    s.text(20, 42, "chat_template.render; 橙色: 词表里专门新增的标记, 各占 1 个 token; 蓝底: assistant 自己要写的部分 (微调时只在这里算 loss)",
           12, INK2, "start")

    # 左: 消息卡片
    colors = {"system": OP_FILL, "user": "#ffffff", "assistant": Q_FILL, "tool": GREEN_FILL}
    labels = ["system: 你是小车。", "user: 往前走 0.5 米", "assistant: tool_calls walk(0.5)",
              "tool: {\"ok\": true}", "assistant: 走好了。"]
    s.text(130, 74, "messages (5 条)", 13, INK, weight="bold")
    for i, (m, lab) in enumerate(zip(CONV, labels)):
        y = 90 + i * 44
        s.rect(20, y, 220, 32, colors[m["role"]], MUTED, rx=3, sw=0.8)
        s.text(30, y + 16, esc(lab), 11, INK, "start")
    s.line(250, 190, 300, 190)
    s.text(275, 178, "render", 10, INK2)

    # 右: 拼出来的文字
    x0, y0 = 312, 70
    s.text(x0, y0 + 4, "模型看到的文字 (一整段, 这里按换行折开)", 13, INK, "start", "bold")
    for i, (ln, trainable) in enumerate(lines):
        y = y0 + 22 + i * 26
        if trainable:
            s.rect(x0 - 4, y - 2, 560, 24, Q_FILL, Q_LINE, rx=2, sw=0.6)
        parts = SPECIAL.split(ln)
        tspans = "".join(
            f'<tspan fill="{KV_LINE}" font-weight="bold">{esc(p)}</tspan>' if SPECIAL.fullmatch(p or "x")
            else f'<tspan fill="{INK}">{esc(p)}</tspan>' for p in parts if p)
        s.items.append(f'<text x="{x0}" y="{y + 11}" font-size="12.5" font-family="{MONO}" '
                       f'dominant-baseline="middle">{tspans or " "}</text>')
    s.text(20, H - 16, esc("<tool_response> 不是新增标记, 会被切成 < tool _response > 四个普通 token; 没给 system 时模板自动补一句 \"You are Qwen...\" (讲义第 3 节)"),
           11, MUTED, "start")
    s.save(out("fig_render.svg"))


def chart_budget():
    with open(os.path.join(UNIT, "experiments", "results", "tokens.json"), encoding="utf-8") as f:
        b = json.load(f)["budget"]
    rows = [("工具说明 (5 个工具)", b["tools_block"], SERIES[1]),
            ("系统提示", b["system_prompt"], SERIES[0]),
            ("模板标记", b["template_markers"], SERIES[2]),
            ("用户指令", b["user"], SERIES[3])]
    total = b["total_with_tools"]
    s = Svg(820, 350)
    s.text(20, 22, f"v0.4 小车的一次提示共 {total} 个 token, 七成是工具说明", 15, INK, "start", "bold")
    s.text(20, 42, "Qwen2.5 分词器实测 (experiments/measure_tokens.py); 每轮对话都要把这些重新喂给模型", 12, INK2, "start")
    x0, w = 190, 520
    for i, (lab, v, c) in enumerate(rows):
        y = 80 + i * 44
        s.text(x0 - 12, y + 13, lab, 13, INK, "end")
        bw = max(w * v / total, 2)
        s.items.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="26" rx="3" fill="{c}"/>')
        s.text(x0 + bw + 8, y + 13, f"{v}  ({v / total:.0%})", 12, INK2, "start")
    # 工具逐个
    y = 80 + 4 * 44 + 18
    s.text(20, y, "工具说明里每个工具的 JSON (不含模板固定的说明文字 "
           f"{b['tools_block'] - b['tools_json_only']} 个 token):", 12, INK2, "start")
    xs, tw = x0, w + 80                              # 这一行单独按工具 JSON 的总数铺满
    for name, v in sorted(b["per_tool"].items(), key=lambda kv: -kv[1]):
        bw = tw * v / b["tools_json_only"]
        s.items.append(f'<rect x="{xs:.1f}" y="{y + 16}" width="{bw - 3:.1f}" height="36" rx="2" fill="{KV_FILL}" stroke="{KV_LINE}" stroke-width="0.8"/>')
        s.text(xs + bw / 2 - 1.5, y + 27, name, 11, INK)
        s.text(xs + bw / 2 - 1.5, y + 42, str(v), 11, INK2)
        xs += bw
    s.save(out("chart_budget.svg"))


if __name__ == "__main__":
    fig_render()
    chart_budget()
