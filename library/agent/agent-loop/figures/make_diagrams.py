"""
make_diagrams.py — C1 讲义里的结构图 (SVG)

  fig_loop.svg    Agent 循环: 对话 → 大模型 → 安全层 → 小车 → 观测追加回对话 (讲义第 1 节)
  fig_modes.svg   五种模式各自去掉 / 改动了循环的哪一段 (讲义第 2、3、6 节)

运行: python figures/make_diagrams.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, ROOT)

from common.svgfig import (INK, INK2, KV_FILL, KV_LINE, MUTED, OP_FILL, OP_LINE, Q_FILL,  # noqa: E402
                           Q_LINE, Svg)

GREEN_FILL, GREEN_LINE = "#dcf3e9", "#1baf7a"


def out(name):
    return os.path.join(HERE, name)


def fig_loop():
    s = Svg(860, 400)
    s.text(20, 22, "Agent 循环: 大模型只负责\"说要做什么\", 做没做、做成什么样, 由循环把结果喂回去", 15, INK, "start", "bold")
    s.text(20, 42, "agent.py 的 run_episode; 一轮 = 问一次模型 + 执行它这一轮的工具调用", 12, INK2, "start")
    # 对话 (左)
    s.rect(30, 70, 230, 290, OP_FILL, OP_LINE)
    s.text(145, 88, "对话 messages", 14, INK, weight="bold")
    rows = [("system", "场地、工具用法、成功标准"), ("user", "\"去红杯子那里, 停在它跟前\""),
            ("assistant", "tool_calls: look()"), ("tool", "{objects: [red_cup 0.9m -12°]}"),
            ("assistant", "tool_calls: turn(-12)"), ("tool", "{ok: true}"), ("…", "越来越长"),
            ("assistant", "\"已到达红杯子跟前\"")]
    for i, (role, txt) in enumerate(rows):
        y = 112 + i * 30
        fill = {"tool": GREEN_FILL, "assistant": Q_FILL}.get(role, "#ffffff")
        s.rect(42, y - 11, 206, 24, fill, MUTED, rx=3, sw=0.8)
        s.text(50, y + 1, role, 10, INK2, "start")
        s.text(112, y + 1, txt, 10, INK, "start")
    # 大模型
    s.box(330, 90, 170, 60, "大模型", Q_FILL, Q_LINE, 14, None)
    s.text(415, 162, "llm.chat(messages, tools)", 11, MUTED)
    s.line(260, 120, 328, 120)
    s.text(294, 110, "全部", 10, INK2)
    # 回复分叉
    s.box(580, 70, 230, 40, "只说话 → 结束", OP_FILL, OP_LINE, 12)
    s.box(580, 140, 230, 40, "tool_calls → 执行", Q_FILL, Q_LINE, 12)
    s.line(500, 112, 578, 92)
    s.line(500, 128, 578, 158)
    # 安全层与小车
    s.box(580, 220, 230, 46, "安全层 validate_call", KV_FILL, KV_LINE, 13)
    s.text(695, 278, "工具名、参数、范围不对 → 不执行, 回错误说明", 10, INK2)
    s.line(695, 180, 695, 218)
    s.box(580, 300, 230, 46, "小车 (sim/car2d.py)", GREEN_FILL, GREEN_LINE, 13)
    s.text(695, 358, "动作有误差; look 有噪声和遮挡", 10, INK2)
    s.line(695, 288, 695, 298)
    # 观测回到对话
    s.path("M578,323 L330,323 L262,323", arrow=True)
    s.text(420, 312, "观测 (role=tool, tool_call_id 对上)", 11, INK2)
    s.text(420, 338, "追加进对话, 下一轮模型就能看到", 11, INK2)
    s.save(out("fig_loop.svg"))


def fig_modes():
    s = Svg(860, 380)
    s.text(20, 22, "五种模式 = 同一个循环, 各自去掉或改动一段", 15, INK, "start", "bold")
    s.text(20, 42, "讲义的案例顺序; 每种模式在 agent.py 里都只有十来行", 12, INK2, "start")
    rows = [
        ("只聊天 chat", "不给工具", "模型只能回一段话, 车不会动", "第 2 节"),
        ("开环 plan", "先 look 一次, 要求一次给出全部动作", "按顺序执行, 中途不再问模型; 走偏了也不知道", "第 2 节"),
        ("闭环 loop", "标准循环", "每轮的结果都喂回去; 但一轮里发多个调用会全部照做", "第 3 节"),
        ("一步一停 step", "每轮只执行第一个调用", "其余的退回, 逼模型先看结果再决定", "第 3 节"),
        ("技能 go_to", "只给 go_to(target) 和 look", "大模型只管听懂去哪; 怎么开过去交给控制代码", "第 6 节"),
    ]
    fills = [OP_FILL, KV_FILL, Q_FILL, GREEN_FILL, "#fbe4ee"]
    lines = [OP_LINE, KV_LINE, Q_LINE, GREEN_LINE, "#e87ba4"]
    for i, (name, change, effect, sec) in enumerate(rows):
        y = 64 + i * 60
        s.rect(30, y, 150, 46, fills[i], lines[i])
        s.text(105, y + 23, name, 13, INK, weight="bold")
        s.text(200, y + 14, change, 12, INK, "start")
        s.text(200, y + 34, effect, 11, INK2, "start")
        s.text(830, y + 23, sec, 11, MUTED, "end")
    s.save(out("fig_modes.svg"))


if __name__ == "__main__":
    fig_loop()
    fig_modes()
