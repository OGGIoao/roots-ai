"""
make_diagrams.py — 结构化输出讲义的结构图

  fig_trie.svg   小车 4 个合法工具调用组成的前缀树; 写到 "target": " 时, 哪些 token 放行、哪些被拦 (讲义第 2、4 节)
  fig_dfa.svg    约束数字格式的自动机 (讲义第 5.1 节, 练习 3 的参考答案)

运行: python figures/make_diagrams.py   (只用标准库)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, ROOT)

from common.svgfig import INK, INK2, KV_FILL, KV_LINE, MUTED, OP_FILL, OP_LINE, Q_FILL, Q_LINE, Svg  # noqa: E402

GREEN_FILL, GREEN_LINE = "#dcf3e9", "#1baf7a"
MONO = "Consolas,'Courier New',monospace"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mono(s, x, y, text, size=12, color=INK, anchor="start"):
    s.items.append(f'<text x="{x}" y="{y}" font-size="{size}" font-family="{MONO}" fill="{color}" '
                   f'text-anchor="{anchor}" dominant-baseline="middle">{esc(text)}</text>')


def fig_trie():
    s = Svg(900, 430)
    s.text(20, 22, "约束解码: 合法输出装进前缀树, 每一步只放行 \"接上去还在树上\" 的 token", 15, INK, "start", "bold")
    s.text(20, 42, "小车 v0.4 go_to 模式第一轮的 4 个合法回复 (constrained.py 的 car_tool_calls)", 12, INK2, "start")
    # 树
    s.rect(20, 66, 300, 34, OP_FILL, OP_LINE)
    mono(s, 30, 83, '<tool_call>\\n{"name": "', 12)
    s.rect(380, 66, 330, 34, Q_FILL, Q_LINE)
    mono(s, 390, 83, 'go_to", "arguments": {"target": "', 12)
    s.rect(380, 214, 330, 34, Q_FILL, Q_LINE)
    mono(s, 390, 231, 'look", "arguments": {}}\\n</tool_call>', 12)
    s.path("M320,83 L378,83")
    s.path("M320,90 C350,90 350,231 378,231")
    s.text(349, 70, "g", 11, MUTED)
    s.text(349, 170, "l", 11, MUTED)
    for i, t in enumerate(["red_cup", "blue_ball", "green_box"]):
        y = 124 + i * 28
        s.rect(740, y - 12, 140, 24, GREEN_FILL, GREEN_LINE, rx=3)
        mono(s, 748, y, t + '"}}…', 11)
        s.path(f"M710,90 C726,90 726,{y} 738,{y}")
    s.text(810, 214, "每条路走到底 = 一个完整的合法回复", 10, MUTED)
    # token 级的一步
    s.rect(20, 290, 860, 120, "#ffffff", MUTED, rx=4, sw=0.8)
    s.text(36, 308, "token 级的一步: 已经写了 …\"target\": \" , Qwen 词表里 15 万个 token 逐个试接", 12, INK, "start", "bold")
    ok = [("red", "接上是 red… 的前缀"), ("blue", ""), ("green", ""), ("g", "单字符也行")]
    bad = [("re", "→ \"re\" 不在树上? 在! red 的前缀也放行"), ("yellow", "树上没有"), ('"}}\\n', "目标还没写"), ("<|im_end|>", "还没写完, 不许停")]
    x = 36
    s.text(36, 334, "放行:", 11, GREEN_LINE, "start", "bold")
    for tok, _ in ok:
        w = 14 + 8 * len(tok)
        s.rect(x + 40, 324, w, 22, GREEN_FILL, GREEN_LINE, rx=3)
        mono(s, x + 47, 335, tok, 11)
        x += w + 8
    s.text(36, 370, "拦下:", 11, KV_LINE, "start", "bold")
    x = 36
    for tok in ("yellow", '"}}\\n', "<|im_end|>", "x"):
        w = 14 + 8 * len(tok)
        s.rect(x + 40, 360, w, 22, KV_FILL, KV_LINE, rx=3)
        mono(s, x + 47, 371, tok, 11)
        x += w + 8
    s.text(36, 398, "被拦的 token 分数设成 -inf, 剩下的照常按分数挑: 格式由约束保证, 选哪个目标仍由模型决定", 11, INK2, "start")
    del bad
    s.save(os.path.join(HERE, "fig_trie.svg"))


def fig_dfa():
    s = Svg(900, 300)
    s.text(20, 22, "约束数字格式的自动机: -?(0|[1-9][0-9]*)(\\.[0-9]+)?", 15, INK, "start", "bold")
    s.text(20, 42, "双框 = 可以停 (已经是完整的数字); 读到没有箭头的字符 = 这个前缀不可能合法 (number_dfa.py)", 12, INK2, "start")
    pos = {"start": (70, 160), "sign": (230, 230), "zero": (390, 100), "int": (390, 230), "dot": (570, 160), "frac": (760, 160)}
    accept = {"zero", "int", "frac"}
    for name, (x, y) in pos.items():
        if name in accept:
            s.rect(x - 52, y - 24, 104, 48, Q_FILL, Q_LINE, rx=22)
            s.rect(x - 47, y - 19, 94, 38, "none", Q_LINE, rx=18, sw=1)
        else:
            s.rect(x - 50, y - 22, 100, 44, OP_FILL, OP_LINE, rx=22)
        s.text(x, y, name, 13, INK)
    edges = [("start", "sign", "负号 -", 0.5), ("start", "zero", "0", 0.4), ("start", "int", "1-9", 0.25), ("sign", "zero", "0", 0.6),
             ("sign", "int", "1-9", 0.5), ("zero", "dot", "小数点 .", 0.5), ("int", "dot", "小数点 .", 0.5), ("dot", "frac", "0-9", 0.5)]
    for a, b, lab, t in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        dx, dy = x2 - x1, y2 - y1
        n = (dx * dx + dy * dy) ** 0.5
        sx, sy = x1 + dx / n * 52, y1 + dy / n * 26
        ex, ey = x2 - dx / n * 54, y2 - dy / n * 28
        s.line(sx, sy, ex, ey)
        s.text(sx + (ex - sx) * t + 10, sy + (ey - sy) * t - 11, lab, 12, KV_LINE, weight="bold")
    # 自环
    for name, lab in (("int", "0-9"), ("frac", "0-9")):
        x, y = pos[name]
        s.path(f"M{x - 18},{y + 24} C{x - 30},{y + 66} {x + 30},{y + 66} {x + 18},{y + 26}")
        s.text(x, y + 64, lab, 12, KV_LINE, weight="bold")
    s.save(os.path.join(HERE, "fig_dfa.svg"))


if __name__ == "__main__":
    fig_trie()
    fig_dfa()
