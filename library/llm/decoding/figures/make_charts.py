"""
make_charts.py — 解码讲义的数据图 (数字来自 experiments/results/toy_experiments.json)

  chart_tradeoff.svg     多样性 (不重复 4 字片段占比) vs 像不像 (模型给的平均对数概率), 每种解码设置一个点
  chart_truncation.svg   分布越确定 (熵越低), top-k / top-p / min-p 各留下几个候选

运行: python figures/make_charts.py   (只用标准库)
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(UNIT)))
sys.path.insert(0, ROOT)

from common.svgfig import INK, INK2, MUTED, SERIES, Chart  # noqa: E402

with open(os.path.join(UNIT, "experiments", "results", "toy_experiments.json"), encoding="utf-8") as f:
    R = json.load(f)

SHORT = {
    "贪心": "贪心", "贪心 + 重复惩罚 1.1": "贪心+惩罚1.1", "贪心 + 重复惩罚 1.3": "贪心+惩罚1.3",
    "贪心 + 重复惩罚 2.0": "贪心+惩罚2.0", "温度 0.5": "T=0.5", "温度 0.7": "T=0.7", "温度 1.0": "T=1.0",
    "温度 1.5": "T=1.5", "温度 1.0 + top-k 5": "T=1 top-k5", "温度 1.0 + top-p 0.9": "T=1 top-p0.9",
    "温度 1.0 + min-p 0.1": "T=1 min-p0.1", "温度 1.5 + min-p 0.1": "T=1.5 min-p0.1",
}


def out(name):
    return os.path.join(HERE, name)


def chart_tradeoff():
    c = Chart(900, 470, left=80, right=330, top=66, bottom=64)
    c.scales(0, 1.0, -2.7, -0.9)
    c.axes([0, 0.2, 0.4, 0.6, 0.8, 1.0], [-2.6, -2.2, -1.8, -1.4, -1.0], "多样性: 不重复的 4 字片段占比 (越右越不重复)",
           "像不像: 平均对数概率 (越上越像)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    ref = R["corpus_reference"]
    c.add(f'<line x1="{c.X(ref["distinct4"]):.1f}" y1="{c.y1}" x2="{c.X(ref["distinct4"]):.1f}" y2="{c.y0}" '
          f'stroke="{MUTED}" stroke-dasharray="4 3"/>')
    c.add(f'<line x1="{c.x0}" y1="{c.Y(ref["avg_logprob"]):.1f}" x2="{c.x1}" y2="{c.Y(ref["avg_logprob"]):.1f}" '
          f'stroke="{MUTED}" stroke-dasharray="4 3"/>')
    c.marker(c.X(ref["distinct4"]), c.Y(ref["avg_logprob"]), "diamond", INK, 6)
    c.text(c.X(ref["distinct4"]) - 8, c.Y(ref["avg_logprob"]) + 16, "语料本身", 12, INK, "end", "bold")
    lx = c.x1 + 28
    c.text(lx + 18, c.y1 + 2, "编号  设置", 11, INK, "start", "bold")
    c.text(lx + 196, c.y1 + 2, "多样性", 11, INK, "start", "bold")
    c.text(lx + 240, c.y1 + 2, "像不像", 11, INK, "start", "bold")
    for i, row in enumerate(R["tradeoff"], 1):
        name = row["setting"]
        x, y = row["distinct4"], row["avg_logprob"]
        if name.startswith("Qwen"):
            color, mark, label = SERIES[3], "square", "Qwen2.5 自带设置"
        elif name.startswith("贪心"):
            color, mark, label = SERIES[1], "triangle", SHORT[name]
        else:
            color, mark, label = SERIES[0], "circle", SHORT[name]
        c.marker(c.X(x), c.Y(y), mark, color, 5)
        # 编号: 贪心那几个点挤在左上角, 编号往下错开
        if name.startswith("贪心"):
            c.text(c.X(x), c.Y(y) + 14 + 11 * (i - 1), str(i), 10, INK2)
        else:
            c.text(c.X(x) + 9, c.Y(y) - 7, str(i), 10, INK2, "start")
        ty = c.y1 + 22 + (i - 1) * 21
        c.marker(lx + 6, ty, mark, color, 4)
        c.text(lx + 18, ty, f"{i:>2}  {label}", 11, INK2, "start")
        c.text(lx + 196, ty, f"{x:.2f}", 11, INK2, "start")
        c.text(lx + 240, ty, f"{y:.2f}", 11, INK2, "start")
    c.text(c.X(0.2), c.Y(-1.0), "← 贪心类 (1~4): 300 字里全部转圈", 11, MUTED, "start")
    c.save(out("chart_tradeoff.svg"), "解码设置的取舍: 越多样, 越不像; 贪心最 \"像\", 却在原地转圈",
           "字符级三元模型 ToyLM, 5 个开头 × 20 个种子 × 300 字; 虚线交点是语料本身 (experiments/toy_experiments.py)")


def chart_truncation():
    pts = R["truncation"]
    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)]
    names = [("top_k5", "top-k = 5", SERIES[1], "triangle"), ("top_p0.9", "top-p = 0.9", SERIES[0], "circle"),
             ("min_p0.1", "min-p = 0.1", SERIES[2], "square")]
    c = Chart(860, 420, left=80, right=150, top=66, bottom=70)
    c.scales(0, 5, 0, 50)
    xs = []
    for lo, hi in bins:
        xs.append((lo + hi) / 2)
    counts = [sum(lo <= p["entropy_bits"] < hi for p in pts) for lo, hi in bins]
    tick = {xm: f"{lo}–{hi} ({n} 处)" for (lo, hi), xm, n in zip(bins, [0.5, 1.5, 2.5, 3.5, 5.0], counts)}
    c.axes([0.5, 1.5, 2.5, 3.5, 5.0], [0, 10, 20, 30, 40, 50], "下一个字符分布的熵 (比特, 越右越没把握; 括号里是落在这一段的位置数)",
           "平均留下几个候选", xfmt=lambda v: tick[v], yfmt=lambda v: f"{v:g}")
    for key, label, color, mark in names:
        line = []
        for (lo, hi), xm in zip(bins, [0.5, 1.5, 2.5, 3.5, 5.0]):
            sel = [p[key] for p in pts if lo <= p["entropy_bits"] < hi]
            if sel:
                line.append((xm, sum(sel) / len(sel)))
        c.series(line, color, mark)
        c.end_label(line[-1][0], line[-1][1], label, dy={"top_k5": 10, "top_p0.9": -6, "min_p0.1": 8}[key])
    c.text(c.X(0.6), c.Y(47), "top-k 在这里留下几十个: 三元模型给没见过的字符打同样的小分,", 11, INK2, "start")
    c.text(c.X(0.6), c.Y(43), "它们和第 5 名并列, 而并列的一律保留 (讲义第 4.4 节)", 11, INK2, "start")
    c.save(out("chart_truncation.svg"), "截断留下几个候选: top-p 和 min-p 随把握大小伸缩, top-k 是死数",
           "在语料里随机取 2000 个位置, 看 ToyLM 对下一个字符的分布 (experiments/toy_experiments.py)")


if __name__ == "__main__":
    chart_tradeoff()
    chart_truncation()
