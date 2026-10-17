"""
make_charts.py — 由实验结果生成注意力线讲义里的数据图 (SVG)

数据来源 (都在 ../results/):
  overfit_mha_3000.json   → fig_overfit.svg   一条完整的 train / val 曲线 (为什么要早停)
  per_run/*.json          → fig_curves.svg    6 个变体的验证曲线 (3 个种子的均值)
  decode.txt              → fig_decode.svg    生成一个 token 的耗时随上下文长度的变化

运行: python figures/make_charts.py   (只依赖标准库; 实验重跑后再运行一次即可更新图)

画法约定 (dataviz 规范):
  - 颜色按固定顺序分配给变体, 同一个变体在所有图里颜色相同;
  - 配色已用 validate_palette.js 验证 (浅色背景, 色盲可分); 其中 3 种颜色对比度偏低,
    所以每条线末端都直接标注名字, 并给每个变体不同的标记形状, 不只靠颜色区分;
  - 线宽 2px, 网格和坐标轴颜色很淡; 只有一个 y 轴;
  - 讲义里每张图旁边都配有对应的数据表 (静态图片做不了悬停提示, 用表格代替)。
"""

import glob
import json
import os
import re
import statistics

from make_diagrams import BG, FONT, INK, INK2, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

# 固定的类别色顺序 (dataviz 参考调色板, 浅色模式)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ORDER = ["MHA", "GQA-4", "GQA-2", "MQA", "MLA-128", "MLA-64"]
COLOR = dict(zip(ORDER, SERIES))
MARK = dict(zip(ORDER, ["circle", "square", "triangle", "diamond", "circle-open", "square-open"]))
GRID, AXIS = "#e9e8e3", "#c9c8c2"


class Chart:
    """一个只有一组坐标轴的折线图。"""

    def __init__(self, w, h, left=70, right=150, top=50, bottom=60):
        self.w, self.h = w, h
        self.x0, self.x1, self.y0, self.y1 = left, w - right, h - bottom, top   # 绘图区: y0 在下
        self.items = []

    def scales(self, xmin, xmax, ymin, ymax, xlog=False):
        import math
        self.xmin, self.xmax, self.ymin, self.ymax, self.xlog = xmin, xmax, ymin, ymax, xlog
        f = math.log2 if xlog else (lambda v: v)
        self._fx = f
        self.X = lambda v: self.x0 + (f(v) - f(xmin)) / (f(xmax) - f(xmin)) * (self.x1 - self.x0)
        self.Y = lambda v: self.y0 - (v - ymin) / (ymax - ymin) * (self.y0 - self.y1)

    def add(self, s):
        self.items.append(s)

    def text(self, x, y, s, size=12, color=INK2, anchor="middle", weight="normal"):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                 f'font-weight="{weight}" dominant-baseline="middle">{s}</text>')

    def axes(self, xticks, yticks, xlabel, ylabel, xfmt=str, yfmt=str):
        for v in yticks:                                  # 水平网格 + y 刻度
            y = self.Y(v)
            self.add(f'<line x1="{self.x0}" y1="{y:.1f}" x2="{self.x1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            self.text(self.x0 - 8, y, yfmt(v), 11, MUTED, "end")
        for v in xticks:                                  # x 刻度
            x = self.X(v)
            self.add(f'<line x1="{x:.1f}" y1="{self.y0}" x2="{x:.1f}" y2="{self.y0 + 4}" stroke="{AXIS}"/>')
            self.text(x, self.y0 + 16, xfmt(v), 11, MUTED)
        self.add(f'<line x1="{self.x0}" y1="{self.y0}" x2="{self.x1}" y2="{self.y0}" stroke="{AXIS}" stroke-width="1"/>')
        self.text((self.x0 + self.x1) / 2, self.y0 + 38, xlabel, 12, INK2)
        self.add(f'<text x="{self.x0 - 50}" y="{(self.y0 + self.y1) / 2:.1f}" font-size="12" fill="{INK2}" '
                 f'text-anchor="middle" transform="rotate(-90 {self.x0 - 50} {(self.y0 + self.y1) / 2:.1f})">{ylabel}</text>')

    def marker(self, x, y, kind, color, size=4.5):
        open_ = kind.endswith("-open")
        fill = BG if open_ else color
        k = kind.replace("-open", "")
        if k == "circle":
            self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size}" fill="{fill}" stroke="{color}" stroke-width="2"/>')
        elif k == "square":
            self.add(f'<rect x="{x - size:.1f}" y="{y - size:.1f}" width="{2 * size}" height="{2 * size}" '
                     f'fill="{fill}" stroke="{color}" stroke-width="2"/>')
        elif k == "triangle":
            p = f"{x:.1f},{y - size - 1:.1f} {x - size - 1:.1f},{y + size:.1f} {x + size + 1:.1f},{y + size:.1f}"
            self.add(f'<polygon points="{p}" fill="{fill}" stroke="{color}" stroke-width="2"/>')
        elif k == "diamond":
            p = f"{x:.1f},{y - size - 1.5:.1f} {x + size + 1.5:.1f},{y:.1f} {x:.1f},{y + size + 1.5:.1f} {x - size - 1.5:.1f},{y:.1f}"
            self.add(f'<polygon points="{p}" fill="{fill}" stroke="{color}" stroke-width="2"/>')

    def series(self, pts, color, mark=None, every=1, dash=False, width=2):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.X(x):.1f},{self.Y(y):.1f}" for i, (x, y) in enumerate(pts))
        da = ' stroke-dasharray="6 4"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{da}/>')
        if mark:
            for i, (x, y) in enumerate(pts):
                if i % every == 0 or i == len(pts) - 1:
                    self.marker(self.X(x), self.Y(y), mark, color)

    def end_label(self, x, y, label, dy=0):
        """线末端的直接标签: 文字用正文色, 不用系列色 (dataviz 规范)。"""
        self.text(self.X(x) + 10, self.Y(y) + dy, label, 12, INK, "start")

    def save(self, name, title, subtitle=None):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" font-family="{FONT}">\n<rect width="100%" height="100%" fill="{BG}"/>\n')
        t = f'<text x="20" y="22" font-size="15" font-weight="bold" fill="{INK}" dominant-baseline="middle">{title}</text>\n'
        if subtitle:
            t += f'<text x="20" y="40" font-size="12" fill="{INK2}" dominant-baseline="middle">{subtitle}</text>\n'
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            f.write(head + t + "\n".join(self.items) + "\n</svg>\n")
        print("写入", name)


def spread(labels, min_gap=15):
    """末端标签按 y 坐标排开, 避免重叠。labels: [(y, ...)] → 返回调整后的 y 列表 (保持原顺序)。"""
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    ys = [labels[i] for i in order]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    out = [0] * len(labels)
    for k, i in enumerate(order):
        out[i] = ys[k]
    return out


# ── 图: 过拟合 ──────────────────────────────────────────────────────

def fig_overfit():
    with open(os.path.join(RES, "overfit_mha_3000.json"), encoding="utf-8") as f:
        r = json.load(f)
    curve = r["curve"]                                     # [(步数, train loss, val loss)]
    c = Chart(820, 360, right=120, top=62)
    c.scales(0, 3000, 0, 3.5)
    c.axes(range(0, 3001, 500), [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], "训练步数", "loss",
           yfmt=lambda v: f"{v:g}")
    train = [(s, t) for s, t, _ in curve]
    val = [(s, v) for s, _, v in curve]
    c.series(train, SERIES[0], width=1.5)
    c.series(val, SERIES[1], width=2)
    best = min(curve, key=lambda x: x[2])
    bx, by = c.X(best[0]), c.Y(best[2])
    c.add(f'<line x1="{bx:.1f}" y1="{c.y1}" x2="{bx:.1f}" y2="{c.y0}" stroke="{MUTED}" stroke-dasharray="4 3"/>')
    c.marker(bx, by, "circle", SERIES[1], 5)
    c.text(bx + 8, by - 26, f"最低 val_loss {best[2]:.3f} (第 {best[0]} 步)", 12, INK, "start")
    c.text(bx + 8, c.y1 + 10, "早停点: 对照实验比较的就是这里", 11, INK2, "start")
    ys = spread([c.Y(val[-1][1]), c.Y(train[-1][1])])
    c.text(c.x1 + 10, ys[0], f"val {val[-1][1]:.2f}", 12, INK, "start")
    c.text(c.x1 + 10, ys[1], f"train {train[-1][1]:.2f}", 12, INK, "start")
    c.save("fig_overfit.svg", "MHA 训练 3000 步: 约 1000 步后开始背训练集",
           "Tiny Shakespeare 字符级, 4 层 256 维; 每 100 步评估一次 (数据: results/overfit_mha_3000.json)")


# ── 图: 6 个变体的验证曲线 ───────────────────────────────────────────

def fig_curves():
    runs = {}
    for path in glob.glob(os.path.join(RES, "per_run", "*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        if r["name"] in ORDER:
            runs.setdefault(r["name"], []).append(r["curve"])
    c = Chart(820, 400, right=110, top=62)
    c.scales(100, 1500, 1.45, 2.2)
    c.axes(range(100, 1501, 200), [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2], "训练步数", "验证 loss (3 个种子的均值)",
           yfmt=lambda v: f"{v:.1f}")
    ends = []
    for name in ORDER:
        curves = runs[name]
        steps = [s for s, _, _ in curves[0]]
        mean = [statistics.mean(cv[i][2] for cv in curves) for i in range(len(steps))]
        pts = list(zip(steps, mean))
        c.series(pts, COLOR[name], MARK[name], every=2)
    # 6 条线的末端挤在约 25 像素内, 直接标签会离线太远而误导; 按规范 >4 个系列只用图例 + 标记形状
    # 图例 (≥ 2 个系列时始终保留), 放在图内右上空白处
    lx, ly = c.x1 - 250, c.y1 + 6
    for i, name in enumerate(ORDER):
        x = lx + (i % 3) * 85
        y = ly + (i // 3) * 20
        c.add(f'<line x1="{x}" y1="{y}" x2="{x + 18}" y2="{y}" stroke="{COLOR[name]}" stroke-width="2"/>')
        c.marker(x + 9, y, MARK[name], COLOR[name], 3.5)
        c.text(x + 24, y, name, 11, INK, "start")
    c.save("fig_curves.svg", "6 种注意力的验证曲线几乎重合",
           "同一解码器只换注意力模块; 最低点都在第 700~900 步附近, 之后一起过拟合 (数据: results/per_run/)")


# ── 图: 推理速度随上下文长度 ─────────────────────────────────────────

def fig_decode():
    with open(os.path.join(RES, "decode.txt"), encoding="utf-8") as f:
        text = f.read()
    rows = {}
    for line in text.splitlines():
        m = re.match(r"\s+(MHA|GQA-4|MQA|MLA \(naive\)|MLA \(absorbed\))\s+\S+KB\s+\S+MB\s+([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)ms", line)
        if m:
            rows.setdefault(m.group(1), []).append([float(m.group(i)) for i in (2, 3, 4)])
    names = ["MHA", "GQA-4", "MQA", "MLA (naive)", "MLA (absorbed)"]
    colors = dict(zip(names, SERIES[:5]))
    marks = dict(zip(names, ["circle", "square", "triangle", "diamond", "circle-open"]))
    lens = [1024, 4096, 8192]
    c = Chart(820, 400, right=150, top=62)
    c.scales(1024, 8192, 0, 22, xlog=True)
    c.axes(lens, [0, 5, 10, 15, 20], "上下文长度 (对数刻度)", "生成一个 token 的耗时 (ms)",
           xfmt=lambda v: f"{v:,}", yfmt=lambda v: f"{v:g}")
    ends = []
    for name in names:
        runs = rows[name]                                 # 两次运行, 取均值画线
        mean = [statistics.mean(r[i] for r in runs) for i in range(3)]
        c.series(list(zip(lens, mean)), colors[name], marks[name])
        for r in runs:                                    # 两次运行的原始点画成小空心点, 显示波动
            for x, v in zip(lens, r):
                c.add(f'<circle cx="{c.X(x):.1f}" cy="{c.Y(v):.1f}" r="2.2" fill="none" '
                      f'stroke="{colors[name]}" stroke-width="1" opacity="0.7"/>')
        ends.append((c.Y(mean[-1]), name))
    ys = spread([e[0] for e in ends], 15)
    for (y0, name), y in zip(ends, ys):
        c.text(c.x1 + 10, y, name, 12, INK, "start")
    lx, ly = c.x0 + 16, c.y1 + 6                          # 图例: 左上角空白处
    for i, name in enumerate(names):
        x, y = lx + (i % 3) * 130, ly + (i // 3) * 20
        c.add(f'<line x1="{x}" y1="{y}" x2="{x + 18}" y2="{y}" stroke="{colors[name]}" stroke-width="2"/>')
        c.marker(x + 9, y, marks[name], colors[name], 3.5)
        c.text(x + 24, y, name, 11, INK, "start")
    c.save("fig_decode.svg", "长上下文下, 缓存越小越快; 朴素 MLA 被解压拖慢, 吸收版几乎不随长度变化",
           "8 层 1024 维 16 头, bf16, RTX 5060; 线为两次运行的均值, 小圆点为每次的原始值 (数据: results/decode.txt)")


if __name__ == "__main__":
    fig_overfit()
    fig_curves()
    fig_decode()
