"""
svgfig.py — 讲义配图用的极简 SVG 工具 (只依赖标准库)

两类图:
  Svg    结构图: 方框、箭头、文字 (数据流、张量形状、谁和谁对应)
  Chart  数据图: 单坐标轴折线图 (实测曲线、对比)

用法见 projects/llm-tree/attention/figures/ 与 projects/llm-minigpt/figures/ 下的生成脚本。

约定 (dataviz 规范):
  - 背景用不透明浅色 BG, GitHub 深色模式下也能看清;
  - 类别色按 SERIES 的固定顺序分配 (已用 validate_palette.js 验证色盲可分);
  - 文字用正文色 INK / INK2 / MUTED, 不用系列色;
  - 图生成后必须渲染出来亲眼检查 (例如 msedge --headless --screenshot)。
"""

import math
import os

FONT = "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC','Source Han Sans SC',sans-serif"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BG = "#fcfcfb"
Q_LINE, Q_FILL = "#2a78d6", "#dbe9f9"
KV_LINE, KV_FILL = "#eb6834", "#fde3d7"
CACHE = "#008300"
OP_LINE, OP_FILL = "#52514e", "#efeeea"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRID, AXIS = "#e9e8e3", "#c9c8c2"


class Svg:
    """一个极简的 SVG 拼装器: 每个方法追加一个图元, save() 时补上外壳。"""

    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def rect(self, x, y, w, h, fill, stroke, dash=False, rx=4, sw=1.5):
        d = ' stroke-dasharray="6 4"' if dash else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=14, color=INK, anchor="middle", weight="normal", italic=False):
        style = ' font-style="italic"' if italic else ""
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                          f'font-weight="{weight}" dominant-baseline="middle"{style}>{s}</text>')

    def box(self, x, y, w, h, label, fill=OP_FILL, stroke=OP_LINE, size=14, sub=None, dash=False):
        """带文字的方框; sub 是方框下方的小字说明 (通常写形状)。"""
        self.rect(x, y, w, h, fill, stroke, dash)
        self.text(x + w / 2, y + h / 2, label, size)
        if sub:
            self.text(x + w / 2, y + h + 13, sub, 11, MUTED)

    def line(self, x1, y1, x2, y2, color=OP_LINE, arrow=True, width=1.4, dash=False):
        m = ' marker-end="url(#arrow)"' if arrow else ""
        d = ' stroke-dasharray="4 3"' if dash else ""
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{width}"{m}{d}/>')

    def path(self, d, color=OP_LINE, arrow=True, width=1.4):
        m = ' marker-end="url(#arrow)"' if arrow else ""
        self.items.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{m}/>')

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" font-family="{FONT}">\n'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{OP_LINE}"/>'
                '</marker></defs>\n'
                f'<rect width="100%" height="100%" fill="{BG}"/>\n')
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + "\n".join(self.items) + "\n</svg>\n")
        print("写入", os.path.basename(path))



class Chart:
    """一个只有一组坐标轴的折线图。"""

    def __init__(self, w, h, left=70, right=150, top=50, bottom=60):
        self.w, self.h = w, h
        self.x0, self.x1, self.y0, self.y1 = left, w - right, h - bottom, top   # 绘图区: y0 在下
        self.items = []

    def scales(self, xmin, xmax, ymin, ymax, xlog=False, ylog=False):
        """设定坐标范围。xlog / ylog: 对数刻度 (以 2 为底; 适合"每次翻倍"的数据)。"""
        self.xmin, self.xmax, self.ymin, self.ymax, self.xlog = xmin, xmax, ymin, ymax, xlog
        f = math.log2 if xlog else (lambda v: v)
        g = math.log2 if ylog else (lambda v: v)
        self._fx = f
        self.X = lambda v: self.x0 + (f(v) - f(xmin)) / (f(xmax) - f(xmin)) * (self.x1 - self.x0)
        self.Y = lambda v: self.y0 - (g(v) - g(ymin)) / (g(ymax) - g(ymin)) * (self.y0 - self.y1)

    def bars(self, groups, series, values, colors, width=0.8, labels=True, fmt=lambda v: f"{v:g}"):
        """分组柱状图 (x 轴为类别)。

        groups: 组名列表 (x 轴上的类别); series: 每组里的柱子名; values[i][j]: 第 i 组第 j 根柱子。
        调用前用 scales(-0.5, len(groups) - 0.5, ...) 设定坐标, 让第 i 组居中在 x = i。
        柱子之间留 2px 背景色间隙, 数据端 (顶部) 圆角 (dataviz 规范)。
        """
        n = len(series)
        slot = (self.X(1) - self.X(0)) * width / n          # 每根柱子的宽度 (像素)
        for i in range(len(groups)):
            left = self.X(i) - slot * n / 2
            for j in range(n):
                v = values[i][j]
                x, y_top, y_base = left + j * slot + 1, self.Y(v), self.Y(self.ymin)
                h = max(y_base - y_top, 0.5)
                w = slot - 2
                r = min(4, w / 2, h)
                # 只把顶部做成圆角: 用路径画出"上圆下方"的矩形
                d = (f"M{x:.1f},{y_base:.1f} L{x:.1f},{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} "
                     f"L{x + w - r:.1f},{y_top:.1f} Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f} "
                     f"L{x + w:.1f},{y_base:.1f} Z")
                self.add(f'<path d="{d}" fill="{colors[j]}"/>')
                if labels:
                    self.text(x + w / 2, y_top - 9, fmt(v), 10, INK2)

    def heatmap(self, x, y, matrix, cell, row_labels=None, col_labels=None, vmax=1.0, color="#2a78d6", fmt=None):
        """热力图 (单一色相, 由浅到深 = 由小到大; 0 显示为背景色)。

        matrix: 二维列表; cell: 每格边长 (像素); fmt: 给定时在格子里写数值。
        """
        for i, row in enumerate(matrix):
            for j, v in enumerate(row):
                a = max(0.0, min(1.0, v / vmax))
                self.add(f'<rect x="{x + j * cell:.1f}" y="{y + i * cell:.1f}" width="{cell - 1}" height="{cell - 1}" '
                         f'fill="{color}" fill-opacity="{a:.3f}" stroke="{GRID}" stroke-width="0.5"/>')
                if fmt:
                    ink = BG if a > 0.55 else INK
                    self.text(x + j * cell + cell / 2 - 0.5, y + i * cell + cell / 2, fmt(v), 10, ink)
        for i, lab in enumerate(row_labels or []):
            self.text(x - 6, y + i * cell + cell / 2, lab, 10, INK2, "end")
        for j, lab in enumerate(col_labels or []):
            self.text(x + j * cell + cell / 2, y - 8, lab, 10, INK2)

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

    def save(self, path, title, subtitle=None):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" font-family="{FONT}">\n<rect width="100%" height="100%" fill="{BG}"/>\n')
        t = f'<text x="20" y="22" font-size="15" font-weight="bold" fill="{INK}" dominant-baseline="middle">{title}</text>\n'
        if subtitle:
            t += f'<text x="20" y="40" font-size="12" fill="{INK2}" dominant-baseline="middle">{subtitle}</text>\n'
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + t + "\n".join(self.items) + "\n</svg>\n")
        print("写入", os.path.basename(path))



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

