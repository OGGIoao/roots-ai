"""
make_charts.py — 生成 miniGPT 讲义里的数据图 (SVG)

数据来源:
  ../experiments/results/nano_curve.json        训练曲线                  (由 experiments/figure_data.py 生成)
  ../experiments/results/nano_attention.json    训练好的模型的注意力权重  (同上)
  ../experiments/results/nano_temperature.json  不同温度下的下一个字符概率 (同上)
  ../experiments/results/topics_measured.json   专题 A~E 引用的实测数字
  softmax 缩放、学习率曲线: 本脚本直接计算

运行: python figures/make_charts.py   (用到 common/svgfig.py; 实验重跑后再运行一次即可更新图)

画法约定与 mainline/A3-attention/figures/make_charts.py 相同 (dataviz 规范):
  固定类别色顺序、线宽 2px、单一 y 轴、图例 + 直接标签 + 标记形状、讲义里每张图配数据表。
"""

import json
import math
import os
import sys

from common.svgfig import BG, GRID, INK, INK2, MUTED, SERIES, Chart, spread

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "experiments", "results")
sys.path.insert(0, os.path.dirname(HERE))


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def out(name):
    return os.path.join(HERE, name)


def legend(c, items, x, y, step=120):
    """items: [(名字, 颜色, 标记)]; 横向排一行。"""
    for i, (name, color, mark) in enumerate(items):
        lx = x + i * step
        c.add(f'<line x1="{lx}" y1="{y}" x2="{lx + 18}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        if mark:
            c.marker(lx + 9, y, mark, color, 3.5)
        c.text(lx + 24, y, name, 11, INK, "start")


def legend_boxes(c, items, x, y, step=110):
    """柱状图用的方块图例。items: [(名字, 颜色)]"""
    for i, (name, color) in enumerate(items):
        lx = x + i * step
        c.add(f'<rect x="{lx}" y="{y - 6}" width="12" height="12" rx="2" fill="{color}"/>')
        c.text(lx + 18, y, name, 11, INK, "start")


def hline(c, v, label, color=MUTED, left=False):
    y = c.Y(v)
    c.add(f'<line x1="{c.x0}" y1="{y:.1f}" x2="{c.x1}" y2="{y:.1f}" stroke="{color}" stroke-dasharray="5 4"/>')
    if left:
        c.text(c.x0 + 6, y - 9, label, 11, INK2, "start")
    else:
        c.text(c.x1 - 4, y - 9, label, 11, INK2, "end")


# ── 1.2 节: nano 的训练曲线 ─────────────────────────────────────────

def fig_train_curve():
    d = load("nano_curve.json")
    steps, tr, va = d["steps"], d["train_loss_avg100"], d["val_loss"]
    c = Chart(820, 360, right=110, top=62)
    c.scales(0, 5000, 1.5, 4.5)
    c.axes(range(0, 5001, 1000), [1.5, 2, 2.5, 3, 3.5, 4, 4.5], "训练步数", "loss", yfmt=lambda v: f"{v:g}")
    hline(c, math.log(65), "ln 65 = 4.17: 在 65 个字符里瞎猜的 loss (第 8.3 节)")
    c.series(list(zip(steps, tr)), SERIES[0], width=1.5)
    c.series(list(zip(steps, va)), SERIES[1], width=2)
    ys = spread([c.Y(va[-1]), c.Y(tr[-1])])
    c.text(c.x1 + 10, ys[0], f"val {va[-1]:.2f}", 12, INK, "start")
    c.text(c.x1 + 10, ys[1], f"train {tr[-1]:.2f}", 12, INK, "start")
    c.text(c.X(steps[0]) + 10, c.Y(va[0]) + 4, f"第 0 步: {va[0]:.2f}", 11, INK2, "start")
    c.save(out("fig_train_curve.svg"), "nano 训练 5000 步: 从\"瞎猜\"到困惑度约 6.7",
           "train 为每 100 步的均值, val 每 100 步评估一次 (单独的一次运行; 数据: experiments/results/nano_curve.json)")


# ── 5.4 节: softmax 随输入放大逐渐"只选一个" ────────────────────────

def fig_softmax_scale():
    base = [1.0, 2.0, 3.0]
    scales = [1, 4, 8, 16]
    vals = []
    for k in scales:
        e = [math.exp(v * k - max(base) * k) for v in base]
        tot = sum(e)
        vals.append([x / tot for x in e])
    c = Chart(760, 340, left=70, right=40, top=70)
    c.scales(-0.5, len(scales) - 0.5, 0, 1.05)
    c.axes([], [0, 0.25, 0.5, 0.75, 1.0], "", "softmax 输出的概率", yfmt=lambda v: f"{v:g}")
    for i, k in enumerate(scales):
        c.text(c.X(i), c.y0 + 16, f"输入 × {k}", 12, INK2)
    c.bars(scales, ["值 1", "值 2", "值 3"], vals, SERIES[:3], fmt=lambda v: f"{v:.2f}" if v >= 0.005 else "0")
    legend_boxes(c, [("输入为 1 的位置", SERIES[0]), ("输入为 2 的位置", SERIES[1]), ("输入为 3 的位置", SERIES[2])],
                 c.x0 + 10, 52, 150)
    c.save(out("fig_softmax_scale.svg"), "同一组分数 [1, 2, 3] 放大后, softmax 越来越接近\"只选最大的\"",
           None)


# ── 5.5 节: 训练好的模型的真实注意力 ─────────────────────────────────

def fig_attention_maps():
    d = load("nano_attention.json")
    # 格子只有 12 像素宽: 换行写成 ↵, 空格写成 _ (␣ 在这个字号下几乎看不见)
    chars = ["↵" if ch == "\n" else ("_" if ch == " " else ch) for ch in d["chars"]]
    n, cell = len(chars), 12
    panel = n * cell
    c = Chart(2 * panel + 220, 2 * panel + 200)
    vmax = 0.6                                       # 每行权重分散在许多位置上, 上限取 0.6 才看得清; ≥0.6 都画成最深
    titles = [[f"第 {l + 1} 层 · 头 {h}" for h in range(2)] for l in range(2)]
    for l in range(2):
        for h in range(2):
            x = 60 + h * (panel + 90)
            y = 110 + l * (panel + 50)
            c.text(x + panel / 2, y - 26, titles[l][h], 13, INK, weight="bold")
            c.heatmap(x, y, d["layers"][l][h], cell, vmax=vmax,
                      row_labels=chars if h == 0 else None, col_labels=chars if l == 0 else None)
    lx, ly = c.w - 150, 52                            # 色阶图例
    c.text(lx - 8, ly + 6, "权重", 11, INK2, "end")
    for i, v in enumerate([0, 0.15, 0.3, 0.45, 0.6]):
        c.add(f'<rect x="{lx + i * 22}" y="{ly}" width="20" height="12" fill="#2a78d6" fill-opacity="{v / vmax:.2f}" '
              f'stroke="{GRID}" stroke-width="0.5"/>')
        c.text(lx + i * 22 + 10, ly + 24, f"{v:g}" if v < vmax else "≥0.6", 9, MUTED)
    c.text(c.w / 2, c.h - 16, "第 i 行 = 位置 i 的注意力分配到各位置上的比例 (每行和为 1); 颜色越深权重越大; 右上角恒为空白 = 因果掩码; _ 表示空格",
           11, INK2)
    c.save(out("fig_attention_maps.svg"), "训练好的 nano 模型在 \"ROMEO:↵I will go thither.\" 上的注意力",
           "2 层 × 2 个头 (数据: experiments/results/nano_attention.json)")


# ── 10.2 节: 学习率调度 ──────────────────────────────────────────────

def fig_lr():
    import train as train_module                     # 直接调用 train.py 里的 get_lr, 保证图和代码一致
    xs = list(range(0, 6001, 10))
    ys = [train_module.get_lr(it, 100, 5000, 1e-3, 1e-4) * 1e3 for it in xs]
    c = Chart(820, 340, right=40, top=62)
    c.scales(0, 6000, 0, 1.1)
    c.axes(range(0, 6001, 1000), [0, 0.25, 0.5, 0.75, 1.0], "训练步数", "学习率 (×10⁻³)", yfmt=lambda v: f"{v:g}")
    c.series(list(zip(xs, ys)), SERIES[0], width=2)
    for it, lab, dy in [(100, "热身结束: 1.0", -14), (2550, "正中间: 0.55", -14), (5000, "衰减结束: 0.1", -14)]:
        v = train_module.get_lr(it, 100, 5000, 1e-3, 1e-4) * 1e3
        c.marker(c.X(it), c.Y(v), "circle", SERIES[0], 4)
        c.text(c.X(it) + 8, c.Y(v) + dy, lab, 11, INK, "start")
    c.text(c.X(5500), c.Y(0.1) + 16, "之后保持最低值", 11, INK2)
    c.save(out("fig_lr.svg"), "学习率: 100 步线性热身, 再余弦下降到峰值的 10%",
           "train.py 的 get_lr(), lr=1e-3, warmup=100, 共 5000 步, min_lr=1e-4")


# ── 11.2 节: 温度 ────────────────────────────────────────────────────

def fig_temperature():
    d = load("nano_temperature.json")
    chars = ["↵" if ch == "\n" else ("空格" if ch == " " else ch) for ch in d["chars"]]
    temps = ["0.5", "1.0", "1.5"]
    groups = chars + ["其它 57 个"]
    vals = [[d["probs"][t][i] for t in temps] for i in range(len(chars))] + [[d["rest"][t] for t in temps]]
    c = Chart(860, 380, left=70, right=30, top=92)
    c.scales(-0.5, len(groups) - 0.5, 0, 0.9)
    c.axes([], [0, 0.2, 0.4, 0.6, 0.8], "下一个字符", "概率", yfmt=lambda v: f"{v:g}")
    for i, g in enumerate(groups):
        c.text(c.X(i), c.y0 + 16, g, 13 if i < len(chars) else 11, INK)
    c.bars(groups, temps, vals, [SERIES[0], SERIES[1], SERIES[2]], labels=False)
    legend_boxes(c, [("温度 0.5", SERIES[0]), ("温度 1.0", SERIES[1]), ("温度 1.5", SERIES[2])], c.x0 + 10, 66)
    c.save(out("fig_temperature.svg"), "前缀 \"ROMEO:↵I will th\" 之后, 温度越低越集中到最可能的 \"e\"",
           "训练好的 nano 模型; 按温度 1.0 时的概率取前 8 个字符 (数据: experiments/results/nano_temperature.json)")


# ── 专题 A: KV Cache 的加速 ─────────────────────────────────────────

def fig_kv_speedup():
    d = load("topics_measured.json")["kv_cache_gpt2_long"]
    L, a, b = d["lengths"], d["no_cache"], d["cache"]
    c = Chart(780, 340, right=140, top=62)
    c.scales(0, 1050, 0, 50)
    c.axes([0, 128, 256, 512, 768, 1000], [0, 10, 20, 30, 40, 50], "生成的 token 数", "总耗时 (秒)",
           yfmt=lambda v: f"{v:g}")
    c.series([(0, 0)] + list(zip(L, a)), SERIES[1], "square")
    c.series([(0, 0)] + list(zip(L, b)), SERIES[0], "circle")
    ys = spread([c.Y(a[-1]), c.Y(b[-1])])
    c.text(c.x1 + 10, ys[0], f"不用 cache {a[-1]}s", 12, INK, "start")
    c.text(c.x1 + 10, ys[1], f"用 cache {b[-1]}s", 12, INK, "start")
    sp = d["speedup"]                                  # 实验脚本用未舍入的时间算出的倍数
    c.text(c.X(512) + 8, c.Y(a[1]) - 12, f"快 {sp[1]} 倍", 11, INK2, "start")
    c.text(c.X(1000) - 10, c.Y(a[2]) + 4, f"快 {sp[2]} 倍", 11, INK2, "end")
    c.save(out("fig_kv_speedup.svg"), "GPT-2 Small 规模, GPU 上生成越长, KV Cache 省得越多",
           "不用 cache 按长度的平方增长, 用 cache 接近线性 (数据: experiments/results/topics_measured.json)")


# ── 专题 B: 注意力饱和 ──────────────────────────────────────────────

def fig_saturation():
    d = load("topics_measured.json")["attention_saturation"]
    groups = [f"d_k = {k}" for k in d["d_k"]]
    vals = [[d["scaled"]["effective_share"][i], d["unscaled"]["effective_share"][i]] for i in range(3)]
    c = Chart(720, 360, left=70, right=30, top=92)
    c.scales(-0.5, 2.5, 0, 100)
    c.axes([], [0, 25, 50, 75, 100], "", "收到有效梯度的位置 (%)", yfmt=lambda v: f"{v:g}%")
    for i, g in enumerate(groups):
        c.text(c.X(i), c.y0 + 16, g, 12, INK2)
    c.bars(groups, ["÷√d_k", "不缩放"], vals, [SERIES[0], SERIES[1]], fmt=lambda v: f"{v:.0f}%")
    legend_boxes(c, [("除以 √d_k", SERIES[0]), ("不缩放", SERIES[1])], c.x0 + 10, 66)
    c.save(out("fig_saturation.svg"), "不除以 √d_k 时, d_k 越大, 越少的位置能收到梯度",
           "单头注意力, S=32, 200 次随机试验 (数据: experiments/results/topics_measured.json)")


# ── 专题 D: 显存随序列长度 ──────────────────────────────────────────

def fig_flash_mem():
    d = load("topics_measured.json")["flash_attention"]
    S = d["S"]
    c = Chart(800, 380, right=150, top=62)
    c.scales(512, 8192, 8, 16384, xlog=True, ylog=True)
    c.axes(S, [8, 32, 128, 512, 2048, 8192], "序列长度 S (对数刻度)", "额外峰值显存 MB (对数刻度)",
           xfmt=lambda v: f"{v:,}", yfmt=lambda v: f"{v:,}")
    hline(c, 8192, "这张卡的显存 8GB", left=True)
    names = [("手写", SERIES[1], "square"), ("MATH", SERIES[3], "diamond"), ("MEM_EFFICIENT", SERIES[0], "circle")]
    ends = []
    for name, color, mark in names:
        pts = [(s, m) for s, m in zip(S, d[name]["mb"]) if m is not None]
        c.series(pts, color, mark)
        ends.append((c.Y(pts[-1][1]), name, pts[-1]))
    ys = spread([e[0] for e in ends], 16)
    for (_, name, (s, m)), y in zip(ends, ys):
        c.text(c.X(s) + 10, y, f"{name} {m:,}MB", 12, INK, "start")
    c.text(c.X(8192) - 6, c.Y(16) , "S = 8192 时手写与 MATH 显存不足 (没有点)", 11, INK2, "end")
    c.save(out("fig_flash_mem.svg"), "手写注意力的显存按 S² 增长, 分块实现 (MEM_EFFICIENT) 按 S 增长",
           "前向+反向, B=4 H=8 D=64, bf16; 每次 S 翻倍: 手写 ×4, MEM_EFFICIENT ×2 (数据: topics_measured.json)")


# ── 专题 E: 显存随 batch ────────────────────────────────────────────

def fig_mem_batch():
    d = load("topics_measured.json")["memory_activation"]
    batch, act, static, peak = d["batch"], d["activation_mb"], d["static_mb"], d["peak_mb"]
    c = Chart(760, 380, left=80, right=40, top=92)
    c.scales(-0.5, len(batch) - 0.5, 0, 9000)
    c.axes([], [0, 2000, 4000, 6000, 8000], "batch size", "显存 (MB)", yfmt=lambda v: f"{v:,}")
    slot = (c.X(1) - c.X(0)) * 0.55
    for i, (b, a, pk) in enumerate(zip(batch, act, peak)):
        x = c.X(i) - slot / 2
        y_static, y_top = c.Y(static), c.Y(pk)          # 柱顶 = 实测峰值 (不用静态 + 激活重算, 避免舍入误差)
        c.add(f'<rect x="{x:.1f}" y="{y_static:.1f}" width="{slot:.1f}" height="{c.Y(0) - y_static:.1f}" fill="{SERIES[0]}"/>')
        c.add(f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{slot:.1f}" height="{y_static - y_top - 2:.1f}" rx="3" fill="{SERIES[1]}"/>')
        c.text(c.X(i), y_top - 10, f"{pk:,}", 11, INK2)
        c.text(c.X(i), c.y0 + 16, str(b), 12, INK2)
    hline(c, 8192, "这张卡的显存 8GB", left=True)
    legend_boxes(c, [("静态: 参数 + 梯度 + AdamW 状态", SERIES[0]), ("激活: 随 batch 线性增长", SERIES[1])],
                 c.x0 + 10, 66, 230)
    c.save(out("fig_mem_batch.svg"), "GPT-2 Small 训练显存: 静态部分固定, 激活每个样本约 378MB",
           "fp32 + AdamW, block_size=256 (数据: experiments/results/topics_measured.json)")


if __name__ == "__main__":
    fig_train_curve()
    fig_softmax_scale()
    fig_attention_maps()
    fig_lr()
    fig_temperature()
    fig_kv_speedup()
    fig_saturation()
    fig_flash_mem()
    fig_mem_batch()
