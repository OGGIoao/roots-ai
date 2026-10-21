"""
make_charts.py — 由实验结果生成老虎机讲义里的数据图 (SVG)

数据来源 (都在 ../results/, 由 ../experiments/ 下的脚本生成):
  greedy.json          → fig_greedy.svg        贪心 vs ε-greedy 的最优臂比例
                       → fig_rank.svg          贪心最后停在第几好的臂
  explore.json         → fig_explore.svg       四种探索方法的最优臂比例 (1000 步)
                       → fig_spike.svg         前 30 步放大: 第 11 步的尖峰
                       → fig_regret.svg        六种方法的累计遗憾
                       → fig_long.svg          10000 步: ε=0.01 反超 ε=0.1
  sweep.json           → fig_sweep.svg         超参数敏感性
  nonstationary.json   → fig_nonstationary.svg 收益在变时: 样本平均 vs 固定步长

运行: python figures/make_charts.py   (实验重跑后再运行一次即可更新图)

画法约定同 mainline/A3-attention/figures/make_charts.py (dataviz 规范):
  同一个方法在所有图里颜色、标记相同; 线末端直接标注名字, 不只靠颜色区分;
  只有一个 y 轴; 讲义里每张图旁边配数据表。
"""

import json
import os

from common.svgfig import INK, INK2, MUTED, SERIES, Chart, spread

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

# 固定的类别色: 同一个方法在所有图里同一种颜色
COLOR = {
    "greedy": SERIES[3], "eps=0.01": SERIES[4], "eps=0.1": SERIES[0],
    "optimistic Q0=5": SERIES[1], "realistic eps=0.1": SERIES[5], "UCB c=2": SERIES[2],
    "sample-average": SERIES[0], "alpha=0.1": SERIES[1],
}
LABEL = {
    "greedy": "贪心", "eps=0.01": "ε=0.01", "eps=0.1": "ε=0.1",
    "optimistic Q0=5": "乐观初值 Q0=5", "realistic eps=0.1": "ε=0.1 固定步长", "UCB c=2": "UCB c=2",
    "sample-average": "样本平均", "alpha=0.1": "固定步长 α=0.1",
}


def load(name):
    with open(os.path.join(RES, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def smooth(ys, w):
    """居中滑动平均 (两端窗口缩短), 只用于画线, 数据表仍用原始值。"""
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - w // 2), min(len(ys), i + w // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def lines(c, curves, names, key, xs, w=1, fmt=lambda v: f"{v:.0%}"):
    """画若干条线并在右端标注名字和末端数值。"""
    ends = []
    for name in names:
        ys = smooth(curves[name][key], w) if w > 1 else curves[name][key]
        c.series(list(zip(xs, ys)), COLOR[name])
        ends.append((c.Y(ys[-1]), name, ys[-1]))
    for y, (_, name, v) in zip(spread([e[0] for e in ends]), ends):
        c.text(c.x1 + 10, y, f"{LABEL[name]} {fmt(v)}", 12, INK, "start")


def pct(v):
    return f"{v:.0%}"


# ── 第 4 节 ────────────────────────────────────────────────────────

def fig_greedy():
    g = load("greedy")
    c = Chart(820, 360, right=150, top=62)
    c.scales(0, 1000, 0, 1)
    c.axes(range(0, 1001, 200), [0, 0.2, 0.4, 0.6, 0.8, 1], "步数", "选中最优臂的比例", yfmt=pct)
    lines(c, g["curves"], ["eps=0.1", "eps=0.01", "greedy"], "optimal_frac", range(1, 1001), w=9)
    c.save(os.path.join(HERE, "fig_greedy.svg"), "贪心很快停在一个低位, 之后再也上不去",
           "10 臂测试台, 2000 局平均, 9 步滑动平均 (数据: results/greedy.json)")


def fig_rank():
    g = load("greedy")
    hist = g["lock"]["rank_hist"]
    total = sum(hist)
    c = Chart(820, 340, right=40, top=62)
    c.scales(-0.5, 9.5, 0, 0.4)
    c.axes([], [0, 0.1, 0.2, 0.3, 0.4], "", "局数占比", yfmt=pct)
    c.bars([str(i) for i in range(1, 11)], ["局数"], [[h / total] for h in hist], [SERIES[3]],
           fmt=lambda v: f"{v:.1%}")
    for i in range(10):
        c.text(c.X(i), c.y0 + 16, f"第 {i + 1}", 11, MUTED)
    c.text((c.x0 + c.x1) / 2, c.y0 + 38, "贪心最后 100 步最常拉的臂, 在本局 10 个臂里按真实均值排第几", 12, INK2)
    c.save(os.path.join(HERE, "fig_rank.svg"), f"只有 {hist[0] / total:.1%} 的局停在了最好的臂上",
           "贪心, 2000 局, 1000 步 (数据: results/greedy.json 的 lock.rank_hist)")


# ── 第 5~7 节 ──────────────────────────────────────────────────────

def fig_explore():
    e = load("explore")
    c = Chart(820, 380, right=190, top=62)
    c.scales(0, 1000, 0, 1)
    c.axes(range(0, 1001, 200), [0, 0.2, 0.4, 0.6, 0.8, 1], "步数", "选中最优臂的比例", yfmt=pct)
    lines(c, e["curves"], ["UCB c=2", "optimistic Q0=5", "eps=0.1", "realistic eps=0.1", "greedy"],
          "optimal_frac", range(1, 1001), w=9)
    c.save(os.path.join(HERE, "fig_explore.svg"), "四种探索办法都把贪心甩开了",
           "2000 局平均, 9 步滑动平均 (数据: results/explore.json, env_seed=0)")


def fig_spike():
    e = load("explore")
    names = ["UCB c=2", "optimistic Q0=5", "eps=0.1"]
    c = Chart(820, 360, right=150, top=62)
    c.scales(1, 30, 0, 0.5)
    c.axes([1, 5, 11, 15, 20, 25, 30], [0, 0.1, 0.2, 0.3, 0.4, 0.5], "步数", "选中最优臂的比例", yfmt=pct)
    marks = {"UCB c=2": "circle", "optimistic Q0=5": "square", "eps=0.1": "triangle"}
    ends = []
    for name in names:
        ys = e["curves"][name]["optimal_frac"][:30]
        c.series(list(zip(range(1, 31), ys)), COLOR[name], mark=marks[name])
        ends.append((c.Y(ys[-1]), name, ys[-1]))
    for y, (_, name, v) in zip(spread([x[0] for x in ends]), ends):
        c.text(c.x1 + 10, y, f"{LABEL[name]} {v:.0%}", 12, INK, "start")
    x11 = c.X(11)
    c.add(f'<line x1="{x11:.1f}" y1="{c.y1}" x2="{x11:.1f}" y2="{c.y0}" stroke="{MUTED}" stroke-dasharray="4 3"/>')
    c.text(x11 + 8, c.y1 + 10, "第 11 步: 10 个臂刚好各试过一次", 11, INK2, "start")
    c.save(os.path.join(HERE, "fig_spike.svg"), "前 30 步放大: 乐观初值和 UCB 在第 11 步都有一个尖峰",
           "不做平滑, 每个点是 2000 局在这一步的平均 (数据: results/explore.json)")


def fig_regret():
    e = load("explore")
    names = ["greedy", "eps=0.01", "realistic eps=0.1", "optimistic Q0=5", "eps=0.1", "UCB c=2"]
    c = Chart(820, 400, right=200, top=62)
    c.scales(0, 1000, 0, 550)
    c.axes(range(0, 1001, 200), range(0, 551, 100), "步数", "累计遗憾")
    lines(c, e["curves"], names, "cum_regret", range(1, 1001), fmt=lambda v: f"{v:.0f}")
    c.save(os.path.join(HERE, "fig_regret.svg"), "累计遗憾: 越平越好, 贪心是一条不弯的直线",
           "每步遗憾 = 最优臂真实均值 − 被拉臂真实均值, 2000 局平均后累加 (数据: results/explore.json)")


def fig_long():
    e = load("explore")["long"]
    every = e["every"]
    names = ["UCB c=2", "eps=0.01", "eps=0.1"]
    n = len(e["curves"][names[0]]["avg_reward"])
    xs = [every * (i + 1) for i in range(n)]
    c = Chart(820, 360, right=150, top=62)
    c.scales(0, 10000, 1.0, 1.6)
    c.axes(range(0, 10001, 2000), [1.0, 1.2, 1.4, 1.6], "步数", "平均收益", yfmt=lambda v: f"{v:.1f}")
    lines(c, e["curves"], names, "avg_reward", xs, w=21, fmt=lambda v: f"{v:.2f}")
    c.save(os.path.join(HERE, "fig_long.svg"), "拉到 10000 步: 探索少的 ε=0.01 后来居上",
           "每 10 步取一个点, 21 点滑动平均 (数据: results/explore.json 的 long)")


def fig_sweep():
    s = load("sweep")
    fam = {"eps-greedy": ("ε-greedy (参数 ε)", SERIES[0], "circle"),
           "UCB": ("UCB (参数 c)", SERIES[2], "square"),
           "optimistic": ("乐观初值 (参数 Q0)", SERIES[1], "triangle")}
    c = Chart(820, 380, right=170, top=62)
    c.scales(1 / 128, 4, 1.0, 1.5, xlog=True)
    ticks = [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]
    names = {1 / 128: "1/128", 1 / 64: "1/64", 1 / 32: "1/32", 1 / 16: "1/16", 1 / 8: "1/8", 1 / 4: "1/4",
             1 / 2: "1/2", 1: "1", 2: "2", 4: "4"}
    c.axes(ticks, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "参数取值 (对数刻度)", "前 1000 步平均收益",
           xfmt=lambda v: names[v], yfmt=lambda v: f"{v:.1f}")
    for key, (label, color, mark) in fam.items():
        pts = s[key]
        c.series(pts, color, mark=mark)
        # 三条线的终点 x 不同, 标签紧跟各自的最后一个点, 而不是统一放在右边
        c.text(c.X(pts[-1][0]) + 12, c.Y(pts[-1][1]), label, 12, INK, "start")
    c.save(os.path.join(HERE, "fig_sweep.svg"), "每种方法都有一个最佳参数区间, 两头都会变差",
           "2000 局, 前 1000 步的平均收益 (数据: results/sweep.json)")


# ── 第 8 节 ────────────────────────────────────────────────────────

def fig_nonstationary():
    d = load("nonstationary")
    every = d["every"]
    names = ["alpha=0.1", "UCB c=2", "sample-average"]
    n = len(d["curves"][names[0]]["optimal_frac"])
    xs = [every * (i + 1) for i in range(n)]
    c = Chart(820, 360, right=170, top=62)
    c.scales(0, 10000, 0, 1)
    c.axes(range(0, 10001, 2000), [0, 0.2, 0.4, 0.6, 0.8, 1], "步数", "选中(当前)最优臂的比例", yfmt=pct)
    lines(c, d["curves"], names, "optimal_frac", xs, w=21)
    c.save(os.path.join(HERE, "fig_nonstationary.svg"), "收益在悄悄变: 样本平均越往后越跟不上",
           "每步所有臂的真实均值做 N(0, 0.01²) 随机游走; ε=0.1; 21 点滑动平均 (数据: results/nonstationary.json)")


if __name__ == "__main__":
    for fig in (fig_greedy, fig_rank, fig_explore, fig_spike, fig_regret, fig_long, fig_sweep, fig_nonstationary):
        fig()
