"""
make_charts.py — 解码讲义的数据图 (数字全部来自 experiments/results/ 和小车 v0.4 的 results/)

  chart_tradeoff.svg      多样性 vs 像不像, 每种解码设置一个点                         (第 4 节, toy_experiments.json)
  chart_entropy_temp.svg  温度越高, 下一个字符的分布越平 (平均熵)                       (第 4 节, toy_more.json)
  chart_truncation.svg    分布越确定 (熵越低), top-k / top-p / min-p 各留下几个候选      (第 5 节, toy_experiments.json)
  chart_shift.svg         整体平移所有分数: 乘法的重复惩罚变, 减法的出现惩罚不变         (第 6.2、7 节, toy_more.json)
  chart_penalties.svg     五种惩罚写法 vs 贪心转圈                                      (第 7 节, toy_more.json)
  chart_beam.svg          束宽越大总分越高, 写出来的照样是 the the                      (第 9 节, toy_more.json)
  chart_orders.svg        三家框架的顺序: 留下几个候选、抽签分布差多少                  (专题 A, toy_more.json)
  chart_real.svg          真实小模型: 重复惩罚改了几条回复、采样下回复不变的概率        (专题 B, real_model_*.json)
  chart_car.svg           小车 v0.4: 关掉重复惩罚, 成功从 14 升到 18                    (实践, 小车 decoding_compare.json)

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
with open(os.path.join(UNIT, "experiments", "results", "toy_more.json"), encoding="utf-8") as f:
    M = json.load(f)

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
    c.text(c.X(0.6), c.Y(43), "它们和第 5 名并列, 而并列的一律保留 (讲义第 5.2 节)", 11, INK2, "start")
    c.save(out("chart_truncation.svg"), "截断留下几个候选: top-p 和 min-p 随把握大小伸缩, top-k 是死数",
           "在语料里随机取 2000 个位置, 看 ToyLM 对下一个字符的分布 (experiments/toy_experiments.py)")




def chart_entropy_temp():
    rows = M["temperature_entropy"]
    c = Chart(820, 400, left=80, right=170, top=66, bottom=64)
    c.scales(0, 3.2, 0, 6.5)
    c.axes([0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0], [0, 1, 2, 3, 4, 5, 6], "温度 T", "下一个字符分布的平均熵 (比特)",
           xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    u = M["uniform_entropy_bits"]
    c.add(f'<line x1="{c.x0}" y1="{c.Y(u):.1f}" x2="{c.x1}" y2="{c.Y(u):.1f}" stroke="{MUTED}" stroke-dasharray="4 3"/>')
    c.text(c.x1 + 8, c.Y(u), f"65 个字符完全均匀: {u:.2f}", 11, MUTED, "start")
    pts = [(r["temp"], r["mean_entropy_bits"]) for r in rows]
    c.series(pts, SERIES[0], "circle")
    for x, y in pts:
        c.text(c.X(x), c.Y(y) - 14, f"{y:.2f}", 10, INK2)
    c.add(f'<line x1="{c.X(1.0):.1f}" y1="{c.y0}" x2="{c.X(1.0):.1f}" y2="{c.y1}" stroke="{MUTED}" stroke-dasharray="2 3"/>')
    c.text(c.X(1.0) + 6, c.y1 + 10, "T = 1: 模型自己学到的分布", 11, INK2, "start")
    c.save(out("chart_entropy_temp.svg"), "温度是一个旋钮: 往左拧接近贪心 (熵→0), 往右拧接近乱抽 (熵→均匀)",
           "ToyLM, 语料里随机 500 个位置的平均熵; 1 比特约等于 \"在两个候选之间犹豫\" (experiments/toy_more.py H)")


def chart_shift():
    rows = M["shift"]["rows"]
    c = Chart(860, 400, left=80, right=210, top=66, bottom=64)
    c.scales(-20, 20, 0, 0.5)
    c.axes([-20, -10, 0, 10, 20], [0, 0.1, 0.2, 0.3, 0.4, 0.5], "所有分数同时加上的常数 (softmax 之后概率完全不变)",
           "惩罚之后 \"e\" 的概率", xfmt=lambda v: f"{v:+g}" if v else "0", yfmt=lambda v: f"{v:g}")
    p0 = M["shift"]["p_before"]
    c.add(f'<line x1="{c.x0}" y1="{c.Y(p0):.1f}" x2="{c.x1}" y2="{c.Y(p0):.1f}" stroke="{MUTED}" stroke-dasharray="4 3"/>')
    c.end_label(20, p0, f"惩罚之前 {p0:.2f}", dy=-8)
    a = [(r["shift"], r["repetition_1.3"]) for r in rows]
    b = [(r["shift"], r["presence_0.5"]) for r in rows]
    c.series(b, SERIES[2], "square", every=2)
    c.series(a, SERIES[1], "circle", every=2)
    c.end_label(20, b[-1][1], "出现惩罚 0.5 (减法): 一条平线", dy=6)
    c.end_label(20, a[-1][1], "重复惩罚 1.3 (乘除法)", dy=0)
    c.save(out("chart_shift.svg"), "同一个分布, 同样的惩罚: 乘除法的效果随分数整体平移大起大落, 减法纹丝不动",
           "ToyLM, \"th\" 之后的 \"e\" 被当作说过 (experiments/toy_more.py F)")


SHORT_E = {"不加惩罚": "不加惩罚", "重复惩罚 1.3 (看全部历史, transformers)": "重复 1.3 全部历史",
           "重复惩罚 1.3 (只看最近 64 个, llama.cpp)": "重复 1.3 最近 64",
           "重复惩罚 1.3 (只看最近 8 个)": "重复 1.3 最近 8", "出现惩罚 0.5 (加法)": "出现 0.5",
           "频率惩罚 0.3 (加法, 按次数)": "频率 0.3", "禁止重复 4 元组": "禁 4 元组", "禁止重复 6 元组": "禁 6 元组"}


def chart_penalties():
    rows = M["penalties"]
    c = Chart(900, 430, left=80, right=40, top=80, bottom=80)
    c.scales(-0.5, len(rows) - 0.5, 0, 1.1)
    c.axes([], [0, 0.2, 0.4, 0.6, 0.8, 1.0], "", "比例", yfmt=lambda v: f"{v:g}")
    c.bars([SHORT_E[r["setting"]] for r in rows], ["转圈", "多样性"],
           [[r["loop_rate"], r["distinct4"]] for r in rows], [SERIES[1], SERIES[0]], fmt=lambda v: f"{v:.2f}")
    for i, r in enumerate(rows):
        c.text(c.X(i), c.y0 + 16, SHORT_E[r["setting"]], 11, INK)
        c.text(c.X(i), c.y0 + 34, f"像不像 {r['avg_logprob']:.2f}", 10, MUTED)
    lx = c.x0
    c.add(f'<rect x="{lx}" y="56" width="12" height="12" fill="{SERIES[1]}"/>')
    c.text(lx + 18, 62, "300 字里转圈的比例", 11, INK2, "start")
    c.add(f'<rect x="{lx + 160}" y="56" width="12" height="12" fill="{SERIES[0]}"/>')
    c.text(lx + 178, 62, "多样性: 不重复的 4 字片段占比", 11, INK2, "start")
    c.save(out("chart_penalties.svg"), "哪种惩罚能打破贪心的转圈: 只有 \"按次数累加\" 和 \"禁止重复 n 元组\"",
           "ToyLM 贪心, 5 个开头 × 300 字; \"像不像\" = 平均对数概率, 越接近 0 越像 (experiments/toy_more.py E)")


def chart_beam():
    rows = M["beam"]
    c = Chart(900, 380, left=80, right=300, top=66, bottom=64)
    c.scales(1, 32, -42, -39.5, xlog=True)
    c.axes([1, 2, 4, 8, 16, 32], [-42, -41, -40], "束宽 B (对数刻度)", "40 个字符的总分 (log 概率之和)",
           xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    pts = [(r["width"], r["score"]) for r in rows]
    c.series(pts, SERIES[0], "circle")
    for r in rows:
        c.text(c.X(r["width"]), c.Y(r["score"]) + 16, f"{r['score']:.2f}", 10, INK2)
    seen = []
    for r in rows:
        head = r["text"][:22].replace("\n", "↵")
        if head not in seen:
            seen.append(head)
    for k, h in enumerate(seen):
        c.text(c.x1 + 40, c.y1 + 20 + 22 * k, f"\"{h}…\"", 11, INK2, "start")
    c.text(c.x1 + 40, c.y1 + 20 + 22 * len(seen), "全部以 the the the 结尾", 11, SERIES[1], "start", "bold")
    c.save(out("chart_beam.svg"), "束搜索: 束越宽总分越高, 到 4 以后就到顶了; 写出来的照样在转圈",
           "ToyLM, 开头 \"ROMEO:↵\", 写 40 个字符; 右边是不同束宽写出的开头 (experiments/toy_more.py I)")


def chart_orders():
    rows = M["orders"]
    names = ["transformers", "vllm", "llama.cpp"]
    labels = ["T0.7 k20 p0.8", "T0.7 p0.9 min-p0.05", "T1.5 p0.9 min-p0.1", "T1.0 k20 p0.8 min-p0.05"]
    c = Chart(900, 430, left=80, right=40, top=80, bottom=86)
    c.scales(-0.5, len(rows) - 0.5, 0, 12.5)
    c.axes([], [0, 2, 4, 6, 8, 10, 12], "", "平均留下几个候选", yfmt=lambda v: f"{v:g}")
    c.bars(labels, names, [[r["kept_mean"][n] for n in names] for r in rows], [SERIES[0], SERIES[2], SERIES[1]],
           fmt=lambda v: f"{v:.1f}")
    for i, r in enumerate(rows):
        c.text(c.X(i), c.y0 + 16, labels[i], 11, INK)
        d = r["kept_set_differs_from_transformers"]
        c.text(c.X(i), c.y0 + 34, f"集合不同: {d['vllm']} / {d['llama.cpp']} 处", 10, MUTED)
        t = r["mean_tvd_vs_transformers"]
        c.text(c.X(i), c.y0 + 50, f"总变差: {t['vllm']:.3f} / {t['llama.cpp']:.3f}", 10, MUTED)
    for k, (n, col) in enumerate(zip(["transformers", "vLLM", "llama.cpp"], [SERIES[0], SERIES[2], SERIES[1]])):
        c.add(f'<rect x="{c.x0 + k * 130}" y="56" width="12" height="12" fill="{col}"/>')
        c.text(c.x0 + k * 130 + 18, 62, n, 11, INK2, "start")
    c.save(out("chart_orders.svg"), "同样的参数, 三家的顺序不同, 留下的候选和抽签的分布也不同",
           "ToyLM, 随机 2000 个上下文; 每组下面是 vLLM / llama.cpp 和 transformers 比: 留下的候选集合不同的上下文数, 平均总变差 (0 一样, 1 完全不重叠)")


def chart_real():
    files = [("Qwen2.5-0.5B-Instruct", "0.5B"), ("Qwen2.5-1.5B-Instruct", "1.5B")]
    data = []
    for f, short in files:
        with open(os.path.join(UNIT, "experiments", "results", f"real_model_{f}.json"), encoding="utf-8") as fh:
            data.append((short, json.load(fh)))
    qs = [r["question"] for r in data[0][1]["rows"]]
    c = Chart(920, 470, left=230, right=40, top=96, bottom=40)
    c.scales(0, 1, 0, 1)
    cell_w, cell_h = 150, 26
    cols = [("调没调工具", lambda r: ("调" if r["calls"] else "不调") + (" ✓" if r["calls"] == r["should_call"] else " ✗")),
            ("惩罚 1.1 改了回复", lambda r: "改了" if not r["same_reply_with_penalty"] else "—"),
            ("采样下回复不变", lambda r: f"{r['p_whole_reply_same_under_sampling']:.2f}")]
    x0, y0 = c.x0, c.y1
    for m, (short, d) in enumerate(data):
        for k, (name, f) in enumerate(cols):
            x = x0 + (m * len(cols) + k) * cell_w * 0.62
            if k == 0:
                c.text(x + cell_w * 0.62 * 1.5, y0 - 34, short, 13, INK, "middle", "bold")
            c.text(x + cell_w * 0.31, y0 - 12, name, 10, INK2)
            for i, r in enumerate(d["rows"]):
                v = f(r)
                y = y0 + i * cell_h
                bad = v.endswith("✗") or v == "改了"
                fill = "#fde3d7" if bad else ("#dbe9f9" if k == 2 and r["p_whole_reply_same_under_sampling"] >= 0.9 else "none")
                c.add(f'<rect x="{x:.1f}" y="{y}" width="{cell_w * 0.62 - 3:.1f}" height="{cell_h - 3}" rx="3" fill="{fill}" '
                      f'stroke="#e9e8e3"/>')
                c.text(x + cell_w * 0.31, y + cell_h / 2 - 1, v, 11, INK)
    for i, q in enumerate(qs):
        c.text(x0 - 10, y0 + i * cell_h + cell_h / 2 - 1, q + ("  [该调]" if data[0][1]["rows"][i]["should_call"] else ""),
               11, INK, "end")
    c.save(out("chart_real.svg"), "真实小模型上: 自带的重复惩罚改了哪些回复, 采样会不会把回复抽歪",
           "Qwen2.5 Instruct, CPU 贪心; 橙色 = 调错工具或被惩罚改了回复; 蓝色 = 采样下 ≥ 0.9 的概率和贪心一字不差 (experiments/real_model.py)")


def chart_car():
    path = os.path.join(ROOT, "projects", "car", "v0.4-text-commands", "results", "decoding_compare.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)["settings"]
    vals = list(d.values())
    labels = ["贪心+惩罚1.1", "同设置重跑", "贪心, 关掉惩罚", "采样 种子0", "采样 种子1", "采样 种子2"]
    c = Chart(860, 400, left=80, right=40, top=80, bottom=60)
    c.scales(-0.5, len(vals) - 0.5, 0, 30)
    c.axes([], [0, 5, 10, 15, 20, 25, 30], "", "30 个任务里的局数", yfmt=lambda v: f"{v:g}")
    c.bars(labels, ["成功", "没动"], [[v["success"], v["outcomes"].get("no_action", 0)] for v in vals],
           [SERIES[2], SERIES[1]])
    for i, lab in enumerate(labels):
        c.text(c.X(i), c.y0 + 16, lab, 11, INK, "middle", "bold" if i == 2 else "normal")
    for k, (n, col) in enumerate([("成功", SERIES[2]), ("没动 (第一轮没写成工具调用)", SERIES[1])]):
        c.add(f'<rect x="{c.x0 + k * 90}" y="56" width="12" height="12" fill="{col}"/>')
        c.text(c.x0 + k * 90 + 18, 62, n, 11, INK2, "start")
    c.save(out("chart_car.svg"), "小车 v0.4 (Qwen2.5-1.5B, go_to): 关掉自带的重复惩罚, \"没动\" 从 12 局降到 4 局",
           "每组 30 个任务, GPU; 数字来自 projects/car/v0.4-text-commands/results/decoding_compare.json")


if __name__ == "__main__":
    chart_tradeoff()
    chart_entropy_temp()
    chart_truncation()
    chart_shift()
    chart_penalties()
    chart_beam()
    chart_orders()
    chart_car()
    if os.path.exists(os.path.join(UNIT, "experiments", "results", "real_model_Qwen2.5-1.5B-Instruct.json")):
        chart_real()
