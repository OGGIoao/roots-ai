"""
make_diagrams.py — 解码讲义的结构图 (SVG, 只用标准库; 数字都由本单元的代码现场算出)

  fig_pipeline.svg    一步解码的全流程: 分数 → 惩罚 → 温度 → 截断 → 挑一个 → 接到后面 → 再问下一步      (第 1、8 节)
  fig_handcalc.svg    "今天天气很" 后面的 6 个候选: 每一步把概率改成什么样                                (第 2 节)
  fig_cycle.svg       ToyLM 的贪心为什么出不来: 四个状态首尾相接                                         (第 3 节)
  fig_beam.svg        束宽 2 的束搜索前三步: 每步展开、排序、只留两条                                     (第 9 节)
  fig_orders.svg      transformers / vLLM / llama.cpp 三家的执行顺序                                    (专题 A)
  fig_stop.svg        三种停止条件                                                                       (第 8.4 节)

运行: python figures/make_diagrams.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(UNIT)))
sys.path.insert(0, ROOT)
sys.path.insert(0, UNIT)

import sampling as S  # noqa: E402
from beam import log_softmax  # noqa: E402
from common.svgfig import (INK, INK2, KV_FILL, KV_LINE, MUTED, OP_FILL, OP_LINE, Q_FILL, Q_LINE, SERIES)  # noqa: E402
from common.svgfig import Svg as _Svg  # noqa: E402
from toy_lm import ToyLM  # noqa: E402

GREEN_FILL, GREEN_LINE = "#dcf3e9", "#1baf7a"
MONO = "Consolas,'Courier New',monospace"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg(_Svg):
    def text(self, x, y, s, *a, **k):
        super().text(x, y, esc(s), *a, **k)

    def title(self, t, sub=None):
        self.text(20, 22, t, 15, INK, "start", "bold")
        if sub:
            self.text(20, 42, sub, 12, INK2, "start")


def out(name):
    return os.path.join(HERE, name)


def show(c):
    return {" ": "␣", "\n": "↵"}.get(c, c)


def fig_pipeline():
    s = Svg(980, 330)
    s.title("一步解码: 分数进来, 一个 token 出去; 接到后面, 再来一步",
            "每一步都只是 \"改分数\": 不想要的设成 -∞; 最后一步才真正挑。贪心跳过 ② ③, 但 ① 惩罚照做 (第 8.3 节)")
    y, h = 120, 54
    boxes = [("模型", "打分 (logits)", OP_FILL, OP_LINE, 90),
             ("① 惩罚", "说过的压低", KV_FILL, KV_LINE, 110),
             ("② 温度", "除以 T", Q_FILL, Q_LINE, 100),
             ("③ 截断", "top-k / top-p / min-p", Q_FILL, Q_LINE, 150),
             ("④ 挑一个", "贪心取最大 / 按概率抽", GREEN_FILL, GREEN_LINE, 150),
             ("停不停?", "停止 token / 字符串 / 长度", OP_FILL, OP_LINE, 160)]
    x = 30
    xs = []
    for label, sub, fill, line, w in boxes:
        s.rect(x, y, w, h, fill, line)
        s.text(x + w / 2, y + 20, label, 14, INK, "middle", "bold")
        s.text(x + w / 2, y + 39, sub, 11, INK2)
        xs.append((x, w))
        x += w + 34
    for (x0, w0), (x1, _) in zip(xs, xs[1:]):
        s.line(x0 + w0 + 2, y + h / 2, x1 - 3, y + h / 2)
    # 回到模型
    lx, lw = xs[-1]
    s.path(f"M{lx + lw / 2},{y + h} L{lx + lw / 2},{y + h + 60} L{xs[0][0] + xs[0][1] / 2},{y + h + 60} "
           f"L{xs[0][0] + xs[0][1] / 2},{y + h + 4}")
    s.text((xs[0][0] + lx + lw) / 2, y + h + 74, "不停: 把挑中的 token 接到输入后面, 再问模型下一步 (这就是 generate.py 的循环)", 12, INK2)
    s.text(lx + lw / 2, y - 14, "停 → 输出", 12, INK, "middle", "bold")
    # 贪心跳过的范围
    a, b = xs[2][0], xs[3][0] + xs[3][1]
    s.rect(a - 8, y - 34, b - a + 16, h + 46, "none", MUTED, dash=True, rx=8, sw=1.2)
    s.text((a + b) / 2, y - 22, "贪心 (do_sample=False) 跳过 ② ③", 12, MUTED)
    s.text(xs[1][0] + xs[1][1] / 2, y - 22, "贪心也照做", 12, KV_LINE, "middle", "bold")
    s.save(out("fig_pipeline.svg"))


def fig_handcalc():
    vocab = ["好", "热", "冷", "晴", "差", "怪"]
    logits = [2.0, 1.5, 1.0, 0.5, -1.0, -2.0]
    panels = [("原始分数 → softmax", logits),
              ("重复惩罚 1.5 (说过 \"好\")", S.repetition_penalty(logits, [0], 1.5)),
              ("温度 0.5", S.temperature(logits, 0.5)),
              ("温度 2.0", S.temperature(logits, 2.0)),
              ("top-k 3", S.top_k(logits, 3)),
              ("top-p 0.8", S.top_p(logits, 0.8)),
              ("min-p 0.3", S.min_p(logits, 0.3)),
              ("温度 0.5 → top-p 0.8", S.top_p(S.temperature(logits, 0.5), 0.8))]
    cols, pw, ph = 4, 225, 170
    s = Svg(20 + cols * pw + 10, 70 + 2 * ph + 10)
    s.title("手算一遍: \"今天天气很___\" 的 6 个候选, 每一步把概率改成什么样",
            "原始分数 2.0, 1.5, 1.0, 0.5, -1.0, -2.0; 灰色虚框是原始概率, 彩色是处理后; 划掉的写 0 (python sampling.py 打印同样的数)")
    base = S.softmax(logits)
    for k, (name, x) in enumerate(panels):
        px, py = 20 + (k % cols) * pw, 66 + (k // cols) * ph
        s.text(px + 4, py + 10, name, 12, INK, "start", "bold")
        probs = S.softmax(x)
        bw, top, hmax = 30, py + 30, 100
        for i, (v, p0, p) in enumerate(zip(vocab, base, probs)):
            bx = px + 8 + i * (bw + 4)
            hh0 = p0 * hmax / 0.8
            s.rect(bx, top + hmax - hh0, bw, hh0, "none", MUTED, dash=True, rx=2, sw=1)
            hh = p * hmax / 0.8
            color = SERIES[0] if p > 0 else MUTED
            if p > 0:
                s.rect(bx, top + hmax - hh, bw, hh, color, color, rx=2, sw=0)
            s.text(bx + bw / 2, top + hmax - max(hh, hh0) - 8, f"{p:.2f}".lstrip("0") if p > 0 else "0", 10, INK2)
            s.text(bx + bw / 2, top + hmax + 12, v, 12, INK)
    s.save(out("fig_handcalc.svg"))


def fig_cycle():
    lm = ToyLM.from_corpus()
    s = Svg(900, 400)
    s.title("贪心为什么出不来: ToyLM 只看最后两个字符, 四个状态首尾相接",
            "每个方框是 \"最后两个字符\", 箭头是贪心选的下一个字符和它的概率; 走到 \"th\" 就再也离不开 (toy_lm.py 现场算)")
    states = ["th", "he", "e ", " t"]
    pos = [(380, 90), (640, 190), (380, 290), (120, 190)]
    w, h = 130, 56
    for st, (x, y) in zip(states, pos):
        s.rect(x, y, w, h, Q_FILL, Q_LINE, rx=10)
        s.text(x + w / 2, y + 20, f'"{st.replace(" ", "␣")}"', 16, INK, "middle", "bold")
        probs = S.softmax(lm.logits(st))
        top = sorted(range(lm.V), key=lambda i: -probs[i])[:3]
        s.text(x + w / 2, y + 41, "  ".join(f"{show(lm.vocab[i])} {probs[i]:.2f}" for i in top), 11, INK2)
    for k, st in enumerate(states):
        nxt = states[(k + 1) % 4]
        probs = S.softmax(lm.logits(st))
        g = S.greedy(lm.logits(st))
        (x0, y0), (x1, y1) = pos[k], pos[(k + 1) % 4]
        cx0, cy0, cx1, cy1 = x0 + w / 2, y0 + h / 2, x1 + w / 2, y1 + h / 2
        # 从框边出发
        dx, dy = cx1 - cx0, cy1 - cy0
        f0 = min(abs((w / 2 + 4) / dx) if dx else 9, abs((h / 2 + 4) / dy) if dy else 9)
        f1 = min(abs((w / 2 + 8) / dx) if dx else 9, abs((h / 2 + 8) / dy) if dy else 9)
        s.line(cx0 + dx * f0, cy0 + dy * f0, cx1 - dx * f1, cy1 - dy * f1, KV_LINE, width=2)
        mx, my = (cx0 + cx1) / 2, (cy0 + cy1) / 2
        ox, oy = mx - 445, my - 218                       # 从图中心往外推, 标签放在箭头外侧
        n = (ox * ox + oy * oy) ** 0.5
        s.text(mx + ox / n * 70, my + oy / n * 34, f'贪心选 "{show(lm.vocab[g])}" ({probs[g]:.2f})', 12, KV_LINE, "middle", "bold")
        assert lm.vocab[g] == nxt[-1], (st, lm.vocab[g])
    s.text(450, 375, "写出来的就是 the the the …; 每一步都选了最可能的字符, 合起来却是最没用的文字", 12, INK2)
    s.save(out("fig_cycle.svg"))


def fig_beam():
    lm = ToyLM.from_corpus()
    prompt = "ROMEO:\n"
    B, steps, show_n = 2, 3, 4
    beams = [("", 0.0)]
    columns = []
    for _ in range(steps):
        cands = []
        for seq, sc in beams:
            lp = log_softmax(lm.logits(prompt + seq))
            for t, l in enumerate(lp):
                cands.append((seq + lm.vocab[t], sc + l, seq))
        cands.sort(key=lambda c: (-c[1], c[0]))
        columns.append(cands[:show_n])
        beams = [(c[0], c[1]) for c in cands[:B]]
    s = Svg(960, 360)
    s.title("束搜索 (束宽 2): 每一步把留下的路各展开一格, 按总分排序, 只留前两条",
            f"开头 \"ROMEO:↵\"; 总分 = 每一步 log 概率之和; 绿色是留下的, 灰色是被挤掉的 (每步只画前 {show_n} 名, beam.py 的规则)")
    cw, bh, gap = 220, 40, 12
    loc = {"": (140, 150 + bh / 2)}
    s.rect(30, 150, 110, bh, OP_FILL, OP_LINE)
    s.text(85, 150 + bh / 2, "ROMEO:↵", 13, INK)
    for j, col in enumerate(columns):
        x = 200 + j * (cw + 30)
        for r, (seq, sc, parent) in enumerate(col):
            y = 70 + r * (bh + gap + 10)
            kept = r < B
            s.rect(x, y, cw - 40, bh, GREEN_FILL if kept else OP_FILL, GREEN_LINE if kept else MUTED, dash=not kept)
            s.text(x + 10, y + bh / 2, "".join(show(c) for c in seq), 14, INK if kept else MUTED, "start", "bold")
            s.text(x + cw - 50, y + bh / 2, f"{sc:.2f}", 12, INK2 if kept else MUTED, "end")
            px, py = loc.get(parent, (None, None))
            if px is not None:
                s.line(px, py, x - 3, y + bh / 2, GREEN_LINE if kept else MUTED, width=1.4 if kept else 1)
            if kept:
                loc[seq] = (x + cw - 40, y + bh / 2)
    s.text(480, 340, "贪心 (束宽 1) 只看眼前一步; 束宽 2 会保留一条 \"现在稍差、后面可能更好\" 的路", 12, INK2)
    s.save(out("fig_beam.svg"))


def fig_orders():
    s = Svg(940, 300)
    s.title("同样几个参数, 三家框架的执行顺序不一样 (读源码, 2026-10-29)",
            "橙色是惩罚, 蓝色是温度, 绿色是截断; llama.cpp 默认把温度放在最后, 所以截断看到的是 \"没被温度改过\" 的概率")
    rows = [("transformers", ["惩罚", "温度", "top-k", "top-p", "min-p"]),
            ("vLLM", ["惩罚", "温度", "min-p", "top-k", "top-p"]),
            ("llama.cpp", ["惩罚", "top-k", "top-p", "min-p", "温度"])]
    color = {"惩罚": (KV_FILL, KV_LINE), "温度": (Q_FILL, Q_LINE)}
    for r, (name, steps) in enumerate(rows):
        y = 80 + r * 66
        s.text(150, y + 20, name, 14, INK, "end", "bold")
        for k, st in enumerate(steps):
            x = 170 + k * 140
            fill, line = color.get(st, (GREEN_FILL, GREEN_LINE))
            s.rect(x, y, 104, 40, fill, line)
            s.text(x + 52, y + 20, st, 13, INK)
            if k:
                s.line(x - 34, y + 20, x - 3, y + 20)
        s.line(170 + 4 * 140 + 106, y + 20, 170 + 5 * 140 - 10, y + 20)
        s.text(170 + 5 * 140, y + 20, "抽签", 13, INK, "start")
    s.text(470, 285, "llama.cpp 默认链里还有 DRY、top-n-σ、typical、XTC 等步骤, 这里只画三家共有的五步", 11, MUTED)
    s.save(out("fig_orders.svg"))


def fig_stop():
    s = Svg(940, 270)
    s.title("什么时候停: 三种停止条件 (generate.py)",
            "停止原因要记下来: 以 length 结束的工具调用, 往往是 JSON 写到一半被截断")
    rows = [("stop_token", ["台", "北", "26", "度", "<|im_end|>"], 4, "生成了停止 token; 输出里包含它"),
            ("stop_string", ["第", "一", "段", "\\n", "\\n"], 3, "新文字里出现了 \"\\n\\n\"; transformers 保留, vLLM 默认去掉"),
            ("length", ['{"', "city", '":', ' "', "台"], None, "写满 max_new_tokens=5 还没停: 半截 JSON")]
    for r, (name, toks, hit, note) in enumerate(rows):
        y = 70 + r * 64
        s.text(140, y + 17, name, 13, INK, "end", "bold")
        x = 160
        for k, t in enumerate(toks):
            w = 26 + 9 * len(t)
            on = hit is not None and k >= hit
            s.rect(x, y, w, 34, KV_FILL if on else Q_FILL, KV_LINE if on else Q_LINE)
            s.text(x + w / 2, y + 17, t, 12, INK)
            x += w + 6
        if hit is None:
            s.text(x + 4, y + 17, "… ✂", 14, KV_LINE, "start", "bold")
            x += 30
        s.text(x + 14, y + 17, note, 12, INK2, "start")
    s.save(out("fig_stop.svg"))


if __name__ == "__main__":
    fig_pipeline()
    fig_handcalc()
    fig_cycle()
    fig_beam()
    fig_orders()
    fig_stop()
