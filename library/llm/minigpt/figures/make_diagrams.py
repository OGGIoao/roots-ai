"""
make_diagrams.py — 生成 miniGPT 讲义里的结构图 (SVG)

结构图用代码生成: 坐标由循环算出, 对齐准确, 改起来也方便。输出与本文件同目录。

运行: python figures/make_diagrams.py   (用到 common/svgfig.py, 需先 pip install -e . --no-deps)

配色约定 (与数据图、注意力线讲义一致):
  蓝  #2a78d6  查询 / 输入数据 / 主要对象      橙  #eb6834  键值 / 第二类对象
  绿  #008300  缓存 / "好"的路径                灰  #52514e  一般运算
"""

import os

from common.svgfig import (  # noqa: F401
    BG, CACHE, INK, INK2, KV_FILL, KV_LINE, MUTED, OP_FILL, OP_LINE, Q_FILL, Q_LINE, Svg,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RED = "#e34948"
GOOD_FILL = "#e3f1e3"


def out(name):
    return os.path.join(HERE, name)


# ── 图: 一次训练的全流程 (讲义 1.4 节) ──────────────────────────────

def fig_pipeline():
    s = Svg(1000, 470)
    s.text(20, 24, "数据", 15, anchor="start", weight="bold")
    s.box(20, 42, 150, 46, "tinyshakespeare.txt", Q_FILL, Q_LINE, 13, sub="约 111 万字符")
    s.line(170, 65, 208, 65)
    s.box(210, 42, 150, 46, "CharTokenizer", OP_FILL, OP_LINE, 13, sub="每个字符 → 一个编号 (65 种)")
    s.line(360, 65, 398, 65)
    s.box(400, 42, 170, 46, "[18, 47, 56, 57, ...]", Q_FILL, Q_LINE, 13, sub="整篇语料变成一串整数")
    s.line(570, 65, 608, 65)
    s.box(610, 42, 170, 46, "切块: x 与 y", Q_FILL, Q_LINE, 13, sub="y 是 x 右移一格, 都是 (B, S)")
    s.text(790, 65, "← 第 2 节", 11, MUTED, anchor="start")

    s.path("M695,106 L695,124 L95,124 L95,152", arrow=True)
    s.text(20, 140, "模型 (GPT.forward)", 15, anchor="start", weight="bold")
    s.box(20, 155, 150, 52, "wte(x) + wpe(位置)", OP_FILL, OP_LINE, 13, sub="(B, S, C)   第 4 节")
    s.line(170, 181, 208, 181)
    s.rect(210, 150, 300, 62, "#ffffff", OP_LINE)
    s.text(360, 166, "Block × n_layer", 13, weight="bold")
    s.text(360, 186, "x = x + Attn(LN(x))   ← 第 5、7 节", 12, INK2)
    s.text(360, 203, "x = x + FFN(LN(x))    ← 第 6、7 节", 12, INK2)
    s.line(510, 181, 548, 181)
    s.box(550, 155, 150, 52, "LN → lm_head", OP_FILL, OP_LINE, 13, sub="(B, S, V) 每个位置的打分")
    s.line(700, 181, 738, 181)
    s.box(740, 155, 170, 52, "cross_entropy(logits, y)", KV_FILL, KV_LINE, 12, sub="一个标量 loss   第 8 节")

    s.path("M825,224 C825,255 600,250 600,272")
    s.text(20, 262, "更新", 15, anchor="start", weight="bold")
    s.box(430, 274, 170, 44, "loss.backward()", OP_FILL, OP_LINE, 13, sub="算出每个参数的梯度")
    s.line(430, 296, 372, 296)
    s.box(220, 274, 150, 44, "梯度裁剪", OP_FILL, OP_LINE, 13, sub="总长度 ≤ 1.0")
    s.line(220, 296, 172, 296)
    s.box(20, 274, 150, 44, "AdamW 更新参数", OP_FILL, OP_LINE, 13, sub="学习率按余弦调度")
    s.text(640, 296, "← 第 10 节", 11, MUTED, anchor="start")
    s.path("M20,296 C-5,296 -5,65 18,65", CACHE)
    s.text(26, 242, "↑ 重复 max_iters 次", 11, CACHE, anchor="start")

    s.line(20, 360, 980, 360, "#e3e2dd", arrow=False, width=1)
    s.text(20, 388, "训练完之后: 生成 (第 11 节)", 15, anchor="start", weight="bold")
    s.box(20, 404, 130, 40, "开头的字符", Q_FILL, Q_LINE, 13)
    s.line(150, 424, 188, 424)
    s.box(190, 404, 130, 40, "整段过一遍模型", OP_FILL, OP_LINE, 13)
    s.line(320, 424, 358, 424)
    s.box(360, 404, 200, 40, "只看最后一个位置的打分", OP_FILL, OP_LINE, 13)
    s.line(560, 424, 598, 424)
    s.box(600, 404, 170, 40, "按温度 / top-k 抽一个", KV_FILL, KV_LINE, 13)
    s.path("M685,444 C685,462 85,462 85,446", CACHE)
    s.text(800, 424, "接到末尾, 再来一轮", 12, CACHE, anchor="start")
    s.save(out("fig_pipeline.svg"))


# ── 图: 一段文本怎么变成 S 道题 (讲义 2.3 节) ───────────────────────

def fig_shift():
    text = "First Cit"
    S = len(text) - 1                                   # 8
    s = Svg(860, 330)
    w, x0 = 52, 175

    def row(y, chars, label, fill, line, start=0):
        s.text(20, y + 18, label, 13, anchor="start", weight="bold")
        for i, ch in enumerate(chars):
            shown = "␣" if ch == " " else ch
            s.box(x0 + (i + start) * w + 2, y, w - 6, 36, shown, fill, line, 15)

    row(24, text, "原文 (S+1 = 9 个字符)", OP_FILL, OP_LINE)
    row(110, text[:-1], "x (输入, 前 8 个)", Q_FILL, Q_LINE)
    row(220, text[1:], "y (答案, 后 8 个)", KV_FILL, KV_LINE)
    for i in range(S):                                  # 每个位置: x_i 猜 y_i
        cx = x0 + i * w + w / 2 - 1
        s.line(cx, 148, cx, 216, Q_LINE, width=1.2)
    s.text(x0 + S * w + 10, 183, "每个位置都在猜", 12, INK2, anchor="start")
    s.text(x0 + S * w + 10, 201, "\"它后面那个字符\"", 12, INK2, anchor="start")
    # 位置 3 能看到的范围
    s.rect(x0 - 2, 104, 4 * w, 48, "none", CACHE, dash=True, rx=6)
    s.text(x0 + 2 * w, 92, "位置 3 只能看这 4 个字符 (因果掩码, 第 5.5 节)", 11, CACHE)
    s.text(410, 300, "一段长 S+1 的文本同时提供 S 道题: 位置 0 看 \"F\" 猜 \"i\", 位置 1 看 \"Fi\" 猜 \"r\", …, 位置 7 看 \"First Ci\" 猜 \"t\"",
           12, INK2)
    s.save(out("fig_shift.svg"))


# ── 图: 张量形状追踪 (讲义 3.2 节) ──────────────────────────────────

def fig_shapes():
    rows = [
        ("输入 idx", "(2, 8)", 8, "整数编号", OP_FILL, OP_LINE),
        ("wte (+ wpe)", "(2, 8, 64)", 64, "每个编号换成 64 维向量, 再加上位置向量", Q_FILL, Q_LINE),
        ("ln_1", "(2, 8, 64)", 64, "归一化不改形状", OP_FILL, OP_LINE),
        ("attn.c_attn", "(2, 8, 192)", 192, "Q、K、V 拼在一起 (64 × 3)", KV_FILL, KV_LINE),
        ("attn 输出", "(2, 8, 64)", 64, "注意力之后回到 64 维", Q_FILL, Q_LINE),
        ("mlp.c_fc", "(2, 8, 256)", 256, "FFN 升到 4 倍", KV_FILL, KV_LINE),
        ("mlp 输出", "(2, 8, 64)", 64, "再降回 64 维", Q_FILL, Q_LINE),
        ("× 第 2 层 Block", "(2, 8, 64)", 64, "每个 Block 进出形状相同, 所以能任意叠加", OP_FILL, OP_LINE),
        ("ln_f → lm_head", "(2, 8, 65)", 65, "每个位置对 65 个字符各打一个分", KV_FILL, KV_LINE),
    ]
    s = Svg(900, 70 + len(rows) * 48)
    s.text(20, 24, "B = 2 条样本, 每条 S = 8 个字符; 方框宽度 ∝ 最后一维的大小", 13, INK2, anchor="start")
    x0, scale = 330, 1.6
    for i, (name, shape, dim, note, fill, line) in enumerate(rows):
        y = 48 + i * 48
        s.text(x0 - 120, y + 16, name, 13, anchor="end")
        s.text(x0 - 12, y + 16, shape, 12, INK2, anchor="end")
        s.rect(x0, y + 2, dim * scale, 28, fill, line)
        s.text(x0 + dim * scale + 10, y + 16, note, 11, MUTED, anchor="start")
        if i < len(rows) - 1:
            s.line(x0 + 10, y + 31, x0 + 10, y + 49, OP_LINE, width=1)
    s.save(out("fig_shapes.svg"))


# ── 图: 多头的切分 (讲义 5.6 节) ───────────────────────────────────

def fig_multihead():
    s = Svg(1000, 380)
    S, cw, ch = 8, 3.0, 16                              # 每列 3 像素, 每行 16 像素

    def matrix(x, y, cols, colors, label, sub):
        """画一个 S 行 × cols 列的矩阵, 列按 colors 分段着色。"""
        seg = cols // len(colors)
        for k, (fill, line) in enumerate(colors):
            s.rect(x + k * seg * cw, y, seg * cw, S * ch, fill, line, rx=2, sw=1.2)
        for r in range(1, S):
            s.line(x, y + r * ch, x + cols * cw, y + r * ch, "#ffffff", arrow=False, width=0.8)
        s.text(x + cols * cw / 2, y - 14, label, 13, weight="bold")
        s.text(x + cols * cw / 2, y + S * ch + 16, sub, 11, MUTED)

    both = [(Q_FILL, Q_LINE), (KV_FILL, KV_LINE)]
    matrix(30, 70, 64, both, "q: (S, C) = (8, 64)", "左 32 维给头 0, 右 32 维给头 1")
    s.line(230, 134, 300, 134)
    s.text(265, 112, "view", 12, INK2)
    s.text(265, 156, "transpose", 12, INK2)
    matrix(310, 36, 32, both[:1], "头 0: (8, 32)", "")
    matrix(310, 196, 32, both[1:], "头 1: (8, 32)", "")
    s.text(358, 344, "形状 (H, S, d_k) = (2, 8, 32)", 11, MUTED)
    s.line(420, 100, 490, 100)
    s.line(420, 260, 490, 260)
    s.box(492, 76, 150, 48, "各自做注意力", "#ffffff", Q_LINE, 13, sub="(8, 8) 的分数矩阵")
    s.box(492, 236, 150, 48, "各自做注意力", "#ffffff", KV_LINE, 13, sub="(8, 8) 的分数矩阵")
    s.line(642, 100, 716, 120)
    s.line(642, 260, 716, 150)
    matrix(720, 70, 64, both, "拼回: (8, 64)", "transpose → contiguous → view")
    s.text(816, 238, "再经过 c_proj, 两个头的信息才混合", 11, MUTED)
    s.text(500, 370, "多头不增加参数: 只是把同一个 64 维向量切成 2 段, 每段独立做注意力, 最后拼回去", 12, INK2)
    s.save(out("fig_multihead.svg"))


# ── 图: Pre-LN 与 Post-LN (讲义 7.5 节) ─────────────────────────────

def fig_block():
    s = Svg(860, 470)

    def plus(x, y):
        s.items.append(f'<circle cx="{x}" cy="{y}" r="11" fill="#ffffff" stroke="{OP_LINE}" stroke-width="1.5"/>')
        s.text(x, y, "+", 16)

    # 左: Pre-LN
    X = 150
    s.text(X + 50, 26, "Pre-LN (GPT-2 起, 本项目)", 15, weight="bold")
    s.line(X, 440, X, 60, CACHE, arrow=False, width=4)   # 残差主干
    s.text(X - 12, 446, "x", 14, anchor="end")
    for k, (name, y) in enumerate([("Attn", 330), ("FFN", 170)]):
        s.path(f"M{X},{y + 70} C{X + 40},{y + 70} {X + 60},{y + 60} {X + 70},{y + 50}")
        s.box(X + 50, y + 10, 70, 32, "LN", OP_FILL, OP_LINE, 12)
        s.line(X + 120, y + 26, X + 140, y + 26)
        s.box(X + 140, y + 6, 80, 40, name, Q_FILL if name == "Attn" else KV_FILL,
              Q_LINE if name == "Attn" else KV_LINE, 13)
        s.path(f"M{X + 180},{y + 6} C{X + 180},{y - 30} {X + 40},{y - 40} {X + 13},{y - 40}")
        plus(X, y - 40)
    s.text(X - 20, 250, "主干上没有任何运算挡着:", 11, CACHE, anchor="end")
    s.text(X - 20, 266, "梯度可以直通到第一层", 11, CACHE, anchor="end")
    s.box(X - 40, 40, 80, 28, "最后的 ln_f", OP_FILL, OP_LINE, 11)

    # 右: Post-LN
    X = 600
    s.text(X + 40, 26, "Post-LN (2017 原版)", 15, weight="bold")
    s.text(X, 50, "输出", 12, INK2)
    s.line(X, 440, X, 60, OP_LINE, arrow=False, width=4)
    s.text(X - 12, 446, "x", 14, anchor="end")
    for name, y in [("Attn", 330), ("FFN", 170)]:
        s.path(f"M{X},{y + 70} C{X + 40},{y + 70} {X + 60},{y + 46} {X + 70},{y + 46}")
        s.box(X + 70, y + 26, 80, 40, name, Q_FILL if name == "Attn" else KV_FILL,
              Q_LINE if name == "Attn" else KV_LINE, 13)
        s.path(f"M{X + 110},{y + 26} C{X + 110},{y + 4} {X + 40},{y + 4} {X + 13},{y + 4}")
        plus(X, y + 4)
        s.box(X - 35, y - 50, 70, 30, "LN", "#fdecec", RED, 12)
    s.text(X + 60, 250, "主干本身要穿过 LN:", 11, RED, anchor="start")
    s.text(X + 60, 266, "层数一多, 梯度难以传回去", 11, RED, anchor="start")
    s.text(430, 462, "x 是贯穿始终的残差主干 (竖线); 每个子层只算一个\"修改量\"加回主干 (⊕)", 12, INK2)
    s.save(out("fig_block.svg"))


# ── 图: KV Cache 每一步算了什么 (专题 A) ─────────────────────────────

def fig_kvcache():
    n, c = 6, 30
    s = Svg(860, 370)
    for p, (title, cached) in enumerate([("不用 cache: 每一步把整段重算一遍", False),
                                         ("用 cache: 每一步只算新来的 1 个位置", True)]):
        x0 = 70 + p * 420
        s.text(x0 + n * c / 2 + 60, 26, title, 14, weight="bold")
        s.text(x0 - 10, 54, "生成第几步 ↓   位置 →", 11, MUTED, anchor="start")
        computed = 0
        for t in range(1, n + 1):
            y = 66 + (t - 1) * (c + 4)
            s.text(x0 - 10, y + c / 2, f"第 {t} 步", 11, INK2, anchor="end")
            for j in range(1, n + 1):
                x = x0 + (j - 1) * (c + 4)
                if j > t:
                    s.rect(x, y, c, c, "#ffffff", "#e3e2dd", rx=3, sw=1)
                elif j == t:
                    s.rect(x, y, c, c, Q_LINE, Q_LINE, rx=3, sw=1)
                    computed += 1
                elif cached:
                    s.rect(x, y, c, c, GOOD_FILL, CACHE, rx=3, sw=1, dash=True)
                else:
                    s.rect(x, y, c, c, Q_FILL, Q_LINE, rx=3, sw=1)
                    computed += 1
        s.text(x0 + n * (c + 4) / 2, 290, f"一共计算 {computed} 个位置", 13, weight="bold")
        s.text(x0 + n * (c + 4) / 2, 310,
               "n(n+1)/2: 随长度平方增长" if not cached else "n: 随长度线性增长", 12, INK2)
    # 图例
    lx = 260
    for i, (fill, line, dash, lab) in enumerate([(Q_LINE, Q_LINE, False, "新 token"), (Q_FILL, Q_LINE, False, "重新计算"),
                                                 (GOOD_FILL, CACHE, True, "从缓存读出")]):
        s.rect(lx + i * 120, 342, 14, 14, fill, line, rx=2, sw=1, dash=dash)
        s.text(lx + i * 120 + 20, 349, lab, 11, INK2, anchor="start")
    s.save(out("fig_kvcache.svg"))


# ── 图: 三种浮点格式的位结构 (专题 C) ───────────────────────────────

def fig_floats():
    fmts = [("fp32", 1, 8, 23, "最大 3.4e38", "1 之后的间距 1.2e-7"),
            ("fp16", 1, 5, 10, "最大 65504", "1 之后的间距 9.8e-4"),
            ("bf16", 1, 8, 7, "最大 3.4e38 (同 fp32)", "1 之后的间距 7.8e-3")]
    bit = 22
    s = Svg(960, 300)
    for i, (name, sg, ex, ma, rng, eps) in enumerate(fmts):
        y = 40 + i * 78
        s.text(60, y + 15, name, 15, anchor="end", weight="bold")
        x = 80
        for k in range(sg + ex + ma):
            if k < sg:
                fill, line = OP_FILL, OP_LINE
            elif k < sg + ex:
                fill, line = KV_FILL, KV_LINE
            else:
                fill, line = Q_FILL, Q_LINE
            s.rect(x + k * bit, y, bit - 2, 30, fill, line, rx=2, sw=1)
        s.text(x + sg * bit / 2, y + 44, "符号", 10, MUTED)
        s.text(x + sg * bit + ex * bit / 2, y + 44, f"指数 {ex} 位", 11, KV_LINE)
        s.text(x + (sg + ex) * bit + ma * bit / 2, y + 44, f"尾数 {ma} 位", 11, Q_LINE)
        right = x + (sg + ex + ma) * bit + 14
        s.text(right, y + 8, rng, 11, INK2, anchor="start")
        s.text(right, y + 24, eps, 11, INK2, anchor="start")
    s.text(480, 280, "指数位数决定范围 (能表示多大、多小) · 尾数位数决定精度 (相邻两个数隔多远)  ——  bf16 = fp32 砍掉 16 位尾数", 12, INK2)
    s.save(out("fig_floats.svg"))


# ── 图: 手写注意力 vs 分块计算 (专题 D) ─────────────────────────────

def fig_tiling():
    n, c = 8, 22
    s = Svg(1000, 360)
    # 左: 手写
    x0, y0 = 60, 70
    s.text(x0 + n * c / 2 + 70, 30, "手写注意力: 先算出完整的 S×S 分数矩阵", 14, weight="bold")
    for i in range(n):
        for j in range(n):
            fill = "#ffffff" if j > i else KV_FILL
            s.rect(x0 + j * c, y0 + i * c, c - 2, c - 2, fill, KV_LINE if j <= i else "#e3e2dd", rx=2, sw=1)
    s.text(x0 + n * c / 2, y0 + n * c + 18, "整张矩阵写进显存, 再读出来做 softmax", 11, INK2)
    s.text(x0 + n * c / 2, y0 + n * c + 36, "显存 ∝ S²", 12, RED, weight="bold")
    s.box(x0 + n * c + 40, y0 + 60, 120, 44, "显存 (慢, 大)", "#fdecec", RED, 12)
    s.line(x0 + n * c + 4, y0 + 82, x0 + n * c + 38, y0 + 82, RED)

    # 右: 分块
    x0 = 560
    s.text(x0 + n * c / 2 + 90, 30, "分块计算: 一次只算一小块, 算完就丢", 14, weight="bold")
    for i in range(n):
        for j in range(n):
            bi, bj = i // 2, j // 2
            if j > i:
                fill, line = "#ffffff", "#e3e2dd"
            elif (bi, bj) == (2, 1):
                fill, line = Q_LINE, Q_LINE                  # 当前正在算的块
            elif bi < 2 or (bi == 2 and bj < 1):
                fill, line = GOOD_FILL, CACHE               # 已经算完、已经丢掉的块
            else:
                fill, line = "#ffffff", "#c9c8c2"
            s.rect(x0 + j * c, y0 + i * c, c - 2, c - 2, fill, line, rx=2, sw=1)
    s.box(x0 + n * c + 40, y0 + 92, 150, 44, "片上缓存 (快, 小)", GOOD_FILL, CACHE, 12)
    s.line(x0 + 2 * c + 2 * c, y0 + 5 * c + 6, x0 + n * c + 38, y0 + 114, Q_LINE)
    s.text(x0 + n * c + 115, y0 + 156, "只放得下当前这一块", 11, INK2)
    s.text(x0 + n * c / 2, y0 + n * c + 18, "边算边更新 softmax 的最大值 m 与指数和 ℓ", 11, INK2)
    s.text(x0 + n * c / 2, y0 + n * c + 36, "显存 ∝ S", 12, CACHE, weight="bold")
    # 图例
    lx = 360
    for i, (fill, line, lab) in enumerate([(Q_LINE, Q_LINE, "正在计算的块"), (GOOD_FILL, CACHE, "已算完并丢弃"),
                                           ("#ffffff", "#e3e2dd", "因果掩码, 不用算")]):
        s.rect(lx + i * 140, 320, 14, 14, fill, line, rx=2, sw=1)
        s.text(lx + i * 140 + 20, 327, lab, 11, INK2, anchor="start")
    s.save(out("fig_tiling.svg"))


if __name__ == "__main__":
    fig_pipeline()
    fig_shift()
    fig_shapes()
    fig_multihead()
    fig_block()
    fig_kvcache()
    fig_floats()
    fig_tiling()
