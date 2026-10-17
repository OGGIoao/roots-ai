"""
make_diagrams.py — 生成注意力线讲义里的结构图 (SVG)

结构图用代码生成而不是手工绘制: 坐标由循环算出, 对齐准确, 改起来也方便。
输出与本文件同目录: fig_heads.svg、fig_mla.svg、fig_absorb.svg

运行: python figures/make_diagrams.py   (只依赖标准库)

配色约定 (与数据图一致, 来自 dataviz 参考调色板):
  查询 Q  蓝    #2a78d6 / 浅底 #dbe9f9
  K、V    橙    #eb6834 / 浅底 #fde3d7
  缓存    绿    #008300 (虚线框)
  其它运算 灰   #52514e / 浅底 #efeeea
背景用不透明的浅色 #fcfcfb, 在 GitHub 深色模式下也能看清。
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

FONT = "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC','Source Han Sans SC',sans-serif"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BG = "#fcfcfb"
Q_LINE, Q_FILL = "#2a78d6", "#dbe9f9"
KV_LINE, KV_FILL = "#eb6834", "#fde3d7"
CACHE = "#008300"
OP_LINE, OP_FILL = "#52514e", "#efeeea"


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

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" font-family="{FONT}">\n'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{OP_LINE}"/>'
                '</marker></defs>\n'
                f'<rect width="100%" height="100%" fill="{BG}"/>\n')
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            f.write(head + "\n".join(self.items) + "\n</svg>\n")
        print("写入", name)


# ── 图 1: MHA / GQA / MQA 的头与 KV 的对应关系 ──────────────────────────

def fig_heads():
    H = 8
    panels = [("MHA", 8, "每个头一组 K、V"), ("GQA (G = 2)", 2, "每 4 个头共用一组"), ("MQA", 1, "所有头共用一组")]
    pw, gap, x_margin = 290, 25, 20
    s = Svg(x_margin * 2 + 3 * pw + 2 * gap, 360)
    bw, step = 26, 33                                   # 查询头方框宽度与间距
    for p, (title, G, note) in enumerate(panels):
        x0 = x_margin + p * (pw + gap)
        s.rect(x0, 10, pw, 340, "#ffffff", "#e3e2dd", rx=8, sw=1)
        s.text(x0 + pw / 2, 34, title, 17, weight="bold")
        s.text(x0 + pw / 2, 56, note, 12, INK2)
        left = x0 + (pw - (H * step - (step - bw))) / 2
        q_cx = [left + i * step + bw / 2 for i in range(H)]
        for i, cx in enumerate(q_cx):                   # 第一行: 8 个查询头
            s.box(cx - bw / 2, 76, bw, 26, f"Q{i}", Q_FILL, Q_LINE, 11)
        per = H // G
        kv_cx = [sum(q_cx[g * per:(g + 1) * per]) / per for g in range(G)]   # 每组 KV 放在组的正下方
        kw = 26 if G == H else 44
        for g, cx in enumerate(kv_cx):                  # 第二行: G 组 K、V
            s.rect(cx - kw / 2, 190, kw, 44, KV_FILL, KV_LINE)
            s.text(cx, 203, "K", 12)
            s.text(cx, 221, "V", 12)
        for i, cx in enumerate(q_cx):                   # 每个查询头连到它所属组的 K、V
            s.line(cx, 104, kv_cx[i // per], 188, Q_LINE, arrow=True, width=1.1)
        # 缓存框: 推理时要存下来的就是第二行
        s.rect(x0 + 14, 180, pw - 28, 64, "none", CACHE, dash=True, rx=6)
        s.text(x0 + pw - 20, 254, "推理时缓存", 11, CACHE, anchor="end")
        s.text(x0 + pw / 2, 290, f"每个 token 缓存 {G} 组 K、V", 14, weight="bold")
        s.text(x0 + pw / 2, 314, f"= 2 × {G} × d_h 个数", 13, INK2)
        s.text(x0 + pw / 2, 334, f"(MHA 的 {G}/8)" if G != H else "(基准)", 12, MUTED)
    s.save("fig_heads.svg")


# ── 图 2: MLA 的数据流 ───────────────────────────────────────────────

def fig_mla():
    s = Svg(1000, 470)
    # 输入
    s.box(20, 215, 56, 40, "x", OP_FILL, OP_LINE, 16, sub="(B, S, C)")

    # 查询路径 (上方): q 拆成两段, nope 段直接进注意力, rope 段先旋转
    s.path("M76,225 C100,225 100,80 120,80")
    s.box(120, 62, 70, 36, "W_q", OP_FILL, OP_LINE)
    s.path("M190,72 C210,72 210,50 236,50")
    s.path("M190,88 C210,88 210,110 236,110")
    s.box(238, 34, 92, 32, "q_nope", Q_FILL, Q_LINE, 13, sub="每头 d_nope 维")
    s.box(238, 94, 92, 32, "q_rope", Q_FILL, Q_LINE, 13, sub="每头 d_rope 维")
    s.line(330, 110, 368, 110)
    s.box(370, 94, 56, 32, "RoPE", "#ffffff", Q_LINE, 12)

    # 键值路径 (中间): 压缩
    s.path("M76,235 L120,235")
    s.box(120, 217, 70, 36, "W_dkv", OP_FILL, OP_LINE, 13)
    s.text(155, 266, "降维 (压缩)", 11, MUTED)
    s.path("M190,230 C210,230 210,190 236,190")
    s.path("M190,240 C210,240 210,300 236,300")
    s.box(238, 172, 92, 36, "c", KV_FILL, KV_LINE, 15, sub="r 维潜向量")
    s.box(238, 282, 92, 36, "k_rope", KV_FILL, KV_LINE, 13, sub="d_rope 维, 所有头共用")
    s.rect(214, 150, 140, 196, "none", CACHE, dash=True, rx=8)
    s.text(284, 362, "推理时只缓存这两样", 12, CACHE, weight="bold")
    s.text(284, 380, "每层每 token: r + d_rope 个数", 11, CACHE)

    # 解压
    s.line(330, 190, 368, 190)
    s.box(370, 172, 72, 36, "RMSNorm", OP_FILL, OP_LINE, 12)
    s.line(442, 190, 478, 190)
    s.box(480, 172, 72, 36, "W_ukv", OP_FILL, OP_LINE, 13)
    s.text(516, 221, "升维 (解压)", 11, MUTED)
    s.line(552, 190, 588, 190)
    s.rect(590, 168, 130, 44, KV_FILL, KV_LINE)
    s.line(655, 168, 655, 212, KV_LINE, arrow=False)
    s.text(622, 190, "k_nope", 12)
    s.text(687, 190, "v", 13)
    s.text(655, 226, "每个头各一份", 11, MUTED)

    # 共享的 rope 键
    s.line(330, 300, 368, 300)
    s.box(370, 282, 56, 36, "RoPE", "#ffffff", KV_LINE, 12)
    s.line(426, 300, 588, 300)
    s.box(590, 282, 130, 36, "k_rope (共享)", KV_FILL, KV_LINE, 12)
    s.text(655, 332, "expand 给每个头, 不复制内存", 11, MUTED)

    # 注意力
    s.rect(770, 150, 120, 180, "#ffffff", OP_LINE)
    s.text(830, 172, "注意力", 15, weight="bold")
    s.text(830, 205, "q_nope · k_nope", 12, Q_LINE)
    s.text(830, 225, "+", 12)
    s.text(830, 245, "q_rope · k_rope", 12, Q_LINE)
    s.text(830, 270, "÷ √(d_nope+d_rope)", 11, INK2)
    s.text(830, 295, "softmax → · v", 12)
    s.path("M330,50 C720,50 750,150 770,185", Q_LINE)   # q_nope 直接进注意力
    s.path("M426,110 C700,110 740,180 770,205", Q_LINE)  # 旋转后的 q_rope 进注意力
    s.line(720, 190, 768, 205)                          # k_nope、v
    s.line(720, 300, 768, 270)                          # k_rope
    s.line(890, 240, 918, 240)
    s.box(920, 222, 60, 36, "W_o", OP_FILL, OP_LINE, 13)

    s.text(500, 420, "rope 段单独走一条路: 位置旋转 R_j 不能和解压矩阵交换次序, 拆出去之后, nope 段才能做吸收 (图 3)", 12, INK2)
    s.text(500, 445, "真实权重名: W_dkv = kv_a_proj_with_mqa · RMSNorm = kv_a_layernorm · W_ukv = kv_b_proj (DeepSeek-V3)", 11, MUTED)
    s.save("fig_mla.svg")


# ── 图 3: 吸收技巧 ─────────────────────────────────────────────────

def fig_absorb():
    s = Svg(1000, 470)
    t = 6                                               # 画 6 个历史位置示意

    def cache_row(x, y):
        for j in range(t):
            s.box(x + j * 34, y, 28, 28, f"c{j + 1}", KV_FILL, KV_LINE, 11)
        s.rect(x - 8, y - 8, t * 34 + 10, 44, "none", CACHE, dash=True, rx=6)

    # 上: 朴素
    s.text(20, 28, "朴素解码: 每一步把全部历史潜向量解压成每个头的 K、V", 15, anchor="start", weight="bold")
    cache_row(40, 62)
    s.text(140, 116, "缓存里的 c (t 个位置)", 11, CACHE)
    s.line(250, 76, 300, 76)
    s.box(302, 58, 120, 36, "× W_ukv", OP_FILL, OP_LINE, 13)
    s.text(362, 108, "t 个位置 × H 个头, 每步都做", 11, "#e34948")
    s.line(422, 76, 470, 76)
    for j in range(t):                                  # 解压出来的 K、V: t 列 × 若干头
        for hh in range(3):
            s.rect(472 + j * 24, 52 + hh * 16, 20, 13, KV_FILL, KV_LINE, rx=2, sw=1)
    s.text(544, 112, "每个头的 K、V (t 个位置)", 11, MUTED)
    s.line(622, 76, 670, 76)
    s.box(672, 58, 150, 36, "与 q 做注意力", "#ffffff", OP_LINE, 13)
    s.line(822, 76, 860, 76)
    s.box(862, 58, 100, 36, "输出", OP_FILL, OP_LINE, 13)
    s.text(500, 150, "每生成一个 token 的计算量 ∝ t × r × H × (d_nope + d_v)  —— 随上下文长度线性增长, 而且乘了头数", 12, INK2)

    s.line(20, 180, 980, 180, "#e3e2dd", arrow=False, width=1)

    # 下: 吸收
    s.text(20, 210, "吸收解码: 把解压矩阵提前乘到 q 和输出上, 历史潜向量从不解压", 15, anchor="start", weight="bold")
    s.box(40, 248, 60, 36, "q", Q_FILL, Q_LINE, 15)
    s.line(100, 266, 140, 266)
    s.box(142, 248, 118, 36, "× W_ukᵀ", OP_FILL, OP_LINE, 13)
    s.text(201, 298, "每步 1 次", 11, CACHE)
    s.line(260, 266, 300, 266)
    s.box(302, 248, 76, 36, "q_lat", Q_FILL, Q_LINE, 13, sub="r 维")
    s.line(378, 266, 418, 266)
    s.rect(420, 238, 150, 56, "#ffffff", OP_LINE)
    s.text(495, 258, "点积 → softmax", 12)
    s.text(495, 278, "得到权重 a_j", 12)
    cache_row(410, 350)
    s.text(510, 404, "缓存里的 c (直接使用)", 11, CACHE)
    s.line(495, 342, 495, 296, KV_LINE)
    s.line(570, 266, 610, 266)
    s.rect(612, 238, 130, 56, "#ffffff", OP_LINE)
    s.text(677, 258, "Σ a_j · c_j", 13)
    s.text(677, 278, "在潜空间加权求和", 11, INK2)
    s.line(600, 352, 650, 296, KV_LINE)                 # 同一份缓存再用一次: 加权求和
    s.line(742, 266, 772, 266)
    s.box(774, 248, 90, 36, "× W_uv", OP_FILL, OP_LINE, 13)
    s.text(819, 298, "每步 1 次", 11, CACHE)
    s.line(864, 266, 894, 266)
    s.box(896, 248, 70, 36, "输出", OP_FILL, OP_LINE, 13)
    s.text(500, 440, "每生成一个 token 的计算量 ∝ t × r × H  —— 少了 (d_nope + d_v) 这个因子; 结果与朴素解码完全相同 (结合律)", 12, INK2)
    s.save("fig_absorb.svg")


if __name__ == "__main__":
    fig_heads()
    fig_mla()
    fig_absorb()
