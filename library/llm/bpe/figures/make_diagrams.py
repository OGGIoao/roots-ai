"""
make_diagrams.py — 生成 BPE 讲义里的结构图 (SVG)

图里的每个 token、每个频率、每一步合并都读自 experiments/results/*.json (实际运行的结果), 不手写。
先运行 python experiments/examples.py fast, 再运行: python figures/make_diagrams.py

配色约定 (与 attention / miniGPT 讲义一致):
  蓝  #2a78d6  主要对象 (token、我们的实现)     橙  #eb6834  被选中 / 第二类对象 (GPT-2)
  绿  #1baf7a  合法、完整                       红  #e34948  不合法、出错
  灰  #52514e  一般运算
"""

import json
import os
from html import escape

from common.svgfig import BG, INK, INK2, MUTED, OP_FILL, OP_LINE, Q_FILL, Q_LINE, KV_FILL, KV_LINE, Svg

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "experiments", "results")
RED, RED_FILL = "#e34948", "#fbe1e1"
GREEN, GREEN_FILL = "#1baf7a", "#dcf3ea"
YEL, YEL_FILL = "#eda100", "#fcefcc"

with open(os.path.join(RES, "examples.json"), encoding="utf-8") as f:
    EX = json.load(f)


def out(name):
    return os.path.join(HERE, name)


def vis(s: str) -> str:
    """让空白可见: 空格 → ␣, 换行 → ↵; 半个汉字这类非法 UTF-8 (显示为 b'..') 改成十六进制。"""
    if s.startswith("b'") or s.startswith('b"'):
        b = eval(s)  # noqa: S307  (内容来自我们自己生成的 json)
        return " ".join(f"{x:02X}" for x in b)
    return s.replace(" ", "␣").replace("\n", "↵")


def tw(s: str, size: int) -> float:
    """估计文字宽度 (像素): 中文约 1 个字号宽, 其它约 0.6 个字号宽。"""
    return sum(size if ord(c) > 0x2E80 else size * 0.6 for c in s)


def chips(s: Svg, x, y, toks, size=13, h=26, gap=3, fill=Q_FILL, stroke=Q_LINE, pick=None, maxw=None):
    """把一串 token 画成相邻的小方块; pick(i, tok) 返回 (fill, stroke) 可单独改色。

    超过 maxw 时换行。返回最后一个方块之后的 (x, y), y 是最后一行的顶边。"""
    x0 = x
    for i, t in enumerate(toks):
        label = vis(t)
        bad = t.startswith("b'")
        w = max(tw(label, size if not bad else size - 2) + 12, 18)
        if maxw and x + w > x0 + maxw:
            x, y = x0, y + h + 8
        f_, s_ = (RED_FILL, RED) if bad else (fill, stroke)
        if pick:
            f_, s_ = pick(i, t) or (f_, s_)
        s.rect(x, y, w, h, f_, s_, rx=4, sw=1.2)
        s.text(x + w / 2, y + h / 2 + 1, escape(label), size if not bad else size - 2, INK)
        x += w + gap
    return x, y


# ── 图: BPE 的三件事 (训练 / 编码 / 解码) ─────────────────────────────

def fig_pipeline():
    s = Svg(980, 360)
    s.text(20, 26, "训练 train()  —— 只做一次, 产出 merges 和 vocab", 15, anchor="start", weight="bold")
    s.box(20, 48, 150, 46, "训练语料", Q_FILL, Q_LINE, 13, sub="莎士比亚 100 万字符")
    s.line(170, 71, 203, 71)
    s.box(205, 48, 150, 46, "预分词", OP_FILL, OP_LINE, 13, sub="切成不跨词的片段")
    s.line(355, 71, 388, 71)
    s.box(390, 48, 160, 46, "词频表", OP_FILL, OP_LINE, 13, sub="{字节元组: 次数}")
    s.line(550, 71, 583, 71)
    s.rect(585, 42, 210, 58, "#ffffff", KV_LINE)
    s.text(690, 60, "重复 vocab_size − 256 次", 12, INK2)
    s.text(690, 82, "数 pair → 合并最高频的", 13, weight="bold")
    s.line(795, 71, 828, 71)
    s.box(830, 48, 130, 46, "merges + vocab", KV_FILL, KV_LINE, 13, sub="存进 tokenizer.json")

    s.text(20, 152, "编码 encode()  —— 每次用", 15, anchor="start", weight="bold")
    s.box(20, 172, 150, 46, '"Hello world"', Q_FILL, Q_LINE, 13)
    s.line(170, 195, 203, 195)
    s.box(205, 172, 150, 46, "预分词", OP_FILL, OP_LINE, 13, sub='["Hello", "␣world"]')
    s.line(355, 195, 388, 195)
    s.box(390, 172, 160, 46, "UTF-8 字节", OP_FILL, OP_LINE, 13, sub="[72, 101, 108, 108, 111] ...")
    s.line(550, 195, 583, 195)
    s.box(585, 172, 210, 46, "按 rank 从小到大合并", OP_FILL, OP_LINE, 13, sub="只用 merges")
    s.line(795, 195, 828, 195)
    s.box(830, 172, 130, 46, "token ID 列表", Q_FILL, Q_LINE, 13)
    s.path("M895,124 L895,150 L690,150 L690,170", arrow=True)

    s.text(20, 276, "解码 decode()  —— 不需要 merges", 15, anchor="start", weight="bold")
    s.box(20, 296, 150, 46, "token ID 列表", Q_FILL, Q_LINE, 13)
    s.line(170, 319, 203, 319)
    s.box(205, 296, 345, 46, "查 vocab: 每个 ID → 它代表的字节, 全部拼起来", OP_FILL, OP_LINE, 13)
    s.line(550, 319, 583, 319)
    s.box(585, 296, 210, 46, "bytes.decode('utf-8')", OP_FILL, OP_LINE, 13)
    s.line(795, 319, 828, 319)
    s.box(830, 296, 130, 46, '"Hello world"', Q_FILL, Q_LINE, 13)
    s.save(out("fig_pipeline.svg"))


# ── 图: Sennrich 经典例子的前 7 轮合并 ─────────────────────────────

def fig_train_rounds():
    rounds = EX["toy_sennrich"]
    words0 = [(list(w), f) for w, f in [("low", 5), ("lower", 2), ("newest", 6), ("widest", 3)]]
    show_rounds = [0, 1, 2, 3, 4, 5, 6, 7]
    col_w, left = 225, 165
    s = Svg(left + 4 * col_w + 10, 70 + 54 * len(show_rounds))
    for j, (w, f) in enumerate(words0):
        s.text(left + j * col_w + 8, 30, f'"{"".join(w)}" × {f}', 13, INK2, anchor="start", weight="bold")
    for r_i, r in enumerate(show_rounds):
        y = 52 + r_i * 54
        if r == 0:
            segs = [w for w, _ in words0]
            s.text(12, y + 13, "开始: 全是单字节", 12, INK2, anchor="start")
            new = None
        else:
            R = rounds[r - 1]
            segs = [w[0] for w in R["words_after"]]
            s.text(12, y + 6, f"第 {r} 轮: {R['pairs'][0][0]}+{R['pairs'][0][1]}", 12, INK, anchor="start", weight="bold")
            s.text(12, y + 24, f"频率 {R['freq']} → 新 ID {R['new_id']}", 11, MUTED, anchor="start")
            new = R["new_token"]
        for j, seg in enumerate(segs):
            pick = (lambda i, t, new=new: (KV_FILL, KV_LINE) if t == new else None)
            chips(s, left + j * col_w + 8, y, seg, size=13, h=26, gap=2, pick=pick)
    s.save(out("fig_train_rounds.svg"))


# ── 图: 一个 token 的合并树 ─────────────────────────────────────────

def fig_merge_tree():
    tree = EX["merge_tree_longest"]
    s = Svg(560, 330)
    # 叶子从左到右排开, 父节点放在两个孩子中间
    pos = {}
    leaves = []
    def leaf_order(n):
        if "kids" not in n:
            leaves.append(n)
        for k in n.get("kids", []):
            leaf_order(k)
    leaf_order(tree)
    for i, n in enumerate(leaves):
        pos[id(n)] = 70 + i * 90
    def place(n):
        if "kids" in n:
            xs = [place(k) for k in n["kids"]]
            pos[id(n)] = sum(xs) / 2
        return pos[id(n)]
    place(tree)
    def y_of(d):
        return 50 + d * 70
    def draw(n, d):
        x, y = pos[id(n)], y_of(d)
        for k in n.get("kids", []):
            s.line(x, y + 16, pos[id(k)], y_of(d + 1) - 16, OP_LINE, arrow=False)
            draw(k, d + 1)
        leaf = "kids" not in n
        w = max(tw(vis(n["tok"]), 13) + 18, 40)
        s.rect(x - w / 2, y - 16, w, 32, Q_FILL if leaf else KV_FILL, Q_LINE if leaf else KV_LINE)
        s.text(x, y - 2, escape(vis(n["tok"])), 13)
        s.text(x, y + 10, str(n["id"]), 9, MUTED)
    draw(tree, 0)
    s.text(20, 312, "蓝 = 初始字节 (ID 0-255); 橙 = 训练中合并出来的 token; 小字是 ID (越大越晚学到)", 11, MUTED,
           anchor="start")
    s.save(out("fig_merge_tree.svg"))


# ── 图: 编码 " lowest" 的每一轮 ─────────────────────────────────────

def fig_encode_trace():
    steps = EX["encode_trace"][" lowest"]
    s = Svg(760, 50 + 60 * len(steps))
    s.text(20, 24, '编码 "␣lowest": 每轮在所有"有规则的相邻对"里挑 rank 最小的', 14, anchor="start", weight="bold")
    for i, st in enumerate(steps):
        y = 50 + i * 60
        seq = st["seq"]
        chosen = st.get("chosen")
        # 选中的 pair 第一次出现的位置
        hit = None
        if chosen:
            for k in range(len(seq) - 1):
                if seq[k] == chosen[0] and seq[k + 1] == chosen[1]:
                    hit = k
                    break
        pick = (lambda k, t, hit=hit: (KV_FILL, KV_LINE) if hit is not None and k in (hit, hit + 1) else None)
        chips(s, 20, y, seq, size=14, h=28, gap=3, pick=pick)
        cands = ", ".join(f"{vis(a)}+{vis(b)}={r}" for a, b, r in st["candidates"])
        if chosen:
            s.text(250, y + 8, escape(f"候选: {cands}"), 12, INK2, anchor="start")
            s.text(250, y + 26, escape(f"选 rank 最小的 {vis(chosen[0])}+{vis(chosen[1])} = {chosen[2]}"), 12, INK,
                   anchor="start", weight="bold")
        else:
            s.text(250, y + 14, "没有任何相邻对在 merges 里 → 结束", 12, INK, anchor="start", weight="bold")
    s.save(out("fig_encode_trace.svg"))


# ── 图: 同一句话, 我们的分词器 vs GPT-2 ───────────────────────────

def fig_tokenized():
    sents = EX["sentences"][:2]
    s = Svg(1000, 300)
    y = 30
    for st in sents:
        s.text(20, y, escape(f"原文: {st['text']}"), 13, INK, anchor="start", weight="bold")
        s.text(20, y + 32, f"我们的 (500)", 12, INK2, anchor="start")
        s.text(20, y + 46, f"{len(st['ours'])} 个 token", 11, MUTED, anchor="start")
        alt = lambda i, t: None if i % 2 == 0 else ("#c4dbf5", Q_LINE)
        _, yy = chips(s, 130, y + 24, st["ours"], size=12, h=24, gap=2, pick=alt, maxw=850)
        s.text(20, yy + 46, "GPT-2 (50257)", 12, INK2, anchor="start")
        s.text(20, yy + 60, f"{len(st['gpt2'])} 个 token", 11, MUTED, anchor="start")
        alt2 = lambda i, t: (KV_FILL, KV_LINE) if i % 2 == 0 else ("#f9c9b3", KV_LINE)
        chips(s, 130, yy + 40, st["gpt2"], size=12, h=24, gap=2, pick=alt2, maxw=850)
        y = yy + 100
    s.text(20, y - 22, "红框 = 不是完整字符的字节 (十六进制); ␣ = 空格", 11, MUTED, anchor="start")
    s.save(out("fig_tokenized.svg"))


# ── 图: UTF-8 的字节布局 ─────────────────────────────────────────────

def fig_utf8():
    rows = [r for r in EX["utf8"] if r["ch"] != "ñ"]
    s = Svg(980, 60 + 70 * len(rows))
    s.text(20, 24, "UTF-8: 前缀位 (橙) 说明\"我是几字节字符的第几个\", 其余位 (蓝) 拼起来就是码点", 14,
           anchor="start", weight="bold")
    prefix_len = {1: [1], 2: [3, 2], 3: [4, 2, 2], 4: [5, 2, 2, 2]}
    for i, r in enumerate(rows):
        y = 50 + i * 70
        n = len(r["bin"])
        s.text(30, y + 14, r["ch"], 20, INK, weight="bold")
        s.text(60, y + 8, r["cp"], 12, INK2, anchor="start")
        s.text(60, y + 26, f"{n} 字节", 11, MUTED, anchor="start")
        x = 150
        payload = ""
        for k, (bits, hx) in enumerate(zip(r["bin"], r["bytes"])):
            p = prefix_len[n][k]
            for j, bit in enumerate(bits):
                pre = j < p
                s.rect(x + j * 19, y, 18, 26, KV_FILL if pre else Q_FILL, KV_LINE if pre else Q_LINE, rx=2, sw=1)
                s.text(x + j * 19 + 9, y + 14, bit, 12)
                if not pre:
                    payload += bit
            s.text(x + 76, y + 40, f"{hx} = {int(hx, 16)}", 11, INK2)
            x += 8 * 19 + 14
        s.text(x + 6, y + 14, f"→ {int(payload, 2):X} = {r['cp']}", 12, INK, anchor="start")
    s.save(out("fig_utf8.svg"))


# ── 图: 0-255 字节表 ─────────────────────────────────────────────────

CLASS_STYLE = {
    "control": ("#efeeea", OP_LINE, "控制字符 0-31, 127"),
    "ascii": (Q_FILL, Q_LINE, "可打印 ASCII 32-126"),
    "cont": (YEL_FILL, YEL, "续字节 128-191 (10xxxxxx)"),
    "lead2": (GREEN_FILL, GREEN, "2 字节首字节 194-223"),
    "lead3": (KV_FILL, KV_LINE, "3 字节首字节 224-239 (汉字在这)"),
    "lead4": ("#f6dbe8", "#e87ba4", "4 字节首字节 240-244 (emoji)"),
    "never": ("#ffffff", RED, "合法 UTF-8 里从不出现"),
}


def fig_byte_map():
    cls = EX["byte_map"]["class"]
    cnt = EX["byte_map"]["shakespeare_counts"]
    cell, x0, y0 = 34, 60, 70
    s = Svg(x0 + 16 * cell + 300, y0 + 16 * cell + 30)
    s.text(20, 24, "初始词表的 256 个字节: 颜色 = 它在 UTF-8 里的角色; 粗框 = 莎士比亚训练集里出现过", 14,
           anchor="start", weight="bold")
    for j in range(16):
        s.text(x0 + j * cell + cell / 2, y0 - 12, f"_{j:X}", 11, MUTED)
    for i in range(16):
        s.text(x0 - 10, y0 + i * cell + cell / 2, f"{i:X}_", 11, MUTED, anchor="end")
        for j in range(16):
            b = i * 16 + j
            f_, st, _ = CLASS_STYLE[cls[b]]
            used = cnt[b] > 0
            s.rect(x0 + j * cell, y0 + i * cell, cell - 2, cell - 2, f_, st if used else "#d9d8d3",
                   rx=3, sw=2.2 if used else 0.8)
            ch = chr(b) if 33 <= b <= 126 else ""
            s.text(x0 + j * cell + cell / 2 - 1, y0 + i * cell + 12, escape(ch) if ch else f"{b:02X}",
                   12 if ch else 9, INK if ch else INK2)
            s.text(x0 + j * cell + cell / 2 - 1, y0 + i * cell + 25, str(b), 8, MUTED)
    lx = x0 + 16 * cell + 24
    for k, (key, (f_, st, label)) in enumerate(CLASS_STYLE.items()):
        y = y0 + 10 + k * 34
        s.rect(lx, y, 22, 22, f_, st, rx=3, sw=1.5)
        n = cls.count(key)
        s.text(lx + 32, y + 11, f"{label}  ({n} 个)", 12, INK, anchor="start")
    used = sum(1 for c in cnt if c > 0)
    s.text(lx, y0 + 10 + 7 * 34 + 14, f"莎士比亚训练集只用到其中 {used} 个字节", 12, INK2, anchor="start")
    s.save(out("fig_byte_map.svg"))


# ── 图: 不同预分词器对同一句话的切法 ─────────────────────────────

def fig_pretok():
    pick_texts = ["It's 2026, hello world!", "价格是 1234567 元", "x = foo(a,b)\n    return x"]
    rows = [("ours", "我们的 _pretokenize"), ("gpt2_regex", "GPT-2"), ("qwen_regex", "Qwen2.5"),
            ("phi4_regex", "phi-4 / GLM-4.5")]
    data = {r["text"]: r for r in EX["pretokenize"]}
    s = Svg(760, 40 + len(pick_texts) * (40 + 36 * len(rows)))
    y = 26
    for t in pick_texts:
        s.text(20, y, escape("原文: " + vis(t)), 13, INK, anchor="start", weight="bold")
        y += 22
        for k, (key, name) in enumerate(rows):
            s.text(20, y + 13, name, 12, INK2, anchor="start")
            alt = (lambda i, tok: None if i % 2 == 0 else ("#c4dbf5", Q_LINE))
            chips(s, 170, y, data[t][key], size=12, h=26, gap=3, pick=alt)
            y += 36
        y += 18
    s.save(out("fig_pretok.svg"))


# ── 图: 中文字节合并跨越了汉字边界 ─────────────────────────────────

def fig_zh_cross():
    toks = EX["chinese"]["repeat_nihao"]["encode_nihao"]          # 如 ["b'\\xe4\\xbd\\xa0\\xe5'", "b'\\xa5\\xbd'"]
    byts = []
    for t in toks:
        byts.append(list(eval(t)))  # noqa: S307
    flat = [b for t in byts for b in t]
    chars = "你好"
    s = Svg(720, 230)
    s.text(20, 24, '在 "你好世界"×100 上训练 (词表 300) 后编码 "你好": token 边界切在了汉字中间', 14,
           anchor="start", weight="bold")
    x0, w = 120, 82
    # 第一行: 字符
    s.text(20, 72, "字符", 12, INK2, anchor="start")
    for k, ch in enumerate(chars):
        s.rect(x0 + k * 3 * w, 56, 3 * w - 4, 32, GREEN_FILL, GREEN)
        s.text(x0 + k * 3 * w + 1.5 * w - 2, 72, ch, 16)
    s.text(20, 120, "UTF-8 字节", 12, INK2, anchor="start")
    for k, b in enumerate(flat):
        s.rect(x0 + k * w, 104, w - 4, 32, OP_FILL, OP_LINE)
        s.text(x0 + k * w + w / 2 - 2, 120, f"{b:02X}", 13)
    s.text(20, 168, "token", 12, INK2, anchor="start")
    x = x0
    for t in byts:
        s.rect(x, 152, len(t) * w - 4, 32, RED_FILL, RED)
        s.text(x + len(t) * w / 2 - 2, 168, " ".join(f"{b:02X}" for b in t), 13)
        x += len(t) * w
    s.line(x0 + 3 * w - 2, 50, x0 + 3 * w - 2, 192, RED, arrow=False, dash=True)
    s.text(x0 + 3 * w - 2, 208, "汉字边界", 11, RED)
    s.save(out("fig_zh_cross.svg"))


if __name__ == "__main__":
    fig_pipeline()
    fig_train_rounds()
    fig_merge_tree()
    fig_encode_trace()
    fig_tokenized()
    fig_utf8()
    fig_byte_map()
    fig_pretok()
    fig_zh_cross()
