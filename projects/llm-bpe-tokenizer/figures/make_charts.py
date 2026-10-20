"""
make_charts.py — 生成 BPE 讲义里的数据图 (SVG)

数据全部读自 experiments/results/ (实际运行的结果):
  vocab_sweep.json        python experiments/examples.py sweep
  examples.json           python experiments/examples.py fast
  tokenizer_anatomy.json  python experiments/tokenizer_anatomy.py
运行: python figures/make_charts.py
"""

import json
import os

from common.svgfig import INK2, MUTED, SERIES, Chart

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "experiments", "results")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


SW = load("vocab_sweep.json")
EX = load("examples.json")
AN = load("tokenizer_anatomy.json")
GPT2_RATIO = EX["compression"]["gpt2_tokens"] / EX["compression"]["val_bytes"]


def out(name):
    return os.path.join(HERE, name)


# ── 图: 压缩率 vs 词表大小 ─────────────────────────────────────────

def chart_vocab_ratio():
    rows = SW["rows"]
    c = Chart(820, 420, left=80, right=190)
    c.scales(256, 65536, 0, 1.05, xlog=True)
    c.axes([256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536], [0, 0.2, 0.4, 0.6, 0.8, 1.0],
           "词表大小 (对数刻度, 每格翻倍)", "压缩率 = token 数 / 字节数 (越低越好)",
           xfmt=lambda v: f"{v:,}", yfmt=lambda v: f"{v:.1f}")
    pts = [(r["vocab"], r["ratio"]) for r in rows]
    c.series(pts, SERIES[0], mark="circle")
    for r in rows:
        if r["vocab"] in (300, 500, 1000, 4096):
            dy = 16 if r["vocab"] == 4096 else -14          # 4096 的数字放到点下方, 避开 GPT-2 参考线
            c.text(c.X(r["vocab"]) - (6 if dy > 0 else -4), c.Y(r["ratio"]) + dy, f"{r['ratio']:.3f}", 11, INK2,
                   "end" if dy > 0 else "start")
    c.end_label(4096, rows[-1]["ratio"], "我们的 BPE (只在莎士比亚上训练)", dy=0)
    # GPT-2: 一个点 + 水平参考线
    c.add(f'<line x1="{c.x0}" y1="{c.Y(GPT2_RATIO):.1f}" x2="{c.x1}" y2="{c.Y(GPT2_RATIO):.1f}" '
          f'stroke="{SERIES[1]}" stroke-width="1.2" stroke-dasharray="5 4"/>')
    c.marker(c.X(50257), c.Y(GPT2_RATIO), "square", SERIES[1])
    c.text(c.X(50257), c.Y(GPT2_RATIO) - 16, f"GPT-2: {GPT2_RATIO:.3f}", 11, INK2)
    c.text(c.X(256) + 8, c.Y(1.0) + 14, "256 = 不合并, 1 字节 1 token", 11, MUTED, "start")
    c.save(out("chart_vocab_ratio.svg"), "压缩率随词表变大而下降, 但越来越慢",
           "同一段莎士比亚验证集 (10,000 字节); 训练一次到 4096, 取前 k 条合并规则")


# ── 图: 每次合并时最高频 pair 的频率 ─────────────────────────────

def chart_merge_freq():
    f = SW["merge_freqs"]
    c = Chart(820, 400, left=80, right=60)
    c.scales(1, len(f), 10, 60000, xlog=True, ylog=True)
    c.axes([1, 4, 16, 64, 256, 1024, len(f)], [10, 100, 1000, 10000],
           "第几次合并 (对数刻度)", "被合并的 pair 出现了多少次 (对数刻度)",
           xfmt=lambda v: f"{v:,}", yfmt=lambda v: f"{v:,}")
    pts = [(i + 1, v) for i, v in enumerate(f)]
    c.series(pts, SERIES[0])
    for i in (1, 15, 244, 1000, len(f)):
        c.marker(c.X(i), c.Y(f[i - 1]), "circle", SERIES[0], 3.5)
        c.text(c.X(i) + 6, c.Y(f[i - 1]) - 12, f"#{i}: {f[i - 1]:,}", 11, INK2, "start")
    c.add(f'<line x1="{c.X(244):.1f}" y1="{c.y1}" x2="{c.X(244):.1f}" y2="{c.y0}" stroke="{MUTED}" '
          f'stroke-dasharray="3 3"/>')
    c.text(c.X(244) - 6, c.y1 + 10, "默认词表 500 到这里为止", 11, MUTED, "end")
    c.save(out("chart_merge_freq.svg"), "越往后学到的合并, 在语料里越少见",
           f"莎士比亚训练集, {len(f)} 次合并; 两个坐标都是对数: 前 100 次降得慢, 之后越来越陡")


# ── 图: 训练耗时 ─────────────────────────────────────────────────

def chart_train_time():
    t = SW["cum_time_s"]
    c = Chart(820, 360, left=80, right=60)
    c.scales(0, len(t), 0, 200)
    c.axes([0, 500, 1000, 1500, 2000, 2500, 3000, 3500], [0, 50, 100, 150, 200],
           "合并次数 (= 词表大小 − 256)", "累计训练时间 (秒)", xfmt=lambda v: f"{v:,}")
    pts = [(i + 1, v) for i, v in enumerate(t)]
    c.series(pts, SERIES[0])
    per = (t[-1] - t[0]) / (len(t) - 1)
    c.text(c.X(len(t)) - 8, c.Y(t[-1]) - 14, f"{len(t)} 次: {t[-1]:.0f} 秒", 11, INK2, "end")
    c.text(c.X(244), c.Y(t[243]) - 14, f"244 次 (词表 500): {t[243]:.1f} 秒", 11, INK2, "start")
    c.marker(c.X(244), c.Y(t[243]), "circle", SERIES[0], 3.5)
    c.save(out("chart_train_time.svg"), f"每次合并的耗时几乎不变 (约 {per * 1000:.0f} 毫秒)",
           "每轮都把 2.5 万个不同的词全部重新扫一遍数 pair, 所以总时间 ∝ 合并次数")


# ── 图: 主流分词器在同一段英文 / 中文上的效率 ─────────────────────

SHORT = {"openai-community/gpt2": "GPT-2", "google-t5/t5-small": "T5", "mistralai/Mistral-7B-v0.1": "Mistral-7B",
         "microsoft/phi-2": "phi-2", "microsoft/Phi-3-mini-4k-instruct": "Phi-3-mini", "microsoft/phi-4": "phi-4",
         "Qwen/Qwen2.5-7B": "Qwen2.5", "deepseek-ai/DeepSeek-V3": "DeepSeek-V3", "zai-org/GLM-4.5": "GLM-4.5",
         "openai/gpt-oss-20b": "gpt-oss", "google-bert/bert-base-chinese": "BERT-zh",
         "google-bert/bert-base-uncased": "BERT-en"}


def chart_anatomy(lang):
    rows = [r for r in AN["rows"] if r["status"] == "ok" and r[f"{lang}_lossless"]]
    if lang == "en":
        vals = [(SHORT[r["repo"]], r["en_bytes"] / r["en_tokens"]) for r in rows]
        sw_rows = {r["vocab"]: r for r in SW["rows"]}
        vals += [("我们的 500", 1 / sw_rows[500]["ratio"]), ("我们的 4096", 1 / sw_rows[4096]["ratio"])]
        ymax, ylab = 4.0, "每个 token 平均几个字节 (越高越省)"
        title = "英文 (莎士比亚验证集): 在同类文本上训练的 4096 词表, 追平了大模型的分词器"
        sub = "只画能无损还原的分词器 (BERT-en 会转小写, BERT-zh 遇到部分英文字符输出 [UNK], 不计入)"
    else:
        vals = [(SHORT[r["repo"]], r["zh_chars"] / r["zh_tokens"]) for r in rows]
        zh = EX["chinese"]["en_tok_on_zh"]
        vals += [("我们的 500", zh["chars"] / zh["tokens"])]
        ymax, ylab = 2.0, "每个 token 平均几个汉字 (越高越省)"
        title = None                                 # 排序后再按数据写标题
        sub = "只画能无损还原的分词器 (BERT-en、T5 会把汉字变成 [UNK]/&lt;unk&gt;, 不计入)"
    vals.sort(key=lambda v: v[1])
    if title is None:
        title = f"中文 (自写的 193 字): 最省的 {vals[-1][0]} 是最费的 {vals[0][0]} 的 {vals[-1][1] / vals[0][1]:.1f} 倍"
    c = Chart(900, 400, left=80, right=30, bottom=80)
    c.scales(-0.5, len(vals) - 0.5, 0, ymax)
    ticks = [i * 0.5 for i in range(int(ymax / 0.5) + 1)]
    c.axes([], ticks, "", ylab, yfmt=lambda v: f"{v:.1f}")
    # 单系列柱状图: 逐根画 (Chart.bars 按系列配色, 这里要让"我们的"单独用第二种颜色); 顶部圆角
    colors = [SERIES[1] if name.startswith("我们") else SERIES[0] for name, _ in vals]
    slot = (c.X(1) - c.X(0)) * 0.7
    for i, (name, v) in enumerate(vals):
        x, y_top, y_base = c.X(i) - slot / 2, c.Y(v), c.Y(0)
        r = 4
        d = (f"M{x:.1f},{y_base:.1f} L{x:.1f},{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} "
             f"L{x + slot - r:.1f},{y_top:.1f} Q{x + slot:.1f},{y_top:.1f} {x + slot:.1f},{y_top + r:.1f} "
             f"L{x + slot:.1f},{y_base:.1f} Z")
        c.add(f'<path d="{d}" fill="{colors[i]}"/>')
        c.text(c.X(i), y_top - 9, f"{v:.2f}", 11, INK2)
        c.text(c.X(i), c.y0 + 16, name, 11, INK2)
    c.save(out(f"chart_anatomy_{lang}.svg"), title, sub)


if __name__ == "__main__":
    chart_vocab_ratio()
    chart_merge_freq()
    chart_train_time()
    chart_anatomy("en")
    chart_anatomy("zh")
