"""
make_diagrams.py — 对话模板讲义的结构图 (SVG, 只用标准库; 数据来自 ../experiments/results/)

  fig_pipeline.svg      全景: 消息 → 模板 → 文字 → token → 模型 → token → 文字 → 解析回消息           (第 1 节)
  fig_render.svg        一段完整对话被拼成的文字; 蓝底是模型自己要写的部分                            (第 3、9 节)
  fig_tokens.svg        两条消息的最小例子切成的 17 个 token, 特殊标记各占 1 个                         (第 3 节)
  fig_tool_flow.svg     一次工具调用的来回: 说明书 → 模型写调用 → 程序执行 → 结果送回 → 模型回答         (第 7 节)
  fig_injection.svg     用户文字里写特殊标记: 默认分词 vs 当普通字符切开                              (第 10 节)
  fig_spans.svg         token 级的 loss 掩码: 哪些 token 算 loss                                     (第 9 节)
  fig_no_template.svg   同一个问题, 不套模板 / 套错模板 / 套对模板, 模型分别写了什么                   (第 2 节)
  fig_zoo.svg           同一段对话, 四个家族的模板拼出来的样子                                         (第 11 节)
  fig_prefix_break.svg  Qwen3: 这一轮模型看到并写出的文字, 到下一轮变了                                (专题 B)
  fig_think.svg         会思考的模型: 关掉思考时模板替模型写了什么                                     (专题 C)

运行: python figures/make_diagrams.py
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(UNIT)))
sys.path.insert(0, ROOT)
sys.path.insert(0, UNIT)

from chat_template import render_with_spans  # noqa: E402
from common.svgfig import (INK, INK2, KV_FILL, KV_LINE, MUTED, OP_FILL, OP_LINE, Q_FILL, Q_LINE)  # noqa: E402
from common.svgfig import Svg as _Svg  # noqa: E402


class Svg(_Svg):
    """所有文字自动把 < > 转义 (图里经常出现 <|im_start|> 这类标记, 不转义 SVG 就坏了)。"""

    def text(self, x, y, s, *a, **k):
        super().text(x, y, s.replace("<", "&lt;").replace(">", "&gt;"), *a, **k)
from examples import CONVERSATION  # noqa: E402

GREEN_FILL, GREEN_LINE = "#dcf3e9", "#1baf7a"
MONO = "Consolas,'Courier New',monospace"
SPECIAL = re.compile(r"(<\|[a-z_]+\|>|<tool_call>|</tool_call>|<think>|</think>|/nothink|<｜[^｜]+｜>|\[/?INST\]|<s>|</s>|\[gMASK\]|<sop>)")
RES = os.path.join(UNIT, "experiments", "results")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def out(name):
    return os.path.join(HERE, name)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mono_line(s, x, y, line, size=12, special_color=KV_LINE):
    """一行等宽文字, 特殊标记用橙色加粗。"""
    parts = SPECIAL.split(line)
    spans = "".join(
        f'<tspan fill="{special_color}" font-weight="bold">{esc(p)}</tspan>' if SPECIAL.fullmatch(p)
        else f'<tspan fill="{INK}">{esc(p)}</tspan>' for p in parts if p)
    s.items.append(f'<text x="{x}" y="{y}" font-size="{size}" font-family="{MONO}" dominant-baseline="middle">'
                   f'{spans or " "}</text>')


def show_nl(t):
    return t.replace("\n", "↵")


def token_row(s, x, y, toks, max_w, special=lambda t: False, fill=None, size=11, h=22):
    """把一串 token 画成一排小方框, 超宽换行; 返回最后的 y。fill(i, t) 可以指定底色。"""
    cx, cy = x, y
    for i, t in enumerate(toks):
        label = show_nl(t) if t.strip() or t == "" else ("␣" * len(t) if "\n" not in t else show_nl(t))
        w = 10 + 7.2 * sum(2 if ord(c) > 0x2E80 else 1 for c in label)
        if cx + w > x + max_w:
            cx, cy = x, cy + h + 6
        f = fill(i, t) if fill else (KV_FILL if special(t) else "#ffffff")
        stroke = KV_LINE if special(t) else MUTED
        s.rect(cx, cy, w, h, f, stroke, rx=3, sw=0.8)
        s.items.append(f'<text x="{cx + w / 2:.1f}" y="{cy + h / 2 + 1:.1f}" font-size="{size}" font-family="{MONO}" '
                       f'fill="{KV_LINE if special(t) else INK}" text-anchor="middle" dominant-baseline="middle">{esc(label)}</text>')
        cx += w + 4
    return cy + h


def fig_pipeline():
    s = Svg(940, 300)
    s.text(20, 22, "对话模板在整条链路里的位置: 程序这边是一条条消息, 模型那边只有一串 token", 15, INK, "start", "bold")
    s.text(20, 42, "去的路上用模板把消息拼成文字 (本单元), 回来的路上把模型写的文字解析回消息 (Agent 领域的 tool-calling)", 12, INK2, "start")
    boxes = [("消息列表", "[{role, content}, …]", OP_FILL, OP_LINE), ("对话模板", "chat_template", Q_FILL, Q_LINE),
             ("一整段文字", "<|im_start|>user …", OP_FILL, OP_LINE), ("分词器", "tokenizer (bpe 单元)", OP_FILL, OP_LINE),
             ("token 序列", "[151644, 872, …]", OP_FILL, OP_LINE)]
    for i, (a, b, f, l) in enumerate(boxes):
        x = 20 + i * 184
        s.box(x, 78, 160, 50, a, f, l, 14)
        s.text(x + 80, 142, b, 10, MUTED)
        if i:
            s.line(x - 24, 103, x - 2, 103)
    s.box(760, 186, 160, 50, "模型", GREEN_FILL, GREEN_LINE, 14)
    s.text(840, 250, "一次只往后写一个 token", 10, MUTED)
    s.path("M840,128 L840,184")
    back = [("新 token", "…, 151645 (<|im_end|>)"), ("解码成文字", "decode"), ("解析", "找 <tool_call>、去掉标记"),
            ("新的一条消息", "{role: assistant, …}")]
    for i, (a, b) in enumerate(back):
        x = 572 - i * 184
        s.box(x, 186, 160, 50, a, OP_FILL, OP_LINE, 13)
        s.text(x + 80, 250, b, 10, MUTED)
        s.line(x + 184 + (0 if i else 4), 211, x + 162, 211)
    s.path("M20,211 C0,211 0,103 18,103")
    s.text(26, 276, "新消息接到消息列表末尾, 进入下一轮", 11, INK2, "start")
    s.save(out("fig_pipeline.svg"))


def fig_render():
    text, spans = render_with_spans(CONVERSATION)
    lines, pos = [], 0
    for ln in text.split("\n"):
        mid = pos + len(ln) // 2
        lines.append((ln, any(a <= mid < b for a, b in spans) or any(a <= pos < b for a, b in spans)))
        pos += len(ln) + 1
    if lines and lines[-1][0] == "":
        lines.pop()
    H = 130 + 26 * len(lines)
    s = Svg(960, H)
    s.text(20, 22, "一段完整对话被模板拼成的文字: 提问 → 调用工具 → 工具返回 → 回答", 15, INK, "start", "bold")
    s.text(20, 42, "chat_template.render (Qwen2.5 的 ChatML 格式); 橙色: 词表里专门新增的标记; 蓝底: 模型自己要写的部分", 12, INK2, "start")
    colors = {"system": OP_FILL, "user": "#ffffff", "assistant": Q_FILL, "tool": GREEN_FILL}
    labels = ["system: 你是一个乐于助人的助手…", "user: 台北现在几度?", "assistant: 调用 get_weather(台北)",
              "tool: {temperature: 26, …}", "assistant: 台北现在 26 度, 多云。"]
    s.text(130, 74, "messages (5 条)", 13, INK, weight="bold")
    for i, (m, lab) in enumerate(zip(CONVERSATION, labels)):
        y = 90 + i * 44
        s.rect(20, y, 230, 32, colors[m["role"]], MUTED, rx=3, sw=0.8)
        s.text(30, y + 16, esc(lab), 11, INK, "start")
    s.line(258, 190, 302, 190)
    s.text(280, 178, "render", 10, INK2)
    x0, y0 = 312, 70
    s.text(x0, y0 + 4, "模型看到的文字 (一整段, 这里按换行折开)", 13, INK, "start", "bold")
    for i, (ln, tr) in enumerate(lines):
        y = y0 + 22 + i * 26
        if tr:
            s.rect(x0 - 4, y - 2, 630, 24, Q_FILL, Q_LINE, rx=2, sw=0.6)
        mono_line(s, x0, y + 11, ln, 12)
    s.text(20, H - 16, esc("<tool_response> 不是新增标记, 会被切成几个普通 token; 第一条 system 消息的位置是模板固定的"), 11, MUTED, "start")
    s.save(out("fig_render.svg"))


def fig_tokens():
    toks = [t for _, t in load("tokens.json")["tiny_tokens"]]
    ids = [i for i, _ in load("tokens.json")["tiny_tokens"]]
    s = Svg(940, 250)
    s.text(20, 22, "两条消息 (system: 你是助手。 user: 你好) 加生成提示, 一共 17 个 token", 15, INK, "start", "bold")
    s.text(20, 42, "Qwen2.5 分词器实测 (experiments/measure_tokens.py); 橙框是词表里新增的特殊标记, 每个只占 1 个 token, id 在下面", 12, INK2, "start")
    x, y = 20, 70
    for t, i in zip(toks, ids):
        sp = t.startswith("<|")
        label = show_nl(t)
        w = 12 + 7.5 * sum(2 if ord(c) > 0x2E80 else 1 for c in label)
        if x + w > 920:
            x, y = 20, y + 70
        s.rect(x, y, w, 26, KV_FILL if sp else "#ffffff", KV_LINE if sp else MUTED, rx=3, sw=0.9)
        s.items.append(f'<text x="{x + w / 2:.1f}" y="{y + 14}" font-size="12" font-family="{MONO}" fill="{KV_LINE if sp else INK}" '
                       f'text-anchor="middle" dominant-baseline="middle">{esc(label)}</text>')
        s.text(x + w / 2, y + 40, str(i), 9, MUTED)
        x += w + 6
    s.text(20, 222, "规律: 每条消息 = <|im_start|> + 角色名 + ↵ + 内容 + <|im_end|> + ↵; 最后的 <|im_start|> assistant ↵ 是 \"轮到你了\"", 12, INK2, "start")
    s.save(out("fig_tokens.svg"))


def fig_tool_flow():
    s = Svg(940, 380)
    s.text(20, 22, "一次工具调用的来回: 模型不直接 \"做\" 任何事, 它只写下 \"我要调用什么\", 由程序去做", 15, INK, "start", "bold")
    s.text(20, 42, "左列是模型看到 / 写出的文字 (第 7 节逐段讲), 右列是程序做的事", 12, INK2, "start")
    steps = [
        ("system 里的说明书", '# Tools … <tools>{"name": "get_weather", …}</tools>', "程序把工具的 JSON Schema 交给模板, 模板拼进 system", OP_FILL, OP_LINE),
        ("模型写出调用", '<tool_call>{"name": "get_weather", "arguments": {"city": "台北"}}</tool_call>', "程序从文字里找出 <tool_call>, 解析 JSON", Q_FILL, Q_LINE),
        ("工具结果送回", '<|im_start|>user <tool_response>{"temperature": 26}</tool_response>', "程序真正去查天气, 把结果当成一条 role=tool 的消息", GREEN_FILL, GREEN_LINE),
        ("模型用结果回答", "台北现在 26 度, 多云。<|im_end|>", "没有 <tool_call>: 这一轮结束, 回答给用户", Q_FILL, Q_LINE),
    ]
    for i, (title, txt, side, f, l) in enumerate(steps):
        y = 70 + i * 74
        s.rect(20, y, 560, 54, f, l, rx=4)
        s.text(32, y + 15, title, 12, INK, "start", "bold")
        mono_line(s, 32, y + 36, txt, 11)
        s.rect(610, y, 310, 54, "#ffffff", MUTED, rx=4, sw=0.8)
        s.text(622, y + 27, esc(side), 11, INK2, "start")
        s.line(580, y + 27, 608, y + 27, arrow=False, dash=True)
        if i < 3:
            s.line(300, y + 54, 300, y + 72)
    s.save(out("fig_tool_flow.svg"))


def fig_injection():
    inj = load("tokens.json")["injection"]
    s = Svg(940, 300)
    s.text(20, 22, "用户文字里写了 \"<|im_end|>…<|im_start|>system\": 默认分词会把它当成真的特殊标记", 15, INK, "start", "bold")
    s.text(20, 42, "同一段用户输入的两种分法 (experiments/measure_tokens.py); 橙框 = 特殊 token, 在模型眼里就是结构", 12, INK2, "start")
    s.text(20, 80, f"默认分词: {inj['user_text_tokens_default']} 个 token, 其中 2 个是真的特殊标记 → 模型以为用户消息到 \"忽略上面\" 就结束了, 后面是一条 system", 12, INK, "start", "bold")
    y = token_row(s, 20, 94, inj["user_pieces_default"], 900, special=lambda t: t.startswith("<|"))
    s.text(20, y + 34, f"split_special_tokens=True: {inj['user_text_tokens_split']} 个普通 token, 没有一个是结构 → 它只是用户打的一串字符", 12, INK, "start", "bold")
    token_row(s, 20, y + 48, inj["user_pieces_split"], 900, special=lambda t: False, fill=lambda i, t: GREEN_FILL)
    s.save(out("fig_injection.svg"))


def fig_spans():
    m = load("token_boundaries.json")["mask_example"]
    s = Svg(940, 230)
    s.text(20, 22, f"token 级的 loss 掩码: {len(m)} 个 token 里, 只有模型自己写的 {sum(x[1] for x in m)} 个算 loss", 15, INK, "start", "bold")
    s.text(20, 42, "render_with_spans 的字符区间 + 分词器的 offset_mapping 换算 (experiments/token_boundaries.py); 蓝底 = 算 loss, 白底 = 题目", 12, INK2, "start")
    token_row(s, 20, 66, [t for t, _ in m], 900, special=lambda t: t.startswith("<|") or t.endswith("tool_call>"),
              fill=lambda i, t: Q_FILL if m[i][1] else "#ffffff", size=10, h=20)
    s.text(20, 212, "两处 <|im_end|> 都算: 模型要学会自己停下来。<|im_start|>assistant↵ 不算: 那是程序替它写好的生成提示", 12, INK2, "start")
    s.save(out("fig_spans.svg"))


def fig_no_template():
    r = load("no_template.json")
    row = next(x for x in r["rows"] if x["question"] == "法国的首都是哪里?")
    s = Svg(940, 330)
    s.text(20, 22, "同一个对话模型, 同一个问题 \"法国的首都是哪里?\": 怎么喂, 决定了它写什么", 15, INK, "start", "bold")
    s.text(20, 42, f"Qwen2.5-0.5B-Instruct, CPU, 贪心, 最多 60 个 token (experiments/no_template.py); 5 个问题里自己停下来的次数: "
           f"不套 {r['summary']['raw']['stopped']}/5, 套错 {r['summary']['wrong']['stopped']}/5, 套对 {r['summary']['right']['stopped']}/5", 12, INK2, "start")
    panels = [("不套模板: 直接喂问题文字", row["raw"], OP_FILL, OP_LINE),
              ("套错模板: [INST] … [/INST] (Mistral 的格式)", row["wrong"], KV_FILL, KV_LINE),
              ("套对模板: Qwen 的 ChatML + 生成提示", row["right"], Q_FILL, Q_LINE)]
    for i, (title, res, f, l) in enumerate(panels):
        y = 66 + i * 86
        s.rect(20, y, 900, 74, f, l, rx=4)
        s.text(32, y + 16, title + ("    → 自己停了" if res["stopped"] else f"    → 写满 {res['n_tokens']} 个 token 还没停"), 12, INK, "start", "bold")
        text = show_nl(res["output"])
        for k in range(2):
            seg = text[k * 62:(k + 1) * 62]
            if seg:
                mono_line(s, 32, y + 38 + k * 18, seg, 11)
    s.save(out("fig_no_template.svg"))


def fig_zoo():
    z = {m["repo"]: m for m in load("template_zoo.json")["models"]}
    picks = [("Qwen2.5 (ChatML)", "Qwen/Qwen2.5-1.5B-Instruct"), ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
             ("Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"), ("gpt-oss (harmony)", "openai/gpt-oss-20b")]
    s = Svg(960, 470)
    s.text(20, 22, "同一段对话 (system + 提问 + 回答 + 再提问), 四个家族的模板拼出来完全不一样", 15, INK, "start", "bold")
    s.text(20, 42, "experiments/template_zoo.py 读各模型发布的模板文件渲染; 橙色是各家的特殊标记 (gpt-oss 只画后半段, 前面是它自动生成的系统说明)", 12, INK2, "start")
    y = 64
    for title, repo in picks:
        text = z[repo]["text"]
        if "gpt-oss" in repo:
            text = text[text.index("<|start|>developer"):]
        lines = []
        for raw in text.split("\n"):
            while len(raw) > 100:
                lines.append(raw[:100])
                raw = raw[100:]
            lines.append(raw)
        lines = [ln for ln in lines if ln != ""][:7]
        h = 26 + 17 * len(lines)
        s.rect(20, y, 920, h, "#ffffff", MUTED, rx=4, sw=0.8)
        s.text(32, y + 13, title, 12, INK, "start", "bold")
        for k, ln in enumerate(lines):
            mono_line(s, 32, y + 32 + k * 17, ln, 11)
        y += h + 10
    s.save(out("fig_zoo.svg"))


def fig_prefix_break():
    e = load("prefix_property.json")["Qwen/Qwen3-8B"]["reasoning"]["first_mismatch"]
    s = Svg(960, 250)
    s.text(20, 22, "Qwen3: 这一轮模型看到并写出的文字, 到了下一轮的提示里变了样", 15, INK, "start", "bold")
    s.text(20, 42, "experiments/prefix_property.py; 上一轮回复里的 <think>…</think> 在下一轮被模板删掉 → 下一轮不能整段复用这一轮算好的缓存", 12, INK2, "start")
    s.rect(20, 66, 920, 70, Q_FILL, Q_LINE, rx=4)
    s.text(32, 82, "这一轮: 提示 + 模型写的回复 (带思考过程)", 12, INK, "start", "bold")
    this = e["this_turn"]
    this = this[this.rfind("<|im_start|>user"):]
    mono_line(s, 32, 110, show_nl(this)[-100:], 11)
    s.rect(20, 152, 920, 70, KV_FILL, KV_LINE, rx=4)
    s.text(32, 168, "下一轮的提示: 同一个位置, 思考过程没了", 12, INK, "start", "bold")
    nxt = e["next_prompt"]
    mono_line(s, 32, 196, show_nl(nxt[:100]), 11)
    s.save(out("fig_prefix_break.svg"))


def fig_think():
    t = load("thinking_templates.json")
    rows = [("Qwen3, 默认", t["Qwen/Qwen3-8B"][0]["tail"]), ("Qwen3, enable_thinking=False", t["Qwen/Qwen3-8B"][1]["tail"]),
            ("GLM-4.5, 默认", t["zai-org/GLM-4.5"][0]["tail"]), ("GLM-4.5, enable_thinking=False", t["zai-org/GLM-4.5"][1]["tail"])]
    s = Svg(960, 330)
    s.text(20, 22, "关掉思考, 模板替模型写好一个空的 <think></think>: 模型 \"以为\" 自己已经想完了", 15, INK, "start", "bold")
    s.text(20, 42, "同一个问题 \"3 乘 9 是多少?\" 的生成提示结尾 (experiments/thinking_templates.py); gpt-oss 的开关不在这里, 在系统消息的 \"Reasoning: low / medium / high\" 一行", 12, INK2, "start")
    for i, (title, tail) in enumerate(rows):
        y = 66 + i * 62
        f, l = (KV_FILL, KV_LINE) if "False" in title else (OP_FILL, OP_LINE)
        s.rect(20, y, 920, 50, f, l, rx=4)
        s.text(32, y + 14, title, 12, INK, "start", "bold")
        mono_line(s, 32, y + 35, "…" + show_nl(tail[-92:]), 11)
    s.save(out("fig_think.svg"))


if __name__ == "__main__":
    for f in (fig_pipeline, fig_render, fig_tokens, fig_tool_flow, fig_injection, fig_spans, fig_no_template, fig_zoo,
              fig_prefix_break, fig_think):
        f()
