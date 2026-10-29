"""
prefix_property.py — 延伸专题 B: 多轮对话里, "前几轮渲染出的提示" 是不是 "整段对话渲染结果" 的前缀?

为什么关心: 推理时每一轮都要把整段对话重新喂给模型。如果第 k 轮的提示正好是第 k+1 轮提示的开头 (前缀),
前面那段已经算过的 KV 缓存就能直接复用 (前缀缓存, 见 LLM 领域的 prefix-caching 单元); 不是前缀的话, 每一轮都得从头算。

做法: 对 template_zoo.py 里的 8 个模型家族, 各用 200 段随机多轮对话 (user / assistant 交替)。对每个不是最后一条的
assistant 回复 (第 k 条) 检查:
    这一轮模型看到并写出的全部文字  render(前 k+1 条消息)          (提示 + 它自己的回复)
    是不是
    下一轮的提示                    render(前 k+2 条消息, 加生成提示) 的前缀
是前缀, 这一轮算过的 KV 缓存下一轮可以整段复用; 不是, 就得从不一样的地方开始重算。
两种对话各测一遍:
    plain      普通对话
    reasoning  assistant 消息带思考过程 (reasoning_content 字段, gpt-oss 用 thinking 字段)

只用模板文件 (template_zoo.py 第一次运行时已下载), 不加载分词器和模型, 几秒。
    python experiments/prefix_property.py
结果写进 results/prefix_property.json。
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from template_zoo import MODELS, load_template, token_str  # noqa: E402

OUT = os.path.join(HERE, "results", "prefix_property.json")
TEXTS = ["台北现在几度?", "高雄呢?", "谢谢", "3 乘 9 是多少", "好的", "Hello", "再说一遍"]
ANSWERS = ["台北现在 26 度。", "高雄 30 度。", "不客气。", "27。", "Hi!"]


def conversation(rng, reasoning):
    msgs = [{"role": "system", "content": "你是助手。"}] if rng.random() < 0.5 else []
    for _ in range(rng.randint(2, 4)):
        msgs.append({"role": "user", "content": rng.choice(TEXTS)})
        m = {"role": "assistant", "content": rng.choice(ANSWERS)}
        if reasoning:
            m["reasoning_content"] = "用户问的是" + rng.choice(["天气", "算数", "客套话"]) + "。"
            m["thinking"] = m["reasoning_content"]
        msgs.append(m)
    return msgs


def main():
    from transformers.utils.chat_template_utils import render_jinja_template

    def render(tpl, specials, msgs, gen):
        out = render_jinja_template([msgs], chat_template=tpl, add_generation_prompt=gen, **specials)
        return out[0][0] if isinstance(out, tuple) else out

    res = {}
    for repo in MODELS:
        try:
            tpl, cfg = load_template(repo)
        except Exception as e:                       # 需要申请权限的模型
            res[repo] = {"status": type(e).__name__}
            print(f"  跳过 {repo}: {type(e).__name__}")
            continue
        specials = {k: token_str(cfg.get(k)) for k in ("bos_token", "eos_token")}
        specials = {k: v for k, v in specials.items() if v}
        row = {}
        for variant in ("plain", "reasoning"):
            rng = random.Random(0)
            ok = total = 0
            example = None
            for _ in range(200):
                msgs = conversation(rng, variant == "reasoning")
                try:
                    full = render(tpl, specials, msgs, False)
                except Exception as e:
                    row[variant] = {"error": str(e)[:120]}
                    break
                for k, m in enumerate(msgs):
                    if m["role"] != "assistant" or k + 1 >= len(msgs):
                        continue
                    total += 1
                    this_turn = render(tpl, specials, msgs[:k + 1], False)
                    next_prompt = render(tpl, specials, msgs[:k + 2], True)
                    if next_prompt.startswith(this_turn):
                        ok += 1
                    elif example is None:
                        i = next((j for j, (a, b) in enumerate(zip(this_turn, next_prompt)) if a != b),
                                 min(len(this_turn), len(next_prompt)))
                        example = {"this_turn": this_turn[max(0, i - 80):i + 60],
                                   "next_prompt": next_prompt[max(0, i - 80):i + 60]}
            else:
                row[variant] = {"prefix_ok": ok, "total": total, "first_mismatch": example}
        res[repo] = row
        print(f"  {repo:<40} " + "  ".join(
            f"{v}: {r['prefix_ok']}/{r['total']}" if "total" in r else f"{v}: {r.get('error', '?')[:40]}" for v, r in row.items()))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(OUT, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
