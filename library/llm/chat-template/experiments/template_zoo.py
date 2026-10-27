"""
template_zoo.py — 同一段对话, 不同模型家族的模板各拼成什么样 (讲义第 8 节)

只下载每个模型的模板文件 (tokenizer_config.json, 有的还有 chat_template.jinja, 都是几 KB 到几百 KB),
不下载权重。渲染用 transformers 自带的 render_jinja_template, 不加载分词器。

    HF_ENDPOINT=https://hf-mirror.com python experiments/template_zoo.py

结果写进 results/template_zoo.json (每个模型渲染出的完整文字), 讲义里的对照表由它整理而来。
需要申请权限的模型 (Llama、Gemma 等) 下载不到时记为 gated, 不影响其它模型。
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "template_zoo.json")

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-V3",
    "microsoft/phi-4",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "zai-org/GLM-4.5",
    "openai/gpt-oss-20b",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-3-1b-it",
]

CONV = [
    {"role": "system", "content": "你是小车。"},
    {"role": "user", "content": "往前走 0.5 米"},
    {"role": "assistant", "content": "好的, 已经走了 0.5 米。"},
    {"role": "user", "content": "再转个身"},
]


def load_template(repo):
    """返回 (模板字符串, tokenizer_config 字典)。新模型把模板单独放在 chat_template.jinja 里。"""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    cfg = json.load(open(hf_hub_download(repo, "tokenizer_config.json"), encoding="utf-8"))
    tpl = cfg.get("chat_template")
    if not tpl:
        try:
            tpl = open(hf_hub_download(repo, "chat_template.jinja"), encoding="utf-8").read()
        except EntryNotFoundError:
            tpl = None
    if isinstance(tpl, list):                       # 少数模型存了好几个模板, 取名为 default 的
        tpl = next(t["template"] for t in tpl if t["name"] == "default")
    return tpl, cfg


def token_str(v):
    return v.get("content") if isinstance(v, dict) else v


def main():
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    from transformers.utils.chat_template_utils import render_jinja_template

    rows = []
    for repo in MODELS:
        row = {"repo": repo}
        try:
            tpl, cfg = load_template(repo)
            if not tpl:
                row["status"] = "no-template"
            else:
                specials = {k: token_str(cfg.get(k)) for k in ("bos_token", "eos_token", "pad_token", "unk_token")}
                specials = {k: v for k, v in specials.items() if v}
                text = render_jinja_template([CONV], chat_template=tpl, add_generation_prompt=True, **specials)
                text = text[0][0] if isinstance(text, tuple) else text
                row.update(status="ok", text=text if isinstance(text, str) else text[0],
                           eos_token=specials.get("eos_token"), bos_token=specials.get("bos_token"),
                           template_chars=len(tpl))
        except GatedRepoError:
            row["status"] = "gated"
        except RepositoryNotFoundError:
            row["status"] = "not-found"
        except Exception as e:                       # 模板里用了本机 transformers 不支持的写法等
            row["status"] = f"error: {type(e).__name__}: {str(e)[:120]}"
        print(f"  {row['status'][:40]:<40} {repo}")
        rows.append(row)

    quirks = deepseek_tool_quirks(render_jinja_template)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"conversation": CONV, "models": rows, "deepseek_v3_tool_quirks": quirks}, f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(OUT, os.path.dirname(HERE)))
    mistral_golden(render_jinja_template)


def _render(render_jinja_template, tpl, msgs, **kw):
    text = render_jinja_template([msgs], chat_template=tpl, **kw)
    return text[0][0] if isinstance(text, tuple) else text


def deepseek_tool_quirks(render_jinja_template):
    """讲义第 8.3 节: DeepSeek-V3 模板在工具调用上的三处不一致, 渲染出来留证。"""
    tpl, _ = load_template("deepseek-ai/DeepSeek-V3")

    def call(name, args):
        return {"type": "function", "function": {"name": name, "arguments": args}}

    cases = {
        "一次调用一个工具": [{"role": "user", "content": "看"},
                     {"role": "assistant", "content": None, "tool_calls": [call("look", "{}")]},
                     {"role": "tool", "content": "A"}, {"role": "assistant", "content": "没看到"}],
        "一次调用两个工具": [{"role": "user", "content": "看"},
                     {"role": "assistant", "content": None, "tool_calls": [call("look", "{}"), call("walk", '{"d": 1}')]},
                     {"role": "tool", "content": "A"}, {"role": "tool", "content": "B"}, {"role": "assistant", "content": "好了"}],
        "两轮各调用一次": [{"role": "user", "content": "a"},
                    {"role": "assistant", "content": None, "tool_calls": [call("look", "{}")]},
                    {"role": "tool", "content": "A"}, {"role": "user", "content": "b"},
                    {"role": "assistant", "content": None, "tool_calls": [call("walk", "{}")]}],
    }
    out = {name: _render(render_jinja_template, tpl, m, add_generation_prompt=False, bos_token="<｜begin▁of▁sentence｜>")
           for name, m in cases.items()}
    for name, text in out.items():
        print(f"  DeepSeek-V3 {name}: {text!r}")
    return out


def mistral_golden(render_jinja_template):
    """练习 3 的标准答案: Mistral-7B-Instruct-v0.3 模板 (不带工具) 渲染的对话, 以及它拒绝的对话。"""
    import random
    tpl, cfg = load_template("mistralai/Mistral-7B-Instruct-v0.3")
    texts = ["你好", "往前走 0.5 米", "  前后有空格  ", "多行\n文字", "Hello!", "\n换行开头"]
    rng = random.Random(0)
    cases = []
    for i in range(40):
        msgs = []
        if rng.random() < 0.5:
            msgs.append({"role": "system", "content": rng.choice(texts)})
        for k in range(rng.randint(1, 3)):
            msgs.append({"role": "user", "content": rng.choice(texts)})
            if k < 2 and rng.random() < 0.8:
                msgs.append({"role": "assistant", "content": rng.choice(texts)})
        cases.append(msgs)
    cases += [  # 不合法的: 角色没有交替, 或第一条不是 user
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        [{"role": "assistant", "content": "a"}],
        [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}],
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "assistant", "content": "c"}],
    ]
    rows = []
    for msgs in cases:
        try:
            rows.append({"messages": msgs, "expected": _render(render_jinja_template, tpl, msgs,
                                                               bos_token=cfg["bos_token"], eos_token=cfg["eos_token"])})
        except Exception as e:                       # 模板里的 raise_exception
            rows.append({"messages": msgs, "error": str(e)[:200]})
    path = os.path.join(HERE, "results", "mistral_golden.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": "mistralai/Mistral-7B-Instruct-v0.3", "cases": rows}, f, ensure_ascii=False, indent=1)
    n_err = sum("error" in r for r in rows)
    print(f"写入 results/mistral_golden.json: {len(rows) - n_err} 段正常, {n_err} 段被模板拒绝")


if __name__ == "__main__":
    main()
