"""
thinking_templates.py — 延伸专题 C: 会 "思考" 的模型, 模板多了什么

对 Qwen3、GLM-4.5、gpt-oss 三个家族, 同一个问题, 渲染生成提示 (模型从哪里开始写):
  - 不传任何开关 (模板默认)
  - Qwen3 / GLM-4.5: enable_thinking=False      (关掉思考)
  - gpt-oss: reasoning_effort = low / high       (思考多少)
看模板替模型写好了什么开头。只用模板文件, 几秒。

    python experiments/thinking_templates.py
结果写进 results/thinking_templates.json。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from template_zoo import load_template, token_str  # noqa: E402

OUT = os.path.join(HERE, "results", "thinking_templates.json")
MSGS = [{"role": "user", "content": "3 乘 9 是多少?"}]
VARIANTS = {
    "Qwen/Qwen3-8B": [{}, {"enable_thinking": False}],
    "zai-org/GLM-4.5": [{}, {"enable_thinking": False}],
    "openai/gpt-oss-20b": [{}, {"reasoning_effort": "low"}, {"reasoning_effort": "high"}],
}


def main():
    from transformers.utils.chat_template_utils import render_jinja_template
    res = {}
    for repo, variants in VARIANTS.items():
        tpl, cfg = load_template(repo)
        specials = {k: token_str(cfg.get(k)) for k in ("bos_token", "eos_token")}
        specials = {k: v for k, v in specials.items() if v}
        rows = []
        for kw in variants:
            out = render_jinja_template([MSGS], chat_template=tpl, add_generation_prompt=True, **specials, **kw)
            text = out[0][0] if isinstance(out, tuple) else out
            rows.append({"kwargs": kw, "text": text, "tail": text[-80:]})
            print(f"{repo:<20} {json.dumps(kw):<32} …{text[-70:]!r}")
        res[repo] = rows
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(OUT, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
