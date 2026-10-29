"""
practice_car.py — 讲义 "实践: 在项目里用" 一节的数字: 小车项目 v0.4 的提示账本

主干讲义用通用小助手; 这里把同样的量法用到小车上:
  - 小车 v0.4 一次提示里, 模板标记、系统提示、5 个底层工具 (walk / turn / stop / look / get_pose) 各占多少 token
  - skill 模式 (只给 go_to + look 两个高层工具) 的提示有多长

只加载分词器 (CPU)。结果写进 results/practice_car.json。
    python experiments/practice_car.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(UNIT)))
sys.path.insert(0, UNIT)
sys.path.insert(0, os.path.join(ROOT, "library", "agent", "agent-loop"))
sys.path.insert(0, os.path.join(ROOT, "projects", "car", "v0.4-text-commands"))

from chat_template import render  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = os.path.join(HERE, "results", "practice_car.json")


def main():
    from transformers import AutoTokenizer
    from agent import SYSTEM_PROMPT
    from skills import SKILL_SCHEMAS, SKILL_SYSTEM_PROMPT
    from tools import TOOL_SCHEMAS

    tok = AutoTokenizer.from_pretrained(MODEL)
    n = lambda text: len(tok.encode(text))           # noqa: E731
    user = "去红杯子那里, 停在它跟前"

    base = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    plain = n(render(base, add_generation_prompt=True))
    full = n(render(base, tools=TOOL_SCHEMAS, add_generation_prompt=True))
    per_tool = {s["function"]["name"]: n(json.dumps(s, ensure_ascii=False)) for s in TOOL_SCHEMAS}
    res = {"model": MODEL, "budget": {
        "system_prompt": n(SYSTEM_PROMPT), "user": n(user), "template_markers": plain - n(SYSTEM_PROMPT) - n(user),
        "tools_block": full - plain, "tools_json_only": sum(per_tool.values()), "per_tool": per_tool,
        "total_no_tools": plain, "total_with_tools": full}}

    sk = [{"role": "system", "content": SKILL_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    sk_plain = n(render(sk, add_generation_prompt=True))
    sk_full = n(render(sk, tools=SKILL_SCHEMAS, add_generation_prompt=True))
    res["skill"] = {"n_tools": len(SKILL_SCHEMAS), "system_prompt": n(SKILL_SYSTEM_PROMPT),
                    "tools_block": sk_full - sk_plain, "total_with_tools": sk_full}
    print("小车底层工具:", res["budget"])
    print("小车 skill 模式:", res["skill"])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(OUT, UNIT))


if __name__ == "__main__":
    main()
