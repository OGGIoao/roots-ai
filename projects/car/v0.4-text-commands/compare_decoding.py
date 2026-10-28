"""
compare_decoding.py — 汇总 run_decoding_suite.py 的结果 (知识库 llm/decoding 讲义第 8.3 节)

  - 复现检查: 默认设置重跑一遍, 和已公开的结果逐局比较 (成功与否、工具调用次数、最终距离、第一轮回复是否逐字相同)
  - 各设置的成功数、失败分类 (分类规则同 analyze.py)
  - 关掉重复惩罚后, 哪几局的结果变了

    python compare_decoding.py      # 只读 results/, 不跑模型
结果写进 results/decoding_compare.json。
"""

import json
import os
from collections import Counter

from analyze import OUTCOMES, classify

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
P = "local-Qwen-Qwen2.5-1.5B-Instruct_"
FILES = {
    "已公开: 贪心 + 自带重复惩罚 1.1": P + "skill_safe_n30.json",
    "重跑: 同样设置": P + "skill_safe_n30_rerun.json",
    "贪心, 关掉重复惩罚 (1.0)": P + "skill-rep1_safe_n30.json",
    "采样 (Qwen 自带设置), 种子 0": P + "skill-sample-s0_safe_n30.json",
    "采样 (Qwen 自带设置), 种子 1": P + "skill-sample-s1_safe_n30.json",
    "采样 (Qwen 自带设置), 种子 2": P + "skill-sample-s2_safe_n30.json",
}


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def first_reply(ep):
    return ep["turns"][0]["raw"] if ep["turns"] and "raw" in ep["turns"][0] else (
        ep["turns"][0]["content"] + json.dumps(ep["turns"][0]["calls"], ensure_ascii=False) if ep["turns"] else "")


def main():
    data = {k: load(v) for k, v in FILES.items() if os.path.exists(os.path.join(RES, v))}
    res = {"settings": {}}
    for k, d in data.items():
        eps = d["episodes"]
        outcomes = Counter(classify(ep, "skill") for ep in eps)
        res["settings"][k] = {"file": FILES[k], "success": sum(ep["success"] for ep in eps), "n": len(eps),
                              "outcomes": {o: outcomes.get(o, 0) for o in OUTCOMES if outcomes.get(o)},
                              "collision_episodes": sum(ep["collisions"] > 0 for ep in eps),
                              "wall_time_s": d.get("wall_time_s")}
        print(f"{k:<34} 成功 {res['settings'][k]['success']:>2}/{len(eps)}  {res['settings'][k]['outcomes']}")

    pub, rer = data.get("已公开: 贪心 + 自带重复惩罚 1.1"), data.get("重跑: 同样设置")
    if pub and rer:
        same = []
        for a, b in zip(pub["episodes"], rer["episodes"]):
            same.append({"seed": a["seed"], "success_same": a["success"] == b["success"],
                         "tool_calls_same": a["tool_calls"] == b["tool_calls"],
                         "distance_same": a["final_distance_m"] == b["final_distance_m"],
                         "first_reply_same": first_reply(a) == first_reply(b)})
        res["rerun_check"] = {k: sum(r[k] for r in same) for k in ("success_same", "tool_calls_same", "distance_same",
                                                                    "first_reply_same")}
        res["rerun_check"]["n"] = len(same)
        res["rerun_check"]["differing_seeds"] = [r["seed"] for r in same if not all(
            r[k] for k in ("success_same", "tool_calls_same", "distance_same"))]
        print("复现检查:", res["rerun_check"])

    rep = data.get("贪心, 关掉重复惩罚 (1.0)")
    if pub and rep:
        changed = [{"seed": a["seed"], "target": a["target"], "with_penalty": classify(a, "skill"),
                    "without_penalty": classify(b, "skill")}
                   for a, b in zip(pub["episodes"], rep["episodes"]) if a["success"] != b["success"]]
        res["penalty_off_changed"] = changed
        print("关掉重复惩罚后结果变了的局:", changed)

    with open(os.path.join(RES, "decoding_compare.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("写入 results/decoding_compare.json")


if __name__ == "__main__":
    main()
