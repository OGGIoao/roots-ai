"""
evaluate.py — 在任务集上评测一个 "模型 + 模式" 组合 (讲义第 5 节: 只看"跑通一次"不可信)

每个任务: 一个随机场景 (sim.car2d.make_world) + 一句中文指令 ("去红杯子那里, 停在它跟前")。
成功 = 车身离目标不超过 0.25 m 且全程没有碰撞 (sim.car2d.reached)。

用法:
    python evaluate.py --backend local:Qwen/Qwen2.5-1.5B-Instruct --mode loop --n 30
    python evaluate.py --backend local:Qwen/Qwen2.5-1.5B-Instruct --mode loop --no-safety --n 30
    python evaluate.py --backend api:deepseek --mode loop --n 30          # 需要 .env 里的 API Key

结果写入 results/<名字>.json: 每个任务的结果、对话记录、真实轨迹, 以及汇总。
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, os.path.join(ROOT, "library", "agent", "agent-loop"))   # Agent 循环本体在知识库里

from agent import make_tasks, run_episode  # noqa: E402
from car2d import Car2D, make_world, reached  # noqa: E402


def make_llm(backend):
    kind, _, name = backend.partition(":")
    if kind == "local":
        from llm import LocalLLM
        return LocalLLM(name)
    if kind == "api":
        from llm import APILLM
        return APILLM(provider=name or None)
    raise ValueError(f"未知后端 {backend!r}, 应为 local:<模型> 或 api:<供应商>")


def summarize(rows):
    n = len(rows)
    ok = sum(r["success"] for r in rows)
    calls = sum(r["tool_calls"] for r in rows)
    return {
        "n": n,
        "success": ok,
        "success_rate": ok / n,
        "collision_episodes": sum(r["collisions"] > 0 for r in rows),
        "mean_tool_calls": calls / n,
        "invalid_call_rate": sum(r["invalid_calls"] for r in rows) / calls if calls else 0.0,
        "bare_call_rate": sum(r["bare_calls"] for r in rows) / calls if calls else 0.0,
        "mean_latency_s": sum(r["latency_s"] for r in rows) / n,
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / n,
        "mean_completion_tokens": sum(r["completion_tokens"] for r in rows) / n,
        "ended_by": {k: sum(r["ended_by"] == k for r in rows) for k in ("answered", "plan_done", "max_turns")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--mode", choices=["chat", "plan", "loop", "step", "skill"], default="loop")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=0, help="任务集的起始场景编号")
    ap.add_argument("--max-turns", type=int, default=30, help="loop/step 模式最多问模型几轮 (step 模式下也就是最多几次工具调用)")
    ap.add_argument("--no-safety", action="store_true", help="关掉安全层 (只用于对照实验)")
    ap.add_argument("--force-first-tool", action="store_true", help="第一轮强制调用工具 (讲义第 6 节)")
    ap.add_argument("--name", help="结果文件名 (默认由参数拼成)")
    args = ap.parse_args()

    llm = make_llm(args.backend)
    toolset = None
    if args.mode == "skill":
        from skills import SKILL_TOOLSET
        toolset = SKILL_TOOLSET
    safety = not args.no_safety
    mode_tag = args.mode + ("-forced" if args.force_first_tool else "")
    name = args.name or "_".join([args.backend.replace(":", "-").replace("/", "-"), mode_tag,
                                  "safe" if safety else "nosafe", f"n{args.n}"])
    rows = []
    t_start = time.time()
    for i, task in enumerate(make_tasks(args.n, args.seed0)):
        world = make_world(task["seed"])
        car = Car2D(world, seed=task["seed"])
        rec = run_episode(llm, car, task, mode=args.mode, max_turns=args.max_turns, safety=safety,
                          force_first_tool=args.force_first_tool, toolset=toolset)
        row = {**task, "success": reached(car, task["target"]),
               "final_distance_m": round(car.distance_to(task["target"]), 3), "collisions": car.collisions,
               **{k: rec[k] for k in ("tool_calls", "invalid_calls", "bare_calls", "skipped_calls", "prompt_tokens",
                                      "completion_tokens", "ended_by")},
               "latency_s": round(rec["latency_s"], 3), "turns": rec["turns"],
               "trace": [(round(x, 3), round(y, 3)) for x, y in car.trace[::5]] + [tuple(round(v, 3) for v in car.trace[-1])]}
        rows.append(row)
        print(f"[{i + 1}/{args.n}] seed={task['seed']} {task['target']:<9} {'成功' if row['success'] else '失败'} "
              f"距离 {row['final_distance_m']:.2f} m  碰撞 {row['collisions']}  工具 {row['tool_calls']} "
              f"(被拦 {row['invalid_calls']}, 无标签 {row['bare_calls']})  {row['latency_s']:.1f} s  [{row['ended_by']}]", flush=True)
    summary = summarize(rows)
    out = {"backend": args.backend, "model": getattr(llm, "model_id", args.backend), "mode": args.mode,
           "force_first_tool": args.force_first_tool,
           "safety": safety, "max_turns": args.max_turns, "seed0": args.seed0,
           "wall_time_s": round(time.time() - t_start, 1), "summary": summary, "episodes": rows}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("写入", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
