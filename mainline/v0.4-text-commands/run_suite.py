"""
run_suite.py — 依次跑讲义用到的全部评测组合 (已经有结果文件的跳过, 中断后重跑会续上)。

    python run_suite.py              # 本地模型 (讲义主数据)
    python run_suite.py --api        # 再加上云端 API 对照 (需要 .env 里的 Key, 会产生少量费用)

每组 30 个任务 (场景 seed 0~29)。本地模型用贪心解码, 结果可复现。
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

LOCAL = [
    ("Qwen/Qwen2.5-1.5B-Instruct", ["chat", "plan", "loop", "step", "skill", "skill-forced"]),
    ("Qwen/Qwen2.5-0.5B-Instruct", ["plan", "loop", "step", "skill", "skill-forced"]),
    ("Qwen/Qwen2.5-3B-Instruct", ["plan", "loop", "step", "skill", "skill-forced"]),
]
NOSAFETY = [("Qwen/Qwen2.5-1.5B-Instruct", "step")]       # 讲义第 4 节: 关掉安全层的对照
API = [("deepseek", ["plan", "loop", "step", "skill"])]       # API 的 skill 一般不需要强制, 需要时加 "skill-forced"
N = 30


def name_of(backend, mode, safety):
    return "_".join([backend.replace(":", "-").replace("/", "-"), mode, "safe" if safety else "nosafe", f"n{N}"])


def run(backend, mode, safety=True):
    name = name_of(backend, mode, safety)
    if os.path.exists(os.path.join(HERE, "results", f"{name}.json")):
        print("跳过 (已有结果):", name)
        return
    base_mode, forced = mode.removesuffix("-forced"), mode.endswith("-forced")
    cmd = [sys.executable, os.path.join(HERE, "evaluate.py"), "--backend", backend, "--mode", base_mode, "--n", str(N), "--name", name]
    if forced:
        cmd.append("--force-first-tool")       # skill-forced: 第一轮强制调用工具 (讲义第 6 节)
    if not safety:
        cmd.append("--no-safety")
    print("运行:", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--only", help="只跑名字里包含这个字符串的组合, 例如 0.5B")
    args = ap.parse_args()
    jobs = [(f"local:{m}", mode, True) for m, modes in LOCAL for mode in modes]
    jobs += [(f"local:{m}", mode, False) for m, mode in NOSAFETY]
    if args.api:
        jobs += [(f"api:{p}", mode, True) for p, modes in API for mode in modes]
    for backend, mode, safety in jobs:
        if args.only and args.only not in backend:
            continue
        run(backend, mode, safety)
