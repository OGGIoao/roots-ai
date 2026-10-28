"""
run_decoding_suite.py — 解码设置对 v0.4 的影响 (知识库 llm/decoding 讲义第 8.3 节的数据来源)

依次 (一次只跑一个, 显存约 4 GB) 在 skill 模式、30 个任务上跑:
  1. 默认设置复现 (贪心 + 模型自带的重复惩罚 1.1): 和已公开的结果逐局比对, 检查贪心能不能复现
  2. 关掉重复惩罚 (--rep-penalty 1.0)
  3. 采样, 用 Qwen2.5 自带的设置 (温度 0.7、top-k 20、top-p 0.8、重复惩罚 1.1), 种子 0 / 1 / 2

    python run_decoding_suite.py            # 约 15~20 分钟 (RTX 5060 Laptop)
"""

import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "local:Qwen/Qwen2.5-1.5B-Instruct"
RUNS = [
    ["--name", "local-Qwen-Qwen2.5-1.5B-Instruct_skill_safe_n30_rerun"],
    ["--rep-penalty", "1.0"],
    ["--sample", "--seed", "0"],
    ["--sample", "--seed", "1"],
    ["--sample", "--seed", "2"],
]

for extra in RUNS:
    cmd = [sys.executable, "evaluate.py", "--backend", MODEL, "--mode", "skill", "--n", "30", *extra]
    print(">>>", " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(r.returncode)
