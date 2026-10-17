"""
overfit_check.py — 先看一条完整的验证曲线, 再决定对照实验比较哪个时刻

用 MHA 变体训练 3000 步 (是 compare.py 的 2 倍), 每 100 步记录 train / val loss,
结果写入 results/overfit_mha_3000.json。讲义第 9 节坑 2 和图 6 用的就是这份数据。

运行: python attention/overfit_check.py   (本机约 3 分钟)
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from compare import VARIANTS  # noqa: E402

from common.bench import BenchConfig, run  # noqa: E402

if __name__ == "__main__":
    result = run("MHA", VARIANTS["MHA"], BenchConfig(steps=3000, eval_every=100))
    out = os.path.join(HERE, "results", "overfit_mha_3000.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    best = min(result["curve"], key=lambda c: c[2])
    print(f"最低 val_loss {best[2]:.4f} @ 第 {best[0]} 步; 第 3000 步 val_loss {result['curve'][-1][2]:.4f}")
    print(f"结果写入 {out}")
