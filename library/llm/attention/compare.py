"""
compare.py — 注意力线的对照实验: MHA / GQA / MQA / MLA

所有变体共用 base.py 的同一个解码器 (4 层, 256 维, 8 个查询头, 每头 32 维),
只替换注意力模块; 同样的数据、训练设置与随机种子。比较:
  - 最佳 val_loss (早停, 3 个种子的均值 ± 标准差)
  - 推理时每个 token 的 KV Cache
  - 参数量与训练速度

运行: python compare.py          (本机 RTX 5060 约 25 分钟)
      python compare.py --quick  (1 个种子, 少步数, 用来检查能不能跑)
结果写入 results/compare.json (每组的原始结果在 results/per_run/, 中断后重新运行会续跑)
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # base.py 与 variants.py 都在本目录

from base import Decoder  # noqa: E402
from variants import GroupedQueryAttention, MultiHeadLatentAttention  # noqa: E402

from dataclasses import replace  # noqa: E402

from common.bench import BenchConfig, print_table, run, summarize  # noqa: E402

DIM, LAYERS, HEADS = 256, 4, 8          # 每头 32 维


def decoder(make_attn):
    return lambda vocab, block: Decoder(vocab, block, DIM, LAYERS, make_attn)


VARIANTS = {
    #                          每层每 token 缓存的数
    "MHA":    decoder(lambda: GroupedQueryAttention(DIM, HEADS, 8)),                 # 2×8×32 = 512
    "GQA-4":  decoder(lambda: GroupedQueryAttention(DIM, HEADS, 4)),                 # 2×4×32 = 256
    "GQA-2":  decoder(lambda: GroupedQueryAttention(DIM, HEADS, 2)),                 # 2×2×32 = 128
    "MQA":    decoder(lambda: GroupedQueryAttention(DIM, HEADS, 1)),                 # 2×1×32 =  64
    "MLA-128": decoder(lambda: MultiHeadLatentAttention(DIM, HEADS, 128, 32, 16, 32)),  # 128+16 = 144
    "MLA-64":  decoder(lambda: MultiHeadLatentAttention(DIM, HEADS, 64, 32, 16, 32)),   #  64+16 =  80
}


# 消融实验 (--ablation): 标准对比里 MLA-128 在 3 个种子上都比 MHA 差约 0.014。
# 假设: MLA 每个头 48 维 Q·K 里只有 16 维带 RoPE, 位置信息比 GQA (32 维全部旋转) 弱。
# 检验: 只把 rope 段加宽到 32 维, 其余不变。
ABLATIONS = {
    "MLA-128-rope32": decoder(lambda: MultiHeadLatentAttention(DIM, HEADS, 128, 32, 32, 32)),  # 128+32 = 160
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ablation", action="store_true", help="额外跑消融实验里的变体")
    args = ap.parse_args()
    if args.ablation:
        VARIANTS.update(ABLATIONS)
    if args.quick:
        cfg, seeds = BenchConfig(steps=200, eval_every=100), (1337,)
    else:
        # 曲线显示这个规模的模型约 800~1000 步后开始过拟合, 1500 步足够覆盖最低点
        cfg, seeds = BenchConfig(steps=1500, eval_every=100), (1337, 2024, 7)
    # 断点续跑: 每跑完一组 (变体, 种子) 立即存盘, 重新运行时跳过已完成的组
    run_dir = os.path.join(HERE, "results", "per_run_quick" if args.quick else "per_run")
    os.makedirs(run_dir, exist_ok=True)
    rows = []
    for name, build in VARIANTS.items():
        for seed in seeds:
            path = os.path.join(run_dir, f"{name}_seed{seed}.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    rows.append(json.load(f))
                print(f"  跳过 {name} (seed={seed}), 已有结果", flush=True)
                continue
            print(f"  训练 {name} (seed={seed}) ...", flush=True)
            row = run(name, build, replace(cfg, seed=seed))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=1)
            rows.append(row)
    print_table(rows)

    out = os.path.join(HERE, "results", "compare_quick.json" if args.quick else "compare.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summarize(rows), "runs": rows}, f, ensure_ascii=False, indent=1)
    print(f"\n  结果已写入 {out}")


if __name__ == "__main__":
    main()
