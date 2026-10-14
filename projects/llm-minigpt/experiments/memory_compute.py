"""
专题 E 实验: 训练的显存账与算力账

1. 静态显存: 参数 + 梯度 + AdamW 两个状态, 预测 16 字节/参数, 与实测对比
2. 激活显存: 随 batch size 线性增长
3. 算力: 每个 token 约 6N 次浮点运算, 实测吞吐与达到的 TFLOPS

全部用 fp32、不开混合精度, 让账目最简单。运行: python experiments/memory_compute.py   (需要 GPU)
"""

import contextlib
import io
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GPTConfig  # noqa: E402
from model import GPT  # noqa: E402

MB = 2 ** 20
CONFIGS = [
    ("micro",       GPTConfig(vocab_size=65, block_size=256, n_layer=4, n_head=4, n_embd=128, dropout=0.0)),
    ("6层384维",    GPTConfig(vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.0)),
    ("GPT-2 Small", GPTConfig(vocab_size=50257, block_size=256, n_layer=12, n_head=12, n_embd=768, dropout=0.0)),
]


def build(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return GPT(cfg).cuda()


def one_step(model, opt, B, cfg):
    x = torch.randint(0, cfg.vocab_size, (B, cfg.block_size), device="cuda")
    _, loss = model(x, x)
    loss.backward()
    opt.step()


def static_memory():
    print("[1] 静态显存: 参数 + 梯度 + AdamW 的 m、v  (预测 = 16 字节 × 参数量)")
    print(f"    {'配置':<12} {'参数量':>12} {'预测':>10} {'实测':>10}")
    for name, cfg in CONFIGS:
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        model = build(cfg)
        n = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        one_step(model, opt, 1, cfg)                 # 第一步之后梯度和 m、v 才真正分配出来
        torch.cuda.synchronize()
        used = (torch.cuda.memory_allocated() - base) / MB
        print(f"    {name:<12} {n:>12,} {16 * n / MB:>8.1f}MB {used:>8.1f}MB")
        del model, opt


def activation_memory():
    name, cfg = CONFIGS[2]
    print(f"\n[2] 激活显存 ({name}, block_size={cfg.block_size}): 峰值 - 静态")
    model = build(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    one_step(model, opt, 1, cfg)
    opt.zero_grad(set_to_none=False)
    torch.cuda.synchronize()
    static = torch.cuda.memory_allocated()
    print(f"    {'batch':<6} {'峰值':>10} {'激活':>10} {'激活/样本':>10}")
    for B in (1, 2, 4, 8, 16):
        torch.cuda.reset_peak_memory_stats()
        try:
            one_step(model, opt, B, cfg)
            opt.zero_grad(set_to_none=False)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            act = (peak - static) / MB
            print(f"    {B:<6} {peak / MB:>8.0f}MB {act:>8.0f}MB {act / B:>8.0f}MB")
        except torch.cuda.OutOfMemoryError:
            print(f"    {B:<6} {'显存不足':>10}")
            break
    del model, opt
    torch.cuda.empty_cache()


def throughput():
    print("\n[3] 算力: 每个 token 约 6N 次浮点运算 (前向 2N + 反向 4N)")
    print(f"    {'配置':<12} {'B':>3} {'tokens/s':>10} {'TFLOPS':>8}")
    for name, cfg in CONFIGS:
        B = 8
        model = build(cfg)
        n = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(3):                           # 预热
            one_step(model, opt, B, cfg)
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        steps, t0 = 10, time.perf_counter()
        for _ in range(steps):
            one_step(model, opt, B, cfg)
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok_s = steps * B * cfg.block_size / dt
        print(f"    {name:<12} {B:>3} {tok_s:>10,.0f} {6 * n * tok_s / 1e12:>8.2f}")
        del model, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 GPU"
    static_memory()
    activation_memory()
    throughput()
