"""
专题 A 实验: KV Cache

1. 用 model.py 里 GPT 的同一套权重, 手写一个带 KV Cache 的逐 token 前向;
2. 验证: 每一步的 logits 与"整段重算"完全一致;
3. 测量: 生成 N 个 token 时, 有无 cache 的耗时对比。

运行: python experiments/kv_cache.py
"""

import contextlib
import io
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GPTConfig  # noqa: E402
from model import GPT  # noqa: E402


def build(cfg):
    with contextlib.redirect_stdout(io.StringIO()):  # 屏蔽 GPT() 打印的参数量
        return GPT(cfg)


@torch.no_grad()
def step_with_cache(model, token, pos, cache):
    """只喂一个新 token (B, 1), 复用 cache 里前面所有位置的 K、V。

    cache: 每层一个 [K, V], 形状 (B, H, 已缓存长度, d_k); 首次调用传空列表。
    返回: 这个新位置的 logits (B, V)
    """
    B = token.size(0)
    x = model.wte(token) + model.wpe(torch.tensor([pos], device=token.device))  # (B, 1, C)
    for i, block in enumerate(model.blocks):
        attn = block.attn
        h = block.ln_1(x)
        q, k, v = attn.c_attn(h).split(model.config.n_embd, dim=-1)       # 各 (B, 1, C)
        # (B, 1, C) -> (B, H, 1, d_k)
        q, k, v = (t.view(B, 1, attn.n_heads, attn.d_k).transpose(1, 2) for t in (q, k, v))
        if len(cache) <= i:
            cache.append([k, v])
        else:  # 把新位置的 K、V 接到缓存末尾
            cache[i][0] = torch.cat([cache[i][0], k], dim=2)
            cache[i][1] = torch.cat([cache[i][1], v], dim=2)
        K, V = cache[i]                                                   # (B, H, pos+1, d_k)
        # 新 token 是当前最后一个位置, 前面全都可以看, 所以不需要掩码
        w = F.softmax(q @ K.transpose(-2, -1) / math.sqrt(attn.d_k), dim=-1)   # (B, H, 1, pos+1)
        out = (w @ V).transpose(1, 2).reshape(B, 1, -1)                   # (B, 1, C)
        x = x + attn.c_proj(out)
        x = x + block.mlp(block.ln_2(x))
    return model.lm_head(model.ln_f(x))[:, -1, :]


@torch.no_grad()
def check_equivalence(device):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=65, block_size=64, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    model = build(cfg).to(device).eval()
    seq = torch.randint(0, 65, (1, 40), device=device)
    full_logits, _ = model(seq)                       # 整段一次算完: (1, 40, 65)
    cache, worst = [], 0.0
    for pos in range(seq.size(1)):
        lg = step_with_cache(model, seq[:, pos:pos + 1], pos, cache)
        worst = max(worst, (lg - full_logits[:, pos]).abs().max().item())
    print(f"[等价性] 40 个位置逐个用 cache 计算, 与整段前向的最大差异 = {worst:.2e}")


@torch.no_grad()
def time_generate(model, n_new, use_cache, device):
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if use_cache:
        cache, tok = [], idx
        for pos in range(n_new):
            logits = step_with_cache(model, tok, pos, cache)
            tok = logits.argmax(-1, keepdim=True)
    else:
        for _ in range(n_new):
            logits, _ = model(idx)
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def benchmark(device):
    rows = [
        ("nano",  GPTConfig(vocab_size=65, block_size=256, n_layer=2, n_head=2, n_embd=64)),
        ("micro", GPTConfig(vocab_size=65, block_size=256, n_layer=4, n_head=4, n_embd=128)),
        ("6层384维", GPTConfig(vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384)),
    ]
    n_new = 255
    print(f"\n[速度] 设备={device}, 生成 {n_new} 个 token (贪心), 各取 3 次中最快的一次")
    print(f"  {'配置':<10} {'无 cache':>10} {'有 cache':>10} {'加速':>7}")
    for name, cfg in rows:
        model = build(cfg).to(device).eval()
        time_generate(model, 8, True, device)            # 预热 (首次运行会加载 kernel)
        time_generate(model, 8, False, device)
        t_full = min(time_generate(model, n_new, False, device) for _ in range(3))
        t_kv = min(time_generate(model, n_new, True, device) for _ in range(3))
        print(f"  {name:<10} {t_full * 1000:>8.0f}ms {t_kv * 1000:>8.0f}ms {t_full / t_kv:>6.1f}x")


def benchmark_long():
    """GPT-2 Small 规模 (12 层 768 维, 上下文 1024) 在 GPU 上生成不同长度。"""
    cfg = GPTConfig(vocab_size=65, block_size=1024, n_layer=12, n_head=12, n_embd=768)
    model = build(cfg).to("cuda").eval()
    time_generate(model, 8, True, "cuda")
    time_generate(model, 8, False, "cuda")
    print("\n[速度] GPT-2 Small 规模, cuda, 生成长度递增")
    print(f"  {'长度':<6} {'无 cache':>10} {'有 cache':>10} {'加速':>7}")
    for n in (128, 512, 1000):
        t_full = time_generate(model, n, False, "cuda")
        t_kv = time_generate(model, n, True, "cuda")
        print(f"  {n:<6} {t_full:>9.1f}s {t_kv:>9.1f}s {t_full / t_kv:>6.1f}x")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    check_equivalence(dev)
    benchmark("cpu")
    if dev == "cuda":
        benchmark("cuda")
        benchmark_long()
