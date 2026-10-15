"""
decode.py — 注意力线的推理实验: 逐 token 生成时的 KV Cache

对照实验 (compare.py) 比的是训练质量; 但这条线真正要解决的问题发生在推理时。本脚本:

1. 为每种注意力写一条"带缓存的逐 token 前向", 验证与整段前向结果一致;
2. MLA 实现两种解码方式:
     naive    缓存潜向量 c, 但每一步把所有历史位置都解压成 K、V 再算注意力
     absorbed 把解压矩阵 W_uk、W_uv 提前乘进 Q 和输出, 直接在 c 上算注意力 (DeepSeek 推理用的技巧)
3. 在长上下文下实测缓存显存与每个 token 的生成耗时。

吸收技巧 (第 h 个头, c_j 为第 j 个位置缓存的潜向量):
    分数:  q_h · (W_uk_h c_j) = (W_uk_hᵀ q_h) · c_j       → 先把 q 变换一次, 再和 c 点积
    输出:  Σ_j a_j (W_uv_h c_j) = W_uv_h (Σ_j a_j c_j)    → 先对 c 加权求和, 最后再乘一次 W_uv

运行: python attention/decode.py
"""

import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from base import Decoder, apply_rope  # noqa: E402
from variants import GroupedQueryAttention, MultiHeadLatentAttention  # noqa: E402


# ── 单层: 带缓存的一步 ─────────────────────────────────────────────

def gqa_step(attn, h, cos, sin, cache):
    """h: (B, 1, C)。cache: {"k": (B, Hkv, t, d), "v": ...}, 缓存的是 Hkv 组而不是 H 个头。"""
    B = h.size(0)
    q = attn.wq(h).view(B, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = attn.wk(h).view(B, 1, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.wv(h).view(B, 1, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    cache["k"] = k if "k" not in cache else torch.cat([cache["k"], k], dim=2)
    cache["v"] = v if "v" not in cache else torch.cat([cache["v"], v], dim=2)
    group = attn.n_heads // attn.n_kv_heads
    K = cache["k"].repeat_interleave(group, dim=1)
    V = cache["v"].repeat_interleave(group, dim=1)
    out = F.scaled_dot_product_attention(q, K, V)        # 新 token 能看到全部缓存, 不需要掩码
    return attn.wo(out.transpose(1, 2).reshape(B, 1, -1))


def _mla_new_entries(attn, h, cos, sin, cache):
    """算出新位置要缓存的 c (已归一化) 和共享 RoPE 键, 接到缓存末尾。"""
    c, k_rope = attn.w_dkv(h).split([attn.kv_rank, attn.d_rope], dim=-1)   # (B, 1, r), (B, 1, d_rope)
    c = attn.kv_norm(c)
    k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin).squeeze(1)          # (B, 1, d_rope)
    cache["c"] = c if "c" not in cache else torch.cat([cache["c"], c], dim=1)
    cache["kr"] = k_rope if "kr" not in cache else torch.cat([cache["kr"], k_rope], dim=1)


def _mla_query(attn, h, cos, sin):
    B, H = h.size(0), attn.n_heads
    q = attn.wq(h).view(B, 1, H, attn.d_nope + attn.d_rope).transpose(1, 2)
    q_nope, q_rope = q.split([attn.d_nope, attn.d_rope], dim=-1)
    return q_nope, apply_rope(q_rope, cos, sin)                           # (B, H, 1, ·)


def mla_step_naive(attn, h, cos, sin, cache):
    """每一步都把全部历史 c 解压成每个头的 K、V: 计算量随上下文长度 × 头数增长。"""
    B, H = h.size(0), attn.n_heads
    _mla_new_entries(attn, h, cos, sin, cache)
    q_nope, q_rope = _mla_query(attn, h, cos, sin)
    t = cache["c"].size(1)
    kv = attn.w_ukv(cache["c"]).view(B, t, H, attn.d_nope + attn.d_v).transpose(1, 2)
    k_nope, v = kv.split([attn.d_nope, attn.d_v], dim=-1)
    k = torch.cat([k_nope, cache["kr"].unsqueeze(1).expand(B, H, t, attn.d_rope)], dim=-1)
    out = F.scaled_dot_product_attention(torch.cat([q_nope, q_rope], dim=-1), k, v)
    return attn.wo(out.transpose(1, 2).reshape(B, 1, -1))


def mla_step_absorbed(attn, h, cos, sin, cache):
    """不解压: 注意力直接在潜向量 c 上算。"""
    B, H = h.size(0), attn.n_heads
    _mla_new_entries(attn, h, cos, sin, cache)
    q_nope, q_rope = _mla_query(attn, h, cos, sin)
    W = attn.w_ukv.weight.view(H, attn.d_nope + attn.d_v, attn.kv_rank)     # 每个头的 [W_uk; W_uv]
    W_uk, W_uv = W[:, :attn.d_nope], W[:, attn.d_nope:]                   # (H, d_nope, r), (H, d_v, r)
    q_lat = torch.einsum("bhqd,hdr->bhqr", q_nope, W_uk)                  # q 变换到潜空间: (B, H, 1, r)
    c, kr = cache["c"], cache["kr"]                                       # (B, t, r), (B, t, d_rope)
    scores = (torch.einsum("bhqr,btr->bhqt", q_lat, c)
              + torch.einsum("bhqd,btd->bhqt", q_rope, kr)) / math.sqrt(attn.d_nope + attn.d_rope)
    a = scores.softmax(dim=-1)                                            # (B, H, 1, t)
    ctx = torch.einsum("bhqt,btr->bhqr", a, c)                            # 先在潜空间加权求和
    out = torch.einsum("bhqr,hvr->bhqv", ctx, W_uv)                       # 最后解压一次: (B, H, 1, d_v)
    return attn.wo(out.transpose(1, 2).reshape(B, 1, -1))


# ── 整个模型: 逐 token 解码 ───────────────────────────────────────

def make_stepper(model, mla_mode="absorbed"):
    attn0 = model.blocks[0].attn
    if isinstance(attn0, GroupedQueryAttention):
        step = gqa_step
    elif isinstance(attn0, MultiHeadLatentAttention):
        step = mla_step_absorbed if mla_mode == "absorbed" else mla_step_naive
    else:
        raise TypeError(type(attn0))

    def forward_one(token, pos, caches):
        cos, sin = model.cos[pos:pos + 1], model.sin[pos:pos + 1]
        x = model.embed(token)                                            # (B, 1, C)
        for block, cache in zip(model.blocks, caches):
            x = x + step(block.attn, block.norm1(x), cos, sin, cache)
            x = x + block.ffn(block.norm2(x))
        return model.lm_head(model.norm(x))[:, -1]
    return forward_one


def cache_bytes(caches):
    return sum(t.numel() * t.element_size() for c in caches for t in c.values())


# ── 实验 ─────────────────────────────────────────────────────────

SMALL = dict(dim=256, layers=4, heads=8)        # 与 compare.py 相同的规模


def build(kind, vocab, block, dim, layers, heads):
    d = dim // heads
    make = {
        "MHA":    lambda: GroupedQueryAttention(dim, heads, heads),
        "GQA-4":  lambda: GroupedQueryAttention(dim, heads, 4),
        "MQA":    lambda: GroupedQueryAttention(dim, heads, 1),
        "MLA":    lambda: MultiHeadLatentAttention(dim, heads, 4 * d, d, d // 2, d),
    }[kind]
    return Decoder(vocab, block, dim, layers, make)


@torch.no_grad()
def check_equivalence():
    print("[1] 逐 token 解码 与 整段前向 的最大差异 (fp32, CPU)")
    torch.manual_seed(0)
    seq = torch.randint(0, 65, (2, 40))
    for kind, modes in [("MHA", [None]), ("GQA-4", [None]), ("MQA", [None]), ("MLA", ["naive", "absorbed"])]:
        model = build(kind, 65, 64, **SMALL).eval()
        full, _ = model(seq)
        for mode in modes:
            step, caches, worst = make_stepper(model, mode or "absorbed"), [{} for _ in model.blocks], 0.0
            for pos in range(seq.size(1)):
                worst = max(worst, (step(seq[:, pos:pos + 1], pos, caches) - full[:, pos]).abs().max().item())
            label = kind if mode is None else f"{kind} ({mode})"
            print(f"    {label:<16} {worst:.2e}")


@torch.no_grad()
def long_context(device):
    cfg = dict(dim=1024, layers=8, heads=16)     # 每头 64 维; 放大到能看出差别的规模
    S = 8192
    print(f"\n[2] 长上下文解码: {cfg['layers']} 层, {cfg['dim']} 维, {cfg['heads']} 头, bf16, {device}")
    print(f"    先用逐 token 解码填满 {S} 个位置的缓存, 再测量在不同长度处生成一个 token 的耗时")
    print(f"    {'变体':<16} {'缓存/token':>10} {'缓存@' + str(S):>11} {'@1024':>9} {'@4096':>9} {'@8192':>9}")
    for kind, mode in [("MHA", None), ("GQA-4", None), ("MQA", None), ("MLA", "naive"), ("MLA", "absorbed")]:
        torch.manual_seed(0)
        model = build(kind, 65, S, **cfg).to(device, torch.bfloat16).eval()
        step, caches = make_stepper(model, mode or "absorbed"), [{} for _ in model.blocks]
        tok = torch.zeros((1, 1), dtype=torch.long, device=device)
        times = {}
        for pos in range(S):
            if pos + 1 in (1024, 4096, 8192):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(5):                      # 同一位置重复测 5 次取均值 (测完丢弃多余的缓存)
                    snapshot = [dict(c) for c in caches]
                    step(tok, pos, snapshot)
                if device == "cuda":
                    torch.cuda.synchronize()
                times[pos + 1] = (time.perf_counter() - t0) / 5 * 1000
            step(tok, pos, caches)
        total = cache_bytes(caches)
        label = kind if mode is None else f"{kind} ({mode})"
        print(f"    {label:<16} {total / S / 1024:>8.1f}KB {total / 2**20:>9.1f}MB "
              f"{times[1024]:>7.2f}ms {times[4096]:>7.2f}ms {times[8192]:>7.2f}ms")
        del model, caches
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    check_equivalence()
    long_context("cuda" if torch.cuda.is_available() else "cpu")
