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
图解见讲义图 3; 实测结果见讲义第 7 节与图 5。

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
#
# 训练时整段序列一次算完 (variants.py 的 forward); 生成时一次只来一个新 token。
# 下面每个 *_step 函数只处理这一个新 token:
#   h      (B, 1, C)      新 token 经过 norm1 之后的输入
#   cos    (1, d_rope/2)  这个位置的 RoPE 表; sin 同理
#   cache  dict           这一层的缓存, 第一次调用时为空, 每调用一次往里追加一个位置
# 返回这一层注意力的输出 (B, 1, C)。

def gqa_step(attn, h, cos, sin, cache):
    """MHA / GQA / MQA 的一步。cache: {"k": (B, G, t, d), "v": (B, G, t, d)}, 存的是 G 组而不是 H 个头。"""
    B = h.size(0)
    # 新 token 的 q、k、v, 和 variants.py 的 forward 一样, 只是序列长度为 1
    q = attn.wq(h).view(B, 1, attn.n_heads, attn.head_dim).transpose(1, 2)      # (B, H, 1, d)
    k = attn.wk(h).view(B, 1, attn.n_kv_heads, attn.head_dim).transpose(1, 2)   # (B, G, 1, d)
    v = attn.wv(h).view(B, 1, attn.n_kv_heads, attn.head_dim).transpose(1, 2)   # (B, G, 1, d)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)                      # 用的是这个位置的 RoPE
    # 追加到缓存末尾 (沿序列维 dim=2)。缓存的是旋转之后的 k: 旧位置的旋转角度不会再变
    # 注意: torch.cat 每次都复制整个缓存, 真实推理框架会预先分配好空间来避免这次复制
    cache["k"] = k if "k" not in cache else torch.cat([cache["k"], k], dim=2)  # (B, G, t, d)
    cache["v"] = v if "v" not in cache else torch.cat([cache["v"], v], dim=2)
    # 和训练时一样, 把每组复制给组内的查询头; 这只是临时张量, 缓存里仍是 G 组
    group = attn.n_heads // attn.n_kv_heads
    K = cache["k"].repeat_interleave(group, dim=1)                              # (B, H, t, d)
    V = cache["v"].repeat_interleave(group, dim=1)
    # 新 token 是当前最后一个位置, 前面的位置它全都可以看, 所以不需要因果掩码
    out = F.scaled_dot_product_attention(q, K, V)                                # (B, H, 1, d)
    return attn.wo(out.transpose(1, 2).reshape(B, 1, -1))                       # (B, 1, C)


def _mla_new_entries(attn, h, cos, sin, cache):
    """MLA: 算出新位置要缓存的两样东西, 接到缓存末尾 (讲义图 2 的绿色虚线框)。

    cache["c"]:  (B, t, r)       已归一化的潜向量 (存归一化之后的, 解码时就不用每步重复归一化)
    cache["kr"]: (B, t, d_rope)  已旋转的共享 rope 键
    """
    c, k_rope = attn.w_dkv(h).split([attn.kv_rank, attn.d_rope], dim=-1)   # (B, 1, r), (B, 1, d_rope)
    c = attn.kv_norm(c)
    # apply_rope 需要 (B, 头, S, d) 的形状: 先插入一个头维度, 旋转完再去掉
    k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin).squeeze(1)          # (B, 1, d_rope)
    cache["c"] = c if "c" not in cache else torch.cat([cache["c"], c], dim=1)
    cache["kr"] = k_rope if "kr" not in cache else torch.cat([cache["kr"], k_rope], dim=1)


def _mla_query(attn, h, cos, sin):
    """MLA: 新 token 的查询, 拆成 nope / rope 两段, 只旋转 rope 段。"""
    B, H = h.size(0), attn.n_heads
    q = attn.wq(h).view(B, 1, H, attn.d_nope + attn.d_rope).transpose(1, 2)   # (B, H, 1, d_nope+d_rope)
    q_nope, q_rope = q.split([attn.d_nope, attn.d_rope], dim=-1)
    return q_nope, apply_rope(q_rope, cos, sin)                           # (B, H, 1, d_nope), (B, H, 1, d_rope)


def mla_step_naive(attn, h, cos, sin, cache):
    """朴素解码 (讲义图 3 上半部分): 缓存虽然只存了 c, 但每一步都把全部历史 c 解压成每个头的 K、V。

    解压的计算量 ∝ t × r × H × (d_nope + d_v), 随上下文长度 t 线性增长 (讲义图 5 里那条上升的黄线)。
    """
    B, H = h.size(0), attn.n_heads
    _mla_new_entries(attn, h, cos, sin, cache)
    q_nope, q_rope = _mla_query(attn, h, cos, sin)
    t = cache["c"].size(1)                                                # 目前缓存了几个位置
    # 关键的一行: 对缓存里全部 t 个位置调用解压矩阵 (练习 3 的测试就是靠拦截这一行来抓朴素写法)
    kv = attn.w_ukv(cache["c"]).view(B, t, H, attn.d_nope + attn.d_v).transpose(1, 2)   # (B, H, t, d_nope+d_v)
    k_nope, v = kv.split([attn.d_nope, attn.d_v], dim=-1)
    # 共享的 rope 键: (B, t, d_rope) → 插入头维度 → expand 给每个头 (不复制内存)
    k = torch.cat([k_nope, cache["kr"].unsqueeze(1).expand(B, H, t, attn.d_rope)], dim=-1)
    out = F.scaled_dot_product_attention(torch.cat([q_nope, q_rope], dim=-1), k, v)   # (B, H, 1, d_v)
    return attn.wo(out.transpose(1, 2).reshape(B, 1, -1))


def mla_step_absorbed(attn, h, cos, sin, cache):
    """吸收解码 (讲义图 3 下半部分): 不解压, 注意力直接在潜向量 c 上算。

    einsum 的下标约定 (同一个字母代表同一个维度, 只出现在输入里的字母会被求和掉):
        b batch    h 头    q 查询位置 (这里是 1)    t 历史位置
        r 潜向量维度    d nope/rope 段的维度    v 每头 v 的维度
    """
    B, H = h.size(0), attn.n_heads
    _mla_new_entries(attn, h, cos, sin, cache)
    q_nope, q_rope = _mla_query(attn, h, cos, sin)

    # 1) 把解压矩阵按头拆开。w_ukv.weight 的形状是 (H*(d_nope+d_v), r) (nn.Linear 的权重是 输出×输入),
    #    按"头"切开后, 每个头的前 d_nope 行是 W_uk (生成 k_nope), 后 d_v 行是 W_uv (生成 v)
    W = attn.w_ukv.weight.view(H, attn.d_nope + attn.d_v, attn.kv_rank)     # (H, d_nope+d_v, r)
    W_uk, W_uv = W[:, :attn.d_nope], W[:, attn.d_nope:]                   # (H, d_nope, r), (H, d_v, r)

    # 2) 吸收 W_uk: q_lat[h] = W_uk[h]ᵀ q_nope[h]。对 d 求和, 把 q 从 d_nope 维变到 r 维 (潜空间)
    #    每生成一个 token 只做一次, 和上下文长度无关
    q_lat = torch.einsum("bhqd,hdr->bhqr", q_nope, W_uk)                  # (B, H, 1, r)

    # 3) 注意力分数 = nope 段 + rope 段, 两段都直接和缓存做点积:
    #    nope 段: q_lat · c_j      对 r 求和  (等价于 q_nope · k_nope_j, 但不需要算出 k_nope_j)
    #    rope 段: q_rope · kr_j    对 d 求和  (共享的 rope 键, 所有头用同一份)
    c, kr = cache["c"], cache["kr"]                                       # (B, t, r), (B, t, d_rope)
    scores = (torch.einsum("bhqr,btr->bhqt", q_lat, c)
              + torch.einsum("bhqd,btd->bhqt", q_rope, kr)) / math.sqrt(attn.d_nope + attn.d_rope)
    a = scores.softmax(dim=-1)                                            # (B, H, 1, t) 每行和为 1

    # 4) 吸收 W_uv: Σ_j a_j (W_uv c_j) = W_uv (Σ_j a_j c_j)
    #    先在潜空间对 c 加权求和 (对 t 求和), 再只乘一次 W_uv (对 r 求和)
    ctx = torch.einsum("bhqt,btr->bhqr", a, c)                            # (B, H, 1, r)
    out = torch.einsum("bhqr,hvr->bhqv", ctx, W_uv)                       # (B, H, 1, d_v)
    return attn.wo(out.transpose(1, 2).reshape(B, 1, -1))                 # (B, 1, C)


# ── 整个模型: 逐 token 解码 ───────────────────────────────────────

def make_stepper(model, mla_mode="absorbed"):
    """返回一个函数 forward_one(token, pos, caches) -> logits, 每调用一次前进一个位置。

    它复用 model 里的全部权重 (嵌入、每层的归一化和 FFN、输出层), 只把注意力换成带缓存的一步。
    """
    attn0 = model.blocks[0].attn
    if isinstance(attn0, GroupedQueryAttention):
        step = gqa_step
    elif isinstance(attn0, MultiHeadLatentAttention):
        step = mla_step_absorbed if mla_mode == "absorbed" else mla_step_naive
    else:
        raise TypeError(type(attn0))

    def forward_one(token, pos, caches):
        """token: (B, 1) 新 token; pos: 它在序列中的位置; caches: 每层一个 dict 的列表。"""
        cos, sin = model.cos[pos:pos + 1], model.sin[pos:pos + 1]         # 只取这一个位置的 RoPE 表
        x = model.embed(token)                                            # (B, 1, C)
        for block, cache in zip(model.blocks, caches):                    # 与 base.Block.forward 相同的结构
            x = x + step(block.attn, block.norm1(x), cos, sin, cache)
            x = x + block.ffn(block.norm2(x))
        return model.lm_head(model.norm(x))[:, -1]                        # (B, V) 下一个 token 的打分
    return forward_one


def cache_bytes(caches):
    """所有层缓存的总字节数 (实测, 不是公式估算)。"""
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
                # 笔记本 GPU 的频率随负载跳变, 单次计时噪声很大:
                # 先空跑 10 次让 GPU 进入稳定状态, 再测 20 次, 每次单独同步, 取中位数
                samples = []
                for i in range(30):
                    snapshot = [dict(c) for c in caches]   # 在副本上跑, 不改变真正的缓存
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    step(tok, pos, snapshot)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    if i >= 10:
                        samples.append((time.perf_counter() - t0) * 1000)
                times[pos + 1] = sorted(samples)[len(samples) // 2]
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
