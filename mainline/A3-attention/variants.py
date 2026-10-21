"""
variants.py — 注意力线上的节点

    MHA ──→ MQA ──→ GQA ──→ MLA
    (每个头一组 K、V)  (所有头共用一组)  (分组共用)  (把 K、V 压缩成一个低维潜向量再缓存)

这条线上每一步要解决的都是同一个问题: 推理时的 KV Cache 太大
(A2-minigpt 讲义专题 A 已经实测过: GPT-2 Small 一条 1024 长度的序列要缓存 72MB)。
讲义: TUTORIAL.md 第 2~5 节; 图 1 (头与 KV 的对应)、图 2 (MLA 数据流)。

两个类都满足 base.py 约定的接口: forward(x, cos, sin)、rope_dim、kv_numbers_per_token。
形状记号: B batch, S 序列长度, C 模型维度, H 查询头数, G KV 组数, d 每头维度, r 潜向量维度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import RMSNorm, apply_rope


class GroupedQueryAttention(nn.Module):
    """n_kv_heads == n_heads 时为 MHA, == 1 时为 MQA, 介于两者之间为 GQA。

    n_heads 个查询头被分成 n_kv_heads 组, 同一组里的查询头共用一组 K、V (讲义图 1)。
    """

    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        # 每个头的维度必须是整数; 每组的查询头数也必须是整数
        assert dim % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = dim // n_heads                                       # d
        # Q 仍然每个头一份: 输出 H * d 维
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # K、V 只有 G 组: 输出 G * d 维。和 MHA 相比, 这两个矩阵缩小到 G/H, 缓存也缩小到 G/H
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        # 输出投影: 把 H 个头拼起来的 H * d 维映射回模型维度
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.rope_dim = self.head_dim                                        # 每个头的全部维度都参与旋转
        # 推理时每个 token 在这一层缓存: K 和 V 各 G 组, 每组 d 个数
        self.kv_numbers_per_token = 2 * n_kv_heads * self.head_dim

    def forward(self, x, cos, sin):
        B, S, _ = x.shape                                                    # x: (B, S, C)
        # 1) 投影, 再把最后一维拆成"头 × 每头维度", 然后把"头"挪到第 1 维,
        #    这样 scaled_dot_product_attention 会对每个头分别算注意力
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)      # (B, H, S, d)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, G, S, d)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, G, S, d)
        # 2) RoPE 只作用在 Q 和 K 上 (V 不需要位置信息: 它是"被取走的内容", 不参与比较)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # 3) 把每组 K、V 复制给组内的每个查询头: (B, G, S, d) -> (B, H, S, d)
        #    repeat_interleave 让相邻的头属于同一组: G=2, H=8 时排成 [g0 g0 g0 g0 g1 g1 g1 g1]
        #    复制只发生在计算时; 推理时缓存的仍然只是 G 组, 这正是省显存的地方
        group = self.n_heads // self.n_kv_heads                              # 每组几个查询头
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        # 4) 标准的缩放点积注意力: softmax(q·kᵀ/√d + 因果掩码) · v
        #    is_causal=True 让每个位置只能看到自己和之前的位置
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)       # (B, H, S, d)
        # 5) 把头挪回去并拼接: (B, H, S, d) -> (B, S, H, d) -> (B, S, H*d), 再输出投影
        return self.wo(out.transpose(1, 2).reshape(B, S, -1))               # (B, S, C)


class MultiHeadLatentAttention(nn.Module):
    """MLA (DeepSeek-V2 提出, V3、Kimi-K2 沿用)。数据流见讲义图 2。

    K、V 不再按头单独投影, 而是先把 x 压缩成一个 kv_rank 维的潜向量 c, 推理时只缓存 c;
    计算注意力时再由 c "解压" 出每个头的 K、V。

    难点在 RoPE: 位置旋转不能和"解压"矩阵交换次序, 所以每个头的 Q、K 被拆成两段:
      - nope 段 (d_nope 维): 不旋转, K 由潜向量 c 解压得到;
      - rope 段 (d_rope 维): 旋转, K 直接由 x 投影得到, 并且**所有头共用一份** (所以缓存很小)。
    推理时每个 token 每层缓存 kv_rank + d_rope 个数。

    字段名对应真实模型 (见 electives/llm-anatomy 的张量名):
      w_dkv  ↔ self_attn.kv_a_proj_with_mqa   x → [潜向量 c, 共享 RoPE 键]
      kv_norm↔ self_attn.kv_a_layernorm       对潜向量做 RMSNorm
      w_ukv  ↔ self_attn.kv_b_proj            c → 每个头的 [K_nope, V]
    (DeepSeek 对 Q 也做了低秩压缩 q_a_proj / q_b_proj, 那是为了省训练时的激活, 与 KV Cache 无关, 这里省略。)
    """

    def __init__(self, dim, n_heads, kv_rank, d_nope, d_rope, d_v):
        super().__init__()
        self.n_heads, self.kv_rank = n_heads, kv_rank                        # H, r
        self.d_nope, self.d_rope, self.d_v = d_nope, d_rope, d_v
        # Q: 每个头 d_nope + d_rope 维 (两段, 后面会拆开)
        self.wq = nn.Linear(dim, n_heads * (d_nope + d_rope), bias=False)
        # 压缩: 一次投影同时得到 r 维潜向量 c 和 d_rope 维的共享 rope 键 (所以输出 r + d_rope 维)
        self.w_dkv = nn.Linear(dim, kv_rank + d_rope, bias=False)
        self.kv_norm = RMSNorm(kv_rank)                                      # 只对 c 归一化, 不含 rope 键
        # 解压: c (r 维) → 每个头的 [k_nope (d_nope 维), v (d_v 维)]
        self.w_ukv = nn.Linear(kv_rank, n_heads * (d_nope + d_v), bias=False)
        self.wo = nn.Linear(n_heads * d_v, dim, bias=False)                  # 输出投影: H 个头的 v 拼起来
        self.rope_dim = d_rope                                               # 只有 rope 段参与旋转
        self.kv_numbers_per_token = kv_rank + d_rope                         # 只缓存 c 和共享的 RoPE 键

    def forward(self, x, cos, sin):
        B, S, _ = x.shape                                                    # x: (B, S, C)
        H = self.n_heads

        # ── 查询: 每个头一份, 拆成 nope / rope 两段, 只旋转 rope 段 ─────────
        q = self.wq(x).view(B, S, H, self.d_nope + self.d_rope).transpose(1, 2)   # (B, H, S, d_nope+d_rope)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)        # (B, H, S, d_nope), (B, H, S, d_rope)
        q_rope = apply_rope(q_rope, cos, sin)

        # ── 键值: 先压缩 ─────────────────────────────────────────────
        # c 和 k_rope 就是推理时要缓存的全部内容 (讲义图 2 的绿色虚线框)
        c, k_rope = self.w_dkv(x).split([self.kv_rank, self.d_rope], dim=-1)   # (B, S, r), (B, S, d_rope)
        # k_rope 没有"头"这一维: 插入一个长度为 1 的头维度再旋转, 之后由所有头共用
        k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin)                     # (B, 1, S, d_rope)

        # ── 键值: 再解压出每个头的 k_nope 和 v ─────────────────────────
        kv = self.w_ukv(self.kv_norm(c)).view(B, S, H, self.d_nope + self.d_v).transpose(1, 2)
        k_nope, v = kv.split([self.d_nope, self.d_v], dim=-1)               # (B, H, S, d_nope), (B, H, S, d_v)

        # ── 把两段拼回去, 做标准注意力 ────────────────────────────────
        q = torch.cat([q_nope, q_rope], dim=-1)                              # (B, H, S, d_nope+d_rope)
        # expand 让共享的 rope 键"看起来"每个头都有一份, 但不复制内存 (只改了读取方式)
        k = torch.cat([k_nope, k_rope.expand(B, H, S, self.d_rope)], dim=-1)   # (B, H, S, d_nope+d_rope)
        # 分数 = q_nope·k_nope + q_rope·k_rope (拼接后的点积正好是两段点积之和)
        # 缩放默认为 1/√(q 的最后一维) = 1/√(d_nope+d_rope); v 的维度 d_v 可以和 q、k 不同
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)       # (B, H, S, d_v)
        return self.wo(out.transpose(1, 2).reshape(B, S, -1))               # (B, S, C)
