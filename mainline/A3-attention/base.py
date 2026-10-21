"""
base.py — 树的"LLaMA 配方"节点: 一个组件可插拔的 decoder-only 模型

结构与 A4-modern-small-model 的 LlamaModel 相同 (Pre-RMSNorm + RoPE + SwiGLU + 无 bias + 权重绑定),
区别是**注意力模块从外部传入**。树上的每个注意力节点 (MHA、GQA、MQA、MLA……) 都是一个
满足下面接口的模块, 替换进来后其余部分完全不变, 对比实验的差异就只来自注意力本身。

注意力模块的接口:
    attn(x, cos, sin) -> (B, S, C)        x: (B, S, C); cos/sin: RoPE 表, 形状 (S, d_rope/2)
    attn.rope_dim                          这个模块需要多宽的 RoPE (每个头里参与旋转的维度)
    attn.kv_numbers_per_token              推理时每个 token 在这一层要缓存多少个数

形状记号 (全文统一):
    B  batch 大小            S  序列长度            C  模型维度 (dim)
    H  查询头数              d  每个头的维度         V  词表大小
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm: 只除以均方根, 不减均值 (比 LayerNorm 少一步, 效果相当)。

        RMSNorm(x)_i = x_i / sqrt(mean(x²) + eps) * weight_i
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps                                   # 防止除以 0
        self.weight = nn.Parameter(torch.ones(dim))      # 可学习的逐维缩放, 初始为 1 (即不缩放)

    def forward(self, x):                                # x: (..., dim)
        # x.float(): 在 fp32 下算平方的均值, bf16 下直接算容易丢精度
        # rsqrt = 1 / sqrt; keepdim=True 让结果形状为 (..., 1), 才能和 x 逐元素相乘
        # .to(x.dtype): 算完再转回输入的精度 (混合精度训练时 x 可能是 bf16)
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rope_table(dim, max_len, theta=10000.0, device=None):
    """预先算好 RoPE 要用的 cos、sin 表, 形状都是 (max_len, dim/2)。

    RoPE 把每个头的 dim 维看成 dim/2 对, 第 i 对在位置 p 处旋转角度 p * θ_i,
    其中 θ_i = theta^(-2i/dim): i 越大频率越低, 前几对转得快 (区分相邻位置),
    后几对转得慢 (区分远距离位置)。
    """
    # 每一对的旋转频率: (dim/2,)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # 外积: 角度[p, i] = p * inv_freq[i], 形状 (max_len, dim/2)
    angles = torch.outer(torch.arange(max_len, device=device).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """按位置旋转 x 的最后一维。

    x:   (B, H, S, dim)          cos/sin: (S, dim/2), 会自动广播到 (B, H, S, dim/2)

    相邻两维 (x1, x2) 组成一对, 当作平面上的一个点, 绕原点旋转角度 a:
        x1' = x1·cos(a) - x2·sin(a)
        x2' = x1·sin(a) + x2·cos(a)
    旋转后, 位置 m 的 q 和位置 n 的 k 的点积只取决于角度差, 也就是相对位置 m - n
    (tests/test_tree.py 的 test_rope_depends_only_on_relative_position 验证了这一点)。
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]                  # 偶数维、奇数维, 各 (B, H, S, dim/2)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)   # (B, H, S, dim/2, 2)
    return out.flatten(-2).type_as(x)                    # 把每一对重新交错拼回: (B, H, S, dim)


class SwiGLU(nn.Module):
    """SwiGLU 前馈层: 一条支路做门控 (SiLU 激活), 另一条支路提供内容, 两者逐元素相乘。

        FFN(x) = W_down( SiLU(W_gate x) ⊙ (W_up x) )
    """

    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)   # 门控支路: 决定每个隐藏单元"开多大"
        self.up = nn.Linear(dim, hidden, bias=False)     # 内容支路
        self.down = nn.Linear(hidden, dim, bias=False)   # 投影回模型维度

    def forward(self, x):                                # x: (B, S, C)
        return self.down(F.silu(self.gate(x)) * self.up(x))   # 中间 (B, S, hidden), 输出 (B, S, C)


class Block(nn.Module):
    """一个 Transformer 层 (Pre-Norm): 先归一化再进子层, 子层的输出加回残差主干。"""

    def __init__(self, dim, attn, ffn_hidden):
        super().__init__()
        self.norm1, self.attn = RMSNorm(dim), attn       # attn 由外部传入, 这是"可插拔"的地方
        self.norm2, self.ffn = RMSNorm(dim), SwiGLU(dim, ffn_hidden)

    def forward(self, x, cos, sin):                      # x: (B, S, C)
        x = x + self.attn(self.norm1(x), cos, sin)       # 注意力: 位置之间交换信息
        return x + self.ffn(self.norm2(x))               # 前馈: 每个位置独立加工


class Decoder(nn.Module):
    """完整的解码器。make_attn: 无参函数, 每调用一次返回一个新的注意力模块 (每层一个)。

    之所以传"函数"而不是传"模块": 每一层要有自己独立的参数, 如果把同一个模块对象
    放进每一层, 所有层就会共享同一套注意力参数。
    """

    def __init__(self, vocab_size, block_size, dim, n_layers, make_attn, ffn_hidden=None, rope_theta=10000.0):
        super().__init__()
        # SwiGLU 有三个矩阵 (普通 FFN 是两个), 为了参数量与 4 倍宽的普通 FFN 相当,
        # 隐藏维度取约 8/3 倍; 再向上对齐到 256 的倍数, 对 GPU 的矩阵运算更友好
        ffn_hidden = ffn_hidden or 256 * math.ceil(8 * dim / 3 / 256)
        self.block_size = block_size                     # 最大序列长度 (RoPE 表只算到这么长)
        self.embed = nn.Embedding(vocab_size, dim)       # token 编号 → 向量: (V, C)
        self.blocks = nn.ModuleList(Block(dim, make_attn(), ffn_hidden) for _ in range(n_layers))
        self.norm = RMSNorm(dim)                         # Pre-Norm 结构在最后要再归一化一次
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight          # 权重绑定: 输入嵌入和输出层共用一张表
        # RoPE 表按注意力模块声明的 rope_dim 生成 (MLA 只旋转一部分维度, 所以不一定等于每头维度)
        cos, sin = rope_table(self.blocks[0].attn.rope_dim, block_size, rope_theta)
        # register_buffer: 跟着 .to(device) 移动, 但不是可训练参数; persistent=False: 不存进 checkpoint
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)                           # 对所有子模块调用 _init

    @staticmethod
    def _init(m):
        # 线性层和嵌入都用 std=0.02 的正态分布初始化 (GPT-2 的做法), 让初始 logits 接近 0,
        # 于是初始 loss ≈ ln(V)。RMSNorm 的 weight 保持为 1。
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        """idx: (B, S) token 编号; targets: (B, S) 每个位置的正确答案 (应是 idx 右移一位)。"""
        S = idx.size(1)
        x = self.embed(idx)                              # (B, S, C)
        for block in self.blocks:
            x = block(x, self.cos[:S], self.sin[:S])     # 只取前 S 个位置的 RoPE 表
        logits = self.lm_head(self.norm(x))              # (B, S, V): 每个位置对每个 token 的打分
        # cross_entropy 要求 (N, V) 和 (N,), 所以把 B、S 两维摊平成 N = B*S
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def kv_cache_bytes_per_token(self, dtype_bytes=2):
        """推理时每生成一个 token, 所有层一共要新增多少字节的缓存 (默认按 bf16, 每个数 2 字节)。"""
        return sum(b.attn.kv_numbers_per_token for b in self.blocks) * dtype_bytes
