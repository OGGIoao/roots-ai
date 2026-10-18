"""
model.py — GPT-2 架构完整实现 (Pre-LN Decoder-only)

一个文件读懂 GPT-2 (nanoGPT 风格):
  - 词嵌入 + 可学习的位置嵌入
  - Causal Self-Attention (多头, 因果掩码)
  - GELU 前馈层
  - Pre-LayerNorm 的 Transformer Block + 残差连接
  - 输出层与词嵌入共享权重
  - 训练时的 loss 计算 + 推理时的自回归生成 (不带 KV Cache, 带缓存的写法见讲义专题 A)

讲义: TUTORIAL.md 第 3~9、11 节。
参考: Karpathy nanoGPT
运行: python model.py  (快速功能测试)

形状记号 (全文统一):
    B  batch 大小        S  序列长度 (≤ block_size)       C  嵌入维度 (n_embd)
    H  注意力头数        d_k  每个头的维度 (= C / H)       V  词表大小
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from config import GPTConfig  # 模型超参的唯一定义在 config.py

# Windows 控制台默认 GBK, 无法编码脚本结尾处的 ✅/❌ 等符号,
# print() 会抛 UnicodeEncodeError 让脚本在跑通后异常退出。
for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r:
        try:
            _r(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass



# ============================================================
# 1. Scaled Dot-Product Attention (with Causal Mask)
# ============================================================

def scaled_dot_product_attention(
    query: torch.Tensor,   # (B, n_heads, S, d_k)
    key: torch.Tensor,     # (B, n_heads, S, d_k)
    value: torch.Tensor,   # (B, n_heads, S, d_v)
    mask: torch.Tensor,    # (1, 1, S, S) 上三角 True=需屏蔽
) -> tuple:
    """
    Scaled Dot-Product Attention

    公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    Causal Mask: 防止位置 i 看到位置 j > i 的信息
    mask 为上三角 True 矩阵, 由 masked_fill(-inf) 实现
    """
    d_k = query.size(-1)
    # 分数矩阵: 第 i 行第 j 列 = 位置 i 的 q 与位置 j 的 k 的点积 (讲义 5.3 节)
    # key.transpose(-2, -1): (B, H, S, d_k) -> (B, H, d_k, S), 这样 matmul 得到 (B, H, S, S)
    # 除以 √d_k: 让分数的标准差保持在 1 左右, 避免 softmax 饱和 (讲义 5.4 节)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 因果掩码: mask 按最大长度 block_size 建好, 这里切出左上角 S×S 来用;
    # True 的位置 (未来) 填成 -inf, softmax 之后权重恰好为 0 (讲义 5.5 节)
    scores = scores.masked_fill(mask[:, :, :scores.size(-2), :scores.size(-1)], float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)                 # (B, H, S, S) 每一行和为 1
    output = torch.matmul(attn_weights, value)               # (B, H, S, d_v) 按权重对 v 加权求和
    return output, attn_weights


# ============================================================
# 2. Causal Self-Attention (Multi-Head)
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with Causal Mask

    GPT-2 的 Attention 特点:
    1. Causal mask: 每个位置只能看到自己和之前的位置
    2. 无 encoder-decoder cross-attention (纯 decoder)
    3. Q, K, V 投影写在一个大矩阵中 (比 3 个小矩阵更快)
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0            # C 必须能平均分给每个头

        self.n_heads = config.n_head                         # H
        self.d_k = config.n_embd // config.n_head            # 每个头的维度

        # QKV 投影: 一次矩阵乘法生成 Q, K, V。输出 3C 维, 之后再切成三份
        # (数学上等价于三个 C×C 的矩阵, 一次大乘法在 GPU 上更快)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影: 把多个头拼起来的结果再混合一次, 让各头的信息互相流通
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # 注意: attn_dropout 定义了但 forward 里没有使用 (注意力权重上没有 dropout),
        # 讲义练习 6 的答案里专门讨论了这一点; 保留它是为了不改变已有 checkpoint 的结构
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)      # 作用在输出投影之后

        # Causal mask: 注册为 buffer (不参与训练, 但跟随模型移动)
        # masked_fill 只接受 bool mask, 且填的是 True 处;
        # 因此这里直接标记「要屏蔽」的上三角, 下三角(含对角)保持 False
        mask = torch.triu(torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1)
        mask = mask.view(1, 1, config.block_size, config.block_size)   # 前两维留给 B、H 广播
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, S, C = x.shape                                    # x: (B, S, C)

        # QKV 投影
        qkv = self.c_attn(x)                                 # (B, S, 3C)
        q, k, v = qkv.split(C, dim=-1)                       # 各 (B, S, C)

        # 拆分成多头: (B, S, C) -> (B, S, H, d_k) -> (B, H, S, d_k)
        # view: 把最后的 C 维看成 H 个头 × d_k 维 (不搬动数据)
        # transpose(1, 2): 把"头"挪到前面, matmul 就会对每个 (样本, 头) 分别算注意力
        q = q.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_out, _ = scaled_dot_product_attention(q, k, v, self.mask)   # (B, H, S, d_k)

        # 拼接多头: (B, H, S, d_k) -> (B, S, H, d_k) -> (B, S, C)
        # transpose 之后内存不再连续, view 要求连续, 所以中间要 contiguous() 复制一份
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, C)

        # 输出投影 + residual dropout
        return self.resid_dropout(self.c_proj(attn_out))     # (B, S, C)


# ============================================================
# 3. Feed-Forward Network (GELU)
# ============================================================

class FeedForward(nn.Module):
    """
    GPT-2 风格的 FFN

    FFN(x) = GELU(x * W_up + b_up) * W_down + b_down

    中间维度 = 4 * d_model (GPT-2 的标准设计)
    使用 GELU 激活 (比 ReLU 更平滑)
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)    # 升维 C -> 4C
        self.gelu = nn.GELU()                                                         # 非线性, 没有它两层线性等于一层
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)  # 降维 4C -> C
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):                                    # x: (B, S, C), 每个位置独立计算
        x = self.c_fc(x)                                     # (B, S, 4C)
        x = self.gelu(x)
        x = self.c_proj(x)                                   # (B, S, C)
        return self.dropout(x)


# ============================================================
# 4. Transformer Block (Pre-LN)
# ============================================================

class Block(nn.Module):
    """
    GPT-2 Transformer Block (Pre-LayerNorm)

    Pre-LN vs Post-LN:
      Pre-LN (GPT-2+): x + Attn(LN(x))  — 训练更稳定
      Post-LN (原始): LN(x + Attn(x))    — 训练不稳定, 需要 warmup

    Pre-LN 已成为现代 LLM 的标配
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)    # 注意力之前的归一化
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)    # 前馈层之前的归一化
        self.mlp = FeedForward(config)

    def forward(self, x):
        # Pre-LN: 先 norm, 再 attention, 再 residual
        # x 是贯穿所有层的"残差主干": 每个子层只算一个修改量加上去, 不替换 x (讲义 7.3 节)
        x = x + self.attn(self.ln_1(x))                      # 位置之间交换信息
        x = x + self.mlp(self.ln_2(x))                       # 每个位置独立加工
        return x


# ============================================================
# 5. GPT Model
# ============================================================

class GPT(nn.Module):
    """
    GPT-2 模型

    结构:
      token_embedding + position_embedding
      -> N × Transformer Block (Pre-LN)
      -> LayerNorm
      -> Linear (-> vocab_size)

    两种用法:
      1. 训练:  forward(idx, targets) 返回 logits + loss (每个位置都预测下一个字符, teacher forcing)
      2. 推理:  forward(idx) 只返回 logits (loss 为 None)

    生成:
      generate(): 自回归采样, temperature + top-k
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token + Position Embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # (V, C): 第 i 行是字符 i 的向量
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # (block_size, C): 第 s 行是位置 s 的向量
        self.drop = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final LayerNorm: Pre-LN 结构里主干上的数值会逐层累加变大, 最后要归一化一次
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Language Model Head: 与 token embedding 共享权重 (GPT-2 标准做法,
        # GPT-2 Small 因此是 124M 而不是 163M)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight                # 两者指向同一块内存, 只算一份参数

        # 初始化权重
        self.apply(self._init_weights)                       # 对每个子模块调用 _init_weights

        # 特殊初始化: 残差投影缩放
        # 每个 Block 往残差主干上加 2 次 (注意力、FFN 各一次), 一共 2*n_layer 次;
        # 把每次加上去的量缩小 √(2*n_layer) 倍, 主干的方差就不随层数增长 (讲义第 9 节)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # 参数统计 (共享的权重只算一次)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT 模型初始化完成: {n_params / 1e6:.1f}M 参数")

    def _init_weights(self, module):
        # std=0.02 是 GPT-2 论文的取值: 足够小, 让初始 logits 接近 0、概率接近均匀,
        # 于是初始 loss ≈ ln(V) (讲义 8.3 节的自检)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: (B, S) token IDs
            targets: (B, S) 目标 token IDs (用于计算 loss), 应是 idx 右移一位
        """
        B, S = idx.shape
        assert S <= self.config.block_size, f"序列长度 {S} > block_size {self.config.block_size}"

        # Token + Position Embedding
        pos = torch.arange(0, S, dtype=torch.long, device=idx.device)    # [0, 1, ..., S-1]
        tok_emb = self.wte(idx)       # (B, S, C)   查表: 每个字符编号换成一个向量
        pos_emb = self.wpe(pos)       # (S, C)      每个位置一个向量
        x = self.drop(tok_emb + pos_emb)   # 广播相加: (B, S, C) + (S, C) -> (B, S, C)

        # Transformer Blocks: 每层进出都是 (B, S, C), 所以可以任意叠加
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # 计算 logits: 每个位置对 V 个字符各打一个分 (还不是概率, softmax 之后才是)
        logits = self.lm_head(x)  # (B, S, vocab_size)

        # 计算 loss (如果提供了 targets)
        loss = None
        if targets is not None:
            # cross_entropy 要求 (N, V) 和 (N,): 把 B、S 两维摊平成 N = B*S 个独立的"猜下一个字符"
            # 用 reshape 而不是 view: targets 可能是切片得到的不连续张量 (例如 x[:, 1:]),
            # view 遇到不连续张量会报错, reshape 会在需要时自动复制一份
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,  # 答案为 -1 的位置不计入 loss (为补齐 padding 预留; 本项目用不到)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成: 每次只预测一个 token, 把它接到末尾, 再预测下一个

        Args:
            idx: (B, T) 起始 token IDs
            max_new_tokens: 最多生成多少个 token
            temperature: 采样温度 (1.0=默认, <1.0=更确定, >1.0=更随机)
            top_k: 只从概率最高的 top_k 个 token 中采样
        """
        for _ in range(max_new_tokens):
            # 截断到 block_size: 位置表只有 block_size 行, 更早的内容模型看不到
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # 前向传播: 整段重新算一遍 (没有 KV Cache; 带缓存的写法见讲义专题 A)
            logits, _ = self(idx_cond)                       # (B, T, V)
            logits = logits[:, -1, :] / temperature  # 只取最后一个位置的预测: (B, V); 除以温度
            # 温度 < 1 放大分数差距 (更确定), > 1 缩小差距 (更随机) (讲义 11.2 节)

            # Top-k 采样: 只保留分数最高的 k 个, 其余设成 -inf (softmax 后概率为 0)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))   # v: 每行前 k 大的值, 降序
                logits[logits < v[:, [-1]]] = float("-inf")             # v[:, [-1]]: 第 k 大的值, 作为门槛

            probs = F.softmax(logits, dim=-1)                # (B, V) 概率分布
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) 按概率随机抽一个

            # 拼接: 新 token 接到末尾, 作为下一轮的输入
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # 快速测试配置
    config = GPTConfig(
        block_size=32,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )

    model = GPT(config)

    # Forward pass
    x = torch.randint(0, 256, (1, 16))
    logits, loss = model(x, targets=x)
    print(f"\nForward: input {x.shape} -> logits {logits.shape}, loss={loss.item():.4f}")

    # Generation
    gen = model.generate(x, max_new_tokens=10, temperature=0.8, top_k=50)
    print(f"Generate: {x.shape} -> {gen.shape}")

    # 参数统计
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    print("\n✅ GPT 模型功能测试通过!")
