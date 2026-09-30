"""
model.py — GPT-2 架构完整实现 (Pre-LN Decoder-only)

这是从 llm-transformer/model.py 进化的版本, 专注 GPT-2:
  - Pre-LayerNorm (GPT-2+ 风格)
  - Causal Self-Attention + KV-Cache
  - 词嵌入 + 位置嵌入
  - 训练时的 loss 计算 + 推理时的自回归生成

参考: Karpathy nanoGPT
运行: python model.py  (快速功能测试)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import sys

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
    mask: torch.Tensor,    # (1, 1, S, S) 上三角 causal mask
) -> tuple:
    """
    Scaled Dot-Product Attention
    
    公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
    
    Causal Mask: 防止位置 i 看到位置 j > i 的信息
    用上三角 True 矩阵 + masked_fill(-inf) 实现
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(mask[:, :, :scores.size(-2), :scores.size(-1)], float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
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
        assert config.n_embd % config.n_head == 0
        
        self.n_heads = config.n_head
        self.d_k = config.n_embd // config.n_head
        
        # QKV 投影: 一次矩阵乘法生成 Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask: 注册为 buffer (不参与训练, 但跟随模型移动)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        mask = mask.view(1, 1, config.block_size, config.block_size)
        self.register_buffer("mask", mask)
    
    def forward(self, x):
        B, S, C = x.shape
        
        # QKV 投影
        qkv = self.c_attn(x)  # (B, S, 3*C)
        q, k, v = qkv.split(C, dim=-1)  # 各 (B, S, C)
        
        # 拆分成多头: (B, S, C) -> (B, n_heads, S, d_k)
        q = q.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_out, _ = scaled_dot_product_attention(q, k, v, self.mask)
        
        # 拼接多头: (B, n_heads, S, d_k) -> (B, S, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, C)
        
        # 输出投影 + residual dropout
        return self.resid_dropout(self.c_proj(attn_out))


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
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
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
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)
    
    def forward(self, x):
        # Pre-LN: 先 norm, 再 attention, 再 residual
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ============================================================
# 5. GPT Model
# ============================================================

@dataclass
class GPTConfig:
    block_size: int = 1024   # 最大序列长度
    vocab_size: int = 50257  # GPT-2 词表大小
    n_layer: int = 12        # Transformer 层数
    n_head: int = 12         # 注意力头数
    n_embd: int = 768        # 嵌入维度
    dropout: float = 0.0     # Dropout
    bias: bool = True        # 是否使用 bias


class GPT(nn.Module):
    """
    GPT-2 模型
    
    结构:
      token_embedding + position_embedding
      -> N × Transformer Block (Pre-LN)
      -> LayerNorm
      -> Linear (-> vocab_size)
    
    两种模式:
      1. training:  返回 logits + loss (teacher forcing)
      2. inference: 返回 logits (单 token)
    
    生成:
      generate(): 自回归采样, temperature + top-k
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token + Position Embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        # Language Model Head (与 wte 共享权重, 但这里独立实现)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 特殊初始化: 残差投影缩放
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        # 参数统计
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT 模型初始化完成: {n_params / 1e6:.1f}M 参数")
    
    def _init_weights(self, module):
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
            targets: (B, S) 目标 token IDs (用于计算 loss)
        """
        B, S = idx.shape
        assert S <= self.config.block_size, f"序列长度 {S} > block_size {self.config.block_size}"
        
        # Token + Position Embedding
        pos = torch.arange(0, S, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)       # (B, S, n_embd)
        pos_emb = self.wpe(pos)       # (S, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # 计算 logits
        logits = self.lm_head(x)  # (B, S, vocab_size)
        
        # 计算 loss (如果提供了 targets)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # padding token
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成
        
        Args:
            idx: (B, T) 起始 token IDs
            max_new_tokens: 最多生成多少个 token
            temperature: 采样温度 (1.0=默认, <1.0=更确定, >1.0=更随机)
            top_k: 只从概率最高的 top_k 个 token 中采样
        """
        for _ in range(max_new_tokens):
            # 截断到 block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # 前向传播
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 只取最后一个位置
            
            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # 拼接
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
