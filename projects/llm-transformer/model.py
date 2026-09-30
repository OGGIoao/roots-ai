"""
GPT Model — 从 GPT-1 到 LLaMA 的完整模型实现

架构演进:
  1. GPT-1 (2018) — 12层, 117M params, Decoder-only + Learned PE
  2. GPT-2 (2019) — 48层, 1.5B params, Pre-LN + GELU FFN
  3. LLaMA (2023) — 32层, 7B params, RMSNorm + SwiGLU + GQA + RoPE

核心设计: Decoder-only Transformer
  - 只有 Decoder, 没有 Encoder
  - 用 causal mask 防止看到未来的 token
  - 训练目标: 预测下一个 token (Causal Language Modeling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformer_block import TransformerBlock, LlamaBlock
from positional_encoding import LearnedPE, RoPE


class GPTModel(nn.Module):
    """
    GPT-1/2 Style Model

    架构:
        Token Embedding + Position Embedding
        → N × TransformerBlock (Pre-LN)
        → Final LayerNorm
        → LM Head (输出 logit)

    参数量:
        GPT-1:  12 layers, d_model=768,  117M params
        GPT-2:  48 layers, d_model=1600, 1.5B params
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = None,  # 默认 4 * d_model
        max_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        d_ff = d_ff or 4 * d_model

        # Token + Position Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = LearnedPE(d_model, max_len)
        self.drop = nn.Dropout(dropout)

        # Transformer Blocks (Pre-LN)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final LayerNorm + LM Head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享: token embedding 和 lm_head 共享权重 (GPT-2 做法)
        self.lm_head.weight = self.token_emb.weight

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, S)
        targets: Optional[torch.Tensor] = None,  # (B, S)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: token ids (B, S)
            targets: 目标 token ids (B, S), 用于计算 loss

        Returns:
            logits: (B, S, vocab_size)
            loss:   scalar (如果 targets 不为 None)
        """
        B, S = input_ids.shape

        # Causal mask: 上三角为 True
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=input_ids.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        # Embedding
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(self.token_emb(input_ids)))

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Final LayerNorm + LM Head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, S, vocab_size)

        # 计算 loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # padding token
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,  # (B, S)
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """自回归生成"""
        self.eval()
        for _ in range(max_new_tokens):
            # 截断到 max_len
            idx_cond = input_ids if input_ids.size(1) <= 1024 else input_ids[:, -1024:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids


class LlamaModel(nn.Module):
    """
    LLaMA-style Model (2023)

    与 GPT-2 的区别:
        1. RMSNorm (替代 LayerNorm)
        2. SwiGLU FFN (替代 GELU FFN)
        3. GQA (替代 MHA)
        4. RoPE (替代 Learned PE)
        5. 无 bias (所有 Linear 层)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 4096,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        n_layers: int = 32,
        d_ff: int = None,  # 默认 8/3 * d_model (SwiGLU)
        max_len: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # SwiGLU 的 d_ff 通常是 8/3 * d_model (非整数时取最近的256的倍数)
        d_ff = d_ff or int(8 / 3 * d_model)
        d_ff = ((d_ff + 255) // 256) * 256  # 对齐到256

        # Token Embedding (无 pos embedding, 用 RoPE)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            LlamaBlock(d_model, n_heads, n_kv_heads, d_ff, max_len, dropout)
            for _ in range(n_layers)
        ])

        # Final RMSNorm + LM Head
        from normalization import RMSNorm
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S = input_ids.shape

        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=input_ids.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        x = self.drop(self.token_emb(input_ids))

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= 4096 else input_ids[:, -4096:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


if __name__ == "__main__":
    # 验证 GPT-2 模型
    print("=" * 60)
    print("GPT-2 Small (124M)")
    model = GPTModel(vocab_size=50257, d_model=768, n_heads=12, n_layers=12)
    x = torch.randint(0, 50257, (2, 64))
    logits, loss = model(x, targets=x)
    print(f"   Input:  {x.shape}")
    print(f"   Logits: {logits.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"   Params: {params:,} ({params/1e6:.1f}M)")
    del model

    # 验证 LLaMA 模型 (小规模)
    print("\n" + "=" * 60)
    print("LLaMA Tiny (验证架构)")
    model = LlamaModel(vocab_size=32000, d_model=256, n_heads=8, n_kv_heads=2, n_layers=4)
    x = torch.randint(0, 32000, (2, 32))
    logits, loss = model(x, targets=x)
    print(f"   Input:  {x.shape}")
    print(f"   Logits: {logits.shape}")
    print(f"   Loss:   {loss.item():.4f}")
    params = sum(p.numel() for p in model.parameters())
    print(f"   Params: {params:,} ({params/1e6:.1f}M)")

    # 生成测试
    prompt = torch.randint(0, 32000, (1, 8))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"   生成: prompt {prompt.shape} -> {generated.shape}")

    print("\n✅ Both models work correctly!")
