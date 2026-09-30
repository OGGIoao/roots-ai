"""
train.py — GPT-2 / LLaMA 训练脚本

用 Tiny Shakespeare 数据集从零训练一个小 GPT 模型。
支持:
  1. GPT-2 Style (Pre-LN + GELU FFN + Learned PE)
  2. LLaMA Style (RMSNorm + SwiGLU + GQA + RoPE)
"""

import os
import sys
import time
import math
import torch
from model import GPTModel, LlamaModel
from data import load_tiny_shakespeare, prepare_data


# ============================================================
# Config
# ============================================================
class Config:
    # 模型选择: "gpt2" 或 "llama"
    model_type: str = "gpt2"

    # GPT-2 Small 配置 (124M)
    vocab_size: int = 0  # 由数据集决定
    d_model: int = 384   # 缩小用于 CPU 训练, GPT-2 原版是 768
    n_heads: int = 6
    n_layers: int = 6    # GPT-2 原版是 12
    d_ff: int = None     # 默认 4*d_model (GPT-2) 或 8/3*d_model (LLaMA)

    # LLaMA 配置 (可选, 需要更大的模型)
    n_kv_heads: int = 2  # GQA

    # 训练配置
    block_size: int = 128  # 上下文长度
    batch_size: int = 32
    max_iters: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_iters: int = 50
    lr_decay_iters: int = 500
    min_lr: float = 1e-5
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_iters: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10
    save_interval: int = 200


cfg = Config()


# ============================================================
# Learning Rate Schedule: Cosine Decay with Warmup
# ============================================================
def get_lr(it):
    """
    学习率调度: 先线性 warmup, 再 cosine 衰减

    这是 LLM 训练的标准做法:
      1. Warmup: 学习率从 0 线性增长到 peak_lr
      2. Cosine Decay: 从 peak_lr 平滑衰减到 min_lr
    """
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def estimate_loss(model, train_loader, val_loader):
    """在训练集和验证集上估计 loss"""
    model.eval()
    losses = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        for i, (x, y) in enumerate(loader):
            if i >= cfg.eval_iters:
                break
            x, y = x.to(cfg.device), y.to(cfg.device)
            _, loss = model(x, y)
            total_loss += loss.item()
        losses[split] = total_loss / min(cfg.eval_iters, i + 1)
    model.train()
    return losses


# ============================================================
# Training Loop
# ============================================================
def train():
    print("=" * 60)
    print("LLM Transformer Training")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] Loading data...")
    text, tokenizer = load_tiny_shakespeare("data")
    train_loader, val_loader, vocab_size = prepare_data(
        text, tokenizer,
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
    )
    cfg.vocab_size = vocab_size

    # 2. 创建模型
    print(f"\n[2/4] Creating model: {cfg.model_type}")
    if cfg.model_type == "gpt2":
        model = GPTModel(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            max_len=cfg.block_size,
        )
    elif cfg.model_type == "llama":
        d_ff = cfg.d_ff or int(8 / 3 * cfg.d_model)
        d_ff = ((d_ff + 255) // 256) * 256
        model = LlamaModel(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            n_layers=cfg.n_layers,
            d_ff=d_ff,
            max_len=cfg.block_size,
        )
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    model = model.to(cfg.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {cfg.model_type}")
    print(f"   Parameters: {params:,} ({params / 1e6:.1f}M)")
    print(f"   Device: {cfg.device}")

    # 3. 创建优化器 (AdamW, LLM 标配)
    print(f"\n[3/4] Setting up optimizer...")
    # weight decay 只应用于权重矩阵, 不应用于 bias 和 LayerNorm/RMSNorm
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=(0.9, 0.95))

    # 4. 训练
    print(f"\n[4/4] Training for {cfg.max_iters} iterations...")
    print("-" * 60)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(cfg.max_iters):
        # 学习率调度
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 定期评估
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader)
            print(
                f"  step {it:4d} | lr {lr:.2e} | "
                f"train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | "
                f"ppl {math.exp(losses['val']):.2f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), "checkpoints/best_model.pt")

        # 训练步
        x, y = next(iter(train_loader))
        x, y = x.to(cfg.device), y.to(cfg.device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # 日志
        if it % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            if it % cfg.eval_interval != 0:
                print(f"  step {it:4d} | loss {loss.item():.4f} | {dt*1000:.0f}ms")

    # 保存最终模型
    torch.save(model.state_dict(), "checkpoints/final_model.pt")
    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to checkpoints/")

    # 5. 生成文本示例
    print("\n" + "=" * 60)
    print("Generating text...")
    prompt = "To be, or not to be"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=cfg.device)
    generated = model.generate(prompt_ids, max_new_tokens=200, temperature=0.8, top_k=50)
    print(f"\nPrompt: {prompt}")
    print(f"Generated:\n{tokenizer.decode(generated[0].tolist())}")


if __name__ == "__main__":
    # 支持命令行选择模型
    if len(sys.argv) > 1:
        cfg.model_type = sys.argv[1]
    train()
