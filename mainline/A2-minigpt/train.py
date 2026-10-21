"""
train.py — miniGPT 训练脚本

从零训练一个 GPT-2 级别的语言模型。讲义: TUTORIAL.md 第 10 节。

使用方式:
  python train.py                    # 使用默认配置 (nano)
  python train.py --config small     # GPT-2 Small (124M)
  python train.py --config micro     # 微型配置
  python train.py --lr 3e-4          # 自定义学习率

训练技巧:
  1. AdamW 优化器 + Cosine LR Schedule
  2. Gradient Clipping (max_norm=1.0)
  3. Mixed Precision (fp16 autocast + GradScaler) 可选
  4. 验证集评估 + Checkpoint 保存
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.optim as optim

from model import GPT
from config import GPTConfig, MODELS
from data import download_tiny_shakespeare, create_dataloaders


# ============================================================
# 1. 学习率调度 (Cosine with Warmup)
# ============================================================

def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """
    Cosine LR Schedule with Linear Warmup (曲线见讲义 10.2 节的图)

    1. 0 ~ warmup_iters:  线性增长 (0 -> learning_rate)
    2. warmup ~ decay:    余弦衰减 (learning_rate -> min_lr)
    3. decay ~ inf:       保持 min_lr

    为什么需要 warmup?
      - 训练初期梯度不稳定, 大学习率会导致 loss 爆炸
      - 先用小学习率预热, 稳定后再增大学习率
    """
    if it < warmup_iters:                                    # 阶段 1: 从 0 线性升到峰值
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:                                  # 阶段 3: 衰减结束后保持最低值
        return min_lr

    # 阶段 2: decay_ratio 从 0 走到 1; cos(π·ratio) 从 1 走到 -1, 所以 coeff 从 1 平滑降到 0
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)        # coeff=1 时为峰值, coeff=0 时为 min_lr


# ============================================================
# 2. 取 batch 与评估
# ============================================================

def infinite_batches(loader):
    """无限循环地产出 batch: 一个 epoch 用完就重新打乱开始下一个。

    不能每步都写 next(iter(loader)): 那样每步都新建迭代器 (开销大),
    而对不打乱的 loader 来说, 永远只会拿到第一个 batch (讲义 10.4 节的坑 #3)。
    """
    while True:
        for batch in loader:          # 每次进入 for, DataLoader 都会按 shuffle 设置重新排一遍顺序
            yield batch


@torch.no_grad()                      # 评估不需要梯度: 省显存、更快
def evaluate(model, loader, device, use_amp, max_batches=20):
    """在 loader 的前 max_batches 个 batch 上求平均 loss。

    val_loader 不打乱, 所以每次评估看的都是同一批数据 ——
    这正是我们想要的: 不同时刻的 val_loss 才能直接比较 (用于挑最佳 checkpoint)。
    """
    model.eval()                      # 关闭 dropout, 同一个输入每次结果相同
    total, n = 0.0, 0
    for x, y in loader:
        if n >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, enabled=use_amp):
            _, loss = model(x, y)
        total += loss.item()          # .item(): 取出 Python 数值, 不保留计算图
        n += 1
    model.train()                     # 切回训练模式 (重新打开 dropout); 忘了这一步, 后面的训练就没有 dropout 了
    return total / max(n, 1)


# ============================================================
# 3. 训练循环
# ============================================================

def train(args):
    """主训练循环。返回 (每步的 train loss 列表, 每次评估的 val loss 列表)。"""

    # ── 设备 ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # ── 数据 ──
    text = download_tiny_shakespeare()
    train_loader, val_loader, tokenizer = create_dataloaders(
        text,
        block_size=args.block_size,
        batch_size=args.batch_size,
    )

    # ── 模型 ──
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,   # 词表大小由数据决定 (字符级约 65), 不取预设里的值
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config)
    model = model.to(device)

    # ── 优化器 ──
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,               # 只是初始值, 每一步都会被学习率调度覆盖
        betas=(0.9, 0.95),        # β2=0.95 比默认 0.999 记忆更短, 梯度突变时反应更快 (GPT 系列的常用设置)
        eps=1e-8,
        weight_decay=0.1,         # 对所有参数都施加 (更常见的做法是只对二维权重施加, 见讲义练习 3)
    )

    # ── 混合精度 ──
    # 只在 GPU 上且传了 --amp 时启用。参数保持 fp32, 计算用 fp16;
    # GradScaler 先放大 loss 防止小梯度在 fp16 下溢成 0 (讲义专题 C)
    use_amp = device == "cuda" and args.amp
    scaler = torch.amp.GradScaler(device, enabled=use_amp)   # enabled=False 时它的方法什么都不做

    # ── 学习率调度参数 ──
    warmup_iters = args.warmup_iters
    lr_decay_iters = args.max_iters       # 在整个训练过程内完成一次余弦衰减
    learning_rate = args.lr
    min_lr = args.lr * 0.1  # 最低学习率为初始的 10%

    # ── Checkpoint 目录 ──
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # ── 训练状态 ──
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print(f"\n{'=' * 60}")
    print(f"开始训练")
    print(f"  模型: {config.n_layer} 层, {config.n_head} 头, {config.n_embd} 维")
    print(f"  参数: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  数据: {len(train_loader.dataset)} 训练块, {len(val_loader.dataset)} 验证块")
    print(f"  优化: lr={args.lr}, batch_size={args.batch_size}, max_iters={args.max_iters}")
    print(f"{'=' * 60}\n")

    # ── 训练循环 ──
    model.train()
    batches = infinite_batches(train_loader)
    for it in range(args.max_iters):
        t0 = time.time()

        # 学习率调度: AdamW 的学习率存在每个参数组里, 每一步手动改写
        lr = get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 获取 batch: x、y 都是 (B, S), y 是 x 右移一位 (讲义 2.3 节)
        x, y = next(batches)
        x, y = x.to(device), y.to(device)

        # 前向传播 (autocast: 在 enabled=True 时自动把合适的运算换成 fp16)
        with torch.autocast(device_type=device, enabled=use_amp):
            logits, loss = model(x, y)

        # 反向传播
        optimizer.zero_grad()                 # 清掉上一步的梯度 (PyTorch 默认把新梯度累加到旧梯度上)
        scaler.scale(loss).backward()         # 混合精度时先放大 loss 再求梯度; 否则等同于 loss.backward()

        # Gradient Clipping (防止梯度爆炸)
        scaler.unscale_(optimizer)            # 先把梯度缩回真实尺度, 裁剪的阈值 1.0 才有意义
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 所有梯度的总长度超过 1 就整体按比例缩小

        # 更新参数
        scaler.step(optimizer)                # 混合精度时若梯度里有 inf/NaN, 会跳过这一步
        scaler.update()                       # 调整放大倍数

        # 记录 loss 与本步耗时 (只算训练步, 不含下面的评估/生成)
        train_losses.append(loss.item())
        dt = time.time() - t0

        # ── 验证 ──
        if it % args.val_interval == 0 or it == args.max_iters - 1:
            val_loss = evaluate(model, val_loader, device, use_amp)
            val_losses.append(val_loss)

            # 保存最佳 checkpoint: 小语料上模型会过拟合, val_loss 最低的那一版才是最好的
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "iter": it,
                    "val_loss": val_loss,
                }
                torch.save(checkpoint, ckpt_dir / "model_best.pt")

        # ── 日志输出 ──
        if it % args.log_interval == 0:
            print(
                f"iter {it:>5d}/{args.max_iters} | "
                f"train_loss {loss.item():.4f} | "
                f"val_loss {val_loss if val_losses else float('nan'):.4f} | "   # 最近一次评估的值
                f"lr {lr:.2e} | "
                f"{dt*1000:.1f}ms/iter"
            )

        # ── 生成示例文本 ──
        if it % args.gen_interval == 0 and it > 0:
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=device)   # 编号 0 = 换行符, 作为开头
            generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=50)
            text_gen = tokenizer.decode(generated[0].tolist())
            print(f"\n{'─' * 40}")
            print(f"生成示例 (iter {it}):")
            print(f"{text_gen}")
            print(f"{'─' * 40}\n")
            model.train()

    # ── 保存最终模型 ──
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "iter": args.max_iters,
        "val_loss": best_val_loss,
    }
    torch.save(checkpoint, ckpt_dir / "model_final.pt")

    print(f"\n{'=' * 60}")
    print(f"训练完成!")
    print(f"  最佳 val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {ckpt_dir / 'model_best.pt'}")
    print(f"{'=' * 60}")

    return train_losses, val_losses


# ============================================================
# 4. 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="miniGPT 训练")

    # 模型配置 (不传时取 --config 指定的预设)
    parser.add_argument("--config", type=str, default="nano",
                        choices=list(MODELS),
                        help="模型配置名称")
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 训练配置
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--amp", action="store_true", help="使用混合精度训练")

    # 日志配置
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--gen_interval", type=int, default=500)

    # 路径
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    # 应用预定义配置: 命令行没给的项, 用预设的值补上
    # (词表大小由数据集的 tokenizer 决定, 不取预设里的 vocab_size)
    preset = MODELS[args.config]
    for k in ("n_layer", "n_head", "n_embd", "block_size"):
        if getattr(args, k) is None:
            setattr(args, k, getattr(preset, k))

    return args


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    train(args)
