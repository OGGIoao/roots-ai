"""统一的对照实验框架: 树上的每个节点都在同一个任务、同一套训练设置下比较。

任务: Tiny Shakespeare 字符级语言模型 (与 llm-minigpt 相同的数据与切分)。
被比较的模型只需满足一个接口:

    model(idx, targets) -> (logits, loss)          # idx, targets: (B, S) 整数
    model.kv_cache_bytes_per_token(dtype_bytes=2)   # 可选: 推理时每个 token 新增多少缓存

用法:
    from common.bench import BenchConfig, compare
    rows = compare({"MHA": lambda V, S: build_mha(V, S), ...}, BenchConfig(steps=2000))

每个模型用同一个随机种子初始化、看同样顺序的 batch, 差异只来自模型本身。
"""

import math
import time
from dataclasses import asdict, dataclass, replace

import torch

from common.corpus import load_shakespeare
from common.text_data import CharTokenizer, make_loaders


@dataclass
class BenchConfig:
    steps: int = 2000
    block_size: int = 256
    batch_size: int = 32
    lr: float = 1e-3
    warmup: int = 100
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_batches: int = 20
    seed: int = 1337
    eval_every: int = 0       # >0 时每隔这么多步记录一次 val_loss, 用来看曲线 (检查过拟合)
    amp: bool = True          # cuda 上用 bf16 autocast (本机不需要 GradScaler, 见 minigpt 专题 C)


def _lr(it, cfg):
    if it < cfg.warmup:
        return cfg.lr * (it + 1) / cfg.warmup
    ratio = (it - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * ratio)))


def _batches(loader, seed):
    g = torch.Generator().manual_seed(seed)       # 每个模型看到的 batch 顺序完全相同
    while True:
        for i in torch.randperm(len(loader.dataset), generator=g).tolist():
            yield i


@torch.no_grad()
def _evaluate(model, loader, cfg, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        if n >= cfg.eval_batches:
            break
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=cfg.amp and device == "cuda"):
            _, loss = model(x.to(device), y.to(device))
        total, n = total + loss.item(), n + 1
    model.train()
    return total / n


def run(name, build, cfg=BenchConfig(), device=None, log_every=0):
    """训练一个模型并返回指标。build(vocab_size, block_size) -> nn.Module"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    text = load_shakespeare()
    tok = CharTokenizer(text)
    train_loader, val_loader = make_loaders(tok.encode(text), cfg.block_size, cfg.batch_size, stride=cfg.block_size)
    ds = train_loader.dataset

    torch.manual_seed(cfg.seed)
    model = build(tok.vocab_size, cfg.block_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    order = _batches(train_loader, cfg.seed)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    curve = []
    t0 = time.perf_counter()
    for it in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = _lr(it, cfg)
        idx = [next(order) for _ in range(cfg.batch_size)]
        x = torch.stack([ds[i][0] for i in idx]).to(device)
        y = torch.stack([ds[i][1] for i in idx]).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=cfg.amp and device == "cuda"):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if log_every and it % log_every == 0:
            print(f"    [{name}] step {it:>5} loss {loss.item():.4f}")
        if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            t_eval = time.perf_counter()
            curve.append((it + 1, round(loss.item(), 4), round(_evaluate(model, val_loader, cfg, device), 4)))
            t0 += time.perf_counter() - t_eval          # 评估时间不计入训练速度
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    final_val = _evaluate(model, val_loader, cfg, device)
    # Tiny Shakespeare 只有约 100 万字符, 模型很快就会过拟合 (val_loss 先降后升)。
    # 有曲线时取曲线上的最低点 (相当于早停), 否则只能用最后一步。
    best_step, best_val = min(((s, v) for s, _, v in curve), key=lambda t: t[1], default=(cfg.steps, final_val))
    kv = model.kv_cache_bytes_per_token(2) if hasattr(model, "kv_cache_bytes_per_token") else None
    return {
        "name": name,
        "seed": cfg.seed,
        "params": n_params,
        "best_val": best_val,
        "best_step": best_step,
        "final_val": final_val,
        "kv_bytes_per_token": kv,
        "tokens_per_s": cfg.steps * cfg.batch_size * cfg.block_size / elapsed,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
        "curve": curve,                   # [(步数, 当步 train loss, val_loss), ...]
        "config": asdict(cfg),
        "device": device,
    }


def compare(builders, cfg=BenchConfig(), seeds=(1337,), device=None, log_every=0):
    """每个模型用每个种子各训练一次, 打印汇总表并返回全部结果。builders: {名字: build 函数}"""
    rows = []
    for name, build in builders.items():
        for seed in seeds:
            print(f"  训练 {name} (seed={seed}) ...", flush=True)
            rows.append(run(name, build, replace(cfg, seed=seed), device, log_every))
    print_table(rows)
    return rows


def summarize(rows):
    """按模型名汇总多个种子: best_val 的均值与标准差, 其余指标取均值。"""
    out = {}
    for r in rows:
        out.setdefault(r["name"], []).append(r)
    summary = []
    for name, rs in out.items():
        vals = [r["best_val"] for r in rs]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        summary.append({
            "name": name, "n_seeds": len(rs), "params": rs[0]["params"],
            "best_val_mean": mean, "best_val_std": std,
            "best_step_mean": sum(r["best_step"] for r in rs) / len(rs),
            "kv_bytes_per_token": rs[0]["kv_bytes_per_token"],
            "tokens_per_s": sum(r["tokens_per_s"] for r in rs) / len(rs),
            "peak_mem_mb": rs[0]["peak_mem_mb"],
        })
    return summary


def print_table(rows):
    summary = summarize(rows)
    base = summary[0]
    print(f"\n  {'模型':<12} {'参数量':>10} {'最佳 val_loss':>18} {'Δ':>7} {'最佳步':>6} {'KV/token':>9} {'KV 相对':>7} "
          f"{'tokens/s':>9} {'峰值显存':>8}")
    for r in summary:
        kv = r["kv_bytes_per_token"]
        kv_s = "—" if kv is None else f"{kv:,}B"
        rel = "—" if kv is None or not base["kv_bytes_per_token"] else f"{kv / base['kv_bytes_per_token']:.0%}"
        mem = "—" if r["peak_mem_mb"] is None else f"{r['peak_mem_mb']:.0f}MB"
        val = f"{r['best_val_mean']:.4f} ± {r['best_val_std']:.4f}"
        print(f"  {r['name']:<12} {r['params']:>10,} {val:>18} {r['best_val_mean'] - base['best_val_mean']:>+7.3f} "
              f"{r['best_step_mean']:>6.0f} {kv_s:>9} {rel:>7} {r['tokens_per_s']:>9,.0f} {mem:>8}")
    print(f"  (± 为 {base['n_seeds']} 个随机种子的标准差; 最佳 = 训练曲线上 val_loss 的最低点, 相当于早停)")
