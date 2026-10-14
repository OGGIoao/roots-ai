"""
专题 D 实验: 手写注意力 vs PyTorch 融合注意力 (F.scaled_dot_product_attention)

1. 本机能用哪些后端 (flash / mem-efficient / math)
2. 数值是否一致
3. 不同序列长度下, 前向+反向的耗时与峰值显存

运行: python experiments/flash_attention.py   (需要 GPU)
"""

import math
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def manual_attention(q, k, v):
    """与 model.py 中 scaled_dot_product_attention 相同的写法: 显式算出 (S, S) 矩阵。"""
    S = q.size(-2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def fused(backend):
    def run(q, k, v):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return run


BACKENDS = [
    ("手写", manual_attention),
    ("FLASH", fused(SDPBackend.FLASH_ATTENTION)),
    ("MEM_EFFICIENT", fused(SDPBackend.EFFICIENT_ATTENTION)),
    ("MATH", fused(SDPBackend.MATH)),
]


def make(B, H, S, D, dtype):
    return [torch.randn(B, H, S, D, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]


def availability(dtype):
    print(f"[1] 本机可用的后端 (dtype={dtype})")
    q, k, v = make(1, 2, 128, 64, dtype)
    ok = {}
    for name, fn in BACKENDS[1:]:
        try:
            fn(q, k, v)
            ok[name] = True
            print(f"    {name:<14} 可用")
        except RuntimeError as e:
            ok[name] = False
            print(f"    {name:<14} 不可用: {str(e).splitlines()[0][:90]}")
    return ok


def parity(dtype, ok):
    print(f"\n[2] 与手写实现的最大差异 (B=2, H=4, S=256, D=64, {dtype})")
    torch.manual_seed(0)
    q, k, v = make(2, 4, 256, 64, dtype)
    ref = manual_attention(q, k, v)
    for name, fn in BACKENDS[1:]:
        if ok.get(name):
            print(f"    {name:<14} {(fn(q, k, v) - ref).abs().max().item():.2e}")


def bench(name, fn, B, H, S, D, dtype, reps=5):
    q, k, v = make(B, H, S, D, dtype)
    try:
        fn(q, k, v).sum().backward()                       # 预热
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn(q, k, v).sum().backward()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / reps
        peak = (torch.cuda.max_memory_allocated() - base) / 2**20
        return f"{ms:>8.2f}ms {peak:>8.0f}MB"
    except torch.cuda.OutOfMemoryError:
        return f"{'显存不足':>19}"
    finally:
        del q, k, v
        torch.cuda.empty_cache()


def speed(dtype, ok):
    B, H, D = 4, 8, 64
    print(f"\n[3] 前向+反向, B={B} H={H} D={D}, {dtype}  (耗时 / 额外峰值显存)")
    names = [n for n, _ in BACKENDS if n == "手写" or ok.get(n)]
    print("    " + f"{'S':<6}" + "".join(f"{n:>21}" for n in names))
    for S in (512, 1024, 2048, 4096, 8192):
        row = [bench(n, fn, B, H, S, D, dtype) for n, fn in BACKENDS if n in names]
        print("    " + f"{S:<6}" + "".join(f"{r:>21}" for r in row))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)   # 强制指定后端时的 "某内核已禁用" 告警
    assert torch.cuda.is_available(), "需要 GPU"
    print(f"torch {torch.__version__}, GPU {torch.cuda.get_device_name()}\n")
    dt = torch.bfloat16
    ok = availability(dt)
    parity(torch.float32, availability(torch.float32))
    parity(dt, ok)
    speed(dt, ok)
