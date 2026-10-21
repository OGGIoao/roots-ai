"""
专题 C 实验: fp32 / fp16 / bf16 与混合精度

1. 三种格式的位数、范围、精度
2. 三类数值事故: 上溢、下溢、"加了等于没加"
3. loss scaling 为什么能救 fp16 的下溢
4. 复现 C2-tool-finetune 里 "fp16 scaler + bf16 参数" 的崩溃

运行: python experiments/precision.py
"""

import torch

DTYPES = [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]


def formats():
    print("[1] 三种格式 (符号位 + 指数位 + 尾数位)")
    bits = {"fp32": "1 + 8 + 23", "fp16": "1 + 5 + 10", "bf16": "1 + 8 + 7"}
    print(f"    {'格式':<6} {'位分配':<12} {'最大值':>12} {'最小正规数':>12} {'eps (1 之后的间距)':>18}")
    for name, dt in DTYPES:
        fi = torch.finfo(dt)
        print(f"    {name:<6} {bits[name]:<12} {fi.max:>12.4g} {fi.tiny:>12.4g} {fi.eps:>18.4g}")


def accidents():
    print("\n[2] 三类数值事故")
    for name, dt in DTYPES:
        big = torch.tensor(70000.0, dtype=dt)
        small = torch.tensor(1e-8, dtype=dt)
        one = torch.tensor(1.0, dtype=dt)
        print(f"    {name}:  70000 存进来 = {big.item():<10.6g}"
              f"  1e-8 存进来 = {small.item():<10.4g}"
              f"  1 + 0.001 = {(one + torch.tensor(1e-3, dtype=dt)).item():.6f}")

    print("\n    参数更新被吞掉: 参数 = 1.0, 每步加上 lr × grad = 1e-4, 共 1000 步 (理论结果 1.1)")
    for name, dt in DTYPES:
        w = torch.tensor(1.0, dtype=dt)
        for _ in range(1000):
            w = w + torch.tensor(1e-4, dtype=dt)
        print(f"      {name}: {w.item():.6f}")


def loss_scaling():
    print("\n[3] loss scaling: 梯度太小, fp16 存不下怎么办")
    grads = torch.tensor([1e-8, 3e-8, 2e-7, 5e-6, 1e-3])
    s = 2.0 ** 16
    direct = grads.half().float()
    rescued = (grads * s).half().float() / s

    def rel(x):
        return "  —  " if x != x else f"{x:6.1%}"

    print(f"    {'真实梯度':>10} {'直接转 fp16':>12} {'误差':>7}   {'×65536 后转 fp16 再还原':>22} {'误差':>7}")
    for g, d, r in zip(grads.tolist(), direct.tolist(), rescued.tolist()):
        print(f"    {g:>10.3g} {d:>12.3g} {rel(abs(d - g) / g):>7}   {r:>22.3g} {rel(abs(r - g) / g):>7}")


def qlora_crash():
    print("\n[4] 复现 QLoRA 崩溃: fp16 的 GradScaler 遇上 bf16 参数")
    if not torch.cuda.is_available():
        print("    (需要 GPU, 跳过)")
        return
    model = torch.nn.Linear(16, 16).cuda().to(torch.bfloat16)     # 参数本身是 bf16
    opt = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda")                          # 为 fp16 设计的 loss scaler
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)).float().pow(2).mean()
    scaler.scale(loss).backward()
    try:
        scaler.unscale_(opt)
        print("    未报错")
    except Exception as e:  # noqa: BLE001
        print(f"    {type(e).__name__}: {str(e).splitlines()[0]}")

    model = torch.nn.Linear(16, 16).cuda().to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)).float().pow(2).mean()
    loss.backward()                                                # bf16: 不需要 scaler
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print("    改用 bf16 autocast、去掉 scaler 后: 正常完成一步更新")


if __name__ == "__main__":
    formats()
    accidents()
    loss_scaling()
    qlora_crash()
