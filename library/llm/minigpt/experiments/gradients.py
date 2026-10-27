"""
专题 B 实验: 梯度推导的数值验证

1. softmax + 交叉熵对 logits 的梯度 = p - onehot(y)
2. softmax 的雅可比矩阵 J = diag(p) - p pᵀ, 以及分布接近 one-hot 时 J → 0
3. 注意力里不除以 √d_k 时, 梯度在各位置之间怎样分布

运行: python experiments/gradients.py
"""

import math
import statistics

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def ce_gradient():
    z = torch.randn(65, requires_grad=True)
    y = 7
    F.cross_entropy(z.unsqueeze(0), torch.tensor([y])).backward()
    p = torch.softmax(z.detach(), 0)
    onehot = torch.zeros(65)
    onehot[y] = 1.0
    print("[1] 交叉熵对 logits 的梯度")
    print(f"    autograd 与 p - onehot 的最大差异: {(z.grad - (p - onehot)).abs().max():.2e}")
    print(f"    正确答案位置的梯度 = p_y - 1 = {z.grad[y]:.4f}  (p_y = {p[y]:.4f})")
    print(f"    梯度各分量之和 = {z.grad.sum():.2e}  (恒为 0)")


def softmax_jacobian():
    z = torch.tensor([1.0, 2.0, 3.0])
    J_auto = torch.autograd.functional.jacobian(lambda t: torch.softmax(t, 0), z)
    p = torch.softmax(z, 0)
    J_formula = torch.diag(p) - torch.outer(p, p)
    print("\n[2] softmax 的雅可比矩阵 J = diag(p) - p pᵀ")
    print(f"    autograd 与公式的最大差异: {(J_auto - J_formula).abs().max():.2e}")
    print("    输入放大后 (分布越来越接近 one-hot), J 的最大元素:")
    for c in (1, 4, 8, 16):
        p = torch.softmax(z * c, 0)
        J = torch.diag(p) - torch.outer(p, p)
        print(f"      z × {c:<3} p = {[round(v, 4) for v in p.tolist()]}   max|J| = {J.abs().max():.2e}")


def attention_grad(d_k, scaled, S=32, trials=200):
    """单头注意力, 随机 q、k、v 与随机上游梯度。

    统计 softmax 输入 (注意力分数) 上的梯度, 因为饱和压小的正是"穿过 softmax"的这一段。
    返回: 平均最大权重, 收到有效梯度的位置比例, 饱和行比例, 饱和行/其余行的梯度中位数
    """
    maxw = share = sat = 0.0
    sat_g, ok_g = [], []
    for _ in range(trials):
        q, k, v = torch.randn(S, d_k), torch.randn(S, d_k), torch.randn(S, d_k)
        scores = q @ k.T
        if scaled:
            scores = scores / math.sqrt(d_k)
        scores.requires_grad_()
        w = torch.softmax(scores, -1)
        out = w @ v
        (out * torch.randn_like(out)).sum().backward()
        G = scores.grad.abs()
        row_max_w = w.max(-1).values
        maxw += row_max_w.mean().item() / trials
        # 一行里梯度超过该行最大梯度 1% 的位置, 视为"收到了有效梯度"
        share += (G > 0.01 * G.max(-1, keepdim=True).values).float().mean().item() / trials
        sat += (row_max_w > 0.99).float().mean().item() / trials
        row_g = scores.grad.norm(dim=-1)
        sat_g += row_g[row_max_w > 0.99].tolist()
        ok_g += row_g[row_max_w <= 0.99].tolist()
    med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
    return maxw, share, sat, med(sat_g), med(ok_g)


def scaling_effect():
    print("\n[3] 注意力: 除不除以 √d_k 对梯度的影响 (S=32, 200 次)")
    print(f"    {'d_k':<5} {'方式':<8} {'平均最大权重':>10} {'有效梯度位置':>10} {'饱和行':>8} "
          f"{'饱和行梯度':>10} {'其余行梯度':>10}")
    for d_k in (16, 64, 256):
        for scaled in (True, False):
            mw, share, sat, g_sat, g_ok = attention_grad(d_k, scaled)
            tag = "÷√d_k" if scaled else "不缩放"
            g_sat_s = "—" if math.isnan(g_sat) else f"{g_sat:.4f}"
            print(f"    {d_k:<5} {tag:<8} {mw:>10.3f} {share:>10.1%} {sat:>8.1%} "
                  f"{g_sat_s:>10} {g_ok:>10.4f}")
    print("    (饱和行 = 最大注意力权重 > 0.99 的行; 梯度为该行注意力分数梯度的 L2 范数中位数)")


if __name__ == "__main__":
    ce_gradient()
    softmax_jacobian()
    scaling_effect()
