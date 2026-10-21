"""练习 3 的验收测试。"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from base import Decoder  # noqa: E402
from ex3_mla_absorbed import absorbed_step  # noqa: E402
from variants import MultiHeadLatentAttention  # noqa: E402

DIM, HEADS, R, D_NOPE, D_ROPE, D_V, LAYERS = 64, 4, 32, 16, 8, 16, 2


def _model():
    torch.manual_seed(0)
    return Decoder(65, 64, DIM, LAYERS, lambda: MultiHeadLatentAttention(DIM, HEADS, R, D_NOPE, D_ROPE, D_V)).eval()


def _guard_decompression(model):
    """把 w_ukv 的前向换成"一次只能处理一个位置"的版本: 解压历史缓存就会报错。"""
    for block in model.blocks:
        lin = block.attn.w_ukv
        original = lin.forward

        def guarded(x, _orig=original):
            if x.dim() == 3 and x.size(1) > 1:
                raise AssertionError("解码时解压了历史位置的潜向量 (w_ukv 被用在了多个位置上)")
            return _orig(x)
        lin.forward = guarded


@torch.no_grad()
def _decode(model, seq):
    caches = [{} for _ in model.blocks]
    logits = []
    for pos in range(seq.size(1)):
        cos, sin = model.cos[pos:pos + 1], model.sin[pos:pos + 1]
        x = model.embed(seq[:, pos:pos + 1])
        for block, cache in zip(model.blocks, caches):
            x = x + absorbed_step(block.attn, block.norm1(x), cos, sin, cache)
            x = x + block.ffn(block.norm2(x))
        logits.append(model.lm_head(model.norm(x))[:, -1])
    return torch.stack(logits, dim=1), caches


def test_matches_full_forward():
    model = _model()
    seq = torch.randint(0, 65, (2, 40))
    with torch.no_grad():
        full, _ = model(seq)
    stepped, _ = _decode(model, seq)
    assert (stepped - full).abs().max().item() < 1e-4


def test_cache_holds_only_latent_and_rope_key():
    model = _model()
    seq = torch.randint(0, 65, (2, 40))
    _, caches = _decode(model, seq)
    for cache in caches:
        numbers = sum(t.numel() for t in cache.values() if torch.is_tensor(t))
        assert numbers == 2 * 40 * (R + D_ROPE), f"每个位置缓存了 {numbers / 80:.0f} 个数, 应为 {R + D_ROPE}"


def test_no_decompression_of_history():
    model = _model()
    _guard_decompression(model)
    _decode(model, torch.randint(0, 65, (2, 10)))
