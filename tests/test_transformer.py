"""llm-transformer 回归测试 (CPU, 极小模型)。"""

import os
import sys

import pytest
import torch

_PROJ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "projects", "llm-transformer")
sys.path.insert(0, _PROJ)

from model import GPTModel, LlamaModel  # noqa: E402


def _tiny(kind, max_len):
    if kind == "gpt2":
        return GPTModel(vocab_size=50, d_model=32, n_heads=2, n_layers=2, max_len=max_len, dropout=0.0)
    return LlamaModel(vocab_size=50, d_model=32, n_heads=4, n_kv_heads=2, n_layers=2, max_len=max_len)


@pytest.mark.parametrize("kind", ["gpt2", "llama"])
def test_generate_past_max_len(kind):
    # 回归: generate 曾把上下文窗口写死为 1024/4096, max_len 更小时位置索引越界
    model = _tiny(kind, max_len=16)
    out = model.generate(torch.randint(0, 50, (1, 8)), max_new_tokens=40)
    assert out.shape == (1, 48)


def test_gpt_position_embedding_added_once():
    # 回归: token embedding 曾被加两遍 (pos_emb 内部已 x + pos)
    model = _tiny("gpt2", max_len=16).eval()
    ids = torch.randint(0, 50, (1, 5))
    with torch.no_grad():
        expected = model.token_emb(ids) + model.pos_emb.pos_embedding(torch.arange(5))
        got = model.pos_emb(model.token_emb(ids))
    assert torch.allclose(got, expected)
    # forward 的输入嵌入应与 expected 一致: 屏蔽 blocks 后 logits 只由 ln_f(emb) 决定
    model.blocks = torch.nn.ModuleList()
    with torch.no_grad():
        logits, _ = model(ids)
        ref = model.lm_head(model.ln_f(expected))
    assert torch.allclose(logits, ref, atol=1e-6)
