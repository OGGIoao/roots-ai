"""
figure_data.py — 为讲义的数据图生成"需要训练好的模型"才有的数据

训练一个 nano 模型 (与 `python train.py --config nano` 相同的设置, 本机约 1 分钟), 然后记录:
  1. 训练曲线: 每 100 步的 train loss 均值与 val loss        → results/nano_curve.json
  2. 注意力热力图: 训练好的模型在一句台词上的注意力权重      → results/nano_attention.json
  3. 温度的影响: 同一个前缀在不同温度下的下一个字符概率       → results/nano_temperature.json

画图: python figures/make_charts.py
运行: python experiments/figure_data.py
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import model as model_module  # noqa: E402
import train as train_module  # noqa: E402
from data import CharTokenizer  # noqa: E402
from common.corpus import load_shakespeare  # noqa: E402

RES = os.path.join(HERE, "results")
CKPT = os.path.join(RES, "ckpt")            # .pt 文件被 .gitignore 忽略, 不入库


def train_nano():
    """与 train.py 的默认 nano 设置相同, 只是每 100 步评估一次, 好画出完整曲线。"""
    args = argparse.Namespace(
        config="nano", n_layer=2, n_head=2, n_embd=64, block_size=64, dropout=0.1,
        batch_size=32, max_iters=5000, lr=1e-3, warmup_iters=100, amp=False,
        log_interval=10**9, val_interval=100, gen_interval=10**9, ckpt_dir=CKPT,
    )
    torch.manual_seed(1337)
    with contextlib.redirect_stdout(io.StringIO()):
        train_losses, val_losses = train_module.train(args)
    # val 在第 0、100、200、...、4900 步以及最后一步 (4999) 评估; train loss 每 100 步取均值, 让曲线平滑
    steps = list(range(0, 5000, 100)) + [4999]
    train_avg = [statistics.mean(train_losses[s:s + 100]) for s in range(0, 5000, 100)] + [train_losses[-1]]
    out = {"steps": steps, "train_loss_avg100": train_avg, "val_loss": val_losses,
           "note": "train_loss_avg100[i] 是从第 steps[i] 步起 100 步的 train loss 均值; val_loss 为该步的评估值"}
    with open(os.path.join(RES, "nano_curve.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    best = min(range(len(val_losses)), key=lambda i: val_losses[i])
    print(f"训练完成: 最低 val_loss {val_losses[best]:.4f} @ 第 {steps[best]} 步")


def load_best():
    ck = torch.load(os.path.join(CKPT, "model_best.pt"), weights_only=False)
    with contextlib.redirect_stdout(io.StringIO()):
        m = model_module.GPT(ck["config"])
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def attention_maps(m, tok):
    """截获每一层、每个头的注意力权重 (model.py 的 scaled_dot_product_attention 会返回它, 但 forward 里丢弃了)。"""
    text = "ROMEO:\nI will go thither."
    captured = []
    original = model_module.scaled_dot_product_attention

    def spy(q, k, v, mask):
        out, w = original(q, k, v, mask)
        captured.append(w[0].tolist())          # (H, S, S), 只取第一条样本
        return out, w

    model_module.scaled_dot_product_attention = spy
    try:
        m(torch.tensor([tok.encode(text)]))
    finally:
        model_module.scaled_dot_product_attention = original
    out = {"text": text, "chars": list(text), "layers": captured,
           "note": "layers[l][h][i][j] = 第 l 层第 h 个头里, 位置 i 对位置 j 的注意力权重"}
    with open(os.path.join(RES, "nano_attention.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"注意力: {len(captured)} 层 × {len(captured[0])} 头, 序列长 {len(text)}")


@torch.no_grad()
def temperature(m, tok):
    prefix = "ROMEO:\nI will th"
    logits, _ = m(torch.tensor([tok.encode(prefix)]))
    z = logits[0, -1]
    temps = [0.5, 1.0, 1.5]
    probs = {t: torch.softmax(z / t, -1) for t in temps}
    top = probs[1.0].topk(8).indices.tolist()     # 按 T=1 时的概率取前 8 个字符
    out = {"prefix": prefix, "chars": [tok.decode([i]) for i in top],
           "probs": {str(t): [round(probs[t][i].item(), 4) for i in top] for t in temps},
           "rest": {str(t): round(1 - sum(probs[t][i].item() for i in top), 4) for t in temps}}
    with open(os.path.join(RES, "nano_temperature.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("温度:", {t: out["probs"][str(t)][:3] for t in temps})


if __name__ == "__main__":
    os.makedirs(CKPT, exist_ok=True)
    train_nano()
    model = load_best()
    tokenizer = CharTokenizer(load_shakespeare())
    attention_maps(model, tokenizer)
    temperature(model, tokenizer)
