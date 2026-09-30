# llm-transformer

> 从零手写 Transformer 架构演进 — 从 2017 原始 Transformer 到 2023 LLaMA

## 架构演进时间线

```
2017  原始 Transformer (Encoder-Decoder, sin-cos PE, Post-LN)
  ↓
2018  BERT (Encoder-only)  /  GPT-1 (Decoder-only, Learned PE)
  ↓
2019  GPT-2 (Pre-LN, GELU FFN)
  ↓
2023  LLaMA (RMSNorm + SwiGLU + GQA + RoPE)
  ↓
2024  Qwen-V2 (MLA 多头潜在注意力 + MoE)
```

## 项目结构

```
llm-transformer/
├── attention.py          # 注意力机制演进
│   ├── MultiHeadAttention (2017, 原始 MHA)
│   ├── GroupedQueryAttention (2023, GQA, LLaMA-2)
│   └── MultiLatentAttention (2024, MLA, Qwen-V2)
│
├── positional_encoding.py # 位置编码演进
│   ├── SinusoidalPE (2017, sin-cos)
│   ├── LearnedPE (2018, GPT-1/2)
│   ├── ALiBi (2022, 线性偏置)
│   └── RoPE (2021, 旋转位置编码, 现代标配)
│
├── normalization.py      # 归一化层演进
│   ├── LayerNorm (标准)
│   └── RMSNorm (LLaMA, Qwen)
│
├── feedforward.py        # FFN 演进
│   ├── FFN (2017, ReLU)
│   ├── GELUFFN (GPT-2)
│   └── SwiGLUFFN (LLaMA, PaLM)
│
├── transformer_block.py  # Transformer Block
│   ├── TransformerBlock (GPT-2 Style, Pre-LN)
│   └── LlamaBlock (LLaMA Style, RMSNorm+SwiGLU+GQA+RoPE)
│
├── model.py              # 完整模型
│   ├── GPTModel (GPT-1/2 Style)
│   └── LlamaModel (LLaMA Style)
│
├── data.py               # 数据加载 (Tiny Shakespeare)
└── train.py              # 训练脚本
```

> 依赖统一由仓库根目录 `requirements-mlc.txt` 声明（conda 环境 `mlc`），
> 本目录不再单独维护 requirements 文件，避免版本漂移。

## 每个模块覆盖的架构决策

### 注意力机制

| 版本 | 年份 | KV-Head | KV-Cache | 论文 |
|------|------|---------|----------|------|
| MHA | 2017 | = Q-Head | 全量 | Attention Is All You Need |
| GQA | 2023 | < Q-Head | 减少 | GQA (Ainslie et al.) |
| MLA | 2024 | 低秩压缩 | 最小 | Qwen-V2 |

### 位置编码

| 版本 | 年份 | 类型 | 外推 | 论文 |
|------|------|------|------|------|
| Sinusoidal | 2017 | 绝对 | 理论可行 | Attention Is All You Need |
| Learned PE | 2018 | 绝对 | 不可 | GPT-1 |
| ALiBi | 2022 | 相对偏置 | 强 | Train Short, Test Long |
| RoPE | 2021 | 相对旋转 | 强(+NTK) | RoFormer |

### 归一化

| 版本 | 位置 | bias | 速度 |
|------|------|------|------|
| Post-LN | Attention 之后 | 有 | 慢 |
| Pre-LN | Attention 之前 | 有 | 中 |
| RMSNorm | Attention 之前 | 无 | 快 |

### FFN

| 版本 | 年份 | 激活 | 参数 |
|------|------|------|------|
| FFN | 2017 | ReLU | W1, W2 |
| GELU FFN | 2018 | GELU | W1, W2 |
| SwiGLU | 2020 | Swish+Gate | W_gate, W_up, W_down |

## 快速开始

```bash
# 安装依赖（仓库根目录，全仓库唯一依赖清单）
pip install -r ../../requirements-mlc.txt

# 训练 GPT-2 Style (默认)
python train.py gpt2

# 训练 LLaMA Style
python train.py llama

# 单独测试每个模块
python attention.py
python positional_encoding.py
python normalization.py
python feedforward.py
python model.py
```

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 Transformer
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Pre-LN + GELU
- [LLaMA](https://arxiv.org/abs/2302.13971) — RMSNorm + SwiGLU + RoPE + GQA
- [GQA](https://arxiv.org/abs/2305.13245) — Grouped Query Attention
- [RoFormer](https://arxiv.org/abs/2104.09864) — RoPE
- [GLU Variants](https://arxiv.org/abs/2002.05202) — SwiGLU
- [Qwen-V2](https://arxiv.org/abs/2401.04088) — MLA + MoE
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — 配套书籍
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — 简洁 GPT-2 实现
- [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) — 100+ 架构带注释实现
