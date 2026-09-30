# llm-minigpt — 从零训练 GPT-2

> 复现 Karpathy nanoGPT 流程, 用 Tiny Shakespeare 数据集从零训练一个 GPT-2 级别的语言模型。

## 学习目标

- 理解 GPT-2 的完整训练流程
- 掌握 Pre-LN Decoder-only 架构
- 实践 Cosine LR Schedule + Gradient Clipping
- 观察模型从「乱码」到「生成莎士比亚风格文本」的全过程

## 项目结构

```
llm-minigpt/
├── config.py      # 模型配置 (nano/micro/small/medium)
├── model.py       # GPT-2 架构 (Pre-LN, Causal Attention, FFN)
├── data.py        # 数据加载 (Tiny Shakespeare + 字符级Tokenizer)
├── train.py       # 训练脚本 (AdamW + Cosine LR + AMP)
└── README.md      # 本文件
```

## 快速开始

```bash
# 1. 安装依赖
pip install torch

# 2. 运行模型测试
python model.py

# 3. 开始训练 (nano 配置, ~5秒/iter)
python train.py --config nano --max_iters 2000

# 4. 训练小模型 (micro 配置, ~30秒/iter, 需要更多显存)
python train.py --config micro --max_iters 5000
```

## 预定义配置

| 配置 | 层数 | 头数 | 维度 | 参数量 | 用途 |
|------|------|------|------|--------|------|
| nano | 2 | 2 | 64 | ~50K | 快速调试 |
| micro | 4 | 4 | 128 | ~2M | 小规模实验 |
| small | 12 | 12 | 768 | ~124M | GPT-2 Small |
| medium | 24 | 16 | 1024 | ~350M | GPT-2 Medium |

## 架构演进

```
原始 Transformer (2017)
  Encoder-Decoder, Post-LN, Sinusoidal PE
    ↓
GPT-1 (2018)
  Decoder-only, Learned PE
    ↓
GPT-2 (2019)
  Pre-LN, Zero-shot, 更大规模
    ↓
GPT-3/4 (2020-2023)
  Scaling Law, In-context Learning
```

本项目实现的是 GPT-2 架构, 使用 Pre-LayerNorm。

## 训练技巧

1. **AdamW**: weight_decay=0.1, betas=(0.9, 0.95)
2. **Cosine LR**: warmup 100 步 → 余弦衰减到 10%
3. **Gradient Clipping**: max_norm=1.0
4. **Mixed Precision**: --amp 启用 (需要 GPU)
5. **Top-K Sampling**: top_k=50, temperature=0.8

## 参考

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [karpathy/minGPT](https://github.com/karpathy/minGPT)
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [OpenAI GPT-2 论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
