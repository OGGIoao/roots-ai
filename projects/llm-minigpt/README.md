# llm-minigpt — 从零训练 GPT-2

> 复现 Karpathy nanoGPT 流程, 用 Tiny Shakespeare 数据集从零训练一个 GPT-2 级别的语言模型。

## 定位: 它和 llm-transformer 是什么关系

| | `llm-minigpt` (本项目) | `llm-transformer` |
|---|---|---|
| 目的 | **端到端跑通**: 读一个 `model.py` 就能看懂 GPT-2, 并马上训练出能生成文本的模型 | **组件对比**: MHA/GQA/MLA、Sinusoidal/Learned/ALiBi/RoPE、LN/RMSNorm、GELU/SwiGLU 逐个拆开讲 |
| 代码形态 | 模型集中在一个文件, 不依赖 transformer 子模块 (nanoGPT 风格) | 每个组件一个文件, 可自由组合出 GPT-2 或 LLaMA |
| 架构 | 只有 GPT-2 (Pre-LN, 可学习位置编码, 绑定输出头) | GPT-2 与 LLaMA 两条线 |
| 状态 | ✅ 可用, **功能冻结** (只修 bug, 不再加新架构) | 🟡 持续演进 |

**怎么选**: 想先看到"模型真的在学"就从这里开始; 想理解各组件为什么这样设计, 去 `llm-transformer`。
新架构、新位置编码之类的实验都放在 `llm-transformer`, 不要往这里加, 否则两边又会重复。
两个项目**共享**的只有 `common/` (语料、字符级分词、数据集); 模型和超参定义各自独立。

## 学习目标

- 理解 GPT-2 的完整训练流程
- 掌握 Pre-LN Decoder-only 架构
- 实践 Cosine LR Schedule + Gradient Clipping
- 观察模型从「乱码」到「生成莎士比亚风格文本」的全过程

## 项目结构

```
llm-minigpt/
├── config.py      # GPTConfig + 预设 (nano/micro/small/medium/large), 超参唯一来源
├── model.py       # GPT-2 架构 (Pre-LN, Causal Attention, FFN)
├── data.py        # 数据加载 (Tiny Shakespeare + 字符级Tokenizer)
├── train.py       # 训练脚本 (AdamW + Cosine LR + AMP)
└── README.md      # 本文件
```

## 快速开始

```bash
# 1. 安装依赖 (仓库根目录唯一依赖清单)
pip install -r ../../requirements.txt

# 2. 运行模型测试
python model.py

# 3. 开始训练 (nano 配置, ~5秒/iter)
python train.py --config nano --max_iters 2000

# 4. 训练小模型 (micro 配置, ~30秒/iter, 需要更多显存)
python train.py --config micro --max_iters 5000
```

## 预定义配置

| 配置 | 层数 | 头数 | 维度 | 参数量 (`python config.py` 估算) | 用途 |
|------|------|------|------|--------|------|
| nano | 2 | 2 | 64 | ~0.12M (字符级词表 65) | 快速调试 |
| micro | 4 | 4 | 128 | ~0.8M (字符级词表 65) | 小规模实验 |
| small | 12 | 12 | 768 | ~124M (GPT-2 词表, 输出头与 embedding 共享) | GPT-2 Small |
| medium | 24 | 16 | 1024 | ~355M | GPT-2 Medium |
| large | 36 | 20 | 1280 | ~774M | GPT-2 Large |

> 训练时词表大小由数据集 tokenizer 决定 (字符级约 65), 预设里的 `vocab_size` 仅用于 `config.py` 的参数量估算。
> 预设只提供 `n_layer / n_head / n_embd / block_size`, 命令行同名参数可覆盖。

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
