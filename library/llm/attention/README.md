# A3 注意力与推理延迟 — MHA / MQA / GQA / MLA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OGGIoao/roots-car/blob/master/library/llm/attention/colab.ipynb) 不想在自己电脑上装环境? 点这个按钮, 在浏览器里直接运行本章 ([colab.ipynb](colab.ipynb))。

> 知识库 · 大语言模型单元, 服务于小车 v0.5 (小车用自己的模型): 小车从听到指令到开始动, 一大块时间花在模型生成上, 而生成速度很大程度由注意力的 KV Cache 决定。
> 整棵架构树 (位置编码、MoE、归一化、替代架构等) 在 [LLM 架构生长树](../llm-tree/README.md); 真实模型的数据见 [解剖台](../../frontier/anatomy/ANATOMY.md)。

讲义: [TUTORIAL.md](TUTORIAL.md)。每个注意力变体回答四个问题:

1. 它解决了父节点的什么问题?
2. 代价是什么?
3. 最小实现 + 与父节点在同一任务上的对照实测;
4. 哪些真实模型用了它 (由解剖台核实)。

## 结构

```
A3-attention/
├── base.py              "LLaMA 配方"解码器, 组件可插拔 (RMSNorm + RoPE + SwiGLU)
├── variants.py          MHA / MQA / GQA (同一个类) 与 MLA
├── compare.py           对照实验: 同一任务、同一设置、3 个随机种子
├── decode.py            推理实验: 不同上下文长度下的生成速度与 KV Cache
├── overfit_check.py     过拟合检查
├── results/             实验结果 (JSON)
├── figures/             make_charts.py (数据图) 与 make_diagrams.py (结构图)
└── exercises/           三级练习 + 测试
```

## 为什么要"可插拔"

每个变体只改一个组件。[base.py](base.py) 的解码器把注意力模块当作参数传入, 其余部分 (嵌入、RMSNorm、SwiGLU、RoPE、训练设置) 完全相同。所以对照实验里看到的差异, 只能来自被替换的那个组件。选修里生长树的其它节点也复用这个解码器。

训练与指标统一由 [common/bench.py](../../../common/bench.py) 负责: Tiny Shakespeare 字符级任务, 同样的数据顺序和随机种子, 输出最佳 val_loss、每个 token 的 KV Cache、参数量、速度和峰值显存。

## 运行

```bash
cd library/llm/attention
python compare.py --quick   # 快速检查, 约 1 分钟
python compare.py           # 完整实验, 测量平台上约 25 分钟
python decode.py            # 推理速度实验
```

## 一个必须知道的实验陷阱: 过拟合

Tiny Shakespeare 只有约 100 万字符, 而这里的模型有约 300 万参数。实测 MHA 变体的验证 loss 在约 750~1000 步达到最低 (1.55), 之后一路上升, 3000 步时升到 3.27, 而训练 loss 降到了 0.17: 模型在背训练集。

如果直接比较"训练结束时"的 val_loss, 比的就是谁背得更狠, 参数少的变体 (比如 MQA) 反而会显得更好。所以 `bench.py` 每 100 步评估一次, 比较的是每个模型**自己曲线上的最低点** (相当于早停), 并用 3 个随机种子估计波动。
