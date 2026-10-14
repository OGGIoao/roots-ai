# llm-tree — 架构生长树上的节点

> 总图见 [roadmaps/LLM_TREE.md](../../roadmaps/LLM_TREE.md), 真实模型的数据见 [llm-anatomy](../llm-anatomy/ANATOMY.md)。

这里放树上**新长出来的节点**。每个节点回答四个问题:

1. 它解决了父节点的什么问题?
2. 代价是什么?
3. 最小实现 + 与父节点在同一任务上的对照实测;
4. 哪些真实模型用了它 (由解剖台核实)。

## 结构

```
llm-tree/
├── base.py              "LLaMA 配方"节点: 组件可插拔的解码器 (RMSNorm + RoPE + SwiGLU)
└── attention/           A. 注意力线
    ├── variants.py      MHA / MQA / GQA (同一个类) 与 MLA
    ├── compare.py       对照实验: 同一任务、同一设置、3 个随机种子
    └── results/         实验结果 (JSON)
```

## 为什么要"可插拔"

树上每个节点只改一个组件。[base.py](base.py) 的解码器把注意力模块当作参数传入, 其余部分 (嵌入、RMSNorm、SwiGLU、RoPE、训练设置) 完全相同。所以对照实验里看到的差异, 只能来自被替换的那个组件。

训练与指标统一由 [common/bench.py](../../common/bench.py) 负责: Tiny Shakespeare 字符级任务, 同样的数据顺序和随机种子, 输出最佳 val_loss、每个 token 的 KV Cache、参数量、速度和峰值显存。

## 运行

```bash
cd projects/llm-tree
python attention/compare.py --quick   # 快速检查, 约 1 分钟
python attention/compare.py           # 完整实验, 本机约 25 分钟
```

## 一个必须知道的实验陷阱: 过拟合

Tiny Shakespeare 只有约 100 万字符, 而这里的模型有约 300 万参数。实测 MHA 变体的验证 loss 在约 750~1000 步达到最低 (1.55), 之后一路上升, 3000 步时升到 3.27, 而训练 loss 降到了 0.17: 模型在背训练集。

如果直接比较"训练结束时"的 val_loss, 比的就是谁背得更狠, 参数少的变体 (比如 MQA) 反而会显得更好。所以 `bench.py` 每 100 步评估一次, 比较的是每个模型**自己曲线上的最低点** (相当于早停), 并用 3 个随机种子估计波动。
