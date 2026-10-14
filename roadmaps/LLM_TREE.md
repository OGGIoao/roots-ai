# LLM 架构生长树

> 所有主流大模型都从 Transformer 这一条主干长出来。本文件是 LLM 板块的总图: 每个节点写明它从哪里长出来、解决了父节点的什么问题、代价是什么、哪些模型用了它, 以及在本仓库的哪里学它。
>
> - **"谁用了"一栏只列 [解剖台](../projects/llm-anatomy/ANATOMY.md) 从真实模型的配置和权重里核实过的模型**, 不凭印象。
> - 节点的出处只列原始论文; 前沿模型的细节在写到对应节点时查阅技术报告, 并在节点讲义里注明来源与日期。
> - 状态: ✅ 完成 · 🟡 进行中 · ⬜ 未开始。最后更新: 2026-10-15。

---

## 总图

```
Transformer (Vaswani et al., 2017, encoder-decoder)
├── Encoder-only: BERT 系                                   (旁支, 不展开)
├── Encoder-decoder: T5 系                                  (旁支, 不展开)
└── Decoder-only ─────────────────────────────────────────── 主干
    └── GPT-2 (2019): Pre-LN, 可学习位置编码, 输出头权重绑定       ✅ llm-minigpt
        └── LLaMA 配方 (2023): Pre-RMSNorm + RoPE + SwiGLU + 无 bias   🟡 llm-transformer, llm-tree/base.py
            │
            ├── A. 注意力线 ············ 推理时 KV Cache 太大、长序列计算太贵
            ├── B. 位置编码线 ··········· 训练长度之外怎么办
            ├── C. FFN / MoE 线 ········· 参数要多, 每个 token 的计算要少
            ├── D. 归一化线 ············· 大规模训练不稳定
            ├── E. 架构替代线 ··········· 注意力随长度平方增长
            └── F. 训练与推理效率线 ······ 同样的模型, 更快更省
                                   ↓
                叶子: 具体的前沿模型 = 以上若干节点的组合
```

**输入层与其它板块的接口**:
- 分词 (BPE) 是整棵树的输入端: ✅ [llm-bpe-tokenizer](../projects/llm-bpe-tokenizer/)
- 微调 (QLoRA) 是"拿到一个训好的叶子之后怎么改造它": 🟡 [llm-qlora-finetune](../projects/llm-qlora-finetune/)
- 后训练与推理模型 (RL、GRPO) 在 RL 板块; 多模态与 VLA 在 RL 板块的具身分支。见 [ROADMAP](ROADMAP.md)。

---

## A. 注意力线

**主线问题**: 生成时每个 token 都要缓存每一层的 K、V。llm-minigpt 讲义专题 A 实测: GPT-2 Small 一条 1024 长度的序列要 72MB, 7B 模型一条 4096 长度的序列要 2GB。

| 节点 | 父节点 | 解决了什么 | 代价 | 出处 | 谁用了 (解剖台核实) | 状态 |
|---|---|---|---|---|---|---|
| A1 MHA | — | 多个头并行关注不同模式 | 每个头一组 K、V, 缓存最大 | Vaswani 2017 | GPT-2, OLMo-2-7B | 🟡 |
| A2 MQA | A1 | 所有头共用一组 K、V, 缓存缩到 1/头数 | 质量下降, 训练可能不稳 | Shazeer 2019 | (表中暂无) | 🟡 |
| A3 GQA | A1、A2 | 折中: 分组共用 K、V | 缓存与质量之间取舍 | Ainslie et al. 2023 | Mistral-7B、Qwen2.5/3、phi-4、GLM-4.5、gpt-oss | 🟡 |
| A4 MLA | A1 | 把 K、V 压成低维潜向量再缓存, 缓存小而质量不降 | 结构复杂, 与 RoPE 不兼容需拆分 | DeepSeek-V2 2024 | DeepSeek-V2/V3/V3.2、Kimi-K2、Kimi-Linear | 🟡 |
| A5 滑动窗口 | A3 | 每个位置只看最近 W 个, 缓存有上限 | 窗口外的信息只能逐层间接传递 | Beltagy et al. 2020 (Longformer) | Mistral-7B (全部层)、gpt-oss (与全局层交替) | ⬜ |
| A6 稀疏注意力 | A4 | 每个查询只挑 top-k 个位置算注意力 | 需要额外的索引器 | DeepSeek-V3.2-Exp 2025 | DeepSeek-V3.2-Exp | ⬜ |
| A7 Attention sink | A5 | 给注意力一个可以"什么都不看"的出口 | 多一组参数 | Xiao et al. 2023 (StreamingLLM) | gpt-oss | ⬜ |

学习位置: [projects/llm-tree/attention/](../projects/llm-tree/attention/)

## B. 位置编码线

| 节点 | 父节点 | 解决了什么 | 出处 | 谁用了 (解剖台核实) | 状态 |
|---|---|---|---|---|---|
| B1 正弦位置编码 | — | 不需要训练, 理论上可外推 | Vaswani 2017 | — | ✅ llm-transformer |
| B2 可学习位置编码 | B1 | 让模型自己学位置表示 | GPT 系列 | GPT-2 | ✅ llm-minigpt |
| B3 ALiBi | B1 | 在注意力分数上加距离惩罚, 外推好 | Press et al. 2022 | — | ✅ llm-transformer |
| B4 RoPE | B1 | 用旋转让 q·k 只依赖相对位置 | Su et al. 2021 (RoFormer) | 除纯 SSM 外几乎全部 | 🟡 llm-transformer |
| B5 RoPE 外推 (PI / NTK / YaRN) | B4 | 让模型用上比训练更长的上下文 | Chen et al. 2023; Peng et al. 2023 (YaRN) | DeepSeek-V2/V3 (yarn×40)、gpt-oss (yarn×32)、Kimi-K2 | ⬜ |
| B6 部分维度旋转 | B4 | 只旋转一部分维度, 其余维度保留"与位置无关"的信息 | — | GLM-4.5 (50%)、MiniMax-01 (50%)、Qwen3-Next (25%) | ⬜ |
| B7 不用位置编码 (NoPE) | B4 | 让其它层 (卷积、SSM、线性注意力) 提供顺序信息 | — | Kimi-Linear 的 MLA 层、Nemotron-H 的注意力层 | ⬜ |

## C. FFN / MoE 线

| 节点 | 父节点 | 解决了什么 | 出处 | 谁用了 (解剖台核实) | 状态 |
|---|---|---|---|---|---|
| C1 GELU FFN | — | 平滑的非线性 | GPT 系列 | GPT-2 | ✅ llm-minigpt |
| C2 SwiGLU | C1 | 门控让 FFN 表达力更强 | Shazeer 2020 | 绝大多数稠密模型 | 🟡 llm-transformer |
| C3 MoE (粗粒度) | C2 | 参数量大而每个 token 只激活少数专家 | Shazeer et al. 2017; Fedus et al. 2021 (Switch) | Mixtral-8x7B (8 选 2) | ⬜ |
| C4 细粒度 + 共享专家 | C3 | 专家更多更小、组合更灵活; 共享专家负责通用知识 | DeepSeekMoE 2024 | DeepSeek-V2/V3、Kimi-K2、GLM-4.5、Qwen3-Next | ⬜ |
| C5 无辅助损失负载均衡 | C3 | 不靠额外损失项也能让专家负载均衡 | Wang et al. 2024 | DeepSeek-V3、Kimi-K2、GLM-4.5、Kimi-Linear | ⬜ |

## D. 归一化线

| 节点 | 父节点 | 解决了什么 | 出处 | 谁用了 (解剖台核实) | 状态 |
|---|---|---|---|---|---|
| D1 Post-LN | — | 原版 Transformer | Vaswani 2017 | — | ✅ llm-transformer |
| D2 Pre-LN | D1 | 梯度有直通路径, 深了也能训 | GPT-2 | GPT-2 | ✅ llm-minigpt |
| D3 RMSNorm | D2 | 去掉均值, 更便宜, 效果相当 | Zhang & Sennrich 2019 | 绝大多数现代模型 | 🟡 llm-transformer |
| D4 QK-Norm | D3 | 对 Q、K 再归一化, 防止注意力分数失控 | Henry et al. 2020; Dehghani et al. 2023 | Qwen3、OLMo-2、GLM-4.5、Qwen3-Next | ⬜ |
| D5 归一化移到子层之后 | D2 | 改变残差与归一化的相对位置以稳定训练 | OLMo 2 2024 | OLMo-2 | ⬜ |

## E. 架构替代线

**主线问题**: 注意力的计算量随序列长度平方增长, KV Cache 随长度线性增长。能不能用一个固定大小的状态代替?

| 节点 | 父节点 | 解决了什么 | 出处 | 谁用了 (解剖台核实) | 状态 |
|---|---|---|---|---|---|
| E1 线性注意力 | A1 | 去掉 softmax, 注意力可以写成递归, 状态大小固定 | Katharopoulos et al. 2020 | — | ⬜ |
| E2 SSM (Mamba / Mamba-2) | E1 | 选择性状态空间模型, 推理时状态大小固定 | Gu & Dao 2023; Dao & Gu 2024 | Mamba-2.8B、Falcon-Mamba-7B (纯 SSM) | ⬜ |
| E3 门控 / Delta 规则的线性注意力 | E1 | 让固定大小的状态能"擦写", 记忆更有效 | Yang et al. 2024 (Gated DeltaNet) 等 | Qwen3-Next、Kimi-Linear、MiniMax-01 的线性层 | ⬜ |
| E4 混合架构 | E2、E3、A3 | 大部分层用线性/SSM, 少数层保留全注意力 | — | Qwen3-Next (36+12)、MiniMax-01 (70+10)、Kimi-Linear (20+7)、Nemotron-Nano-9B-v2 (27 SSM + 4 注意力) | ⬜ |

## F. 训练与推理效率线

| 节点 | 解决了什么 | 出处 | 谁用了 (解剖台核实) | 状态 |
|---|---|---|---|---|
| F1 KV Cache | 生成时不重复计算 | — | 全部 | ✅ llm-minigpt 专题 A |
| F2 FlashAttention | 不把 S×S 矩阵写进显存 | Dao et al. 2022 | — (推理框架层面) | ✅ llm-minigpt 专题 D |
| F3 混合精度 (bf16) | 计算用 16 位, 主权重用 32 位 | Micikevicius et al. 2018 | — | ✅ llm-minigpt 专题 C |
| F4 FP8 训练 / 存储 | 更低精度, 更快更省 | DeepSeek-V3 2024 | DeepSeek-V3/V3.2、Kimi-K2 (FP8 权重) | ⬜ |
| F5 低比特权重 (MXFP4 等) | 推理时权重更小 | — | gpt-oss (MXFP4 专家权重) | ⬜ |
| F6 多 token 预测 (MTP) | 一次预测多个 token: 训练信号更密, 推理可做投机解码 | Gloeckle et al. 2024 | DeepSeek-V3/V3.2、GLM-4.5、Qwen3-Next | ⬜ |
| F7 投机解码 | 小模型先猜、大模型一次验证多个 token | Leviathan et al. 2023 | — | ⬜ |
| F8 量化 (INT4/NF4) | 推理与微调的显存 | Dettmers et al. 2023 (QLoRA) | — | 🟡 llm-qlora-finetune |

---

## 叶子: 前沿模型 = 节点的组合

由解剖台的数据直接读出 (完整表格见 [ANATOMY.md](../projects/llm-anatomy/ANATOMY.md)):

| 模型 | 注意力线 | 位置编码线 | FFN 线 | 归一化线 | 架构替代线 | 效率线 |
|---|---|---|---|---|---|---|
| GPT-2 | A1 | B2 | C1 | D2 | — | — |
| Mistral-7B | A3 + A5 | B4 | C2 | D3 | — | — |
| Qwen3-8B | A3 | B4 | C2 | D3 + D4 | — | — |
| DeepSeek-V3 | A4 | B4 + B5 | C4 + C5 | D3 | — | F4 + F6 |
| DeepSeek-V3.2-Exp | A4 + A6 | B4 + B5 | C4 + C5 | D3 | — | F4 + F6 |
| gpt-oss-120b | A3 + A5 + A7 | B4 + B5 | C3 | D3 | — | F5 |
| Qwen3-Next-80B-A3B | A3 | B4 + B6 | C4 | D3 + D4 | E3 + E4 | F6 |
| Kimi-Linear-48B-A3B | A4 | B7 | C4 + C5 | D3 | E3 + E4 | — |
| Nemotron-Nano-9B-v2 | A3 | B7 | C1 的变体 (ReLU², 无门控) | D3 | E2 + E4 | — |

---

## 学习顺序

1. **A. 注意力线** (进行中): MHA → MQA → GQA → MLA, 接着滑动窗口、稀疏注意力。直接承接 llm-minigpt 专题 A 的 KV Cache 问题。
2. B → C → D → E → F, 每条线都按"父节点的问题 → 子节点的解法 → 同任务对照实测"推进。
3. 每完成一个节点, 更新本表的状态, 并在解剖台里加入用到它的新模型。
