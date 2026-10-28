# 知识库 · 大语言模型

> 一条完整的链: 输入 → 架构 → 预训练 → 后训练 (在 [后训练](../post-training/README.md) 领域) → 推理能力 → 推理与部署, 旁边挂评测与可解释性。总目录见 [library/README.md](../README.md#llm)。

**架构段的总地图** 是 [LLM 架构生长树](llm-tree/README.md): 每个节点写明从哪里长出来、解决了什么、代价是什么、哪些真实模型用了它。真实模型的配置用 [解剖台](../frontier/anatomy/README.md) 核实。

**基座线**: [roots-lm](../../base/README.md) 把本领域的单元装成一个自己从零训的模型。每学完一个单元, 就把对应的部件装上去, 测前后变化。

**前沿**: 带 🔥 的单元最后一节是 "前沿对照", 并排比较 DeepSeek、Kimi、智谱、Qwen 等的做法; 按厂商读见 [前沿档案卡](../frontier/README.md)。

**推理与部署段** 里的 paged-kv-cache → batch-invariance 是一个系列: 一步步写出自己的小推理引擎, 是项目 [推理服务](../../projects/serving/README.md) 的核心。

## 单元

<!-- catalog:start -->
| 段 | 单元 | 讲什么 | 状态 | 用到它的地方 |
|---|---|---|---|---|
| 输入 | [bpe](bpe/TUTORIAL.md) | 分词: 把文字变成整数 (字节级 BPE) | ✅ | v0.5, 基座 |
| 架构 (总地图: llm-tree) | [llm-tree](llm-tree/README.md) | LLM 架构生长树: 整个架构段的地图, 每个节点写明来历、代价、谁用了 | 🟡 |  |
|  | [minigpt](minigpt/TUTORIAL.md) | 从零写一个 GPT-2 结构的小模型并训练 | ✅ | v0.5, 基座 |
|  | [attention](attention/TUTORIAL.md) | 注意力变体 (MHA / MQA / GQA / MLA) 与推理延迟 | ✅ | v0.5, 基座 |
|  | [modern-small-model](modern-small-model/TUTORIAL.md) | 现代小模型长什么样 (RoPE、RMSNorm、SwiGLU ……) | 🟡 | v0.5, 基座 |
|  | position-encoding | RoPE 细讲、YaRN 等长度外推、部分维度旋转 | ⬜ | 基座 |
|  | residual-connections | Pre-LN、RMSNorm、QK-Norm → 超连接 (DeepSeek mHC、Kimi Attention Residuals) 🔥 | ⬜ | 基座 |
|  | moe | 混合专家: 路由、细粒度与共享专家、负载均衡 🔥 | ⬜ | 基座 |
|  | sparse-attention | 滑动窗口 → DeepSeek V3.2 稀疏注意力 → V4 的压缩注意力 🔥 | ⬜ | 基座 |
|  | linear-attention-ssm | 线性注意力、Mamba、Gated DeltaNet、Kimi Delta Attention 🔥 | ⬜ | 基座 |
|  | hybrid-architecture | 混合架构: 大部分层线性、少数层全注意力 🔥 | ⬜ | 基座 |
|  | long-context | 长上下文: 外推、滑动窗口、attention sink、怎么测 | ⬜ |  |
|  | diffusion-lm | 扩散语言模型: 整块并行生成 🔥 | ⬜ |  |
| 预训练 | pretraining-data | 数据从哪来: 抓取、去重、过滤、配比 | ⬜ | 基座 |
|  | scaling-laws | 规模定律: 参数、数据、算力怎么分配 | ⬜ | 基座 |
|  | optimizers | AdamW → Muon 🔥 | ⬜ | 基座 |
|  | training-stability | loss 尖峰、学习率、μP | ⬜ | 基座 |
|  | low-precision | bf16 → FP8 训练 🔥 | ⬜ | 基座 |
|  | distributed-training | 数据并行、FSDP、张量并行、流水线并行、专家并行 | ⬜ | 基座 |
|  | mtp | 多 token 预测 🔥 | ⬜ | 基座 |
|  | mid-training | 退火、长上下文扩展 | ⬜ | 基座 |
|  | distillation | 大模型教小模型 | ⬜ | 基座 |
| 推理能力 | chain-of-thought | 思维链: 先想再答为什么有用 | ⬜ |  |
|  | test-time-scaling | 推理时多花算力: 多次采样投票、best-of-N、搜索 🔥 | ⬜ |  |
|  | reasoning-models | 思考模式与思考预算 🔥 | ⬜ |  |
| 推理与部署 | [chat-template](chat-template/TUTORIAL.md) | 对话模板: 模型怎么分清 system / user / assistant / tool | ✅ | v0.4, v0.5 |
|  | [decoding](decoding/TUTORIAL.md) | 解码与采样: 贪心、温度、top-p、重复惩罚、束搜索, 为什么评测用贪心 | ✅ | v0.4 |
|  | [structured-output](structured-output/TUTORIAL.md) | 约束解码: 让模型只能输出合法的 JSON 和工具调用 | ✅ | v0.4 |
|  | quantization | 量化: INT4、FP8、MXFP4/NVFP4, 变小变快的代价 🔥 | ⬜ | v0.6, 推理服务 |
|  | local-inference | 本地推理: llama.cpp、vLLM、SGLang 的用法与延迟 | ⬜ | v0.5, 推理服务 |
|  | paged-kv-cache | 自写推理引擎 ①: 分页 KV 缓存 | ⬜ | 推理服务 |
|  | continuous-batching | 自写推理引擎 ②: 连续批处理 | ⬜ | 推理服务 |
|  | prefix-caching | 自写推理引擎 ③: 前缀缓存 | ⬜ | 推理服务 |
|  | speculative-decoding | 自写推理引擎 ④: 投机解码, 用 MTP 当草稿 🔥 | ⬜ | 推理服务 |
|  | batch-invariance | 自写推理引擎 ⑤: 批不变与确定性推理 🔥 | ⬜ | 推理服务 |
|  | serving-at-scale | 专家并行、prefill/decode 分离: 读 DeepInfer 源码 🔥 | ⬜ | 推理服务 |
| 旁支 | llm-eval | 困惑度、基准、数据污染、红队、用大模型当评委 | ⬜ |  |
|  | interpretability | logit lens、探针、稀疏自编码器、电路追踪 | ⬜ |  |
<!-- catalog:end -->

**前置**: 基础领域的 linear-algebra、pytorch-basics; bpe 只需要会写 Python。bpe、minigpt、attention 有 Colab 笔记本, 没有显卡也能跑。
