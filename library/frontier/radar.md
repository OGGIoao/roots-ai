# 前沿雷达

> 带日期的流水账。新东西先登记在这里, 写清来源和是否核实过。满足两个条件时升级进单元: ① 能用开源代码、权重或技术报告核实; ② 确实重要 (两家以上采用, 或明显改变了做法)。
> 状态: 📌 已登记 · 🔍 已核实 · ⬆️ 已进单元。

| 登记日期 | 事项 | 来源 | 状态 | 将进入的单元 |
|---|---|---|---|---|
| 2026-10-28 | DeepSeek 开源推理框架 DeepInfer (Apache-2.0; 只支持 NVIDIA SM100 与昇腾 950; 投机解码、批不变推理、结构化输出) | [GitHub 仓库 README](https://github.com/deepseek-ai/DeepInfer) | 🔍 README 已读 | llm/serving-at-scale、speculative-decoding、batch-invariance |
| 2026-10-28 | DeepSeek V4: 压缩稀疏注意力 + 重度压缩注意力、流形约束超连接、Muon、百万上下文 | [V4 模型卡 (PDF)](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf) | 📌 | llm/sparse-attention、residual-connections、optimizers |
| 2026-10-28 | Kimi K3: Kimi Delta Attention 混合线性注意力、Attention Residuals、原生视觉 | 二手报道 (VentureBeat 等) | 📌 待核实 | llm/linear-attention-ssm、residual-connections |
| 2026-10-28 | 智谱 GLM-5 采用 DeepSeek 稀疏注意力; slime 异步强化学习框架 | 二手报道 | 📌 待核实 | llm/sparse-attention、post-training/framework-reading |
| 2026-10-28 | Qwen3.5 混合架构 (Gated DeltaNet + 全注意力)、早期融合多模态 | 二手报道 | 📌 待核实 | llm/hybrid-architecture、multimodal/native-multimodal |
| 2026-10-28 | 扩散语言模型开源: DiffusionGemma (Google)、Nemotron-Labs-Diffusion (NVIDIA) | Google 开发者博客、NVIDIA Research | 📌 | llm/diffusion-lm |
| 2026-10-28 | Agentic AI Foundation 新增 Agent Skills、Open Responses、OpenEnv、OpenSandbox (2026-10-22) | 新闻稿转载 | 📌 | agent/agent-interop |
| 2026-10-28 | 用世界模型生成数据辅助真机器人强化学习 (WorldSample 等) | arXiv 预印本 | 📌 | embodied/world-model-data |
| 2026-10-28 | 小规模全能模型 MiniMind-O (约 0.1B, 文字 / 语音 / 图像输入, 流式语音输出) | arXiv 技术报告 | 📌 待核实 | multimodal/omni-models、基座线阶段 4 |

以上来源在 2026-10-28 的设计讨论中检索到, 详见 [FRAMEWORK.md](../../FRAMEWORK.md)。
