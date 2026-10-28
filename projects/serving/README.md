# 项目 ③ 推理服务

> 用自己写的推理引擎, 把基座线 roots-lm 和开源小模型部署成服务, 压测、量化、优化, 每一步都和成熟框架对照实测。状态: ⬜ 未开始。

**用到的单元**: LLM 领域推理与部署段 (local-inference、quantization、paged-kv-cache → batch-invariance 的自写引擎系列、serving-at-scale)、基础领域的 gpu-basics 和 profiling。见 [LLM 领域](../../library/llm/README.md)。

**练的技能**: 推理优化。

各站的规划在项目开始前写出 (按 [FRAMEWORK.md](../../FRAMEWORK.md) 第 8 节, 第 3 步开始)。
