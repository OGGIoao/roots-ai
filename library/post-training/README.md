# 知识库 · 后训练

> 预训练好的语言模型, 怎么一步步变成能用的助手: 监督微调、偏好对齐、可验证奖励的强化学习、Agent 强化学习、蒸馏、RL 基础设施。总目录见 [library/README.md](../README.md#post-training)。

**分工**: 强化学习基础 (bandit → PPO) 在 [强化学习](../rl/README.md) 领域; 这里只讲 "把语言模型当成策略" 之后多出来的部分。训练数据的合成与筛选 (instruction-data) 放在这里。

**基座线**: [roots-lm](../../base/README.md) 阶段 1 在笔记本上跑通 SFT → DPO → GRPO (用能自动判对错的任务); 阶段 2 把同一套代码放大到云上。

**项目**: 小车 v0.5 用 tool-finetune 和 tool-use-rl (在它们的实践一节); 第二批项目 "领域后训练" 是本领域的综合练习。

## 单元

<!-- catalog:start -->
| 段 | 单元 | 讲什么 | 状态 | 用到它的地方 |
|---|---|---|---|---|
| P0 总览 | post-training-map | 后训练分几个阶段、各家的实际流程 | ⬜ |  |
| P1 监督微调 | sft | 指令微调; loss 只算在模型的回答上 | ⬜ | 基座 |
|  | instruction-data | 训练数据: 人工、合成、拒绝采样、质量过滤 | ⬜ | 基座 |
|  | lora-qlora | 只训一小部分参数; 8GB 也能微调 | ⬜ | v0.5 |
|  | [tool-finetune](tool-finetune/README.md) | 监督微调让小模型稳定输出指定格式的工具调用 (待改造, 现为通用 QLoRA 演示) | 🟡 | v0.5 |
| P2 偏好对齐 | reward-models | 奖励模型、奖励被钻空子 | ⬜ |  |
|  | rlhf-ppo | 经典 RLHF: PPO、KL 约束、价值头 | ⬜ |  |
|  | dpo | 不用奖励模型的偏好优化及其变体 | ⬜ | 基座 |
| P3 可验证奖励的强化学习 | grpo | R1 的方法: 对错能自动判的任务直接当奖励 | ⬜ | 基座 |
|  | grpo-variants | DAPO、GSPO、CISPO 各修了什么 🔥 | ⬜ |  |
|  | process-rewards | 只看结果还是每步打分; 验证器 | ⬜ |  |
|  | rl-dynamics | 探索、熵、训练为什么会崩 🔥 | ⬜ |  |
| P4 Agent 强化学习 | agentic-rl | 多轮、带工具、带环境的强化学习; 长轨迹的功劳分配 🔥 | ⬜ |  |
|  | tool-use-rl | 用任务成败当奖励, 以强化学习训练模型调用工具 | ⬜ | v0.5 |
| P5 蒸馏 | reasoning-distill | 用大模型的推理过程教小模型 | ⬜ |  |
|  | on-policy-distill | 小模型自己生成、大模型逐 token 纠正 🔥 | ⬜ |  |
| P6 RL 基础设施 | rl-infra | 生成与训练两套引擎、异步、权重同步 🔥 | ⬜ |  |
|  | mini-rl-loop | 自己写最小的 RL 训练循环 | ⬜ | 基座 |
|  | framework-reading | 读 slime、OpenRLHF、verl 的源码 | ⬜ |  |
| P7 安全对齐 | safety-alignment | 拒答、越狱、提示注入 | ⬜ |  |
<!-- catalog:end -->

**前置**: LLM 领域的 [minigpt](../llm/minigpt/TUTORIAL.md)、[chat-template](../llm/chat-template/TUTORIAL.md); P2~P4 还需要强化学习领域的 ppo。
