# 选修

> 仓库分三块 (见 [ROADMAP](../ROADMAP.md) 第 1 节): **主线** 造小车; **知识库** 放主线用到的每个知识点; **选修** 放不在造车路上、但值得深入的内容。
> 两种用法: 想把某个领域系统串起来时按图深挖; 不想动手的时候, 当作可以慢慢读的详细文档。

| 选修 | 内容 | 状态 |
|---|---|---|
| [llm-tree/](llm-tree/README.md) | LLM 架构生长树: 从 Transformer 长出的注意力、位置编码、MoE、归一化、替代架构、效率六条分支 | 总图 ✅; 注意力线已在知识库 [attention](../library/llm/attention/TUTORIAL.md) 实现, 其余分支 ⬜ |
| [llm-anatomy/](llm-anatomy/README.md) | 解剖台: 读取 20 多个真实开源模型的配置与张量名, 用数据核对生长树 | ✅ 可用 |
| [agent-frameworks/](agent-frameworks/README.md) | 成熟的 Agent 框架 (LangChain/LangGraph、Microsoft Agent Framework、OpenAI Agents SDK、smolagents、ADK、CrewAI) 与 MCP 协议, 逐一对照知识库 [agent-loop](../library/agent/agent-loop/TUTORIAL.md) 的手写循环 | 讲解 ✅; 移植练习的参考实现 ⬜ |
| [quadruped/](quadruped/README.md) | 四足机器人 Pupper: 本体与经典控制 → RL 步态 → 模仿学习 → 多峰行为 → 语言与视觉 | 规划 ✅, 实现 ⬜ |
| [scale-up/](scale-up/README.md) | 规模化与上云: 把模型做大、多卡训练、推理服务压测 | ⬜ |

选修的写法和知识库一样 (讲义 + 参考实现 + 三级练习, 数字全部实测), 只是不在通往小车 demo 的路上。后训练原先在这里, 2026-10-27 移入知识库 [library/post-training](../library/post-training/README.md)。
