# AGENT课程 — 完整大纲（已确认）

## 设计原则
- **底层优先**：每模块先手写，再决定调包
- **技术+业务链路**：技术链路（M1-M5）→ 业务落地（M6代码生成/数据分析）→ 框架解析（M7）
- **可运行**：每模块 = Notebook + core.py + README + TUTORIAL
- **打通**：理解底层后，知道何时用LangChain/AutoGen/LlamaIndex，为什么迭代

## 模块结构（统一模板）
| 模块 | 底层手写 | 框架对比 | 调包决策 | 业务接入 | 呈现 |
|---|---|---|---|---|---|
| M1 Agent原子循环 | AgentLoop手写 | 无（先手写） | 何时用LangChain | 代码生成Agent | ✅ 已建 |
| M2 Token/Embedding | 自己写tokenizer（参考`llm-bpe-tokenizer`） | tiktoken/BPE | 何时用现成tokenizer | 无 | 待建 |
| M3 Tool+Memory | JSON Tool解析 + dict Memory | LangChain Tools/Memory | 何时用框架Memory | 数据分析Agent | 待建 |
| M4 Planning/子Agent | Plan生成 + Sub-agent拆解 | AutoGen多Agent | 何时用AutoGen | 复杂任务链 | 待建 |
| M5 Eval/Benchmark | 手写准确率/成功率 | LangSmith/评估框架 | 何时用标准评估 | 无 | 待建 |
| M6 业务落地 | 代码生成Agent + 数据分析Agent | 无（综合应用） | 综合调包决策 | 代码生成 + 数据分析 | 待建 |
| M7 主流框架解析 | 无（纯解析） | PyTorch/TF/JAX + LangChain/AutoGen/LlamaIndex | 为什么迭代、何时调包、场景匹配 | 无 | 待建 |

## 你的要求已满足
1. ✅ 基础开发者 + 要底层（每模块先手写）
2. ✅ 了解底层后根据情况调包（每模块有"调包决策"）
3. ✅ 技术+业务链路（M1-M5技术，M6业务，M7框架）
4. ✅ 前者（Notebook交互式，参考你workspace的`TUTORIAL.md`+代码结构）
5. ✅ 主流框架（M7：LangChain/AutoGen/LlamaIndex + PyTorch/TF/JAX，为什么迭代，调包如何用，适合什么场景）

## 下一步
审查M1（已建在`projects/agent-course/M1-agent-atom/`），确认后我继续建M2-M7，或根据你的反馈调整。
