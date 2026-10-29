# 知识库 · Agent

> 让大模型 "动手", 并做成能交付给用户的系统: 工具、上下文与记忆、检索、协议、典型 Agent、多智能体、工程化、评测、安全。总目录见 [library/README.md](../README.md#agent)。

**分工**: LLM 领域管 "模型怎么跑得快"; 这里的 A7 管 "整个系统怎么稳定地交付给用户"。Agent 的强化学习训练在 [后训练](../post-training/README.md) 领域。

**项目**: 小车用 A1、A8、A9; 学习助教基本就是本领域的综合练习 (A2、A3、A7、A8)。

## 单元

<!-- catalog:start -->
| 段 | 单元 | 讲什么 | 状态 | 用到它的地方 |
|---|---|---|---|---|
| A0 总览 | agent-map | 什么时候该用 Agent; 模型 + 外壳 (harness) 的结构 | ⬜ |  |
| A1 核心机制 | [agent-loop](agent-loop/TUTORIAL.md) | Agent 循环: 消息、工具调用、四种循环写法、安全层、评测方法 | 🟡 | v0.4 |
|  | tool-calling | 各家的工具调用格式、并行调用、和约束解码的配合 | ⬜ | v0.4 |
|  | planning | ReAct、先规划再执行、反思与重试 | ⬜ |  |
|  | harness-engineering | 把模型包成能干活的系统: 提示、工具设计、循环控制、出错恢复 🔥 | ⬜ | 助教 |
| A2 上下文与记忆 | context-engineering | 窗口里放什么、怎么压缩、怎么让前缀命中缓存 🔥 | ⬜ | 助教 |
|  | memory | 短期与长期记忆, 什么时候写、什么时候读 🔥 | ⬜ | 助教 |
| A3 检索增强 | embeddings | 文字怎么变成向量、相似度怎么算 | ⬜ | 助教 |
|  | rag-basics | 切块、检索、重排、关键词 + 向量混合检索 | ⬜ | 助教 |
|  | rag-advanced | Agent 式检索、GraphRAG、长上下文与 RAG 的取舍 🔥 | ⬜ | 助教 |
|  | rag-eval | 检索召回率、回答是否忠于资料 | ⬜ | 助教 |
| A4 协议与生态 | mcp | MCP: 把工具做成任何应用都能调用的服务 | ⬜ | v0.4 |
|  | agent-interop | A2A、Agent Skills、AGENTS.md 等互通标准 🔥 | ⬜ |  |
|  | [agent-frameworks](agent-frameworks/README.md) | LangGraph、OpenAI Agents SDK 等框架, 逐个部件对照手写循环 | 🟡 |  |
| A5 典型 Agent | coding-agent | 编程 Agent: 读代码库、跑测试、在沙箱里改代码 | ⬜ |  |
|  | computer-use | 看截图或页面结构操作电脑和网页 | ⬜ |  |
|  | deep-research | 搜索、阅读、综合成报告 | ⬜ |  |
| A6 多智能体 | multi-agent | 主管带工人、分工、辩论; 什么时候真有用 | ⬜ |  |
| A7 工程化落地 | data-pipelines | 把文档变成可检索的知识: 解析、OCR、清洗、增量更新 | ⬜ | 助教 |
|  | agent-deployment | 把 Agent 做成服务: 流式、长任务、重试 | ⬜ | 助教 |
|  | observability | 追踪每一步调用了什么、花了多少、错在哪 | ⬜ | 助教 |
|  | cost-latency | 缓存、按难度选模型、预算控制 | ⬜ | 助教, 推理服务 |
| A8 评测 | agent-eval | 任务集设计、置信区间、回归测试; 常见公开基准各测什么 | ⬜ | v0.5, 助教 |
| A9 安全 | agent-safety | 网页和文件里藏的指令、工具权限、沙箱 | ⬜ | v0.4, 助教 |
<!-- catalog:end -->

**前置**: 会写 Python、认识 JSON。A1 的演示和练习用假模型, 不需要显卡。
