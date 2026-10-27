# 选修: 成熟的 Agent 框架

> 主线 [C1](../../library/agent/agent-loop/) 故意**不用框架**, 用两百来行手写 Agent 循环, 让每一步都看得见。
> 这一篇回答下一个问题: 工业界用的那些框架, 到底替你做了哪一部分? 读完之后, 打开任何一个框架的文档,
> 都应该能指着它的概念说出 "这就是 C1 里的哪一行"。

**信息核对日期: 2026-10-25。** 这个领域变化很快, 版本号和"谁合并了谁"每隔几个月就会变,
下面每一条都附了来源, 用之前请再看一眼官方页面。

## 1. 先认清: 每个框架都在包装同样的五样东西

C1 的手写循环 (`agent.py` 的 `run_episode`) 拆开来就是五个部件。框架的名词五花八门, 但几乎都能对上:

| C1 里的部件 | C1 代码位置 | 它在做什么 | 框架里常见的名字 |
|---|---|---|---|
| 工具描述 | `tools.py` 的 `TOOL_SCHEMAS` | 告诉模型有哪些工具、参数是什么 | tool / function tool / `@tool` 装饰器 (从函数签名和注释自动生成描述) |
| 循环 | `agent.py` 的 `for _ in range(max_turns)` | 问模型 → 执行工具 → 把结果追加进对话 → 再问 | agent loop / runner / executor / graph |
| 安全层 | `tools.py` 的 `validate_call` | 执行前检查参数、拦截危险动作 | guardrails / middleware / tool approval (人工确认) |
| 记忆 | `messages` 列表 | 之前所有动作和观测 | memory / session / state / checkpointer |
| 停止条件与记录 | `ended_by`、`rec["turns"]` | 什么时候结束; 事后能复盘每一步 | max iterations / tracing / observability |

还有一样框架**通常不替你做**: 评测。C1 的 `evaluate.py` (30 个任务、成功率、碰撞、失败分类) 在任何框架里都得自己写。
主线讲义里最有价值的数字都来自这一部分。

## 2. 主流框架一览

### LangChain 1.0 与 LangGraph 1.0

- **是什么**: 生态最大的 Python/JS Agent 框架。LangGraph 是底层: 把 Agent 写成一张"状态图", 节点是步骤, 边是跳转;
  LangChain 1.0 的 `create_agent` 是建在 LangGraph 上的高层入口。
- **关键事实**: 两者在 2025-10-22 同时发布 1.0, 官方承诺到 2.0 之前不做破坏性改动。1.0 引入了 **middleware**,
  可以在"调用模型之前 / 之后""执行工具之前"插入自己的逻辑。
- **对照 C1**: `create_agent` ≈ `run_episode`; middleware 的"工具执行前"钩子 ≈ `validate_call`;
  LangGraph 的状态 ≈ `messages`, 而它的 checkpointer 可以把状态存盘, 中断后续跑 (C1 没有这个能力)。
- 来源: [LangChain 博客: LangChain and LangGraph 1.0](https://blog.langchain.com/langchain-langgraph-1dot0/)

### Microsoft Agent Framework

- **是什么**: 微软把自家两个项目 Semantic Kernel 和 AutoGen 合并成的统一框架, 同时支持 .NET 和 Python, MIT 许可。
- **关键事实**: 2026-02-19 发布 RC, 2026-04-03 发布 1.0 正式版。Semantic Kernel 和 AutoGen 转入维护状态,
  新项目建议直接用 Agent Framework。
- **对照 C1**: 单个 Agent 的部分和 C1 一样; 它更擅长的是"工作流" (多个 Agent、多个步骤按图编排) 和企业部署。
- 来源: [Microsoft 开发者博客 (RC 公告)](https://devblogs.microsoft.com/foundry/microsoft-agent-framework-reaches-release-candidate/)、
  [Visual Studio Magazine (1.0 正式版报道)](https://visualstudiomagazine.com/articles/2026/04/06/microsoft-ships-production-ready-agent-framework-1-0-for-net-and-python.aspx)

### OpenAI Agents SDK

- **是什么**: OpenAI 出的轻量框架, `pip install openai-agents`, MIT 许可。核心概念很少:
  **Agent** (带指令和工具的模型)、**Handoff** (把任务交给另一个 Agent)、**Guardrail** (校验输入输出)、
  **Session** (跨轮次的记忆), 外加内置的 **tracing**。
- **关键事实**: 不只支持 OpenAI 的模型, 可以通过 LiteLLM 等适配接入其他供应商。
- **对照 C1**: Guardrail ≈ `validate_call`; tracing ≈ C1 写进 `results/*.json` 的每一轮记录, 只是它有现成的可视化界面。
  Handoff 是 C1 没有的: 只有一个 Agent 时用不上。
- 来源: [官方文档](https://openai.github.io/openai-agents-python/)、[GitHub](https://github.com/openai/openai-agents-python)

### smolagents (Hugging Face)

- **是什么**: Hugging Face 的极简框架, Apache 2.0 许可, 代码量小, 适合读源码。
- **关键事实**: 两种 Agent。`ToolCallingAgent` 和 C1 一样, 让模型输出 JSON 格式的工具调用;
  `CodeAgent` 让模型**直接写一段 Python 代码**当作动作 (例如在一段代码里循环调用 `look` 和 `walk`), 由沙箱执行。
- **对照 C1**: 这正好对着 C1 讲义第 3 节的问题。C1 的 loop 模式里, 小模型一轮发出十几个调用、中间不看结果;
  CodeAgent 等于**正式允许**模型在一轮里写一个小程序。表达力更强, 但安全层要从"检查一个参数"变成"检查一段代码",
  难度高很多。对一台会撞墙的小车, C1 选择了 step 模式 (一次只执行一个动作)。
- 来源: [smolagents 文档](https://huggingface.co/docs/smolagents/)

### Google ADK (Agent Development Kit)

- **是什么**: Google 的 Agent 框架, Apache 2.0 许可, `pip install google-adk`。
- **关键事实**: 官方说明它"针对 Gemini 优化", 但模型无关, 也可以接其他模型; 支持多 Agent 编排和部署到 Google Cloud。
- **对照 C1**: 单 Agent 部分同上; 它的特色在多 Agent 分层和云上部署。
- 来源: [GitHub: google/adk-python](https://github.com/google/adk-python)

### CrewAI

- **是什么**: 多 Agent 协作框架, MIT 许可, 自述"从零构建, 不依赖 LangChain 或其他 Agent 框架"。
- **关键事实**: 两个核心概念。**Crews**: 一组有角色分工的 Agent 自主协作 (例如"研究员 + 写手");
  **Flows**: 事件驱动的工作流, 由开发者精确控制每一步。
- **对照 C1**: 小车只有一个"大脑", 用不上 Crew。但 Crews 和 Flows 的区别值得记住:
  它就是 C1 讲义里 **loop (让模型自己决定下一步)** 和 **plan (流程事先写死)** 的区别, 只是换到了多个 Agent 的层面。
- 来源: [PyPI: crewai](https://pypi.org/project/crewai/)

## 3. MCP: 不是框架, 是"工具的插座标准"

- **是什么**: Model Context Protocol, 一个开放协议, 规定"大模型应用"和"提供工具/数据的服务"之间怎么对话。
  三个角色: **Host** (大模型应用) 里的 **Client**, 连到外部的 **Server**。Server 可以提供 **Tools** (可调用的函数)、
  **Resources** (数据)、**Prompts** (提示模板)。底层消息格式是 JSON-RPC 2.0。
- **关键事实**: 由 Anthropic 提出, 2025 年 12 月捐给 Linux 基金会旗下的 Agentic AI Foundation。
  截至核对日期, 官方规范的最新版本是 2026-07-28。
- **对照 C1**: C1 的 `TOOL_SCHEMAS` 是"只给自己用的工具描述"。如果把小车的五个工具包装成一个 MCP Server,
  那么任何支持 MCP 的应用 (包括上面这些框架里的大部分) 都能直接控制这台车, 不用再为每个框架写一遍适配。
  这是小车以后可以走的一步, 不在主线上。
- 规范里专门有一节讲安全: 工具等于"任意代码执行", Host 在调用工具前必须征得用户同意。
  这和 C1 安全层的出发点一样: **模型说要做的事, 不等于应该直接去做。**
- 来源: [MCP 规范](https://modelcontextprotocol.io/specification)

## 4. 什么时候该用框架, 什么时候手写

| 情况 | 建议 | 原因 |
|---|---|---|
| 学习 Agent 原理 | 手写 (就是 C1) | 框架把循环藏起来了, 出错时你不知道是模型的问题还是框架的问题 |
| 小模型 + 真实硬件 | 手写或极简框架 | 需要精确控制每一步 (C1 的 step 模式、安全层), 框架的默认行为不一定合适 |
| 要接很多现成工具 (搜索、数据库、网页) | 框架 + MCP | 工具生态是框架最大的价值 |
| 多个 Agent 协作、长流程、要断点续跑 | LangGraph / Microsoft Agent Framework / CrewAI Flows | 状态管理和编排自己写很费劲 |
| 需要看清线上每一次调用 | 任何带 tracing 的框架 | C1 的 json 记录在 30 个任务时够用, 到上千次就需要专门的界面 |

## 5. 练习 (可选)

> 和主线一样分三级。这一篇的练习不影响主线进度。

1. **对照 (读)**: 选一个框架, 读它"最小 Agent 示例"的源码 (smolagents 最短), 在本页第 1 节的表格里给每个部件找到对应的类或函数。
2. **移植 (写)**: 用一个框架重写 C1 的 Agent, 工具仍然调用 `sim/car2d.py` 的 `Car2D`, 模型仍用本地 Qwen2.5。
   注意: 安全层 `validate_call` 要接到框架的 guardrail / middleware 上, 不能丢掉。
3. **评测 (比)**: 用 C1 的 `make_tasks(30)` 跑同样的 30 个任务, 和手写版比较成功率、碰撞次数和 token 用量。
   框架默认的提示词和循环方式不同, 结果可能更好也可能更差; 写下你认为差别从哪里来。

状态: 本页内容 ✅; 移植与评测的参考实现 ⬜。
