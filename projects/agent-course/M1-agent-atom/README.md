# M1 Agent原子循环

## 目标
手写最小Agent循环（Prompt→Tool→Output），不使用LangChain/AutoGen，理解底层。

## 底层解析
- Agent不是"魔法"，而是**循环结构**：输入→处理→工具调用→输出→更新历史
- 对比Transformer attention（你workspace的`attention.py`）：Agent的"上下文"不是Q/K/V矩阵，而是**Memory/History管理**（文本列表）
- 为什么需要Tool？因为LLM是文本生成器，无法直接执行代码/查询数据库

## 工程实现
- `core.py`：**唯一实现**。手写`AgentLoop`，JSON解析工具调用，纯Python无依赖
- `M1_agent_atom.ipynb`：**只做讲解与演示**，`import core` 后逐步调用，不重复实现逻辑
- 运行实现：`python core.py`
- 运行演示：用 conda 环境 `mlc` 打开 notebook（依赖见仓库根 `requirements-mlc.txt`）

> 约定：实现放 `.py`，notebook 只负责讲解和跑通。
> 之前 notebook 里有 3 个只有注释的代码 cell，容易让人误以为里面有实现，现已消除。

## 注意：工具调用必须是 JSON
`AgentLoop._parse_tool_call` 只识别 JSON 形式（`{"name": ..., "args": {...}}`）。
自然语言输入（如 `"Calculate 3+5"`）解析不出工具调用，会被原样回显 —— 这是最小实现
刻意保留的边界，不是 bug。notebook 里两条路径都有演示。

## 调包决策（M7预告）
- 何时用LangChain？当你需要**多工具链、复杂Memory、标准化评估**时
- 何时不用？当你要**理解底层、定制行为、减少依赖**时（本模块）

## 业务接入
- 代码生成Agent：解析需求→调用`calc`/`query_db`→生成代码
- 数据分析Agent：解析数据需求→调用分析工具→输出报告
