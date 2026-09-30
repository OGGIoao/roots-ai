# M1 Agent原子循环

## 目标
手写最小Agent循环（Prompt→Tool→Output），不使用LangChain/AutoGen，理解底层。

## 底层解析
- Agent不是"魔法"，而是**循环结构**：输入→处理→工具调用→输出→更新历史
- 对比Transformer attention（你workspace的`attention.py`）：Agent的"上下文"不是Q/K/V矩阵，而是**Memory/History管理**（文本列表）
- 为什么需要Tool？因为LLM是文本生成器，无法直接执行代码/查询数据库

## 工程实现
- `core.py`：手写`AgentLoop`，JSON解析工具调用，纯Python无依赖
- 运行：`python core.py`

## 调包决策（M7预告）
- 何时用LangChain？当你需要**多工具链、复杂Memory、标准化评估**时
- 何时不用？当你要**理解底层、定制行为、减少依赖**时（本模块）

## 业务接入
- 代码生成Agent：解析需求→调用`calc`/`query_db`→生成代码
- 数据分析Agent：解析数据需求→调用分析工具→输出报告
