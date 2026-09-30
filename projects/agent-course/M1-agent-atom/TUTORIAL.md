# M1 Agent原子循环 — 教程

## 1. 为什么要手写？
调包（LangChain）隐藏了：Prompt模板、Tool解析、Memory管理、循环控制。手写后你才知道每一步在做什么。

## 2. 对比Transformer（你workspace的attention.py）
- Transformer：输入→Embedding→Attention（Q/K/V）→FFN→输出
- Agent：输入→History管理→LLM生成→Tool解析→执行→更新History→输出
- 关键区别：Agent的"上下文"是**可修改的文本历史**，不是固定的attention矩阵

## 3. 运行步骤
```bash
python core.py
```

## 4. 扩展练习（打通链路）
- 修改`core.py`，增加多轮对话（Memory持久化到JSON文件）
- 添加新Tool（如`file_read`），模拟数据分析Agent
- 对比：用LangChain重写同一功能，观察差异
