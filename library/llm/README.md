# 知识库 · 大语言模型

> 大模型内部是怎么工作的: 从把文字变成数字, 到自己训练一个小模型, 再到让它在笔记本上跑得快。总目录见 [library/README.md](../README.md#大语言模型-llm)。

## 单元

| 顺序 | 单元 | 状态 |
|---|---|---|
| 1 | [bpe](bpe/TUTORIAL.md) 分词 | ✅ |
| 2 | [minigpt](minigpt/TUTORIAL.md) 从零写并训练一个 GPT | ✅ |
| 3 | [attention](attention/TUTORIAL.md) 注意力变体与推理延迟 | ✅ |
| 4 | [modern-small-model](modern-small-model/TUTORIAL.md) 现代小模型长什么样 | 🟡 |
| 5 | chat-template 对话模板 | ⬜ |
| 6 | decoding 解码与采样 | ⬜ |
| 7 | local-inference 本地推理服务与延迟 | ⬜ |
| 8 | quantization 量化 | ⬜ |

**前置**: 会写 Python, 第 2 单元起需要会一点 PyTorch (讲义里边用边讲)。1~3 都有 Colab 笔记本, 没有显卡也能跑。

想把 LLM 架构系统串起来 (位置编码、MoE、Mamba 等整棵树), 见选修 [LLM 架构生长树](../../electives/llm-tree/README.md) 和 [解剖台](../../electives/llm-anatomy/README.md)。
