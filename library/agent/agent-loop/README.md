# Agent 循环 (知识库 · Agent)

> 大模型怎么 "动手": 消息与角色、工具调用、四种循环写法、安全层、怎么评测。用仿真小车做例子, 但讲的东西对任何 Agent 都适用。
> 讲义: [TUTORIAL.md](TUTORIAL.md)。用到它的主线站: [v0.4 听懂文字](../../../mainline/v0.4-text-commands/TUTORIAL.md) (在小车上实测本单元的每一种写法)。

**只需要 Python**: 演示和练习都用假模型, 不下载模型, 不需要显卡。

## 结构

```
agent-loop/
├── TUTORIAL.md        讲义
├── demo_minimal.py    最小演示: 假模型开车, 打印整段对话 (一秒内)
├── agent.py           Agent 循环本体 run_episode (chat / plan / loop / step / skill)
├── tools.py           工具说明书 (JSON Schema) 与安全层 validate_call
├── llm.py             模型后端: 本地 transformers / 云端 API / 剧本 (测试用)
├── figures/           make_diagrams.py 生成的结构图
└── exercises/         练习 1 (补全安全层)、练习 2 (写一步一停的循环)
```

## 运行

```bash
cd library/agent/agent-loop
python demo_minimal.py --seed 1
cd exercises && python -m pytest test_ex1.py -q
```

换成真模型跑、在 30 个任务上评测, 去主线 [v0.4 这一站](../../../mainline/v0.4-text-commands/TUTORIAL.md)。
