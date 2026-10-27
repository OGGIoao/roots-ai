# C1 Agent 循环 — 主线 C "让车听懂话" 的第一章

> 主线 C 服务于 v0.4: 对小车说一句话, 它自己完成。本章先在仿真小车上把 "大模型 → 工具调用 → 小车 → 观测 → 大模型" 这个循环手写出来, 并用 30 个任务实测它到底行不行。
> 讲义: [TUTORIAL.md](TUTORIAL.md)。工业界的 Agent 框架和这个手写循环的对应关系见选修 [agent-frameworks](../../electives/agent-frameworks/README.md)。
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OGGIoao/roots-car/blob/master/mainline/C1-agent-loop/colab.ipynb)

本章是一串案例, 后一个都是因为前一个在实测里失败才出现的:

| 节 | 案例 | 朴素办法怎么失败 | 引出的概念 |
|---|---|---|---|
| 2 | 只聊天; 开环一次出完动作 | 车不动; 动了但走偏、撞墙也不知道 | 工具调用, 开环与闭环 |
| 3 | 闭环 loop; 一步一停 step | 一轮发十几个调用不看结果; 说了要做却不调用 | 观测反馈, 每轮执行几个调用 |
| 4 | 关掉安全层 | 模型要求的动作直接执行 | 参数校验, 拦下并解释 |
| 5 | 30 个任务评测, 规则控制器对照 | "跑通一次" 不等于能用; 没到却说到了 | 成功率, 失败分类, 调用预算 |
| 6 | 技能 go_to: 大模型只管选目标 | — | 分工: 大模型听懂话, 控制代码 (以后是 RL 策略) 开车 |

## 结构

```
C1-agent-loop/
├── TUTORIAL.md        讲义: 案例驱动, 数字全部实测
├── tools.py           工具说明书 (JSON Schema) 与安全层 validate_call
├── llm.py             三种模型后端: 本地 transformers / 云端 API / 剧本 (测试用)
├── agent.py           手写的 Agent 循环 (chat / plan / loop / step / skill 五种模式)
├── skills.py          高层技能 go_to (第 6 节)
├── baseline.py        不用大模型的规则控制器 (第 5 节的对照组)
├── evaluate.py        在 30 个任务上评测一个 "模型 + 模式" 组合
├── run_suite.py       依次跑讲义用到的全部组合
├── analyze.py         把每一局归到一类结局 (失败分类)
├── results/           评测结果 (JSON: 每一局的对话、工具调用、真实轨迹)
├── figures/           make_charts.py (数据图) 与 make_diagrams.py (结构图)
└── exercises/         三级练习 + 测试
仿真小车在仓库根目录的 sim/car2d.py (以后的章节也用它)。
```

## 运行

```bash
cd mainline/C1-agent-loop
python baseline.py                     # 规则控制器, 只用 CPU, 约 10 秒
python evaluate.py --backend local:Qwen/Qwen2.5-1.5B-Instruct --mode loop --n 30   # 需要 GPU, 约 3 分钟
python run_suite.py                    # 讲义用到的全部本地组合 (三个模型, 测量平台上约 2 小时)
python run_suite.py --api              # 再加上云端 API 对照 (需要 .env 里的 Key, 会产生少量费用)
python analyze.py                      # 失败分类
python figures/make_charts.py          # 由 results/ 重新生成图
cd exercises && python -m pytest test_ex1.py -q
```

本地模型第一次运行会从 Hugging Face 下载权重 (1.5B 约 3 GB); 国内网络先设置 `HF_ENDPOINT=https://hf-mirror.com`。
