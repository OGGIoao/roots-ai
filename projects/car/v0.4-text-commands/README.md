# 小车 v0.4 听懂文字 (仿真版)

> 对仿真小车打一句 "去红杯子那里, 停在它跟前", 大模型通过工具调用把车开过去; 在 30 个任务上实测各种写法、各种模型的成败。
> 讲义: [TUTORIAL.md](TUTORIAL.md)。**先读** 知识库 [Agent 循环](../../../library/agent/agent-loop/TUTORIAL.md)。
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OGGIoao/roots-car/blob/master/projects/car/v0.4-text-commands/colab.ipynb)

## 几个实测结论 (Qwen2.5-1.5B, 每组 30 个任务)

- 让模型自己一步步开车 (chat / plan / loop / step): **全部 0/30**。loop 里 30 局只有 6 局看到过目标, 还有 4 局 "没到却说到了";
- 不用大模型的规则控制器: 21/30, 零碰撞 —— 工具本身够用;
- 只让大模型选目标 (`go_to`): 14/30; 再强制它第一轮调用工具: **21/30, 零碰撞**。

还没有数据的组合 (3B、关安全层、云端 API、提示里加搜索策略) 见讲义第 7 节, 欢迎补跑。

## 结构

```
v0.4-text-commands/
├── TUTORIAL.md        本站讲义: 环境准备 → 仿真小车与任务 → 四组实测 → 留给你的实验
├── evaluate.py        在 30 个任务上评测一个 "模型 + 写法" 组合
├── run_suite.py       依次跑讲义用到的全部组合 (已有结果自动跳过)
├── analyze.py         把每一局归到一类结局 (失败分类)
├── baseline.py        不用大模型的规则控制器 (对照组)
├── skills.py          go_to 技能
├── results/           评测结果 (每一局的对话、工具调用、真实轨迹)
├── figures/           make_charts.py 生成的数据图
└── exercises/         练习 3: 自己写 go_to
Agent 循环本体 (agent.py / llm.py / tools.py) 在知识库 library/agent/agent-loop/; 仿真小车在 sim/car2d.py。
```

## 运行

```bash
cd projects/car/v0.4-text-commands
python baseline.py                     # 规则控制器, 只用 CPU
python evaluate.py --backend local:Qwen/Qwen2.5-1.5B-Instruct --mode loop --n 30   # 需要 GPU 和下载好的模型
python analyze.py                      # 失败分类
python figures/make_charts.py          # 由 results/ 重新生成图
cd exercises && python -m pytest test_ex3.py -q
```

环境配置、模型下载 (含国内镜像)、云端 API、显存与耗时, 见讲义第 1 节和第 9 节。
