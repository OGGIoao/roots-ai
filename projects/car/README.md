# 项目 ① 小车 — 一站一站造出一台听得懂话的小车

> 本仓库的第一个项目 (全部项目见 [projects/](../README.md))。每一站结束, 小车就多一项本事, 你手里就有一台能演示的车 (先在仿真里, 再在真车上)。
> 每一站只讲 "这一步在车上做什么、怎么跑、实测结果如何"; 用到的原理链接到 [知识库](../../library/README.md), 不在这里重讲。零基础的同学顺着链接下去补, 补完回来接着造。

```mermaid
flowchart LR
    V01["v0.1<br/>能遥控"] --> V02["v0.2<br/>走得准"] --> V03["v0.3<br/>不撞墙"] --> V04["v0.4<br/>听懂文字"]
    V04 --> V05["v0.5<br/>自己的大脑"] --> V06["v0.6<br/>看得见"] --> V07["v0.7<br/>自己学"] --> V10["v1.0<br/>听得懂话"]
    classDef doing fill:#fde3d7,stroke:#eb6834,color:#0b0b0b
    classDef todo fill:#efeeea,stroke:#8a8984,color:#0b0b0b
    class V04 doing
    class V01,V02,V03,V05,V06,V07,V10 todo
```

<sub>橙色: 这一站已经可以跑 (仿真版); 灰色: 还没开始。项目处于**早期阶段**, 真车还没有造出来。</sub>

## 各站

| 站 | 小车多了什么本事 | 先学 (知识库) | 本站 |
|---|---|---|---|
| **v0.1 能遥控** | 手机网页遥控前后左右 | [electronics](../../library/hardware/electronics/TUTORIAL.md) 🟡 · power ⬜ · datasheets-tools ⬜ · mcu-architecture ⬜ · motor-driver ⬜ · wifi-remote ⬜ | ⬜ |
| **v0.2 走得准** | 执行 "走 1 米" "转 90 度" | encoder ⬜ · pid ⬜ · kinematics ⬜ · state-estimation ⬜ | ⬜ |
| **v0.3 不撞墙** | 自己避障; 断网自动停车 | sensors ⬜ · comm-protocols ⬜ · robot-networking ⬜ · safety-firmware ⬜ | ⬜ |
| **v0.4 听懂文字** | 打一句话, 大模型调用工具把车开过去 | [agent-loop](../../library/agent/agent-loop/TUTORIAL.md) 🟡 · [chat-template](../../library/llm/chat-template/TUTORIAL.md) ✅ · [decoding](../../library/llm/decoding/TUTORIAL.md) ✅ · [structured-output](../../library/llm/structured-output/TUTORIAL.md) ✅ | [仿真版](v0.4-text-commands/TUTORIAL.md) 🟡 (真车版 ⬜) |
| **v0.5 自己的大脑** | 换成自己微调的小模型, 在笔记本上本地运行 | [bpe](../../library/llm/bpe/TUTORIAL.md) ✅ · [minigpt](../../library/llm/minigpt/TUTORIAL.md) ✅ · [attention](../../library/llm/attention/TUTORIAL.md) ✅ · [modern-small-model](../../library/llm/modern-small-model/TUTORIAL.md) 🟡 · [tool-finetune](../../library/post-training/tool-finetune/README.md) 🟡 · local-inference ⬜ · agent-eval ⬜; 可选用基座线 [roots-lm](../../base/README.md) | ⬜ |
| **v0.6 看得见** | 用摄像头找到红杯子 | camera ⬜ · detection ⬜ · perception-3d ⬜ · quantization ⬜ | ⬜ |
| **v0.7 自己学** | 仿真里用强化学习训练的控制策略上真车 | [bandit](../../library/rl/bandit/TUTORIAL.md) ✅ · mdp ⬜ · q-learning ⬜ · policy-gradient ⬜ · ppo ⬜ · continuous-control ⬜ · learned-dynamics ⬜ · sim2real ⬜ | ⬜ |
| **v1.0 听得懂话** | 一句语音指挥小车完成任务 | audio ⬜ · keyword-spotting ⬜ · asr ⬜ · tts ⬜ · tinyml ⬜ | ⬜ |

每一站的验收标准、小车的工具接口、安全规则和预算见 [PRODUCT.md](PRODUCT.md); 开发板选型、引脚分配和物料清单见 [hardware/](hardware/README.md); 推进顺序见 [ROADMAP.md](../../ROADMAP.md)。

## 一站的结构

```
projects/car/v0.x-<名字>/
├── TUTORIAL.md    本站讲义: 先学什么 → 准备环境 → 在车上跑 → 实测结果 → 留给你的实验 → 下一站
├── README.md      一页概览和运行命令
├── 实验与评测脚本   在小车 (仿真或真车) 上跑, 结果写进 results/
├── results/  figures/
└── exercises/     本站的练习 (和小车直接相关的那一级; 原理的练习在知识库单元里)
```

仿真小车是所有站共用的, 在仓库根目录的 [sim/](../../sim/car2d.py); 它和真车暴露同一套工具接口。

## 旧路径对照

仓库重构过两次, 旧链接对应的新位置:

| 旧路径 | 新路径 | 时间 |
|---|---|---|
| `mainline/` (主线) | [projects/car/](./) | 2026-10-28 |
| `mainline/v0.4-text-commands` | [projects/car/v0.4-text-commands](v0.4-text-commands/) | 2026-10-28 |
| 根目录 `hardware/` | [projects/car/hardware](hardware/README.md) | 2026-10-28 |
| 根目录 `PRODUCT.md` | [projects/car/PRODUCT.md](PRODUCT.md) | 2026-10-28 |
| `electives/llm-tree` | [library/llm/llm-tree](../../library/llm/llm-tree/README.md) | 2026-10-28 |
| `electives/agent-frameworks` | [library/agent/agent-frameworks](../../library/agent/agent-frameworks/README.md) | 2026-10-28 |
| `electives/llm-anatomy` | [library/frontier/anatomy](../../library/frontier/anatomy/README.md) | 2026-10-28 |
| `electives/scale-up` | [base/cloud-plan.md](../../base/cloud-plan.md) | 2026-10-28 |
| `mainline/A1-bpe` | [library/llm/bpe](../../library/llm/bpe/) | 2026-10-27 |
| `mainline/A2-minigpt` | [library/llm/minigpt](../../library/llm/minigpt/) | 2026-10-27 |
| `mainline/A3-attention` | [library/llm/attention](../../library/llm/attention/) | 2026-10-27 |
| `mainline/A4-modern-small-model` | [library/llm/modern-small-model](../../library/llm/modern-small-model/) | 2026-10-27 |
| `mainline/B1-electronics` | [library/hardware/electronics](../../library/hardware/electronics/) | 2026-10-27 |
| `mainline/C1-agent-loop` | 原理: [library/agent/agent-loop](../../library/agent/agent-loop/); 小车实测: [projects/car/v0.4-text-commands](v0.4-text-commands/) | 2026-10-27 |
| `mainline/C2-tool-finetune` | [library/post-training/tool-finetune](../../library/post-training/tool-finetune/) | 2026-10-27 |
| `mainline/E1-bandit` | [library/rl/bandit](../../library/rl/bandit/) | 2026-10-27 |
| `electives/post-training` | [library/post-training](../../library/post-training/README.md) | 2026-10-27 |
