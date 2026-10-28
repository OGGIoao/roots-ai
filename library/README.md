# 知识库 — 造车要用到的知识, 一个知识点一份讲义

> 仓库分两层 (见 [根目录 README](../README.md)):
> - **主线** ([mainline/](../mainline/README.md)): 一站一站造小车, 每站只讲 "这一步在车上做什么、实测结果如何";
> - **知识库** (本目录): 主线用到的每个知识点, 各写成一份完整的讲义。主线遇到不懂的概念就链接到这里, 读完回去接着造车。
>
> 不在造车路上、但值得深入的内容在 [选修](../electives/README.md) (四足机器人、规模化上云、LLM 架构全景等)。

## 一个单元长什么样

每个单元是一个目录, 一个知识点讲透, 颗粒度参照 [分词 BPE](llm/bpe/TUTORIAL.md):

| 文件 | 内容 |
|---|---|
| `TUTORIAL.md` | 讲义: 从 "为什么需要它" 讲起, 案例驱动 (朴素办法 → 实测它怎么失败 → 引出概念), 数字全部实测 |
| 参考实现 | 从零写的最小实现, 每一行都能读懂 |
| `experiments/`、`results/`、`figures/` | 讲义里的数字和图都出自这里, 可以重跑 |
| `exercises/` | 三级练习: ① 填空 → ② 只给接口 → ③ 白纸, 由测试判对错 |

每份讲义开头写明: **前置** (需要先读哪些单元)、**用到它的主线站**、怎么运行。

## 总目录

状态: ✅ 完成 · 🟡 有内容但待补 · ⬜ 规划中。"用到它的站" 指主线的版本号 (见 [主线总图](../mainline/README.md))。"原编号" 是重构前的章节号, 旧文档里还会看到。

### 硬件 ([hardware/](hardware/README.md))

| 单元 | 讲什么 | 状态 | 用到它的站 | 原编号 |
|---|---|---|---|---|
| [electronics](hardware/electronics/TUTORIAL.md) | 电路与 ESP32 入门: 点灯、按钮消抖、PWM、ADC、非阻塞循环 (浏览器仿真) | 🟡 仿真完成, 真板对照待补 | v0.1 | B1 |
| motor-driver | 直流电机与 H 桥驱动 | ⬜ | v0.1 | B2 |
| wifi-remote | WiFi 与网页遥控 | ⬜ | v0.1 | B2 |
| encoder | 编码器与测速 | ⬜ | v0.2 | B3 |
| pid | PID 控制 | ⬜ | v0.2 | B3 |
| sensors | 测距与 IMU | ⬜ | v0.3 | B4 |
| safety-firmware | 状态机、看门狗与急停 | ⬜ | v0.3 | B4 |
| camera | 摄像头采集 | ⬜ | v0.6 | D1 |

### 大语言模型 ([llm/](llm/README.md))

| 单元 | 讲什么 | 状态 | 用到它的站 | 原编号 |
|---|---|---|---|---|
| [bpe](llm/bpe/TUTORIAL.md) | 分词: 把文字变成整数 (字节级 BPE) | ✅ | v0.5 | A1 |
| [minigpt](llm/minigpt/TUTORIAL.md) | 从零写一个 GPT-2 结构的小模型并训练 | ✅ | v0.5 | A2 |
| [attention](llm/attention/TUTORIAL.md) | 注意力变体 (MHA / MQA / GQA / MLA) 与推理延迟 | ✅ | v0.5 | A3 |
| [modern-small-model](llm/modern-small-model/TUTORIAL.md) | 现代小模型长什么样 (RoPE、RMSNorm、SwiGLU ……) | 🟡 待精简 | v0.5 | A4 |
| [chat-template](llm/chat-template/TUTORIAL.md) | 对话模板: 模型怎么分清 system / user / assistant / tool | ✅ | v0.4, v0.5 | — |
| [decoding](llm/decoding/README.md) | 解码与采样: 贪心、温度、top-p, 为什么评测用贪心 | 🟡 | v0.4 | — |
| local-inference | 本地推理服务与延迟 | ⬜ | v0.5 | C3 |
| quantization | 量化: 把模型变小变快, 代价是什么 | ⬜ | v0.6, v1.0 | D2 / F1 |

### Agent ([agent/](agent/README.md))

| 单元 | 讲什么 | 状态 | 用到它的站 | 原编号 |
|---|---|---|---|---|
| [agent-loop](agent/agent-loop/TUTORIAL.md) | Agent 循环: 消息、工具调用、四种循环写法、安全层、评测方法 | 🟡 初稿 | v0.4 | C1 |
| agent-eval | Agent 评测进阶: 任务集设计、置信区间、回归评测 | ⬜ | v0.5 | C4 |
| mcp | 工具协议 MCP: 把小车的工具做成任何应用都能调用的服务 | ⬜ | 延伸 | — |

工业界的 Agent 框架 (LangChain、OpenAI Agents SDK 等) 在选修 [agent-frameworks](../electives/agent-frameworks/README.md)。

### 强化学习 ([rl/](rl/README.md))

| 单元 | 讲什么 | 状态 | 用到它的站 | 原编号 |
|---|---|---|---|---|
| [bandit](rl/bandit/TUTORIAL.md) | 老虎机: 探索与利用, ε-greedy, UCB | ✅ | v0.7 | E1 |
| q-learning | FrozenLake: 状态、折扣回报、价值、Q-learning | ⬜ | v0.7 | E2 |
| policy-gradient | CartPole: 策略梯度 REINFORCE、baseline | ⬜ | v0.7 | E3 |
| ppo | PPO: 裁剪的替代目标、GAE | ⬜ | v0.7 | E4 |

### 后训练 ([post-training/](post-training/README.md))

| 单元 | 讲什么 | 状态 | 用到它的站 | 原编号 |
|---|---|---|---|---|
| [tool-finetune](post-training/tool-finetune/README.md) | 监督微调与 LoRA: 让 0.5B 小模型只输出小车的工具调用 | 🟡 待改造 (现为通用 QLoRA 演示) | v0.5 | C2 |
| rl-for-lm | 把语言模型当策略做强化学习 | ⬜ | 延伸 | P1 |
| dpo | 偏好对齐 DPO | ⬜ | 延伸 | P2 |
| grpo | 用小车任务的成败作奖励, GRPO 提升工具调用成功率 | ⬜ | v0.5 之后 | P3 |

### 具身智能 ([embodied/](embodied/README.md))

| 单元 | 讲什么 | 状态 | 用到它的站 | 原编号 |
|---|---|---|---|---|
| sim2real | 仿真到真车: 差距在哪、怎么测、域随机化 | ⬜ | v0.7 | E5 |
| imitation-learning | 模仿学习: 从示范里学控制 | ⬜ | 延伸 | — |
| vla | 视觉-语言-动作模型: 一个模型从画面和指令直接出动作 | ⬜ | 延伸 | — |

完整的四足机器人具身路线 (从经典控制到 RL 步态、模仿学习、语言与视觉) 在选修 [quadruped](../electives/quadruped/README.md)。

## 写作顺序

不按领域一口气写完, 而是**主线走到哪一站, 先补那一站要用的单元**。当前顺序见 [ROADMAP.md](../ROADMAP.md)。
