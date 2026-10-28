# 总框架设计

> 2026-10-28 逐个领域讨论后的汇总, 当天确认; 仓库名定为 **roots-ai** (中文名: 扎根 AI)。按第 9 节的步骤一点一点改, 每一步单独一个 PR。
> 状态: ✅ 完成 · 🟡 进行中 · ⬜ 未开始。★ = 项目会用到 · 🔥 = 前沿。

---

## 0. 一句话

**一套自己从头搭起来的 AI 全栈知识库, 加上一个自己从零训出来的小模型, 再用几个真实的作品 (第一个是听得懂话的小车) 把知识串成能跑通的链路。**

不收集别人的教程, 每个知识点都自己写讲义、自己实现、自己实测; 前沿的东西用同样的方式消化进来。

---

## 1. 三层结构

```
┌──────────────────────────────────────────────────────────┐
│ 项目层  projects/                                          │
│   ① 小车   ② 学习助教   ③ 推理服务   (④ 领域后训练, 第二批)    │
│   每个项目是一条"造东西"的主线, 一站一个能用的版本              │
├──────────────────────────────────────────────────────────┤
│ 基座线  base/  roots-lm                                    │
│   自己从零训的小模型: 笔记本 → 云 → 装前沿部件 → 长眼睛和耳朵    │
│   所有项目都可以基于它延伸                                    │
├──────────────────────────────────────────────────────────┤
│ 知识库  library/                                           │
│   8 个领域, 约 190 个单元, 一个单元一个知识点 (颗粒度参照 bpe)   │
│   + frontier/ 前沿档案卡与雷达   + 安全线索引                  │
└──────────────────────────────────────────────────────────┘
```

| 层 | 回答的问题 | 写法 |
|---|---|---|
| 知识库 | 这个知识点是什么、为什么、怎么实现 | 讲义 + 参考实现 + 三级练习 (填空 → 接口 → 白纸), 数字全部实测 |
| 基座线 | 这些知识点合在一起, 能不能训出一个真的模型 | 一份持续生长的代码 + 每个阶段的检查点和实测报告 |
| 项目 | 学到的东西能不能做成一个别人用得上的作品 | 一站一个版本, 每站只讲"做了什么、怎么跑、实测结果", 原理链接到知识库 |

**选内容的规则**: 新知识点进知识库; 哪个项目、哪个基座阶段用到它, 就在单元里注明。知识库可以比任何一个项目都大, 但项目的进度不能被知识库拖住 (第 8 节的写作顺序保证这一点)。

---

## 2. 知识库: 8 个领域

已有 10 个单元 (含进行中的), 其余都是 ⬜。每个领域一张总览图 (`*-map`) 作为入口。

### 2.1 基础 foundations (新领域, 约 21 个)

读者门槛: 会写 Python 函数、会用命令行。数学 "够用为止", 每个概念都用代码写一遍。

| 段 | 单元 |
|---|---|
| F0 | foundations-map (附自测题: 从哪开始、哪些可以跳过) |
| F1 编程与工具 | dev-environment · git-github · python-for-ai · numpy |
| F2 数学 | linear-algebra · calculus · probability · information-theory · optimization |
| F3 机器学习 | ml-basics · autograd · neural-networks · pytorch-basics · training-practice |
| F4 计算机系统 | gpu-basics · profiling · linux-server · cloud-gpu |
| F5 研究方法 | experiment-method · reading-papers |

### 2.2 硬件 hardware (约 24 个)

三层算力: 单片机 (实时控制) / 单板电脑 (端侧 AI) / 电脑与云 (大模型)。

| 段 | 单元 |
|---|---|
| H0 | hardware-map |
| H1 电子基础 | electronics 🟡★ · power ★ · datasheets-tools ★ · pcb-basics |
| H2 单片机与固件 | mcu-architecture ★ · comm-protocols ★ · rtos ★ · firmware-engineering ★ |
| H3 执行与控制 | motor-driver ★ · encoder ★ · pid ★ · servo-stepper |
| H4 感知硬件 | sensors ★ · camera ★ · audio ★ |
| H5 连接 | wifi-remote ★ · robot-networking ★ |
| H6 安全 | safety-firmware ★ |
| H7 端侧 AI 🔥 | tinyml ★ (小车感知板) · edge-npu · edge-gpu · on-device-llm · edge-system-design |

单板电脑不进小车预算, 作为可选; 没买时讲原理 + 笔记本 CPU 模拟, 真板数字标 "没有数据"。

### 2.3 大语言模型 llm (约 37 个)

一条链: 输入 → 架构 → 预训练 → (后训练, 见 2.5) → 推理能力 → 推理与部署; 旁边挂评测和可解释性。选修 llm-tree 移进来, 作为架构段的总地图。

| 段 | 单元 |
|---|---|
| 输入 | bpe ✅ |
| 架构 | llm-tree (地图) · minigpt ✅ · attention ✅ · modern-small-model 🟡 · position-encoding · residual-connections 🔥 · moe 🔥 · sparse-attention 🔥 · linear-attention-ssm 🔥 · hybrid-architecture 🔥 · long-context · diffusion-lm 🔥 |
| 预训练 | pretraining-data · scaling-laws · optimizers 🔥 · training-stability · low-precision 🔥 · distributed-training · mtp 🔥 · mid-training · distillation ★ |
| 推理能力 | chain-of-thought · test-time-scaling 🔥 · reasoning-models 🔥 |
| 推理与部署 | chat-template ✅★ · decoding 🟡★ · structured-output ★ · quantization ★🔥 · local-inference ★ · paged-kv-cache · continuous-batching · prefix-caching · speculative-decoding 🔥 · batch-invariance 🔥 · serving-at-scale (只读 DeepInfer 源码) |
| 旁支 | llm-eval · interpretability |

### 2.4 Agent (约 24 个)

| 段 | 单元 |
|---|---|
| A0 | agent-map |
| A1 核心机制 | agent-loop 🟡★ · tool-calling ★ · planning · harness-engineering 🔥 |
| A2 上下文与记忆 🔥 | context-engineering · memory |
| A3 检索增强 | embeddings ★ · rag-basics ★ · rag-advanced 🔥 · rag-eval ★ |
| A4 协议与生态 🔥 | mcp · agent-interop (MCP / A2A / Agent Skills / AGENTS.md 等) · agent-frameworks (从选修移入) |
| A5 典型 Agent | coding-agent · computer-use · deep-research (写作顺序排最后) |
| A6 多智能体 | multi-agent |
| A7 工程化落地 ★ | data-pipelines · agent-deployment · observability · cost-latency |
| A8 评测 | agent-eval ★ |
| A9 安全 | agent-safety ★ |

分工: LLM 领域管"模型怎么跑得快", Agent 的 A7 管"整个系统怎么稳定地交付给用户"。Agent 强化学习在后训练领域。

### 2.5 后训练 post-training (约 20 个)

强化学习基础在 RL 领域, 这里只讲"把语言模型当策略"之后多出来的部分。

| 段 | 单元 |
|---|---|
| P0 | post-training-map |
| P1 监督微调 | sft · instruction-data · lora-qlora · tool-finetune 🟡★ |
| P2 偏好对齐 | reward-models · rlhf-ppo · dpo |
| P3 可验证奖励 RL | grpo · grpo-variants 🔥 · process-rewards · rl-dynamics 🔥 |
| P4 Agent RL 🔥 | agentic-rl · car-grpo ★ |
| P5 蒸馏 | reasoning-distill · on-policy-distill 🔥 |
| P6 RL 基础设施 🔥 | rl-infra · mini-rl-loop · framework-reading (slime / OpenRLHF / verl) |
| P7 | safety-alignment |

### 2.6 强化学习 rl (约 20 个)

**小车仿真器 (`sim/car2d.py`) 是全领域的练习环境**: 从格子版小车练到连续控制, 学完 PPO 就是小车 v0.7。每个单元另配一个标准 Gymnasium 环境做对照。

| 段 | 单元 |
|---|---|
| R0 | rl-map |
| R1 表格方法 | bandit ✅ · mdp · dynamic-programming · mc-td · q-learning |
| R2 深度价值方法 | dqn · dqn-improvements |
| R3 策略方法 | policy-gradient · actor-critic · ppo ★ · continuous-control ★ |
| R4 模型与数据 🔥 | model-based-rl · offline-rl |
| R5 工程与规模化 | rl-engineering · reward-design · parallel-simulation 🔥 (Windows 能否跑先实测) |
| R6 延伸 | multi-agent-rl · exploration · rl-with-foundation-models |

### 2.7 具身智能 embodied (约 25 个)

| 段 | 单元 |
|---|---|
| E0 | embodied-map |
| E1 本体与感知 | kinematics ★ · state-estimation ★ · perception-3d ★ · slam |
| E2 规划与经典控制 | motion-planning ★ · mpc |
| E3 从示范和试错中学 | imitation-learning · action-chunking · diffusion-policy · sim2real ★ · real-robot-rl |
| **E4 世界模型 🔥 (重点)** | world-model-basics · **car-world-model ★ (给小车训世界模型, 笔记本可做)** · video-world-models · latent-world-models · world-action-models · world-model-data |
| E5 具身基础模型 🔥 | vla · vla-finetune · robot-data |
| E6 系统与部署 | edge-deployment ★ · robot-software · embodied-eval · embodied-safety |

分界: 算法 (运动学、估计、规划) 在这里; 传感器、驱动、单片机上的 PID 在硬件领域。机械臂内容先用仿真写。

### 2.8 多模态 multimodal (新领域, 约 20 个)

生成模型基础 (M3) 放在这里, 具身领域的扩散策略、视频世界模型, LLM 领域的扩散语言模型都链接过来。

| 段 | 单元 |
|---|---|
| M0 | multimodal-map |
| M1 视觉 | image-basics · cnn · detection ★ · vit · self-supervised-vision |
| M2 音频与语音 | audio-basics · keyword-spotting ★ · asr ★ · tts ★ · audio-codecs 🔥 |
| M3 生成模型基础 | vae · diffusion · image-generation · video-generation |
| M4 多模态大模型 🔥 | vision-language-alignment · vlm · native-multimodal · omni-models · multimodal-eval |

### 2.9 两条跨领域的索引线

| 线 | 是什么 | 串起来的单元 |
|---|---|---|
| **安全线** | 一页索引, 安全单元留在各自领域 | chat-template (注入) · safety-alignment · agent-safety · safety-firmware · sim2real · embodied-safety · llm-eval (红队) · interpretability |
| **前沿线** | 见第 4 节 | 所有 🔥 单元 |

---

## 3. 基座线 roots-lm

自己从零训一个小模型, 一切的根。所有阶段用同一套代码, 每个阶段发布检查点 + 实测报告; 每个知识库单元学完, 就把对应的部件装上去, 测前后变化。

| 阶段 | 规模 | 在哪跑 | 做什么 | 用到的领域 |
|---|---|---|---|---|
| **1 本地** | 几千万参数 (先实测再定) | 笔记本 | 全流程跑通: 数据 → 分词 → 预训练 → SFT → DPO / GRPO → 量化 → 自己的推理引擎 | 基础、LLM、后训练 |
| **2 上云** | 几亿到十亿参数 | 云 GPU | 同一套代码放大; 得到真正能用的小基座 | + 分布式训练、cloud-gpu |
| **3 前沿部件** | 同上 | 本地或云 | MoE、混合注意力、MTP、新优化器等逐个装上, 和不装的版本对比 | LLM 架构与预训练的 🔥 单元 |
| **4 长眼睛和耳朵** | 同上 | 本地或云 | 接视觉和语音, 做成小号视觉语言模型或全能模型 (已有约 0.1B 规模的开源全能模型 MiniMind-O 证明小模型也能做, 写时核实) | 多模态 |

**说明**: 自己训的模型通用能力会远弱于 Qwen 这类模型。项目里窄任务 (小车指令、工具调用、格式抽取) 用自己的模型, 需要通用能力时用开源模型, 两者对比本身就是教学内容。

选修 scale-up (规模化与上云) 并入基座线阶段 2 和 LLM 的 distributed-training。

---

## 4. 前沿怎么进来

**技术为纲, 厂商为索引。** 各家在同一批技术问题上竞争, 所以按技术放最清楚。

| 层 | 放在哪 | 做什么 |
|---|---|---|
| 1. 单元里的"前沿对照"节 | 每个 🔥 单元的最后一节 | 先讲原理并在小模型上实现, 再并排对比各家做法。例: residual-connections 里对比 DeepSeek V4 的 mHC 和 Kimi K3 的 Attention Residuals |
| 2. 厂商档案卡 | `library/frontier/labs/` | DeepSeek、Kimi、智谱 GLM、Qwen 各一页 (以后可加), 按时间列每代模型带来了什么, 每条链接到单元。想"沿着 DeepSeek 读一遍"就照着卡走 |
| 3. 前沿雷达 | `library/frontier/radar.md` | 带日期的流水账, 新东西先登记; 能用开源代码或权重核实, 并且确实重要 (两家以上采用或明显改变做法) 时, 升级进单元 |

**核实规则** (沿用仓库现有做法):
- 数字和架构细节以技术报告、模型配置、源码为准; 二手文章只用来找线索。
- 能用解剖台 (`llm-anatomy`, 移到 `library/frontier/anatomy/`) 读真实配置的, 一律读出来核对。
- 跑不了的 (比如 DeepInfer 只支持 SM100 和昇腾) 只读源码, 讲义里写明。
- 没测过的写 "没有数据"。

---

## 5. 项目

| 项目 | 做出来是什么 | 主要用到 | 状态 |
|---|---|---|---|
| **① 小车** | 听得懂话的玩具车, v0.1 → v1.0 | 硬件、具身、RL、Agent、多模态 | v0.4 仿真版 🟡 |
| **② 学习助教** | 能回答本知识库问题、出题、判练习的助手 | Agent (检索、记忆、落地、评测)、后训练 | ⬜ |
| **③ 推理服务** | 用自己写的推理引擎部署基座模型和开源小模型, 压测、量化、优化 | LLM 推理与部署、基础 F4 | ⬜ |
| ④ 领域后训练 (第二批) | 选一个窄领域, SFT + 可验证奖励 RL 把小模型训专, 配完整评测 | 后训练、评测 | 基座线到阶段 2 后再开 |

**待定的候选** (第 10 节): 机械臂 (SO-101 + LeRobot, 补上抓取和 VLA 的真机部分)、端侧推理 (车上或板上直接跑小模型)、四足机器人 (现在是选修)。

小车各站与它用到的单元 (节选, 完整表在 `projects/car/README.md`):

| 站 | 本事 | 关键单元 |
|---|---|---|
| v0.1–v0.3 | 遥控、走得准、不撞墙 | 硬件 H1–H6 |
| v0.4 | 听懂文字 | agent-loop、chat-template、decoding、structured-output |
| v0.5 | 自己的大脑 | tool-finetune、local-inference、agent-eval; 可选用 roots-lm |
| v0.6 | 看得见 | camera、detection、perception-3d、quantization |
| v0.7 | 自己学 | RL R1–R3、car-world-model、sim2real |
| v1.0 | 听得懂话 | audio、keyword-spotting、asr、tts |

---

## 6. 技能地图

你要练的四项技能, 在哪练、产出什么作品:

| 技能 | 在哪练 | 拿得出手的作品 |
|---|---|---|
| **微调与后训练** | 后训练领域 · 基座线阶段 1–2 · 项目 ④ | 同一个模型做 SFT / DPO / GRPO 前后的对比报告; 自己写的 RL 训练循环; 小车工具调用微调 |
| **推理优化** | LLM 推理与部署段 · 基础 F4 · 项目 ③ | 自己写的推理引擎 (分页 KV 缓存、连续批处理、投机解码) + 压测报告; 量化前后的速度与质量对比 |
| **Agent 落地与数据管线** | Agent A3、A7、A8 · 项目 ② | 部署上线的学习助教, 带评测面板和成本监控; 文档入库管线 |
| **物理 AI 与端侧推理** | 具身 · RL · 硬件 H7 · 项目 ① | 小车 v0.7 的 RL 控制器从仿真到真车; 小车世界模型; 感知板上的唤醒词 |

---

## 7. 目录结构 (目标)

```
roots-ai/
├── README.md  ROADMAP.md  FRAMEWORK.md  PRODUCT.md (→ 并入 projects/car)
├── library/
│   ├── README.md                 总目录: 8 个领域、所有单元、状态、被谁用到
│   ├── foundations/ hardware/ llm/ agent/ rl/ post-training/ embodied/ multimodal/
│   ├── frontier/                 labs/ (厂商档案卡) · radar.md · anatomy/ (原 electives/llm-anatomy)
│   └── safety.md                 安全线索引
├── base/                         roots-lm: 代码、各阶段检查点说明、实测报告
├── projects/
│   ├── car/                      原 mainline/ + 原根目录 hardware/ (选型、引脚、物料清单)
│   ├── tutor/
│   └── serving/
├── electives/                    只剩 quadruped (是否升为项目见第 10 节)
├── sim/  common/  tools/  tests/
```

变动汇总:

| 原位置 | 新位置 |
|---|---|
| `mainline/` | `projects/car/` |
| 根目录 `hardware/` | `projects/car/hardware/` |
| `electives/llm-tree/` | `library/llm/llm-tree/` |
| `electives/agent-frameworks/` | `library/agent/agent-frameworks/` |
| `electives/llm-anatomy/` | `library/frontier/anatomy/` |
| `electives/scale-up/` | 并入 `base/` (阶段 2) 和 `library/llm/distributed-training/` |
| 单元 `rl/q-learning` (原计划) | 拆成 mdp、dynamic-programming、mc-td、q-learning |

---

## 8. 写作顺序

原则不变: **项目走到哪, 先补那里要用的单元**; 不按领域一口气写完。

| 阶段 | 内容 |
|---|---|
| **第 0 步: 收尾 + 改结构** | ① 当前分支 (chat-template、decoding 核心) 推送开 PR; ② 按第 9 节改目录和改名, 只搬家不写新内容 |
| **第 1 步: 小车 v0.4 → v0.5 + 基础核心** | decoding 收尾 · structured-output · local-inference · quantization · agent-eval; 基础的 dev-environment、linear-algebra、calculus、probability、autograd、pytorch-basics |
| **第 2 步: 基座线阶段 1 + 小车 v0.7 的前半** | 预训练段 · 后训练 P1、P3 · RL R0–R3 · roots-lm 在笔记本上全流程跑通 |
| **第 3 步: 两个新项目 + 世界模型** | Agent A3、A7、A8 → 学习助教; 推理引擎系列 → 推理服务; 具身 E4 → 小车世界模型 |
| **第 4 步: 上云与前沿** | 基座线阶段 2、3; 各 🔥 单元; 多模态与基座线阶段 4; 硬件随小车各站推进 |

每一步结束都更新本文件的状态。

---

## 9. 改结构的步骤 (确认后执行)

1. 当前分支先合并, 保证起点干净;
2. 新开分支, 只做目录搬迁: 用 `git mv` 保留历史, 改全部相对链接, `tests/test_links.py` 必须全过, 所有测试必须全过;
3. 新建 `library/foundations/`、`library/multimodal/`、`library/frontier/`、`base/`、`projects/tutor/`、`projects/serving/` 的 README 骨架 (只有目录和 ⬜, 不写内容);
4. 更新 `library/README.md` 总目录 (约 190 个单元)、各领域 README、ROADMAP、根 README;
5. 开 PR, 你确认后合并;
6. GitHub 上改仓库名 (旧链接会自动跳转), 更新本地 remote、Colab 链接、README 里的徽章。

每一步都可以单独回退。

---

## 10. 还没定的事

| 事项 | 什么时候定 |
|---|---|
| 单板电脑与总预算 | 硬件各站开始前 |
| 机械臂、端侧推理、四足是否升为项目 | 项目 ①②③ 各做出第一个版本之后 |
| decoding 单元要不要跑模型实测 (CPU 小实验 / GPU 评测) | 第 1 步开始时 |

### 仓库名 (已定: roots-ai)

基座模型已经叫 roots-lm, "roots (根)" 也正好是你说的 "根基"。建议沿用这个词根:

| 候选 | 优点 | 缺点 |
|---|---|---|
| **roots-ai** (中文名: 扎根 AI) | 一看就知道是 AI 全栈; 和 roots-lm、roots-car 成一个家族 | 比较朴素 |
| roots | 最短 | 太泛, 搜不到 |
| ai-from-roots | 直接说出 "从根上学 AI" | 长 |
| rootstack | 有 "全栈" 的意思, 好记 | 看不出是 AI |
| roots-lab | 像一个实验室, 能容纳以后的新项目 | 不如 roots-ai 直白 |

2026-10-28 定为 **roots-ai**, 中文名 **扎根 AI**: 基座叫 roots-lm, 项目叫 roots-car、roots-tutor, 一家人一个姓。
