# 路线图

> 产品目标与各版本的验收标准见 [PRODUCT.md](PRODUCT.md), 讲义与练习的写法见 [CONTRIBUTING.md](CONTRIBUTING.md)。
> 状态: ✅ 完成 · 🟡 进行中或待改造 · ⬜ 未开始。最后更新: 2026-10-27

## 1. 两层结构

| 层 | 目录 | 放什么 |
|---|---|---|
| **主线** | [mainline/](mainline/README.md) | 造小车, 一站一个版本 (v0.1 → v1.0)。每站只讲 "这一步在车上做什么、怎么跑、实测结果如何", 用到的原理链接到知识库 |
| **知识库** | [library/](library/README.md) | 主线用到的每个知识点, 一个单元一份完整讲义 (颗粒度参照 [分词 BPE](library/llm/bpe/TUTORIAL.md)), 分六个领域: 硬件、大语言模型、Agent、强化学习、后训练、具身智能 |
| **选修** | [electives/](electives/README.md) | 不在造车路上、但值得深入的内容: 四足机器人、规模化上云、LLM 架构全景、Agent 框架等 |

**选内容的规则**: 每个知识点先问 "造车的哪一站需要它?" 需要的进知识库, 由主线链接过去; 不需要的进选修。两者都不删, 格式也一样 (讲义 + 参考实现 + 三级练习, 数字全部实测)。

(2026-10-27 之前, 知识单元放在 `mainline/` 下按章节号命名, 如 `A1-bpe`。旧路径对照见 [mainline/README.md](mainline/README.md#旧路径对照)。)

## 2. 主线各站与知识库单元

每一站要用到的知识库单元 (完整目录和每个单元的状态见 [library/README.md](library/README.md)):

| 站 | 小车多了什么本事 | 知识库单元 | 本站状态 |
|---|---|---|---|
| v0.1 能遥控 | 手机网页遥控 | 硬件: electronics 🟡 · motor-driver ⬜ · wifi-remote ⬜ | ⬜ |
| v0.2 走得准 | 走 1 米、转 90 度 | 硬件: encoder ⬜ · pid ⬜ | ⬜ |
| v0.3 不撞墙 | 避障、断网停车 | 硬件: sensors ⬜ · safety-firmware ⬜ | ⬜ |
| v0.4 听懂文字 | 打字指挥, 大模型调用工具 | Agent: agent-loop 🟡; LLM: chat-template ✅ · decoding 🟡 | 仿真版 🟡 (讲义初稿; 3B、关安全层、搜索提示、API 对照待补) · 真车版 ⬜ |
| v0.5 自己的大脑 | 自己微调的小模型, 本地运行 | LLM: bpe ✅ · minigpt ✅ · attention ✅ · modern-small-model 🟡 · chat-template ✅ · local-inference ⬜; 后训练: tool-finetune 🟡; Agent: agent-eval ⬜ | ⬜ |
| v0.6 看得见 | 摄像头找红杯子 | 硬件: camera ⬜; LLM: quantization ⬜ | ⬜ |
| v0.7 自己学 | RL 策略上真车 | RL: bandit ✅ · q-learning ⬜ · policy-gradient ⬜ · ppo ⬜; 具身: sim2real ⬜ | ⬜ |
| v1.0 听得懂话 | 语音指挥 | 待规划 (唤醒词、量化、全链路评测) | ⬜ |

**推进顺序: 软件先行** (2026-10-25 定): 先在仿真里把 "大模型 → Agent → 强化学习" 打通, 再上硬件。

1. v0.4 仿真版: Agent 循环驱动仿真小车 (🟡 进行中);
2. v0.5 仿真版: 补 modern-small-model、tool-finetune、local-inference、agent-eval, 换成自己微调的小模型;
3. v0.7 仿真部分: 补 q-learning → policy-gradient → ppo, 在仿真小车上训练控制策略, 替换 v0.4 里 go_to 的规则控制器;
4. 硬件: v0.1–v0.3 造出真车, 把仿真里做好的 Agent 和策略搬上去 (工具接口不变, 见 PRODUCT.md 第 4 节);
5. v0.6 → v1.0。

**写作顺序**: 主线走到哪一站, 先补那一站要用的知识库单元, 不按领域一口气写完。开发板已选定、暂不购买, 见 [hardware/README.md](hardware/README.md)。

### 知识库单元的三件套

| 组成 | 作用 |
|---|---|
| 讲义 `TUTORIAL.md` | 讲透原理与代码。样章: [minigpt](library/llm/minigpt/TUTORIAL.md)、[bpe](library/llm/bpe/TUTORIAL.md) |
| 参考实现 | 仓库里的代码, 也是练习的标准答案 |
| 手写练习 | 三级递进: 填空 → 接口 → 白纸, 由测试判定对错 |

## 3. 选修

全部在 [electives/](electives/README.md):

| 选修 | 内容 | 和主线的关系 |
|---|---|---|
| [LLM 架构生长树](electives/llm-tree/README.md) | 从 Transformer 长出的整棵架构树: 注意力、位置编码、MoE、归一化、替代架构 (Mamba 等)、效率 | 知识库 LLM 领域是它的主干; 想把 LLM 架构系统串起来时读它 |
| [解剖台](electives/llm-anatomy/README.md) | 读取 20 多个真实开源模型的配置与张量名 | modern-small-model 用它读懂要微调的小模型 |
| [四足机器人 Pupper](electives/quadruped/README.md) | R1–R5: 本体与经典控制 → RL 步态 → 模仿学习 → 多峰行为 → 语言与视觉 | 工具接口与小车相同, 主线的 Agent 可以直接驱动它 |
| [Agent 框架](electives/agent-frameworks/README.md) | LangChain/LangGraph、Microsoft Agent Framework、OpenAI Agents SDK、smolagents、ADK、CrewAI 与 MCP | 和知识库 agent-loop 的手写循环逐个部件对照 |
| [规模化与上云](electives/scale-up/README.md) | 把模型做大、多卡训练、推理服务压测 | 笔记本实测撞墙后再做 |

后训练原先在选修里, 2026-10-27 移入知识库 ([library/post-training](library/post-training/README.md))。

## 4. 运行环境与上云原则

| 用途 | 环境 |
|---|---|
| 测量平台 (讲义里的数字都出自这里) | Windows 11 笔记本, RTX 5060 Laptop 8GB, Python 3.10 |
| 软件部分 (LLM、Agent、RL、后训练) | 主环境, 安装见根目录 README |
| 往主环境装 gymnasium、mujoco 等新包 | 装之前先 `pip install --dry-run`, 确认不会替换已装好的 torch |
| 会锁定 torch 版本的工具链 | 单独的 conda 环境, 不装进主环境 |
| 硬件部分 | ESP32-S3 开发板; 每一课先在浏览器仿真器里跑通 |
| 笔记本放不下的模型、Windows 跑不通的工具链 | 云服务器 |
| 本地模型评测 | **同一时间只跑一个** (8 GB 显存、16 GB 内存, 两个评测并行会占满, 2026-10-26 实测) |

**上云原则**: 只在笔记本**实测**撞墙 (显存溢出、要跑好几天) 之后才上云, 并把这个数字写进讲义。上云的部分也按对照实验来写: 笔记本 vs 云 GPU、单卡 vs 多卡, 让读者看到 "为什么需要多卡", 而不只是 "换了台机器"。

各工具在 Windows 上能否运行, 动手前逐个实测, 不做假设。

## 5. 降低门槛

目标: 读者从点开仓库到第一次跑出结果, 不需要装环境、不需要买硬件、不需要显卡。

| 事项 | 内容 | 状态 | 依赖 |
|---|---|---|---|
| Colab 笔记本 | 知识库 bpe、minigpt、attention 和主线 v0.4 各一本 `colab.ipynb`, README 顶部有 "Open in Colab" 按钮; 由 [tools/make_notebooks.py](tools/make_notebooks.py) 生成 | 🟡 bpe / minigpt / attention 已在全新克隆的仓库里模拟 Colab 逐格运行通过; 在 Colab 上实际运行 (含 v0.4) 待首次确认 | — |
| 不用模型的最小演示 | 知识库 agent-loop 的 `demo_minimal.py`: 只用 Python 标准库, 一秒内看到完整的 Agent 循环 | ✅ | — |
| Wokwi 一键链接 | electronics 的五个案例存成 Wokwi 公开项目, 讲义第 0 节放链接, 点开就能运行, 不用复制粘贴 | ⬜ | 用维护者的 Wokwi 账号保存为公开项目 |
| 物料清单 | 小车的完整物料清单, 带购买链接与价格, 放进 `hardware/` | ⬜ | 硬件各站开始前定下开发板型号 (先核对 electronics 讲义第 8 节的引脚账) |

之后每完成一站, 同时补上这一站的 Colab / Wokwi 入口; 需要显卡的单元 (如 tool-finetune 微调) 先实测 Colab 免费 GPU 能否跑完再承诺。
