# 路线图

> 整体设计 (三层结构、8 个领域、基座线、项目、前沿、技能地图) 见 [FRAMEWORK.md](FRAMEWORK.md); 本文件只记**现在走到哪、下一步做什么**。讲义与练习的写法见 [CONTRIBUTING.md](CONTRIBUTING.md)。
> 状态: ✅ 完成 · 🟡 进行中或待改造 · ⬜ 未开始。最后更新: 2026-10-28

## 1. 当前进度

| 层 | 进度 |
|---|---|
| 知识库 | 192 个单元中 7 个完成、6 个进行中 (逐个单元的状态见 [library/README.md](library/README.md), 根目录 README 顶部有摘要) |
| 基座线 roots-lm | ⬜ 未开始 |
| 项目 ① 小车 | v0.4 仿真版 🟡 (讲义初稿; 解码设置对照已测 (2026-10-29); 3B、关安全层、搜索提示、API 对照待补); 其余各站 ⬜ |
| 项目 ② 学习助教 | ⬜ |
| 项目 ③ 推理服务 | ⬜ |

## 2. 推进顺序

原则: **项目和基座线走到哪, 先补那里要用的知识库单元**; 不按领域一口气写完。分五步 (细节见 [FRAMEWORK.md](FRAMEWORK.md) 第 8 节):

| 步 | 内容 | 状态 |
|---|---|---|
| 0 | 收尾 (对话模板、解码核心) + 改结构 (三层目录) + 仓库改名为 roots-ai | ✅ 2026-10-28 |
| 1 | 小车 v0.4 → v0.5 要用的单元: decoding 收尾 (✅ 2026-10-29)、structured-output (✅ 2026-10-29)、local-inference、quantization、agent-eval; 基础领域核心: dev-environment、linear-algebra、calculus、probability、autograd、pytorch-basics | ⬜ |
| 2 | 基座线阶段 1 (笔记本全流程) + 小车 v0.7 前半: 预训练段、后训练 P1/P3、强化学习 R0–R3 | ⬜ |
| 3 | 学习助教 (Agent A3、A7、A8)、推理服务 (自写推理引擎系列)、小车世界模型 (具身 E4) | ⬜ |
| 4 | 基座线阶段 2–4 (上云、前沿部件、多模态); 各 🔥 单元; 硬件随小车各站推进 | ⬜ |

小车的推进顺序 (2026-10-25 定): **软件先行**。先在仿真里把 "大模型 → Agent → 强化学习" 打通 (v0.4 → v0.5 → v0.7 仿真部分), 再造真车 (v0.1–v0.3), 把仿真里做好的 Agent 和策略搬上去 (工具接口不变, 见 [PRODUCT.md](projects/car/PRODUCT.md) 第 4 节), 然后 v0.6 → v1.0。开发板已选定、暂不购买, 见 [projects/car/hardware](projects/car/hardware/README.md)。

## 3. 运行环境与上云原则

| 用途 | 环境 |
|---|---|
| 测量平台 (讲义里的数字都出自这里) | Windows 11 笔记本, RTX 5060 Laptop 8GB, Python 3.10 |
| 软件部分 (LLM、Agent、RL、后训练) | 主环境, 安装见根目录 README |
| 往主环境装 gymnasium、mujoco 等新包 | 装之前先 `pip install --dry-run`, 确认不会替换已装好的 torch |
| 会锁定 torch 版本的工具链 | 单独的 conda 环境, 不装进主环境 |
| 硬件部分 | ESP32-S3 开发板; 每一课先在浏览器仿真器里跑通 |
| 笔记本放不下的模型、Windows 跑不通的工具链 | 云服务器 (计划见 [base/cloud-plan.md](base/cloud-plan.md)) |
| 本地模型评测 | **同一时间只跑一个** (8 GB 显存、16 GB 内存, 两个评测并行会占满, 2026-10-26 实测) |

**上云原则**: 只在笔记本**实测**撞墙 (显存溢出、要跑好几天) 之后才上云, 并把这个数字写进讲义。上云的部分也按对照实验来写: 笔记本 vs 云 GPU、单卡 vs 多卡, 让读者看到 "为什么需要多卡", 而不只是 "换了台机器"。

各工具在 Windows 上能否运行, 动手前逐个实测, 不做假设。

## 4. 降低门槛

目标: 读者从点开仓库到第一次跑出结果, 不需要装环境、不需要买硬件、不需要显卡。

| 事项 | 内容 | 状态 | 依赖 |
|---|---|---|---|
| Colab 笔记本 | 知识库 bpe、minigpt、attention 和小车 v0.4 各一本 `colab.ipynb`, README 顶部有 "Open in Colab" 按钮; 由 [tools/make_notebooks.py](tools/make_notebooks.py) 生成 | 🟡 bpe / minigpt / attention 已在全新克隆的仓库里模拟 Colab 逐格运行通过; 在 Colab 上实际运行 (含 v0.4) 待首次确认 | — |
| 不用模型的最小演示 | 知识库 agent-loop 的 `demo_minimal.py`、chat-template 的 `chat_template.py`: 只用 Python 标准库 | ✅ | — |
| Wokwi 一键链接 | electronics 的五个案例存成 Wokwi 公开项目, 讲义第 0 节放链接, 点开就能运行 | ⬜ | 用维护者的 Wokwi 账号保存为公开项目 |
| 物料清单 | 小车的完整物料清单, 带购买链接与价格, 放进 `projects/car/hardware/` | ⬜ | 硬件各站开始前定下开发板型号 |

之后每完成一站, 同时补上这一站的 Colab / Wokwi 入口; 需要显卡的单元 (如 tool-finetune 微调) 先实测 Colab 免费 GPU 能否跑完再承诺。
