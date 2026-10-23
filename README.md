# roots-car · 根系

[![CI](https://github.com/OGGIoao/roots-car/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/OGGIoao/roots-car/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **造一台听得懂话的小车, 把大模型、Agent、强化学习和硬件连成一条链, 一起打通。**
>
> *Build a small car that understands spoken commands — learning LLMs, agents, reinforcement learning and embedded hardware as one connected chain, on a single 8 GB laptop and a few hundred RMB of parts. (Docs are in Chinese.)*

**终点 demo** (建设中): 你对小车说"去找那个红色杯子"。芯片上自己训练的唤醒词模型听到指令 → 笔记本上**自己微调的小模型**把这句话拆成工具调用 → 小车转身、前进、用摄像头确认、避开障碍 → 停在杯子前, 用语音回答"找到了"。

---

## 为什么做这个

大模型、Agent、强化学习、嵌入式, 每个方向的资料都很多, 但散落各处, 很难自己串起来; 硬件这一侧门槛又高, 很多人停在"跑通别人的代码"。

roots-car 的做法: **只学通往同一个 demo 的那部分, 每学一章, 小车就多一项本事**。

| 原则 | 具体是什么 |
|---|---|
| 目标导向 | 每一章都回答"小车的哪个版本需要它"; 答不上来的内容放进[选修](electives/README.md), 不删除 |
| 亲手写最小实现 | 不调包了事: BPE、GPT、注意力变体、消抖、PID……都从零写一遍, 再和成熟方案对照 |
| 数字全部实测 | 讲义里没有示意值; 每个数字都能用仓库里的脚本重新算出来, 回归测试会检查 |
| 先仿真, 再上真板 | 硬件章节先在浏览器仿真器里跑通, 不买硬件也能学; 仿真和真板哪里不同, 讲义里专门列出 |
| 三级练习 | 填空 → 只给接口 → 白纸, 由测试判对错, 卡住了按"方向 → 哪一行 → 代码"分级提示 |
| 产品具体, 原理通用 | 车是具体的, 原理讲成通用的: PID 换到机械臂上也成立, 工具调用的安全设计换到别的设备上也成立 |

和同类项目的区别: [nanoGPT](https://github.com/karpathy/nanoGPT) 把一个模型做到底, 各类课程讲得广, 生产框架追求好用; 本仓库做的是**一条链**, 而且明确是玩具规模, 追求每一环都能自己写出来。

---

## 现在做到哪了

> 项目处于**早期阶段**: 大模型基础和电子入门已经可以学, 小车本身还没有造出来。

版本路线 (完整的验收标准见 [PRODUCT.md](PRODUCT.md)):

| 版本 | 小车能做什么 | 状态 |
|---|---|---|
| v0.1 能动 | 手机网页遥控 | 🟡 电子入门 (B1) 已完成, 电机与遥控 (B2) 未开始 |
| v0.2 走得准 | 执行"走 1 米""转 90 度" | ⬜ |
| v0.3 不撞墙 | 自己避障, 安全层生效 | ⬜ |
| v0.4 听懂文字 | 打字指挥, 由 API 大模型调用工具 | ⬜ |
| v0.5 自己的大脑 | 换成自己微调的小模型, 本地运行 | 🟡 模型基础 (A1–A3) 已完成 |
| v0.6 看得见 | 用摄像头找到指定物体 | ⬜ |
| v0.7 自己学 | 仿真里用 RL 训练的策略上真车 | 🟡 强化学习第一章 (E1) 已完成 |
| v1.0 听得懂话 | 语音唤醒 + 全链路 | ⬜ |

已经可以学的章节:

| 章节 | 状态 | 内容 |
|---|---|---|
| [A1 BPE 分词器](mainline/A1-bpe) | ✅ | 字节级 BPE: 训练、编码、解码, 一字不差还原; 讲义、图解、三级练习 |
| [A2 miniGPT](mainline/A2-minigpt) | ✅ | 单文件 GPT-2, 端到端训练与生成; 延伸专题: KV cache、FlashAttention、混合精度、显存账本 |
| [A3 注意力与推理延迟](mainline/A3-attention) | ✅ | MHA / MQA / GQA / MLA 手写实现, 同任务对照实验与推理实验 |
| [B1 电子入门与仿真器](mainline/B1-electronics) | ✅ 仿真部分 | **不用买硬件**: 浏览器仿真 + 逻辑分析仪实测; 点灯、按钮消抖、硬件 PWM、ADC、非阻塞循环; 真板对照待补 |
| [E1 老虎机](mainline/E1-bandit) | ✅ | 强化学习的第一个案例: 探索与利用, ε-greedy、乐观初值、UCB |
| [A4 现代小模型](mainline/A4-modern-small-model) | 🟡 | RoPE、RMSNorm、SwiGLU 等组件; 待精简 |
| [C2 微调工具调用模型](mainline/C2-tool-finetune) | 🟡 | 现为通用 QLoRA 演示, 待改造为"只输出小车工具调用"的小模型 |
| [选修](electives/README.md) | — | LLM 架构生长树、解剖台 (读 20 多个真实模型的配置)、四足机器人、后训练、规模化与上云 |

完整的章节规划见 [ROADMAP.md](ROADMAP.md)。

---

## 从哪里开始

| 你想先学 | 从这里开始 | 需要什么 |
|---|---|---|
| 大模型是怎么回事 | [A1 BPE 分词器](mainline/A1-bpe/TUTORIAL.md) → [A2 miniGPT](mainline/A2-minigpt/TUTORIAL.md) | Python; A1 只用标准库; A2 的 nano 配置 CPU 上也能跑 (实测约 2 分钟, GPU 约 40 秒) |
| 硬件, 但从没碰过 | [B1 电子入门](mainline/B1-electronics/TUTORIAL.md) | 只要浏览器; 不用买任何东西 |
| 强化学习 | [E1 老虎机](mainline/E1-bandit/TUTORIAL.md) | Python + numpy |

每章的入口都是目录里的 `TUTORIAL.md` (讲义) 和 `README.md` (怎么跑); 练习在每章的 `exercises/` 下。

---

## 安装

需要 Python 3.10+。推荐用 conda 建一个独立环境:

```bash
conda create -n roots-car python=3.10 pip -y
conda activate roots-car

# torch 三件套必须用同一条命令一起装。cu128 适用于 RTX 50 系;
# 老显卡换成对应的 CUDA 索引, 没有 GPU 就去掉 --index-url 装 CPU 版
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install -e . --no-deps    # 仓库自带的共享包 common/ (可编辑安装, 在仓库根目录执行)
python verify_env.py --full   # 环境自检, 会告诉你缺什么、该敲什么命令
```

> **同一个环境里不要混用 conda 和 pip 装 torch 系包**: conda 装的 CUDA 版很容易被 pip 装的 CPU 版覆盖, 最后留下一个 ABI 不匹配、import 就报错的 torchaudio。
>
> 国内网络下载 PyPI / Hugging Face 失败时, `verify_env.py` 会检测出网情况并给出建议。

## 快速开始

以下命令都从仓库根目录执行:

```bash
python -m pytest tests -q                                   # 全部回归测试, CPU 上也能跑
python mainline/A1-bpe/train.py                             # 训练 BPE 分词器, 约 15 秒
python mainline/E1-bandit/experiments/greedy_fails.py       # 老虎机: 看贪心怎么失败, 约 15 秒
python mainline/A2-minigpt/train.py --config nano           # 训练 miniGPT 并生成文本, GPU 上约 40 秒
```

(耗时在下面的测量平台上实测; miniGPT 的 checkpoint 写到当前目录的 `checkpoints/`, 不入库。)

硬件章节在浏览器里跑, 步骤见 [B1 讲义第 0 节](mainline/B1-electronics/TUTORIAL.md#0-怎么跑起来)。

## 测量平台

| 用途 | 平台 |
|---|---|
| 软件章节的数字 | Windows 11 笔记本, RTX 5060 Laptop 8GB (Blackwell sm_120), Python 3.10, torch 2.11 + CUDA 12.8 |
| 硬件章节的数字 | [Wokwi](https://wokwi.com) 网页版仿真, ESP32-S3-DevKitC-1, MicroPython v1.28.0 (真板对照会单独标出) |
| 持续集成 | GitHub Actions, Ubuntu + CPU 版 torch, 跑全部回归测试 |

在其它机器上跑, 训练速度和显存数字会不同, 但同种子下的结论应当一致。

---

## 项目结构

```
roots-car/
├── mainline/                  # 主线: 按章节编号, 每章服务于小车的某个版本
│   ├── A1-bpe/  A2-minigpt/  A3-attention/  A4-modern-small-model/     # A 看懂模型
│   ├── B1-electronics/                                                 # B 让车听话
│   ├── C2-tool-finetune/                                               # C 让模型成为小车专家
│   └── E1-bandit/                                                      # E 让车自己学
├── electives/                 # 选修: 更大的地图, 全部保留, 当作详细的阅读文档
├── common/                    # 多章共享: 语料定位、字符级分词、对照实验框架、SVG 作图
├── templates/llm-client/      # LLM API 统一客户端
├── tests/                     # 回归测试 (CI 每次提交都跑)
├── PRODUCT.md                 # 产品规格书: 版本、验收标准、工具接口、安全规则、预算
├── ROADMAP.md                 # 主线章节与选修总图
├── CONTRIBUTING.md            # 讲义与练习的写作规范
├── NOTICE.md                  # 第三方内容与致谢
├── verify_env.py              # 环境健康检查
└── requirements.txt           # 全仓库唯一依赖清单
```

每一章的结构相同: `TUTORIAL.md` (讲义)、参考实现、`experiments/` (实验脚本, 讲义里的数字都出自这里)、`results/` (实测数据)、`figures/` (图和生成脚本)、`exercises/` (三级练习和测试)。

---

## 常见问题

**要买硬件吗?** 现在不用。B1 全部在浏览器里完成。造小车 (B2 起) 需要一套硬件, 预算目标是小车加工具 600 元以内, 清单见 [PRODUCT.md](PRODUCT.md#6-预算-粗估); 具体型号会在 B2 开始前定下。

**要 GPU 吗?** 不是必须的。回归测试、A1、B1、E1 只用 CPU; A2 的 nano 配置在 CPU 上实测 105 秒训完 (GPU 约 40 秒), A3 的对照实验在 CPU 上也能跑, 只是慢。微调 (C2) 需要一张 6~8 GB 显存的 NVIDIA 显卡。

**支持什么系统?** 讲义是在 Windows 上实测的; 回归测试在 Linux (GitHub Actions) 上每次提交都跑。macOS 没有测过。

**为什么芯片上用 MicroPython 而不是 C++?** 先把注意力放在电路上, 控制逻辑也能在电脑上用 pytest 测。到摄像头和芯片上的唤醒词 (D、F 部分) 性能不够时再换 C++, 换的时候会实测两者的差距。

**讲义是什么语言?** 中文。代码注释和提交信息也以中文为主。

---

## 参与

项目还在早期, 欢迎:

- **提 issue**: 讲义里看不懂的地方、跑不通的命令、数字对不上、来源或许可写错了;
- **在真板上复现**: B1 讲义里标了 ⏳ 的"真板对照", 如果你手上有 ESP32-S3 开发板, 欢迎把实测结果提交上来;
- **贡献章节或练习**: 先读 [CONTRIBUTING.md](CONTRIBUTING.md) (讲义规范、三级练习的标准、"数字全部实测"的要求), 开 issue 讨论之后再动手。

提交前请跑 `python -m pytest tests -q`。

## 致谢与许可证

本仓库参考或使用了一些外部项目和数据, 见 [NOTICE.md](NOTICE.md)。

代码、讲义和图以 [MIT 许可](LICENSE) 发布。
