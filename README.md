# roots-car · 根系

[![CI](https://github.com/OGGIoao/roots-car/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/OGGIoao/roots-car/actions/workflows/ci.yml)

> **造一台听得懂话的小车, 把大模型、Agent、强化学习和硬件连成一条链一起打通。**
> 不是把几个领域分开学, 而是每学一样东西, 小车就多一项本事。全部在一台 8GB 显存的笔记本和几百元的硬件上完成; 讲义里的每个数字都实测, 不用示意值。

**终点 demo** (建设中): 你对小车说"去找那个红色杯子"。芯片上自己训练的唤醒词模型听到指令 → 笔记本上**自己微调的小模型**把这句话拆成工具调用 → 小车转身、前进、用摄像头确认、避开障碍 → 停在杯子前, 用语音回答"找到了"。

- **为什么做成一条链**: 大模型、Agent、强化学习、嵌入式的资料各有很多, 但散落各处, 很难自己串起来。这里只学通往同一个 demo 的那部分, 每一章都让小车多一项本事。
- **产品具体, 原理通用**: 车是具体的, 讲义把原理讲成通用的 (PID 换到机械臂上也成立, 工具调用的安全设计换到别的设备上也成立)。
- **学习方式**: 讲义 + 参考实现 + 三级手写练习 (填空 → 接口 → 白纸), 练习由测试判定对错。

| 文档 | 内容 |
|---|---|
| [PRODUCT.md](PRODUCT.md) | 产品规格书: 版本路线 v0.1–v1.0、验收标准、工具接口、安全规则、预算 |
| [ROADMAP.md](ROADMAP.md) | 主线章节与选修总图 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 讲义与练习的写作规范 |

## 版本路线

| 版本 | 小车能做什么 | 状态 |
|---|---|---|
| v0.1 能动 | 手机网页遥控 | ⬜ |
| v0.2 走得准 | 执行"走 1 米""转 90 度" | ⬜ |
| v0.3 不撞墙 | 自己避障, 安全层生效 | ⬜ |
| v0.4 听懂文字 | 打字指挥, 由 API 大模型调用工具 | ⬜ |
| v0.5 自己的大脑 | 换成自己微调的小模型, 本地运行 | 🟡 模型基础 (A1–A3) 已完成 |
| v0.6 看得见 | 用摄像头找到指定物体 | ⬜ |
| v0.7 自己学 | 仿真里用 RL 训练的策略上真车 | 🟡 RL 第一章 (E1) 已完成 |
| v1.0 听得懂话 | 语音唤醒 + 全链路 | ⬜ |

## 现在能跑什么

| 章节 | 状态 | 说明 |
|---|---|---|
| [A1 BPE 分词器](mainline/A1-bpe) | ✅ | 字节级 BPE, 训练 + 编解码; 讲义、图解、三级练习 |
| [A2 miniGPT](mainline/A2-minigpt) | ✅ | 单文件 GPT-2, 端到端训练与生成; 讲义含 KV cache、FlashAttention、精度、显存账本等延伸专题 |
| [A3 注意力与推理延迟](mainline/A3-attention) | ✅ | MHA / MQA / GQA / MLA, 对照实验、推理实验、三级练习 |
| [A4 现代小模型](mainline/A4-modern-small-model) | 🟡 | LLaMA 配方各组件: RoPE、RMSNorm、SwiGLU 等; 待精简 |
| [B1 电子入门与仿真器](mainline/B1-electronics) | 🟡 | 不用买硬件: Wokwi 仿真 + 逻辑分析仪实测, 点灯、按钮消抖、硬件 PWM、ADC、非阻塞循环; 真板对照待补 |
| [C2 微调工具调用模型](mainline/C2-tool-finetune) | 🟡 | 现为通用 QLoRA 演示, 待改造为小车工具调用微调 |
| [E1 老虎机](mainline/E1-bandit) | ✅ | 探索与利用; 讲义、4 个实验、10 张图、三级练习 |
| [选修](electives/README.md) | — | LLM 架构生长树、解剖台、四足机器人、后训练、规模化与上云 |

## 项目结构

```
roots-car/
├── mainline/                  # 主线: 按章节编号, 每章服务于小车的某个版本
│   ├── A1-bpe/  A2-minigpt/  A3-attention/  A4-modern-small-model/
│   ├── B1-electronics/
│   ├── C2-tool-finetune/
│   └── E1-bandit/
├── electives/                 # 选修: 更大的地图, 全部保留
├── common/                    # 多章共享: 语料定位、字符级分词、对照实验框架、SVG 作图
├── templates/llm-client/      # LLM API 统一客户端 (C1 会用到)
├── tests/                     # 回归测试
├── PRODUCT.md  ROADMAP.md  CONTRIBUTING.md
├── verify_env.py              # 环境健康检查 (先跑这个)
└── requirements.txt           # 全仓库唯一依赖清单
```

硬件部分 (物料清单、接线图、固件) 做到主线 B 部分时放进 `hardware/`。

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
python -m pytest tests -q                                   # 全部测试, CPU 上也能跑
python mainline/A1-bpe/train.py                             # 训练 BPE 分词器, 约 15 秒
python mainline/E1-bandit/experiments/greedy_fails.py       # 老虎机: 看贪心怎么失败, 约 15 秒
python mainline/A2-minigpt/train.py --config nano           # 训练 miniGPT 并生成文本, GPU 上约 40 秒
```

(耗时在测量平台上实测; miniGPT 的 checkpoint 写到当前目录的 `checkpoints/`, 不入库。)

## 测量平台

讲义中的数字都出自: Windows 11 笔记本, RTX 5060 Laptop 8GB (Blackwell sm_120), Python 3.10, torch 2.11 + CUDA 12.8。在其它机器上跑, 训练速度和显存数字会不同, 但同种子下的结论应当一致。

## 许可证

[MIT](LICENSE)
