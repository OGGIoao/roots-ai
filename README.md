# llm-roots · 根系

[![CI](https://github.com/OGGIoao/llm-roots/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/OGGIoao/llm-roots/actions/workflows/ci.yml)

> 一套玩具级的完整链路: 在一台 8GB 显存的笔记本上, 亲手搭出小号的大模型、强化学习、仿真四足机器人和 Agent, 边搭边学。
> 每个环节都要能自己写出最小闭环; 讲义里的每个数字都在测量平台上实测, 不用示意值。

**终点 demo** (建设中): 自己预训练的小模型 → 自己做后训练 → 跑在自己写的推理服务上 → 被自己写的 Agent 调用 → 一句话指挥仿真里的四足机器人走过去。

- **LLM 线**以 [架构生长树](roadmaps/LLM_TREE.md) 为主干: 从 GPT-2 到 LLaMA 配方, 再长出注意力、位置编码、MoE、归一化、混合架构、效率六条分支, 叶子是具体的前沿模型。
- **具身线**以 [具身线总图](roadmaps/EMBODIED.md) 为主干: 经典 RL (老虎机 → PPO), 然后在同一台 Pupper 四足上做 本体与经典控制 → RL 步态 → 模仿学习 → 多峰行为 → 语言与视觉, 全部仿真; 与 Datawhale《动手学具身智能》的 CS123 四足课逐章对齐。
- **汇合点**: 后训练 (RL / DPO / GRPO) 需要两条线的成果; "语言与视觉"让 LLM 通过工具调用驱动四足, 同时接入 Agent 板块。
- **学习方式**: 讲义 + 参考实现 + 三级手写练习 (填空 → 接口 → 白纸), 练习由测试判定对错。总体设计见 [ROADMAP](roadmaps/ROADMAP.md), 写作规范见 [CONTRIBUTING](CONTRIBUTING.md)。

## 和同类项目的区别

| 同类 | 它的重点 | 本仓库 |
|---|---|---|
| nanoGPT / minimind | 把一个模型做到底 | 做的是一条**链路**: LLM、RL、具身、Agent 串在一起 |
| 课程类 (CS336、Datawhale) | 讲得广 | 每个节点手写最小实现, 与父节点在同一任务上**对照实测**, 配三级练习 |
| 生产框架 (LangChain、LeRobot) | 好用、性能 | 明确是**玩具**: 不追求性能, 追求每一环都能自己写出来 |

---

## 现在能跑什么

| 项目 | 状态 | 说明 |
|---|---|---|
| [`llm-bpe-tokenizer`](projects/llm-bpe-tokenizer) | ✅ 完成 | 字节级 BPE 分词器, 训练 + 编解码; 讲义、图解、三级练习 |
| [`llm-minigpt`](projects/llm-minigpt) | ✅ 完成 | 单文件 GPT-2, 端到端训练与生成; 讲义含 5 个延伸专题 (KV cache、FlashAttention、精度、显存账本等) |
| [`llm-tree`](projects/llm-tree) | 🟡 进行中 | 生长树的新节点; 注意力线 MHA / MQA / GQA / MLA ✅, 含对照实验、推理实验、三级练习 |
| [`llm-anatomy`](projects/llm-anatomy) | ✅ 可用 | 读取 20 多个真实开源模型的配置与张量名, 生成 [ANATOMY.md](projects/llm-anatomy/ANATOMY.md) 核对生长树 |
| [`llm-transformer`](projects/llm-transformer) | 🟡 代码完整 | LLaMA 配方各组件: MHA/GQA/MLA、RoPE/ALiBi、RMSNorm、SwiGLU |
| [`llm-qlora-finetune`](projects/llm-qlora-finetune) | 🟡 待验证 | 8GB 显存上的 QLoRA 微调, 尚未端到端跑通 |
| [`embodied-rl-basics`](projects/embodied-rl-basics) | 🟡 进行中 | 具身线 R0; ① 老虎机 ✅ (讲义、4 个实验、10 张图、三级练习), ②~④ 未开始 |
| [`templates/llm-client`](templates/llm-client) | ✅ 可用 | DeepSeek / Qwen / OpenAI / Ollama 统一客户端 |
| Agent 板块 | ⬜ 未开始 | 手写 Agent 循环与工具调用, 第一个真实场景是驱动四足 (见 ROADMAP) |

## 项目结构

```
llm-roots/
├── projects/
│   ├── llm-bpe-tokenizer/     # 字节级 BPE 分词器从零实现
│   ├── llm-minigpt/           # 单文件 GPT-2, 端到端可训练
│   ├── llm-transformer/       # LLaMA 配方的各个组件
│   ├── llm-tree/              # 架构生长树上的新节点 (注意力线已完成)
│   ├── llm-anatomy/           # 解剖台: 读真实模型的配置与张量名
│   ├── llm-qlora-finetune/    # QLoRA 微调
│   └── embodied-rl-basics/    # 具身线 R0: 经典 RL 主干
├── common/                    # 多项目共享: 语料定位、字符级分词、数据集、对照实验框架、SVG 作图
├── tests/                     # 冒烟/回归测试
├── templates/llm-client/      # LLM API 统一客户端模板
├── roadmaps/                  # 路线图、LLM 生长树总图、具身线总图
├── verify_env.py              # 环境健康检查 (先跑这个)
├── requirements.txt           # 全仓库唯一依赖清单
└── CONTRIBUTING.md            # 讲义与练习的写作规范
```

## 安装

需要 Python 3.10+。推荐用 conda 建一个独立环境:

```bash
conda create -n llm-roots python=3.10 pip -y
conda activate llm-roots

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
python -m pytest tests -q                                          # 全部测试, CPU 上也能跑
python projects/llm-bpe-tokenizer/train.py                         # 训练 BPE 分词器, 约 15 秒
python projects/embodied-rl-basics/bandit/experiments/greedy_fails.py  # 老虎机: 看贪心怎么失败, 约 15 秒
python projects/llm-minigpt/train.py --config nano                 # 训练 miniGPT 并生成文本, GPU 上约 40 秒
```

(耗时在测量平台上实测; miniGPT 的 checkpoint 写到当前目录的 `checkpoints/`, 不入库。)

每个项目目录下的 `README.md` 和 `TUTORIAL.md` 是入口; 讲义里的命令和输出都能照着复现。

## 测量平台

讲义中的数字都出自: Windows 11 笔记本, RTX 5060 Laptop 8GB (Blackwell sm_120), Python 3.10, torch 2.11 + CUDA 12.8。在其它机器上跑, 训练速度和显存数字会不同, 但同种子下的结论应当一致。

## 许可证

[MIT](LICENSE)
