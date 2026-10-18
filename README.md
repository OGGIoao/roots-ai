# llm-roots · 根系

> 从 Transformer 这个根出发, 亲手长出大模型的整棵树; 另一条线从经典强化学习一路长到具身智能的 VLA。两条线并行, 在后训练和 VLA 两处汇合, 最后走向 Agent。
> 每个节点都要能自己写出最小闭环, 所有数字都在本机实测。

- **LLM 线**以 [架构生长树](roadmaps/LLM_TREE.md) 为主干: 从 GPT-2 到 LLaMA 配方, 再长出注意力、位置编码、MoE、归一化、混合架构、效率六条分支, 叶子是具体的前沿模型。
- **具身线**以 [具身线总图](roadmaps/EMBODIED.md) 为主干: 经典 RL (老虎机 → PPO) → 仿真器解剖台 → 连续控制 → 模仿学习 → 扩散策略 → VLA, 全部仿真。
- **汇合点**: 后训练 (RL / DPO / GRPO) 需要两条线的成果; **Agent 板块** (规划中) 手写 Agent 循环、工具与记忆。
- 学习方式: 讲义 + 参考实现 + 三级手写练习 (填空 → 接口 → 白纸)。总体设计见 [ROADMAP](roadmaps/ROADMAP.md)。

本仓库最初从 [github-claw-for-ioao](https://github.com/OGGIoao/github-claw-for-ioao) 单开 (旧名 `github-claw-llm`)。

---

## 项目结构

```
llm-roots/
├── projects/
│   ├── llm-bpe-tokenizer/     # L1: 字节级 BPE Tokenizer 从零实现 (CS336 风格)
│   ├── llm-transformer/       # L2: Transformer 架构演进 (MHA/GQA/MLA, RoPE, SwiGLU)
│   ├── llm-minigpt/           # L2.5: miniGPT (单文件 GPT-2, 端到端可训练; 功能冻结)
│   ├── llm-qlora-finetune/    # L3: QLoRA 微调实战 (RTX 5060 8GB)
│   ├── llm-tree/              # 架构生长树上的新节点 (注意力线: MHA/GQA/MQA/MLA)
│   ├── llm-anatomy/           # 解剖台: 读真实模型的配置与张量名, 核对生长树
│   ├── embodied-rl-basics/    # 具身线 R0: 经典 RL 主干 (老虎机 → FrozenLake → CartPole → PPO)
│   └── agent-course/          # L4: Agent 课程 (M1-M7, 仅 M1 已建)
├── common/                    # 多项目共享: 语料定位/下载、字符级分词、数据集
├── tests/                     # 冒烟/回归测试 (python -m pytest tests -q)
├── templates/
│   └── llm-client/            # LLM API 统一客户端模板
├── roadmaps/
│   ├── ROADMAP.md             # 学习路线图: 两条并行线、汇合点与练习设计
│   ├── LLM_TREE.md            # LLM 架构生长树总图 (LLM 线的主干)
│   └── EMBODIED.md            # 具身线总图 (R0 RL 主干 → R5 VLA)
├── verify_env.py              # 环境健康检查 (先跑这个)
└── README.md
```

## 项目列表

| 项目 | 状态 | 说明 |
|---|---|---|
| `llm-bpe-tokenizer` | ✅ 完成 | 字节级 BPE Tokenizer,含训练+demo;TUTORIAL 17 章 |
| `llm-transformer` | 🟡 代码完整 | MHA/GQA/MLA + RoPE/ALiBi + RMSNorm/SwiGLU;可离线跑数据 |
| `llm-qlora-finetune` | 🟡 待验证 | RTX 5060 8GB 上 QLoRA 微调(需环境就绪后跑通) |
| `llm-minigpt` | ✅ 可用 (功能冻结) | 单文件 GPT-2, 与 transformer 的定位区别见其 README; 讲义含 5 个延伸专题 |
| `llm-tree` | 🟡 进行中 | 生长树的新节点; 注意力线 (MHA/MQA/GQA/MLA) ✅, 含讲义、对照实验、推理实验、三级练习 |
| `llm-anatomy` | ✅ 可用 | 读取 21 个真实模型的配置与张量名, 生成 [ANATOMY.md](projects/llm-anatomy/ANATOMY.md) |
| `embodied-rl-basics` | 🟡 进行中 | 具身线 R0; ① 老虎机 ✅ (讲义、4 个实验、10 张图、三级练习), ②~④ 未开始 |
| `agent-course` | 🔴 脚手架 | M1 已建(core.py),M2-M7 待建 |
| `llm-client` (模板) | ✅ 可用 | DeepSeek/Qwen/OpenAI/Ollama 统一客户端 |

## 快速开始

```bash
# 0) 环境自检 (先跑这个, 它会告诉你缺什么、该敲什么命令)
python verify_env.py --full

# BPE Tokenizer
cd projects/llm-bpe-tokenizer
python train.py

# Transformer (离线可跑, 直接读取 BPE 项目入库的语料)
cd projects/llm-transformer
python data.py

# LLM Client
cd templates/llm-client
cp .env.example .env   # 填入 API Key
python llm_client.py
```

## 环境

- Python 3.10+
- conda 环境: **`mlc`** (GPU) / ~~`ml`~~ (已废弃, 见下)

### ⚠️ 为什么用 `mlc` 而不是 `ml`

旧 `ml` 环境是**混用 conda 与 pip 的事故现场**: 最初用 conda 装的
`pytorch 2.2.2 + cuda11.8` 被后来 pip 装的 `torch 2.11.0+cpu` **覆盖**,
conda 随后把 CUDA 版清掉, 还留下 ABI 不匹配的 `torchaudio` (import 直接报错)。
而且 cu118 本来就跑不了 RTX 5060 (Blackwell sm_120 需要 CUDA 12.8+)。

**重建方式** (三件套必须同一条命令一起装, 否则又会 ABI 不匹配):

```bash
conda create -n mlc python=3.10 pip -y
conda activate mlc
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
pip install bitsandbytes
pip install -r requirements-mlc.txt
pip install -e . --no-deps           # 仓库自带的共享包 common/ (可编辑安装, 必须在仓库根执行)
python verify_env.py --full          # 应为全绿
```

> **铁律**: 同一个环境里**不要混用 conda 和 pip 装 torch 系包**。
> 要装什么一律先用 pip, 并且 torch / torchvision / torchaudio 三个版本必须配套。

### 网络

本机通过 Clash 代理 (`127.0.0.1:7890`) 出网。代理没启动时 pip 和
huggingface 下载都会失败 (`WinError 10061`), `verify_env.py` 会直接报出来。

