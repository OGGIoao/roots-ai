# github-claw-llm

> LLM 学习与实践独立仓库 — BPE Tokenizer + LLM Client 模板 + 后续 LLM 训练项目

从 [github-claw-for-ioao](https://github.com/OGGIoao/github-claw-for-ioao) 单开,专注大模型方向。

---

## 项目结构

```
github-claw-llm/
├── projects/
│   ├── llm-bpe-tokenizer/     # L1: 字节级 BPE Tokenizer 从零实现 (CS336 风格)
│   ├── llm-transformer/       # L2: Transformer 架构演进 (MHA/GQA/MLA, RoPE, SwiGLU)
│   ├── llm-minigpt/           # L2.5: miniGPT (暂停)
│   ├── llm-qlora-finetune/    # L3: QLoRA 微调实战 (RTX 5060 8GB)
│   └── agent-course/          # L4: Agent 课程 (M1-M7, 仅 M1 已建)
├── templates/
│   └── llm-client/            # LLM API 统一客户端模板
├── roadmaps/                  # 学习路线图 (LLM_ROADMAP.md 待补)
├── verify_env.py              # 环境健康检查 (先跑这个)
└── README.md
```

## 项目列表

| 项目 | 状态 | 说明 |
|---|---|---|
| `llm-bpe-tokenizer` | ✅ 完成 | 字节级 BPE Tokenizer,含训练+demo;TUTORIAL 17 章 |
| `llm-transformer` | 🟡 代码完整 | MHA/GQA/MLA + RoPE/ALiBi + RMSNorm/SwiGLU;可离线跑数据 |
| `llm-qlora-finetune` | 🟡 待验证 | RTX 5060 8GB 上 QLoRA 微调(需环境就绪后跑通) |
| `llm-minigpt` | ⏸️ 暂停 | 代码在,Phase 1.2 暂停 |
| `agent-course` | 🔴 脚手架 | M1 已建(core.py),M2-M7 待建 |
| `llm-client` (模板) | ✅ 可用 | DeepSeek/Qwen/OpenAI/Ollama 统一客户端 |

## 快速开始

```bash
# 0) 环境自检 (先跑这个, 它会告诉你缺什么、该敲什么命令)
python verify_env.py --full

# BPE Tokenizer
cd projects/llm-bpe-tokenizer
python train.py

# Transformer (离线可跑, 会自动复用 BPE 项目的语料)
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
pip install -r projects/llm-qlora-finetune/requirements.txt
python verify_env.py --full          # 应为全绿
```

> **铁律**: 同一个环境里**不要混用 conda 和 pip 装 torch 系包**。
> 要装什么一律先用 pip, 并且 torch / torchvision / torchaudio 三个版本必须配套。

### 网络

本机通过 Clash 代理 (`127.0.0.1:7890`) 出网。代理没启动时 pip 和
huggingface 下载都会失败 (`WinError 10061`), `verify_env.py` 会直接报出来。

