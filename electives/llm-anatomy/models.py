"""
解剖台要读取的模型清单。

只放能在 Hugging Face 上找到官方仓库的模型; 受限仓库 (需要登录并同意协议) 也列出来,
抓取时会记录为 "gated", 设置环境变量 HF_TOKEN 后可以正常读取。

新增模型: 往 MODELS 里加一行即可, 然后重新运行 fetch.py 和 analyze.py。
"""

# (仓库名, 在树上主要用来说明什么)
MODELS = [
    # ── 主干 ──────────────────────────────────────────────
    ("openai-community/gpt2",                  "GPT-2: A2-minigpt 复现的对象"),
    ("meta-llama/Llama-2-7b-hf",               "LLaMA 配方的代表"),
    ("meta-llama/Llama-3.1-8B",                "LLaMA 配方 + GQA"),
    ("mistralai/Mistral-7B-v0.1",              "GQA + 滑动窗口注意力"),
    ("Qwen/Qwen2.5-7B",                        "GQA, 大 rope_theta"),
    ("Qwen/Qwen3-8B",                          "GQA + QK-Norm"),
    ("allenai/OLMo-2-1124-7B",                 "归一化位置的改造 + QK-Norm"),
    ("microsoft/phi-4",                        "GQA 的中等规模稠密模型"),
    ("google/gemma-2-9b",                      "局部/全局注意力交替"),
    ("google/gemma-3-12b-pt",                  "局部/全局注意力交替"),
    # ── MoE ──────────────────────────────────────────────
    ("mistralai/Mixtral-8x7B-v0.1",            "粗粒度 MoE: 8 选 2"),
    ("Qwen/Qwen3-30B-A3B",                     "细粒度 MoE: 128 选 8"),
    ("deepseek-ai/DeepSeek-V2-Lite",           "MLA + 细粒度 MoE + 共享专家"),
    ("deepseek-ai/DeepSeek-V3",                "MLA + MoE + 多 token 预测"),
    ("deepseek-ai/DeepSeek-V3.2-Exp",          "MLA + 稀疏注意力索引"),
    ("moonshotai/Kimi-K2-Instruct",            "沿用 DeepSeek-V3 架构, 更多专家"),
    ("zai-org/GLM-4.5",                        "MoE + QK-Norm + 多 token 预测"),
    ("openai/gpt-oss-20b",                     "MoE + 滑动/全局注意力交替"),
    ("openai/gpt-oss-120b",                    "MoE + 滑动/全局注意力交替"),
    ("meta-llama/Llama-4-Scout-17B-16E",       "MoE"),
    # ── 非 Transformer 与混合架构 ─────────────────────────
    ("state-spaces/mamba-2.8b-hf",             "纯 SSM (Mamba), 没有注意力"),
    ("tiiuae/falcon-mamba-7b",                 "纯 SSM (Mamba)"),
    ("ai21labs/AI21-Jamba-Mini-1.5",           "Mamba + 注意力混合"),
    ("nvidia/NVIDIA-Nemotron-Nano-9B-v2",      "Mamba-2 + 少量注意力层"),
    ("MiniMaxAI/MiniMax-Text-01",              "线性注意力 + softmax 注意力 7:1 混合"),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct",       "线性注意力 (Gated DeltaNet) + 注意力 3:1 混合"),
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "线性注意力 (KDA) + MLA 混合"),
]
