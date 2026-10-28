# 模型解剖表

> 由 [analyze.py](analyze.py) 根据 [snapshots/](snapshots/) 自动生成, 请勿手改。
> 数据来源: 各模型官方仓库的 `config.json` 与权重张量名 (未下载权重)。
> 快照抓取日期: 2026-10-15。按仓库创建时间排序 (近似发布时间)。

## 1. 总表

| 模型 | 创建 | 层数 | 维度 | 层的构成 | 注意力 | KV Cache / token (bf16) | FFN | 位置编码 | 归一化与其它 | 配置与权重一致 |
|---|---|---|---|---|---|---|---|---|---|---|
| `openai-community/gpt2` | 2022-03-02 | 12 | 768 | 注意力 12 | MHA (12 头) | 36.0 KB | 稠密 gelu_new ×3072 | 可学习绝对位置 (1024) | LayerNorm, QKV 带 bias, 输入输出权重绑定 | ✅ |
| `mistralai/Mistral-7B-v0.1` | 2023-09-20 | 32 | 4096 | 滑窗注意力 32 | GQA (32 头 / 8 组 KV), 滑窗 4096 | 128.0 KB (滑窗层最多缓存 4096 个位置) | 稠密 SwiGLU ×14336 | RoPE θ=10000 | RMSNorm | ✅ |
| `mistralai/Mixtral-8x7B-v0.1` | 2023-12-01 | 32 | 4096 | 注意力 32 | GQA (32 头 / 8 组 KV) | 128.0 KB | MoE 8 选 2, 专家宽度 14336 | RoPE θ=1e+06 | RMSNorm | ✅ |
| `state-spaces/mamba-2.8b-hf` | 2024-03-05 | 64 | 2560 | SSM 64 | 无 (纯 SSM) | 0 (固定大小状态) | 无独立 FFN (扩展维度在 Mamba 块内部) | 无显式位置编码 | RMSNorm, 输入输出权重绑定 | ✅ |
| `deepseek-ai/DeepSeek-V2-Lite` | 2024-05-15 | 27 | 2048 | 注意力 27 | MLA (潜向量 512 + RoPE 键 64) | 30.4 KB | MoE 64 选 6 + 2 共享, 前 1 层稠密, 专家宽度 1408 | RoPE θ=10000, 扩展 yarn×40 | RMSNorm | ✅ |
| `tiiuae/falcon-mamba-7b` | 2024-07-17 | 64 | 4096 | SSM 64 | 无 (纯 SSM) | 0 (固定大小状态) | 无独立 FFN (扩展维度在 Mamba 块内部) | 无显式位置编码 | LayerNorm | ✅ |
| `Qwen/Qwen2.5-7B` | 2024-09-15 | 28 | 3584 | 注意力 28 | GQA (28 头 / 4 组 KV) | 56.0 KB | 稠密 SwiGLU ×18944 | RoPE θ=1e+06 | RMSNorm, QKV 带 bias | ✅ |
| `allenai/OLMo-2-1124-7B` | 2024-10-29 | 32 | 4096 | 注意力 32 | MHA (32 头) | 512.0 KB | 稠密 SwiGLU ×11008 | RoPE θ=500000 | RMSNorm, QK-Norm, 后置归一化 (子层之后) | ✅ |
| `microsoft/phi-4` | 2024-12-11 | 40 | 5120 | 注意力 40 | GQA (40 头 / 10 组 KV) | 200.0 KB | 稠密 SwiGLU ×17920 | RoPE θ=250000 | RMSNorm | ✅ |
| `deepseek-ai/DeepSeek-V3` | 2024-12-25 | 61 | 7168 | 注意力 61 | MLA (潜向量 512 + RoPE 键 64) | 68.6 KB | MoE 256 选 8 + 1 共享, 前 3 层稠密, 专家宽度 2048 | RoPE θ=10000, 扩展 yarn×40 | RMSNorm, 多 token 预测 (MTP), 无辅助损失负载均衡, FP8 权重 | ✅ |
| `MiniMaxAI/MiniMax-Text-01` | 2025-01-12 | 80 | 6144 | 线性注意力 70 + 注意力 10 | GQA (64 头 / 8 组 KV) | 40.0 KB | MoE 32 选 2, 专家宽度 9216 | RoPE θ=1e+07, 只旋转 50% 维 | RMSNorm | ✅ |
| `Qwen/Qwen3-30B-A3B` | 2025-04-27 | 48 | 2048 | 注意力 48 | GQA (32 头 / 4 组 KV) | 96.0 KB | MoE 128 选 8, 专家宽度 768 | RoPE θ=1e+06 | RMSNorm, QK-Norm | ✅ |
| `Qwen/Qwen3-8B` | 2025-04-27 | 36 | 4096 | 注意力 36 | GQA (32 头 / 8 组 KV) | 144.0 KB | 稠密 SwiGLU ×12288 | RoPE θ=1e+06 | RMSNorm, QK-Norm | ✅ |
| `moonshotai/Kimi-K2-Instruct` | 2025-07-11 | 61 | 7168 | 注意力 61 | MLA (潜向量 512 + RoPE 键 64) | 68.6 KB | MoE 384 选 8 + 1 共享, 前 1 层稠密, 专家宽度 2048 | RoPE θ=50000, 扩展 yarn×32.0 | RMSNorm, 无辅助损失负载均衡, FP8 权重 | ✅ |
| `zai-org/GLM-4.5` | 2025-07-20 | 92 | 5120 | 注意力 92 | GQA (96 头 / 8 组 KV) | 368.0 KB | MoE 160 选 8 + 1 共享, 前 3 层稠密, 专家宽度 1536 | RoPE θ=1e+06, 只旋转 50% 维 | RMSNorm, QK-Norm, QKV 带 bias, 多 token 预测 (MTP), 无辅助损失负载均衡 | ✅ |
| `openai/gpt-oss-120b` | 2025-08-04 | 36 | 2880 | 滑窗注意力 18 + 注意力 18 | GQA (64 头 / 8 组 KV), 滑窗 128, attention sink | 72.0 KB (滑窗层最多缓存 128 个位置) | MoE 128 选 4, 专家宽度 2880 | RoPE θ=150000, 扩展 yarn×32.0 | RMSNorm, QKV 带 bias, MXFP4 专家权重 | ✅ |
| `openai/gpt-oss-20b` | 2025-08-04 | 24 | 2880 | 滑窗注意力 12 + 注意力 12 | GQA (64 头 / 8 组 KV), 滑窗 128, attention sink | 48.0 KB (滑窗层最多缓存 128 个位置) | MoE 32 选 4, 专家宽度 2880 | RoPE θ=150000, 扩展 yarn×32.0 | RMSNorm, QKV 带 bias, MXFP4 专家权重 | ✅ |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 2025-08-12 | 56 | 4480 | SSM 27 + 注意力 4 + 纯 MLP 25 | GQA (40 头 / 8 组 KV) | 16.0 KB | 稠密 relu2 ×15680 | 无显式位置编码 | LayerNorm | ✅ |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 2025-09-09 | 48 | 2048 | 线性注意力 36 + 注意力 12 | GQA (16 头 / 2 组 KV) | 24.0 KB | MoE 512 选 10 + 1 共享, 专家宽度 512 | RoPE θ=1e+07, 只旋转 25% 维 | RMSNorm, QK-Norm, 多 token 预测 (MTP) | ✅ |
| `deepseek-ai/DeepSeek-V3.2-Exp` | 2025-09-29 | 61 | 7168 | 注意力 61 | MLA (潜向量 512 + RoPE 键 64) + 稀疏索引 top-2048 | 68.6 KB | MoE 256 选 8 + 1 共享, 前 3 层稠密, 专家宽度 2048 | RoPE θ=10000, 扩展 yarn×40 | RMSNorm, 多 token 预测 (MTP), 无辅助损失负载均衡, FP8 权重 | ✅ |
| `moonshotai/Kimi-Linear-48B-A3B-Instruct` | 2025-10-30 | 27 | 2304 | 线性注意力 20 + 注意力 7 | MLA (潜向量 512, 不用 RoPE) | 7.0 KB | MoE 256 选 8 + 1 共享, 前 1 层稠密, 专家宽度 1024 | MLA 层不用 RoPE (mla_use_nope) | RMSNorm, 无辅助损失负载均衡 | ✅ |

## 2. 未能读取的模型

- `meta-llama/Llama-2-7b-hf` (LLaMA 配方的代表): 受限仓库, 需要 HF_TOKEN
- `google/gemma-2-9b` (局部/全局注意力交替): 受限仓库, 需要 HF_TOKEN
- `meta-llama/Llama-3.1-8B` (LLaMA 配方 + GQA): 受限仓库, 需要 HF_TOKEN
- `ai21labs/AI21-Jamba-Mini-1.5` (Mamba + 注意力混合): 受限仓库, 需要 HF_TOKEN
- `google/gemma-3-12b-pt` (局部/全局注意力交替): 受限仓库, 需要 HF_TOKEN
- `meta-llama/Llama-4-Scout-17B-16E` (MoE): 受限仓库, 需要 HF_TOKEN

## 3. 各列的含义与口径

- **层的构成**: 混合架构里各类层的数量, 来自配置文件里描述层类型的字段; 最后一列核对它与权重里真实存在的张量是否一致。
- **KV Cache / token**: 每生成一个 token, 所有**全注意力层**一共要新增多少缓存 (bf16, 2 字节/数)。
  MHA/GQA 为 `2 × KV头数 × 头维度 × 2 字节 × 层数`; MLA 只缓存压缩后的潜向量和 RoPE 键, 为 `(潜向量维度 + RoPE 键维度) × 2 字节 × 层数`。
  滑窗层计入, 但它们最多只缓存窗口内的位置; 线性注意力与 SSM 层是固定大小的状态, 不计入。
- **创建**: Hugging Face 仓库的创建日期, 只是发布时间的近似。早期模型的仓库常晚于真实发布, 例如 GPT-2 实际发布于 2019 年。
- **推断项**: 表中凡是从配置推断而非直接读出的内容会在脚本注释里注明, 例如 `mla_use_nope` 时不缓存 RoPE 键。
- **归一化与其它**: 来自张量名。例如有 `q_norm`/`k_norm` 张量即为 QK-Norm, 有 `weight_scale_inv` 即为 FP8 存储;
  这些信息很多在配置文件里没有写。
