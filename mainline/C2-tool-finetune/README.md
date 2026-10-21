# C2-tool-finetune — QLoRA 微调实战

> 在 RTX 5060 8GB 上用 QLoRA 微调 1.5B-3B 参数的大语言模型。
>
> **主线 C2 (服务 v0.5), 待改造**: 目标是手写 LoRA 层, 把 Qwen2.5-0.5B 微调成**只输出小车工具调用**的专用模型,
> 成功率不低于 v0.4 的 API 模型。现在的代码是通用的 QLoRA 演示, 尚未端到端跑通。

## 设备要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | 6GB VRAM | 8GB+ (RTX 5060) |
| RAM | 8GB | 16GB+ |
| 存储 | 10GB | 20GB+ (含模型下载) |

## 快速开始

```bash
# 1. 安装依赖（仓库根目录，全仓库唯一依赖清单）
pip install -r ../../requirements.txt

# 2. 运行微调 (使用内置示例数据)
python train.py

# 3. 使用自定义数据
python train.py --data my_data.jsonl --epochs 5

# 4. 切换模型
python train.py --model Qwen/Qwen2.5-3B-Instruct
```

## 项目结构

```
C2-tool-finetune/
├── config.py          # 配置参数 (量化/LoRA/训练)
├── data.py            # 数据加载 (JSONL + 内置示例)
├── train.py           # 训练脚本 (QLoRA + SFT)
└── README.md          # 本文件
```

> 依赖统一由仓库根目录 `requirements.txt` 声明，
> 本目录不再单独维护 requirements 文件，避免版本漂移。

## QLoRA 原理

```
标准全量微调:  1.5B 参数 x FP32 = 6GB 显存 (超出 8GB!)
QLoRA 微调:   1.5B 参数 x INT4 = ~800MB + LoRA ~10MB + 激活值 ~2GB = ~3GB
```

核心技巧:
1. **4-bit NF4 量化**: 权重压缩到 4-bit,显存降 75%
2. **LoRA**: 冻结原权重,只训练低秩矩阵,参数量降 99.9%
3. **PagedAdamW**: 分页优化器,避免显存 OOM
4. **Double Quant**: 二次量化,进一步压缩

## 数据格式 (JSONL)

每行一个 JSON 对象:

```json
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

## 常用模型

| 模型 | 参数量 | 显存 (QLoRA) | 特点 |
|------|--------|-------------|------|
| Qwen/Qwen2.5-0.5B | 0.5B | ~1.5GB | 极小,入门 |
| Qwen/Qwen2.5-1.5B | 1.5B | ~3GB | 中文强,推荐 |
| Qwen/Qwen2.5-3B | 3B | ~4GB | 平衡性能和质量 |
| meta-llama/Llama-3.2-3B-Instruct | 3B | ~4GB | 英文强 |

## 参考

- [Qwen2.5](https://huggingface.co/Qwen)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [PEFT](https://github.com/huggingface/peft)
- [HuggingFace TRL](https://github.com/huggingface/trl)
