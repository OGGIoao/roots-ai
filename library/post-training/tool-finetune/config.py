"""
config.py — QLoRA 微调配置

设备: RTX 5060 8GB
策略: 4-bit 量化 + LoRA (只训练 0.1% 参数)

使用方式:
    python config.py  (打印所有配置)
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # 精度: 本机 (RTX 5060 / Blackwell sm_120) 实测 fp16 会在 grad clipping 崩溃,
    # 必须用 bfloat16。train.py 以此处为默认值, 不要改回 float16。
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class QuantizationConfig:
    """4-bit 量化配置 (QLoRA 核心)"""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # NormalFloat4, 比 INT4 更适合正态分布的权重
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True  # 二次量化, 进一步压缩


@dataclass
class LoraConfig:
    """LoRA 配置"""
    r: int = 8  # LoRA 秩 (越大越强, 但显存占用越多)
    lora_alpha: int = 32  # 缩放因子 (通常 = 4*r)
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: str = "./qlora_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # 有效 batch_size = 2 * 4 = 8
    learning_rate: float = 2e-4
    bf16: bool = True    # 本机唯一可用的混合精度
    fp16: bool = False   # 本机实测会崩, 仅供实验
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    max_seq_length: int = 512
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    optim: str = "paged_adamw_32bit"  # 分页优化器, 节省显存
    report_to: str = "none"  # 不上报到 wandb


@dataclass
class DataConfig:
    """数据配置"""
    max_seq_length: int = 512
    system_prompt: str = "You are a helpful AI assistant."


# ── 显存估算 ──────────────────────────────────────────────

def estimate_memory():
    """估算 RTX 5060 8GB 的显存使用"""
    print("=" * 60)
    print("RTX 5060 8GB 显存估算 (QLoRA + 1.5B 模型)")
    print("=" * 60)
    
    items = [
        ("模型权重 (4-bit 量化)", "~800MB"),
        ("LoRA 参数 (r=8)", "~10MB"),
        ("激活值 + 梯度", "~2-3GB"),
        ("优化器状态 (paged_adamw)", "~20MB"),
        ("KV-Cache (推理时)", "~500MB"),
        ("─────────────────────────", "────────"),
        ("总计 (训练)", "~3-4GB"),
        ("总计 (推理)", "~1.5GB"),
        ("剩余显存", "~4-5GB"),
    ]
    
    for name, value in items:
        print(f"  {name:<30} {value:>10}")
    
    print("=" * 60)
    print("  结论: 8GB 显存绰绰有余!")
    print("=" * 60)


# ── 配置字典 ──────────────────────────────────────────────

PRESETS = {
    "qwen-1.5b": ModelConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct"),
    "qwen-0.5b": ModelConfig(model_name="Qwen/Qwen2.5-0.5B-Instruct"),
    "llama-1b": ModelConfig(model_name="meta-llama/Llama-3.2-1B-Instruct"),
}


if __name__ == "__main__":
    estimate_memory()
    print("\n预定义模型配置:")
    for name, cfg in PRESETS.items():
        print(f"  {name}: {cfg.model_name}")
