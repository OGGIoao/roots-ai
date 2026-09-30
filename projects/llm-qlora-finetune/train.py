"""
train.py — QLoRA fine-tuning script for RTX 5060 8GB.

Usage:
  python train.py                          # Default: Qwen2.5-1.5B + sample data
  python train.py --model Qwen/Qwen2.5-3B-Instruct
  python train.py --data my_data.jsonl --epochs 5
  python train.py --lora_r 16 --lr 1e-4

Requirements: pip install -r requirements.txt
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from data import prepare_dataset, load_sample_data


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA Fine-tuning")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="HuggingFace model name")
    p.add_argument("--data", type=str, default=None,
                    help="Path to JSONL training data (default: sample data)")
    p.add_argument("--output_dir", type=str, default="./qlora_output")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    # ⚠️ 精度选择 (RTX 5060 / Blackwell sm_120 实测结论):
    #    fp16 会崩 -> NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda
    #                 not implemented for 'BFloat16'  (发生在 grad clipping)
    #    换成 adamw_torch 也救不了, 因为根因在 AMP scaler 与参数 dtype 不一致。
    #    bf16 实测可正常训练, 且 Blackwell 原生支持 bf16 —— 所以默认用 bf16。
    p.add_argument("--bf16", action="store_true", default=True,
                   help="bf16 混合精度 (本机默认, fp16 不可用)")
    p.add_argument("--fp16", action="store_true", default=False,
                   help="fp16 混合精度 (本机实测会崩, 仅供实验)")
    return p.parse_args()


def main():
    args = parse_args()
    # 统一长度参数的别名, 避免下面误用不存在的 args.max_seq_length
    max_len = args.max_seq_len
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        # 注意: 属性名是 total_memory, 不是 total_mem
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        name = torch.cuda.get_device_name()
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {name} (sm_{cap[0]}{cap[1]})")
        print(f"VRAM: {vram:.1f} GB")

    # 1. 4-bit quantization config (QLoRA core)
    #    compute_dtype 与训练精度保持一致: bf16 (见下方 --bf16/--fp16 说明)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load model + tokenizer
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Prepare dataset
    dataset = prepare_dataset(file_path=args.data, tokenizer=tokenizer)
    print(f"Training samples: {len(dataset)}")

    # 5. Training config
    #
    # 版本适配说明 (trl 1.x / transformers 5.x):
    #   - max_seq_length   -> max_length      (旧名已移除, 传了会 TypeError)
    #   - max_prompt_length -> 已移除
    #   - warmup_ratio     -> 不再是 dataclass 字段, 改为直接算 warmup_steps
    steps_per_epoch = max(1, len(dataset) // (args.batch_size * args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * 0.1))

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        max_length=max_len,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="paged_adamw_32bit",
        report_to="none",
        dataset_text_field="text",
    )
    print(f"Steps: {total_steps} total, {warmup_steps} warmup")

    # 6. Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("Starting QLoRA fine-tuning")
    print(f"  Model: {args.model}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}x{args.grad_accum}={args.batch_size*args.grad_accum}")
    print(f"  LR: {args.lr}, Max seq len: {max_len}")
    print("=" * 60 + "\n")

    train_result = trainer.train()

    # 打印训练指标 (否则只看得到进度条)
    metrics = train_result.metrics if train_result is not None else {}
    print("\n--- Training metrics ---")
    for k in ("train_loss", "train_runtime", "train_samples_per_second", "epoch"):
        if k in metrics:
            v = metrics[k]
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if device == "cuda":
        print(f"  peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # 7. Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    # 8. Quick inference test
    print("\n--- Inference Test ---")
    model.eval()
    messages = [{"role": "user", "content": "What is deep learning?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_k=50)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Q: What is deep learning?")
    print(f"A: {response}")


if __name__ == "__main__":
    main()
