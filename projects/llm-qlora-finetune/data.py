"""
data.py — QLoRA fine-tune data preparation.

ChatML format:
  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Run: python data.py
"""

import json
import os
from datasets import Dataset


SAMPLE_DATA = [
    {"messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a branch of AI that lets computers learn patterns from data automatically."},
    ]},
    {"messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is LoRA?"},
        {"role": "assistant", "content": "LoRA is a parameter-efficient fine-tuning method that freezes pretrained weights and adds low-rank matrices (A and B) to train only a tiny fraction of parameters."},
    ]},
    {"messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is quantization?"},
        {"role": "assistant", "content": "Quantization reduces model weights from high precision (FP32) to low precision (INT4/INT8), dramatically shrinking memory usage while preserving most quality."},
    ]},
    {"messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain the Transformer architecture."},
        {"role": "assistant", "content": "Transformer uses self-attention instead of recurrence. Core components: multi-head attention, feed-forward network, residual connections, and layer normalization."},
    ]},
    {"messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is RLHF?"},
        {"role": "assistant", "content": "RLHF uses human preference data to train a reward model, then optimizes the LLM with PPO reinforcement learning to align outputs with human values."},
    ]},
]


def load_sample_data():
    """Load built-in sample conversations."""
    print(f"Loaded {len(SAMPLE_DATA)} sample conversations")
    return Dataset.from_list(SAMPLE_DATA)


def load_jsonl_data(path):
    """Load conversations from a JSONL file."""
    if not os.path.exists(path):
        print(f"File not found: {path}, using sample data")
        return load_sample_data()
    data = []
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if "messages" in item:
                data.append(item)
        except json.JSONDecodeError:
            continue
    print(f"Loaded {len(data)} conversations from {path}")
    return Dataset.from_list(data) if data else load_sample_data()


def format_chatml(example, tokenizer):
    """Format conversation using tokenizer's chat template."""
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def prepare_dataset(file_path=None, tokenizer=None):
    """Load and optionally tokenize the dataset."""
    ds = load_jsonl_data(file_path) if file_path else load_sample_data()
    if tokenizer:
        ds = ds.map(
            lambda x: format_chatml(x, tokenizer),
            remove_columns=ds.column_names,
        )
    return ds


if __name__ == "__main__":
    ds = load_sample_data()
    print(f"Dataset size: {len(ds)}")
    print(f"Columns: {ds.column_names}")
    msg = ds[0]["messages"][1]
    print(f"Example: [{msg['role']}] {msg['content']}")
