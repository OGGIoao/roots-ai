"""
greedy_still_penalizes.py — 贪心解码时, 模型自带的重复惩罚还在不在?

Qwen2.5-Instruct 发布时附带 generation_config.json:
    do_sample=true, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1
本仓库 v0.4 评测调用 generate(..., do_sample=False)。这个脚本查清楚 transformers 这时到底执行了哪些步骤。

做法: 读 Qwen2.5 的 generation_config (几百字节, 不下载权重), 装到一个随机初始化的迷你 Qwen2 模型上
(1 层, 隐藏维度 32, 几秒建好), 拦截 generate 内部构造的处理器列表。CPU 即可。

    python experiments/greedy_still_penalizes.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "greedy_still_penalizes.json")


def main():
    import torch
    from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

    gc = GenerationConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    shipped = {k: getattr(gc, k) for k in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty")}
    print("模型自带的生成配置:", shipped)

    cfg = Qwen2Config(vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).eval()
    model.generation_config = gc

    seen = []
    original = model._get_logits_processor

    def spy(*a, **k):
        procs = original(*a, **k)
        seen.append([type(p).__name__ for p in procs])
        return procs

    model._get_logits_processor = spy
    x = torch.tensor([[1, 2, 3, 4]])
    runs = {
        "generate(do_sample=False)  (v0.4 的写法)": dict(do_sample=False),
        "generate(do_sample=False, repetition_penalty=1.0)": dict(do_sample=False, repetition_penalty=1.0),
        "generate()  (全用模型自带配置)": dict(),
    }
    res = {"shipped_generation_config": shipped, "runs": {}}
    for name, kw in runs.items():
        model.generate(x, max_new_tokens=2, pad_token_id=0, **kw)
        res["runs"][name] = seen[-1]
        print(f"  {name}\n      → {seen[-1]}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("写入", os.path.relpath(OUT, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
