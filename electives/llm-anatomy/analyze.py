"""
analyze.py — 把 snapshots/ 里的快照整理成对比表 ANATOMY.md

每个模型从两个来源取信息, 并且互相核对:
  - config.json: 作者声明的超参数
  - 张量名模式: 权重里真实存在的结构 (配置文件里不一定写, 比如 QK-Norm)

运行: python analyze.py   (不需要联网, 只读快照)
"""

import json
import os
import re
from glob import glob

from models import MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = dict(MODELS)


# ── 读取 ─────────────────────────────────────────────────────

def load():
    rows = []
    for path in glob(os.path.join(HERE, "snapshots", "*.json")):
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        if s["status"] == "ok":
            cfg = dict(s["config"])
            if isinstance(cfg.get("text_config"), dict):      # 多模态模型: 语言部分在 text_config 里
                cfg = {**cfg, **cfg.pop("text_config")}
            s["cfg"] = cfg
        rows.append(s)
    rows.sort(key=lambda s: (s.get("created") or "9999", s["repo"]))
    return rows


def g(cfg, *keys, default=None):
    """不同家族给同一个超参起了不同的名字, 按顺序取第一个存在的。"""
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return default


def count(pats, regex, exclude=r"^mtp\."):
    """匹配某个正则的张量模式, 取最大出现次数 (= 有这种张量的层数)。"""
    hits = [n for p, n in pats.items() if re.search(regex, p) and not re.search(exclude, p)]
    return max(hits, default=0)


def has(pats, regex):
    return any(re.search(regex, p) for p in pats)


# ── 各分支的分析 ──────────────────────────────────────────────

def layer_mix(s):
    """返回 (各类层的数量 dict, 说明, 与权重核对的结果)。"""
    cfg, pats = s["cfg"], s["tensor_patterns"]
    L = g(cfg, "num_hidden_layers", "n_layer")
    mt = cfg.get("model_type", "")

    if mt in ("mamba", "falcon_mamba"):
        mix, how = {"SSM": L}, "全部是 Mamba 层"
        check = count(pats, r"mixer\.A_log") == L
    elif cfg.get("hybrid_override_pattern"):
        p = cfg["hybrid_override_pattern"]
        mix = {"SSM": p.count("M"), "注意力": p.count("*"), "纯 MLP": p.count("-")}
        how = f"hybrid_override_pattern (M=Mamba, *=注意力, -=MLP)"
        check = (count(pats, r"mixer\.A_log") == mix["SSM"]
                 and count(pats, r"mixer\.q_proj") == mix["注意力"])
    elif cfg.get("attn_type_list"):
        t = cfg["attn_type_list"]
        mix = {"线性注意力": t.count(0), "注意力": t.count(1)}
        how = "attn_type_list (0=线性, 1=softmax)"
        check = (count(pats, r"self_attn\.qkv_proj") == mix["线性注意力"]
                 and count(pats, r"self_attn\.q_proj") == mix["注意力"])
    elif cfg.get("full_attention_interval"):
        k = cfg["full_attention_interval"]
        mix = {"线性注意力": L - L // k, "注意力": L // k}
        how = f"full_attention_interval={k} (每 {k} 层一层全注意力)"
        check = (count(pats, r"linear_attn\.out_proj") == mix["线性注意力"]
                 and count(pats, r"self_attn\.q_proj") == mix["注意力"])
    elif isinstance(cfg.get("linear_attn_config"), dict):
        full = cfg["linear_attn_config"].get("full_attn_layers", [])
        mix = {"线性注意力": L - len(full), "注意力": len(full)}
        how = "linear_attn_config.full_attn_layers"
        check = (count(pats, r"self_attn\.A_log") == mix["线性注意力"]
                 and count(pats, r"kv_a_proj_with_mqa") == mix["注意力"])
    elif cfg.get("layer_types"):
        t = cfg["layer_types"]
        mix = {"滑窗注意力": t.count("sliding_attention"), "注意力": t.count("full_attention")}
        how = "layer_types"
        check = count(pats, r"self_attn\.q_proj\.weight") == L
    else:
        sw = cfg.get("sliding_window")
        use_sw = sw and cfg.get("use_sliding_window", True) is not False
        mix = {("滑窗注意力" if use_sw else "注意力"): L}
        how = "所有层相同"
        check = count(pats, r"(q_proj|qkv_proj|q_a_proj|c_attn|kv_a_proj_with_mqa)\.weight") >= L
    return {k: v for k, v in mix.items() if v}, how, check


def attention(s, mix):
    """注意力类型与每个 token 的 KV Cache (bf16, 只算全注意力层)。"""
    cfg, pats = s["cfg"], s["tensor_patterns"]
    C = g(cfg, "hidden_size", "n_embd")
    H = g(cfg, "num_attention_heads", "n_head")
    if not mix.get("注意力") and not mix.get("滑窗注意力"):
        return "无 (纯 SSM)", None
    kv = g(cfg, "num_key_value_heads", default=H)
    hd = g(cfg, "head_dim", default=C // H)
    n_full = mix.get("注意力", 0) + mix.get("滑窗注意力", 0)   # 滑窗层也要缓存, 只是有上限

    if cfg.get("kv_lora_rank"):
        r, rope = cfg["kv_lora_rank"], cfg["qk_rope_head_dim"]
        if cfg.get("mla_use_nope"):
            # 配置声明 MLA 层不用 RoPE; 据此推断不缓存 RoPE 键 (未从实现代码核实)
            kind = f"MLA (潜向量 {r}, 不用 RoPE)"
            rope = 0
        else:
            kind = f"MLA (潜向量 {r} + RoPE 键 {rope})"
        if cfg.get("index_topk"):
            kind += f" + 稀疏索引 top-{cfg['index_topk']}"
        per_tok = (r + rope) * 2 * n_full
    else:
        if kv == H:
            kind = f"MHA ({H} 头)"
        elif kv == 1:
            kind = f"MQA ({H} 头共用 1 组 KV)"
        else:
            kind = f"GQA ({H} 头 / {kv} 组 KV)"
        per_tok = 2 * kv * hd * 2 * n_full
    if mix.get("滑窗注意力"):
        kind += f", 滑窗 {cfg.get('sliding_window')}"
    if has(pats, r"self_attn\.sinks"):
        kind += ", attention sink"
    return kind, per_tok


def ffn(s):
    cfg, pats = s["cfg"], s["tensor_patterns"]
    E = g(cfg, "n_routed_experts", "num_experts", "num_local_experts")
    if cfg.get("model_type") in ("mamba", "falcon_mamba"):
        return "无独立 FFN (扩展维度在 Mamba 块内部)"
    if not E:
        act = cfg.get("mlp_hidden_act") or cfg.get("hidden_act") or cfg.get("activation_function")
        inter = cfg.get("intermediate_size") or (4 * g(cfg, "hidden_size", "n_embd"))
        gated = has(pats, r"(gate_proj|gate_up_proj|w3)\.weight")
        return f"稠密 {'SwiGLU' if gated and act == 'silu' else act} ×{inter}"
    k = g(cfg, "num_experts_per_tok", "experts_per_token", "num_experts_per_token")
    shared = g(cfg, "n_shared_experts", "num_shared_experts", default=0)
    if not shared and has(pats, r"shared_expert\."):
        shared = 1
    out = f"MoE {E} 选 {k}"
    if shared:
        out += f" + {shared} 共享"
    dense = cfg.get("first_k_dense_replace")
    if dense:
        out += f", 前 {dense} 层稠密"
    inter = g(cfg, "moe_intermediate_size", "intermediate_size")
    return out + f", 专家宽度 {inter}"


def position(s):
    cfg, pats = s["cfg"], s["tensor_patterns"]
    if has(pats, r"wpe\.weight"):
        return f"可学习绝对位置 ({g(cfg, 'n_positions', 'n_ctx')})"
    theta = cfg.get("rope_theta")
    if cfg.get("mla_use_nope"):
        return "MLA 层不用 RoPE (mla_use_nope)"
    if theta is None:
        return "无显式位置编码" if cfg.get("model_type") in ("mamba", "falcon_mamba", "nemotron_h") else "?"
    out = f"RoPE θ={theta:g}"
    frac = cfg.get("partial_rotary_factor")
    if cfg.get("rotary_dim") and cfg.get("head_dim"):
        frac = cfg["rotary_dim"] / cfg["head_dim"]
    if frac and frac < 1:
        out += f", 只旋转 {frac:.0%} 维"
    rs = cfg.get("rope_scaling")
    if isinstance(rs, dict):
        out += f", 扩展 {rs.get('type') or rs.get('rope_type')}×{rs.get('factor')}"
    return out


def extras(s):
    cfg, pats = s["cfg"], s["tensor_patterns"]
    out = []
    if has(pats, r"self_attn\.(q_norm|k_norm)\.weight") or cfg.get("use_qk_norm"):
        out.append("QK-Norm")
    if has(pats, r"post_feedforward_layernorm") and not has(pats, r"input_layernorm"):
        out.append("后置归一化 (子层之后)")
    if has(pats, r"(q_proj|c_attn)\.bias"):
        out.append("QKV 带 bias")
    if g(cfg, "num_nextn_predict_layers", default=0) or has(pats, r"^mtp\.|eh_proj"):
        out.append("多 token 预测 (MTP)")
    if has(pats, r"e_score_correction_bias"):
        out.append("无辅助损失负载均衡")
    if has(pats, r"weight_scale_inv"):
        out.append("FP8 权重")
    if has(pats, r"_blocks$"):
        out.append("MXFP4 专家权重")
    if cfg.get("tie_word_embeddings") or not has(pats, r"lm_head"):
        out.append("输入输出权重绑定")
    norm = "LayerNorm" if (cfg.get("layer_norm_epsilon") and not cfg.get("rms_norm")) else "RMSNorm"
    return norm, out


# ── 输出 ─────────────────────────────────────────────────────

def main():
    rows = load()
    ok = [s for s in rows if s["status"] == "ok"]
    lines = [
        "# 模型解剖表",
        "",
        "> 由 [analyze.py](analyze.py) 根据 [snapshots/](snapshots/) 自动生成, 请勿手改。",
        "> 数据来源: 各模型官方仓库的 `config.json` 与权重张量名 (未下载权重)。",
        f"> 快照抓取日期: {', '.join(sorted({s['fetched'] for s in ok}))}。按仓库创建时间排序 (近似发布时间)。",
        "",
        "## 1. 总表",
        "",
        "| 模型 | 创建 | 层数 | 维度 | 层的构成 | 注意力 | KV Cache / token (bf16) | FFN | 位置编码 | 归一化与其它 | 配置与权重一致 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in ok:
        cfg = s["cfg"]
        mix, how, check = layer_mix(s)
        att, kv = attention(s, mix)
        norm, ex = extras(s)
        mix_str = " + ".join(f"{k} {v}" for k, v in mix.items())
        kv_str = "0 (固定大小状态)" if kv is None else f"{kv / 1024:.1f} KB"
        if kv is not None and mix.get("滑窗注意力"):
            kv_str += f" (滑窗层最多缓存 {cfg.get('sliding_window')} 个位置)"
        lines.append(
            f"| `{s['repo']}` | {s.get('created') or '?'} | {g(cfg, 'num_hidden_layers', 'n_layer')} "
            f"| {g(cfg, 'hidden_size', 'n_embd')} | {mix_str} | {att} | {kv_str} | {ffn(s)} "
            f"| {position(s)} | {', '.join([norm] + ex)} | {'✅' if check else '⚠️ 不一致'} |"
        )

    lines += ["", "## 2. 未能读取的模型", ""]
    for s in rows:
        if s["status"] != "ok":
            reason = "受限仓库, 需要 HF_TOKEN" if s["status"] == "gated" else s["status"]
            lines.append(f"- `{s['repo']}` ({NOTES.get(s['repo'], '')}): {reason}")

    lines += [
        "",
        "## 3. 各列的含义与口径",
        "",
        "- **层的构成**: 混合架构里各类层的数量, 来自配置文件里描述层类型的字段; 最后一列核对它与权重里真实存在的张量是否一致。",
        "- **KV Cache / token**: 每生成一个 token, 所有**全注意力层**一共要新增多少缓存 (bf16, 2 字节/数)。",
        "  MHA/GQA 为 `2 × KV头数 × 头维度 × 2 字节 × 层数`; MLA 只缓存压缩后的潜向量和 RoPE 键, 为 `(潜向量维度 + RoPE 键维度) × 2 字节 × 层数`。",
        "  滑窗层计入, 但它们最多只缓存窗口内的位置; 线性注意力与 SSM 层是固定大小的状态, 不计入。",
        "- **创建**: Hugging Face 仓库的创建日期, 只是发布时间的近似。早期模型的仓库常晚于真实发布, 例如 GPT-2 实际发布于 2019 年。",
        "- **推断项**: 表中凡是从配置推断而非直接读出的内容会在脚本注释里注明, 例如 `mla_use_nope` 时不缓存 RoPE 键。",
        "- **归一化与其它**: 来自张量名。例如有 `q_norm`/`k_norm` 张量即为 QK-Norm, 有 `weight_scale_inv` 即为 FP8 存储;",
        "  这些信息很多在配置文件里没有写。",
    ]
    out = os.path.join(HERE, "ANATOMY.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"写入 {out}: {len(ok)} 个模型, {len(rows) - len(ok)} 个未读取")


if __name__ == "__main__":
    main()
