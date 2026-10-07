# Transformer 架构演进 — 逐文件深度讲解

> 本教程逐行解释 `llm-transformer/` 项目中每个文件的核心代码,
> 帮助你从原理到实现彻底理解 Transformer 架构的演进过程。

---

## 目录

1. [架构全景图](#1-架构全景图)
2. [attention.py — 注意力机制](#2-attentionpy--注意力机制)
3. [positional_encoding.py — 位置编码](#3-positional_encodingpy--位置编码)
4. [normalization.py — 归一化层](#4-normalizationpy--归一化层)
5. [feedforward.py — 前馈网络](#5-feedforwardpy--前馈网络)
6. [transformer_block.py — Transformer 块](#6-transformer_blockpy--transformer-块)
7. [model.py — 完整模型](#7-modelpy--完整模型)
8. [data.py — 数据处理](#8-datapy--数据处理)
9. [train.py — 训练循环](#9-trainpy--训练循环)
10. [动手实验建议](#10-动手实验建议)

---

## 1. 架构全景图

### Transformer 架构演进时间线

```
2017  原始 Transformer (Vaswani et al.)
      ├── Encoder-Decoder 结构
      ├── Sinusoidal PE (固定正弦余弦)
      ├── Post-LN (归一化在 Attention 之后)
      ├── Multi-Head Attention (Q=K=V 头数相同)
      └── ReLU FFN (标准两层 MLP)

2018  BERT (Encoder-only) / GPT-1 (Decoder-only)
      ├── BERT: 双向, 用于理解 (分类/问答)
      └── GPT-1: 单向自回归, 用于生成
          └── Learned PE (可学习的位置嵌入)

2019  GPT-2 (1.5B 参数)
      ├── Pre-LN (归一化移到 Attention 之前 → 训练更稳定)
      ├── GELU FFN (更平滑的激活函数)
      └── 权重共享 (token embedding ↔ LM head)

2021  RoPE (Su et al.)
      └── 旋转位置编码: 对 Q,K 做旋转变换
          → 内积自然编码相对位置
          → 支持更长序列外推

2022  ALiBi + SwiGLU
      ├── ALiBi: 线性偏置, 无需参数即可外推
      └── SwiGLU: 门控 FFN, 表达力更强

2023  LLaMA (Meta)
      ├── RMSNorm (去掉 mean centering, 更快)
      ├── SwiGLU FFN (门控激活)
      ├── GQA (分组查询注意力, KV Cache 更省)
      ├── RoPE (旋转位置编码)
      └── 无 bias (所有 Linear 层)

2024  Qwen-V2
      └── MLA (多头潜在注意力): 低秩压缩 KV Cache
```

### 一个 Transformer Block 的数据流

```
输入 x (B, S, d_model)
    │
    ├──→ LayerNorm / RMSNorm (Pre-Norm)
    │         │
    │    ┌────▼────┐
    │    │ Attention │  ← 位置编码在这里生效
    │    └────┬────┘
    │         │
    ├──→ 残差连接 x = x + attn_output
    │
    ├──→ LayerNorm / RMSNorm (Pre-Norm)
    │         │
    │    ┌────▼────┐
    │    │   FFN    │  ← 知识存储的主要位置
    │    └────┬────┘
    │         │
    └──→ 残差连接 x = x + ffn_output

输出 x (B, S, d_model)
```

### 为什么 Decoder-only 最终胜出?

| 架构 | 代表 | 训练方式 | 生成方式 | 为什么 Decoder-only 胜出 |
|------|------|---------|---------|------------------------|
| Encoder-only | BERT | MLM (随机 mask) | 不能生成 | 无法自回归生成 |
| Encoder-Decoder | T5, BART | Span Corruption | 需要 Encoder 输入 | 两阶段,效率低 |
| Decoder-only | GPT | CLM (预测下一个token) | 自然自回归 | 简单统一,Scaling Law 友好 |

Decoder-only 的核心优势:
1. **训练统一**: 只有一个目标——预测下一个 token
2. **自回归天然**: 训练时的 forward pass 和推理时的 generate 完全一致
3. **Scaling**: 数据量和参数量增大时,性能持续提升(涌现能力)
4. **In-context Learning**: 不需要 fine-tune,直接给 few-shot 示例就能做新任务

---

## 2. attention.py — 注意力机制

### 2.1 Scaled Dot-Product Attention (最核心的公式)

```python
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
```

**这行代码做了什么?**

假设我们有:
- `query`: (B, S, d_k) — "我(每个token)想找什么信息"
- `key`:   (B, S, d_k) — "我(每个token)能提供什么信息"
- `value`: (B, S, d_v) — "我(每个token)实际携带的信息"

`matmul(q, k^T)` 的结果是 (B, S, S) 矩阵:
- `scores[i][j]` = token_i 的 query 和 token_j 的 key 的点积
- 这个分数越高,说明 token_i 应该更多地关注 token_j

**为什么要除以 sqrt(d_k)?**

数学直觉:
- 假设 q 和 k 的每个元素都是独立的标准正态分布
- `q · k = q_1*k_1 + q_2*k_2 + ... + q_d*k_d`
- 这是 d_k 个独立乘积的和,方差 = d_k
- 不缩放的话,当 d_k 很大(如 64, 128),点积值会很大
- 大的 logits → softmax 输出接近 one-hot → 梯度消失

除以 sqrt(d_k) 后:
- `Var(q · k / sqrt(d_k)) = Var(q · k) / d_k = 1`
- 方差回归到 1,softmax 梯度正常

```python
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, value)
```

softmax 将分数转为概率分布(每行和为 1),然后用这个分布对 value 做加权求和:
- 如果 token_i 对 token_j 的注意力权重很高,token_i 的输出就主要来自 token_j 的 value

### 2.2 Multi-Head Attention — 为什么需要多头?

```python
# 一个大矩阵包含所有头的投影
qkv = self.W_qkv(x)  # (B, S, 3*d_model)
q, k, v = qkv.chunk(3, dim=-1)
```

**为什么用一个大矩阵而不是三个小矩阵?**

内存和计算效率:
- 三个小矩阵: 3 × (d_model × d_model) 的参数矩阵 + 3 次矩阵乘法
- 一个大矩阵: 1 × (d_model × 3*d_model) 的参数矩阵 + 1 次矩阵乘法
- 参数量相同,但 GPU 上一次大矩阵乘法比三次小矩阵乘法更快(更少的 kernel launch)

**多头的意义是什么?**

```python
# (B, S, d_model) -> (B, n_heads, S, d_k)
q = q.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
```

将 d_model 维的向量拆成 n_heads 组,每组 d_k 维:
- **Head 1**: 可能学习到 "关注相邻的词"(局部语法)
- **Head 2**: 可能学习到 "关注主语"(语义关系)
- **Head 3**: 可能学习到 "关注标点"(结构信息)
- ...

研究表明,不同的头确实会学到不同类型的依赖关系(这被称为 "attention head specialization")。

### 2.3 Grouped Query Attention (GQA) — KV Cache 优化

**KV Cache 问题:**

自回归生成时,每生成一个新 token:
- 标准 MHA: 需要缓存所有之前 token 的 K 和 V
- Cache 大小 = 2 × n_layers × n_heads × S × d_k × dtype_bytes
- 例如 LLaMA-2 70B (64 heads, 80 layers, FP16): 每个 token 的 KV Cache ≈ 1.25 MB
- 生成 4096 个 token → 单个序列的 KV Cache ≈ 5 GB!

**GQA 的解决方案:**

```python
self.n_heads_per_kv = n_heads // n_kv_heads  # 每个 KV 头对应几个 Q 头
```

- LLaMA-2 70B: 64 个 Q 头, 8 个 KV 头 → 每 8 个 Q 头共享一组 KV
- KV Cache 减少到原来的 8/64 = 12.5%
- 质量损失极小(QA/Benchmark 几乎无差)

**为什么不是 MQA (只有 1 个 KV 头)?**

MQA 虽然 Cache 最小,但质量下降明显。GQA 是 MHA 和 MQA 之间的折中:
- GQA-8 (8 个 KV 头): 质量接近 MHA, 速度接近 MQA

### 2.4 Causal Mask — 防止"作弊"

```python
causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
```

这创建一个上三角矩阵:

```
[[False, True,  True,  True],   # token 0 不能看 token 1,2,3
 [False, False, True,  True],   # token 1 不能看 token 2,3
 [False, False, False, True],   # token 2 不能看 token 3
 [False, False, False, False]]  # token 3 可以看所有之前的位置
```

在 attention scores 中,True 的位置会被设为 -inf:
```python
scores = scores.masked_fill(mask, float("-inf"))
```

softmax(-inf) = 0, 所以 token_i 对未来 token_j 的注意力权重为 0。

**为什么必须用 causal mask?**
- 训练时: 模型同时处理整个序列,如果不 mask,模型可以"偷看"答案
- 推理时: 只能看到已生成的 token,自然满足因果性
- Causal mask 确保训练和推理行为一致

---

## 3. positional_encoding.py — 位置编码

### 3.1 为什么 Self-Attention 需要位置信息?

Self-Attention 是 **置换不变的(permutation invariant)**:

假设输入序列是 [A, B, C], attention 计算的是所有 token 两两之间的关系:
- A 对 B 的注意力 = A_query · B_key
- B 对 A 的注意力 = B_query · A_key

如果把输入打乱成 [B, A, C],attention 的输出也只是被相应地重新排列,注意力权重不变。

这意味着: "猫吃鱼" 和 "鱼吃猫" 在没有位置编码的 attention 中是等价的!

### 3.2 Sinusoidal PE — 为什么用 sin/cos?

```python
pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
```

其中 `div_term = 1 / 10000^(2i/d_model)`

**设计直觉:**

每个维度对应一个不同频率的波:
- 低维度(2i 小): 高频波 → 编码精细的位置差异
- 高维度(2i 大): 低频波 → 编码粗粒度的位置关系

类比:
- 分针(高频): 每分钟都在变,可以区分相邻位置
- 时针(低频): 每小时才变,可以区分远距离位置
- 分针+时针 = 可以唯一确定一个时间点

**为什么不用学习的位置编码?**

学习的位置编码(如 GPT-2)在训练时见过的位置上效果更好,但无法外推到更长的序列:
- 训练时 max_len=1024
- 推理时如果输入 2048 个 token → 位置 1025-2048 没有对应的 embedding → 崩溃

Sinusoidal PE 理论上可以外推(因为函数是连续的),但实际外推能力有限。

### 3.3 RoPE — 旋转位置编码的核心思想

```python
def _apply_rotary(x, cos, sin):
    d2 = x.size(-1) // 2
    x1, x2 = x[..., :d2], x[..., d2:]  # 将维度分成两半
    rotated = torch.cat([-x2, x1], dim=-1)  # rotate_half: (a,b) -> (-b,a)
    return x * cos + rotated * sin
```

**数学原理:**

将向量 x 的每两个维度视为一个复数: `z = x1 + i*x2`

乘以旋转因子 `e^(i*theta*pos)`:
```
z' = z * e^(i*theta*pos)
   = (x1 + i*x2) * (cos(theta*pos) + i*sin(theta*pos))
   = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
```

等价于代码中的:
```
rotated = x * cos + rotate_half(x) * sin
```

**为什么 RoPE 天然编码相对位置?**

两个 token 的 attention score:
```
q_i · k_j = Re[q_i_rotated · k_j_rotated]
          = Re[q_i · e^(i*theta*i) · k_j · e^(-i*theta*j)]  (k 的旋转方向相反)
          = Re[(q_i · k_j) * e^(i*theta*(i-j))]
```

结果只依赖于 `(i-j)` 即相对位置差! 这就是 RoPE 的核心——用旋转让点积自然编码相对位置。

**RoPE 的外推能力:**

训练时只见过位置 0-4096,推理时可以外推到更长序列,因为:
1. 旋转操作对任何位置都有效(函数是连续的)
2. 配合 NTK-aware scaling 或 YaRN,可以将上下文从 4K 扩展到 128K+

### 3.4 ALiBi — 简单但有效的替代方案

```python
alibi = -relative_positions.abs().float() * slopes.view(1, -1, 1, 1)
```

直接在 attention score 上减去一个与距离成正比的偏置:
- 近的 token 权重高,远的 token 权重低
- 不同的头用不同的衰减速度(斜率 m)
- 无需任何额外参数,纯数学计算

**ALiBi vs RoPE:**
- ALiBi: 更简单,无参数,但假设位置关系是线性的(表达力有限)
- RoPE: 更灵活,可以学到非线性的位置关系,是当前主流

---

## 4. normalization.py — 归一化层

### 4.1 LayerNorm 的数学原理

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
x_norm = (x - mean) / torch.sqrt(var + eps)
return self.weight * x_norm + self.bias
```

**每一步的含义:**

1. `x - mean`: 中心化,使输出均值为 0
2. `/ sqrt(var + eps)`: 缩放,使输出方差为 1
3. `* weight + bias`: 可学习的仿射变换,恢复表达力

**为什么需要归一化?**

训练深度网络时,每层的输入分布会随着训练不断变化(Internal Covariate Shift):
- 第 1 层输出 → 第 2 层输入 → 第 2 层输出 → ...
- 分布漂移导致训练不稳定
- 归一化将每层输入拉回标准分布,稳定训练

### 4.2 RMSNorm — 为什么去掉 mean?

```python
rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
return x / rms * self.weight
```

**对比:**

| | LayerNorm | RMSNorm |
|---|---|---|
| 公式 | `(x - mean) / std * w + b` | `x / RMS * w` |
| 参数 | weight + bias (2×d_model) | weight (d_model) |
| 计算 | 减均值 + 除标准差 | 只除 RMS |
| 速度 | 1x | ~1.2-1.5x |

**去掉 mean 为什么可行?**

研究发现: layer normalization 中,缩放不变性(scaling invariance)比中心化更重要。

RMSNorm 只保留了缩放,去掉了中心化:
- 减少了计算(一次 mean 代替两次 mean+var)
- 去掉了 bias 参数(少 d_model 个参数)
- 效果几乎一样(LLaMA 的实验证明了这一点)

### 4.3 Pre-LN vs Post-LN — 为什么位置很重要?

**Post-LN (原始 Transformer):**
```
x → Attention → Add → LN → FFN → Add → LN → output
         ↑              ↑
     残差连接        残差连接
```

**Pre-LN (GPT-2+):**
```
x → LN → Attention → Add → LN → FFN → Add → output
         ↑               ↑
     残差连接         残差连接
```

**为什么 Pre-LN 更好?**

在 Post-LN 中,LN 在残差连接之后:
- 每一层的 LN 会"重新标准化"残差路径
- 导致深层网络的梯度无法直接从输出传回输入
- 训练不稳定,需要仔细的 learning rate warmup

在 Pre-LN 中,LN 在 Attention/FFN 之前:
- 残差路径是干净的直连(没有 LN 打断)
- 梯度可以通过残差连接直接传回任意层
- 训练更稳定,可以训练更深的网络,不需要 warmup

这就是为什么 GPT-2 之后几乎所有模型都用 Pre-LN。

---

## 5. feedforward.py — 前馈网络

### 5.1 标准 FFN — 简单的两层 MLP

```python
def forward(self, x):
    return self.w2(self.dropout(F.relu(self.w1(x))))
```

数据流: `(B, S, d_model) → Linear(d_model, d_ff) → ReLU → Linear(d_ff, d_model)`

**维度变化:**
- 升维: d_model → d_ff (通常 d_ff = 4 × d_model)
- 降维: d_ff → d_model

**为什么 d_ff = 4 × d_model?**

原始论文的实验表明,4 倍是性能和效率的最佳平衡:
- 太小(2×): 表达力不足,模型学不到复杂模式
- 太大(8×): 参数量爆炸,过拟合风险增加
- 4×: 足够的表达力,参数量可控

**FFN 的角色是什么?**

```
Attention: "token 之间如何交换信息"(token mixing)
FFN:       "每个 token 内部如何变换"(channel mixing)
```

研究表明,FFN 层是模型存储知识的主要位置:
- 每个 FFN 可以看作一个 key-value 存储器
- 输入的 token 是 query,FFN 的权重是 key-value 对
- 这就是为什么大模型可以"记忆"大量事实

### 5.2 GELU FFN — 更平滑的激活

```python
return self.w2(self.dropout(F.gelu(self.w1(x))))
```

**GELU vs ReLU:**

```
ReLU:    f(x) = max(0, x)     — 硬阈值, x<0 时梯度为 0
GELU:    f(x) = x * Φ(x)     — 软阈值, Φ(x) 是高斯分布的 CDF
```

GELU 的优势:
1. **平滑**: 没有突然的拐点,梯度更稳定
2. **非单调**: 在 x<0 的小范围内有负值输出(ReLU 完全截断)
3. **概率解释**: GELU(x) ≈ x × P(|N(0,1)| < x),类似随机 dropout 的期望效果

GPT-2 和 BERT 都用 GELU,效果比 ReLU 好。

### 5.3 SwiGLU — 现代 FFN 的标配

```python
gate = F.silu(self.w_gate(x))  # Swish 激活的门控分支
up = self.w_up(x)              # 上投影分支
return self.dropout(self.w_down(gate * up))  # 门控 × 上投影 → 下投影
```

**门控机制的工作原理:**

```
gate = Swish(W_gate * x)  ∈ [0, ∞)  (Swish 输出范围)
up = W_up * x              ∈ (-∞, ∞)

output = gate ⊙ up        (逐元素相乘)
```

- `gate` 决定每个维度"放行多少信息"(0=完全阻断,1=完全通过)
- `up` 提供实际的信息内容
- 两者相乘 = 选择性地放大/抑制信息

**为什么 SwiGLU 比标准 FFN 好?**

1. **选择性**: 门控让网络可以学到"哪些信息重要"
2. **表达力**: 3 个矩阵(3×d_model×d_ff)比 2 个矩阵(2×d_model×d_ff)表达力更强
3. **效率**: 虽然参数多 50%,但 d_ff 通常设为 8/3×d_model,总参数量相近

**为什么用 Swish 而不是 ReLU?**

```
Swish(x) = x * sigmoid(βx)
         = x / (1 + e^(-βx))
```

Swish 是平滑的,在 x<0 时有小的负值输出,比 ReLU 保留了更多信息。

**LLaMA 的 FFN 维度选择:**

```python
d_ff = int(8 / 3 * d_model)
d_ff = ((d_ff + 255) // 256) * 256  # 对齐到 256
```

- 标准 FFN: d_ff = 4 × d_model (2 个矩阵)
- SwiGLU FFN: d_ff = 8/3 × d_model (3 个矩阵)
- 参数量对比: 3 × (8/3) = 8 vs 2 × 4 = 8 → 总参数量相同!

---

## 6. transformer_block.py — Transformer 块

### 6.1 GPT-2 Block — Pre-LN 的标准实现

```python
def forward(self, x, mask=None):
    # Pre-LN: Norm → Attention → Residual
    x = x + self.attn(self.norm1(x), mask)
    # Pre-LN: Norm → FFN → Residual
    x = x + self.ffn(self.norm2(x))
    return x
```

**残差连接的作用:**

```
output = x + sublayer(x)
```

1. **梯度高速公路**: 梯度可以通过 `+` 直接传回,不经过 sublayer
2. **恒等映射**: 如果 sublayer 什么都没学到,输出 = x + 0 = x,至少不退化
3. **深度训练**: 没有残差连接,超过 10 层的 Transformer 几乎无法训练

**代码中的顺序:**

```python
x = x + self.attn(self.norm1(x), mask)
#      ↑           ↑
#   残差连接    先 Norm 再 Attention
```

注意: `norm1(x)` 先归一化,然后送入 attention,最后加上原始 x:
```
output = x + Attention(LayerNorm(x))
```

这就是 "Pre-LN" 的含义——LN 在 sublayer 之前。

### 6.2 LLaMA Block — 现代标配

```python
def forward(self, x, mask=None):
    # Pre-RMSNorm → GQA → Residual
    x = x + self.attn(self.norm1(x), mask)
    # Pre-RMSNorm → SwiGLU FFN → Residual
    x = x + self.ffn(self.norm2(x))
    return x
```

与 GPT-2 Block 的区别(4 个替换):

| 组件 | GPT-2 Block | LLaMA Block | 为什么换 |
|------|------------|-------------|---------|
| 归一化 | LayerNorm | RMSNorm | 更快,效果一样 |
| 注意力 | MHA (全部 KV 头) | GQA (部分 KV 头) | KV Cache 更省 |
| FFN | GELU FFN | SwiGLU FFN | 表达力更强 |
| 位置编码 | Learned PE (加到输入) | RoPE (作用于 Q,K) | 更好外推 |

---

## 7. model.py — 完整模型

### 7.1 GPTModel 的数据流

```python
def forward(self, input_ids, targets=None):
    B, S = input_ids.shape

    # 1. Causal mask
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)

    # 2. Embedding
    x = self.drop(self.pos_emb(self.token_emb(input_ids)))

    # 3. Transformer Blocks
    for block in self.blocks:
        x = block(x, mask=mask)

    # 4. Final LayerNorm + LM Head
    x = self.ln_f(x)
    logits = self.lm_head(x)

    # 5. Loss
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    return logits, loss
```

**逐行解释:**

**Step 1: Causal Mask**
```python
mask = torch.triu(torch.ones(S, S), diagonal=1)
```
创建上三角矩阵,True 的位置会被 mask 掉。diagonal=1 表示对角线不 mask(每个 token 可以看自己)。

**Step 2: Embedding**
```python
x = pos_emb(token_emb(input_ids))      # 等价于 token_emb(ids) + pos_table[0..S-1]
```

关键细节: `LearnedPE.forward(x)` 内部已经做了 `x + pos_embedding(positions)`,
所以 `pos_emb` 接收的是 **token embedding 的输出**, 返回的就是"token 语义 + 位置"的和:
- 位置向量与 token 向量在同一个 d_model 空间里相加
- 与 GPT-2 原版 `wte(ids) + wpe(positions)` 等价

> ⚠️ 不要写成 `token_emb(ids) + pos_emb(token_emb(ids))` —— `pos_emb` 的返回值里
> 已经含有 token_emb, 这样写会把 token embedding 加两遍 (本项目早期版本踩过这个坑,
> 见 `tests/test_transformer.py`)。

**Step 3: 权重共享**
```python
self.lm_head.weight = self.token_emb.weight
```

GPT-2 的一个巧妙设计:
- token embedding 矩阵: (vocab_size, d_model) — 输入时用
- LM head 矩阵: (d_model, vocab_size) — 输出时用
- 两者共享权重(转置关系)

为什么可以共享?
- 输入 embedding: "token_id → d_model 维向量"(语义空间)
- 输出 projection: "d_model 维向量 → token 分数"(回到词表空间)
- 这两个变换是互逆的,共享权重不仅减少参数,还能正则化

### 7.2 自回归生成 (generate 方法)

```python
@torch.no_grad()
def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=50):
    for _ in range(max_new_tokens):
        # 截断到 max_len
        idx_cond = input_ids[:, -max_len:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature  # 只取最后一个位置

        # Top-k sampling
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return input_ids
```

**逐行解释:**

1. `logits[:, -1, :]`: 只取最后一个位置的预测(前面位置的预测已经用过了)
2. `/ temperature`: 温度缩放
   - temperature < 1: 更集中,更有确定性(如 0.5)
   - temperature = 1: 原始分布
   - temperature > 1: 更随机,更多样(如 1.5)
3. Top-k sampling: 只从概率最高的 k 个 token 中采样,排除低概率的噪声
4. `torch.multinomial`: 按概率分布随机选择一个 token

**为什么 @torch.no_grad()?**

生成时不需要计算梯度(不做反向传播),关闭梯度计算可以:
- 减少内存占用(不需要存储中间激活)
- 加速计算

---

## 8. data.py — 数据处理

> 本节的 `CharTokenizer` / `TextDataset` 现在位于仓库根的 `common/text_data.py`
> (与 llm-minigpt 共用); `data.py` 只保留本项目的组装逻辑。下文讲解的原理不变。

### 8.1 CharTokenizer — 最简单的 Tokenizer

```python
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
```

**原理:**

1. 统计文本中所有不同的字符 → 排序 → 编号
2. `encode`: 字符 → 编号
3. `decode`: 编号 → 字符

例如: "hello" → 字符集 {h, e, l, o} → 编号 {h:0, e:1, l:2, o:3} → encode("hello") = [0, 1, 2, 2, 3]

**为什么用字符级而不是词级?**

教学目的:
- 无需外部依赖(不需要训练 BPE)
- 词表小(通常 ~50 个字符),模型可以快速训练
- 可以看到模型从零学习拼写和语法

**实际 LLM 的 Tokenizer:**

实际 LLM 使用 BPE/SentencePiece(你在 `llm-bpe-tokenizer/` 项目中已经实现过):
- GPT-2: 50,257 个 token(字节级 BPE)
- LLaMA: 32,000 个 token(SentencePiece)
- Claude: ~100,000 个 token

### 8.2 TextDataset — 滑动窗口切分

```python
def __getitem__(self, idx):
    x = self.data[idx : idx + self.block_size]      # 输入
    y = self.data[idx + 1 : idx + self.block_size + 1]  # 目标(右移1位)
    return x, y
```

**为什么目标是右移 1 位?**

训练目标是"预测下一个 token":
- 输入: "To be, or not"
- 目标: "o be, or not " (每个位置预测下一个字符)

这实现了 Causal Language Modeling (CLM):
- 位置 i 的预测 = 位置 i+1 的真实 token
- Loss = cross_entropy(logits[i], target[i]) 对所有 i 求和

### 8.3 数据流全景

```
原始文本: "To be, or not to be..."
    ↓
CharTokenizer.encode(): [52, 33, 1, 33, 14, ...]
    ↓
train/val split (90%/10%)
    ↓
TextDataset (滑动窗口):
    x: [52, 33, 1, 33, 14, ...]
    y: [33, 1, 33, 14, 2, ...]
    ↓
DataLoader (shuffle + batch):
    x: (B, block_size)  — batch of input sequences
    y: (B, block_size)  — batch of target sequences
```

---

## 9. train.py — 训练循环

### 9.1 Learning Rate Schedule

```python
def get_lr(it):
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)
```

**学习率变化曲线:**

```
Learning Rate
    │
    │     ╱ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅╲
    │    ╱                    ╲
    │   ╱                      ╲
    │  ╱                        ╲
    │ ╱                          ╲___
    │╱                                ̅ ̅ ̅
    └──────────────────────────────────→ Step
    0   warmup                    decay → min_lr
```

**为什么需要 Warmup?**

训练初期,模型权重是随机初始化的:
- 梯度方向不稳定,可能很大
- 如果一开始就用大学习率,容易"跳过"最优解
- Warmup 让学习率从 0 缓慢增加,给模型时间"稳定方向"

**为什么用 Cosine Decay?**

比线性衰减更平滑:
- 前期: 学习率保持较高,快速学习
- 中期: 平滑衰减,精细调整
- 后期: 接近最小学习率,稳定收敛

### 9.2 Weight Decay 的选择性应用

```python
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
```

**规则:**
- `dim >= 2` (权重矩阵): 应用 weight decay
- `dim < 2` (bias, LayerNorm/RMSNorm weight): 不应用 weight decay

**为什么?**

Weight decay 的本质是 L2 正则化,惩罚大的权重值:
- 权重矩阵: 需要正则化防止过拟合
- Bias: 通常很小,不需要正则化(正则化反而会限制模型偏移能力)
- LayerNorm/RMSNorm 的 weight: 这是 scale 参数,正则化会破坏归一化效果

### 9.3 梯度裁剪

```python
if cfg.grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
```

**为什么需要梯度裁剪?**

训练过程中,偶尔会出现"梯度爆炸"——某个 batch 的梯度特别大:
- 不裁剪: 参数更新过大,loss 飙升,训练崩溃
- 裁剪: 将梯度范数限制在阈值内(如 1.0),安全

```python
clip_grad_norm_(parameters, max_norm)
# 实际操作: total_norm = sqrt(sum(p.grad.norm()^2 for p in parameters))
# 如果 total_norm > max_norm: scale all grads by max_norm / total_norm
```

### 9.4 评估

```python
@torch.no_grad()
def estimate_loss(model, train_loader, val_loader):
    model.eval()
    # 在训练集和验证集上分别计算 loss
    ...
    model.train()
```

**为什么要在验证集上评估?**

- 训练 loss 下降不一定意味着模型在变好(可能过拟合)
- 验 loss 下降 = 模型在未见过的数据上也在变好
- 当 val loss 开始上升而 train loss 还在下降 → 过拟合信号

**Perplexity (困惑度) 的含义:**

```python
ppl = math.exp(val_loss)
```

- Perplexity = e^cross_entropy_loss
- 直觉: 模型在每个位置"平均困惑于多少个 token"
- PPL=1: 完美预测(100% 确定下一个 token)
- PPL=vocab_size: 完全随机(均匀分布)
- PPL=20: 平均在 20 个候选中犹豫

---

## 10. 动手实验建议

### 实验 1: 对比 MHA vs GQA 的 KV Cache

```python
# 在 attention.py 的 __main__ 中添加:
import sys

D, n_heads, S = 64, 8, 1024
mha = MultiHeadAttention(D, n_heads)
gqa = GroupedQueryAttention(D, n_heads, n_kv_heads=2)

# 比较参数量
mha_params = sum(p.numel() for p in mha.parameters())
gqa_params = sum(p.numel() for p in gqa.parameters())
print(f"MHA params: {mha_params:,}")
print(f"GQA params: {gqa_params:,}")
print(f"KV Cache 节省: {1 - gqa_params/mha_params:.1%}")

# 比较推理速度 (需要 GPU)
x = torch.randn(1, S, D, device='cuda')
mha, gqa = mha.cuda(), gqa.cuda()

%timeit mha(x)  # 多次计时
%timeit gqa(x)
```

### 实验 2: 验证 RoPE 的相对位置编码

```python
# 创建两个不同的 Q 向量,但相对位置相同
rope = RoPE(d_k=16, max_len=100)
q1 = torch.randn(1, 1, 16)  # 位置 0
q2 = torch.randn(1, 1, 16)  # 位置 0

# 对 q1 做位置 5 的旋转, q2 做位置 10 的旋转
# 它们的相对位置差 = 5
q1_rot = rope(q1, offset=5)
q2_rot = rope(q2, offset=10)

# 验证: 改变绝对位置但保持相对位置差,注意力分数应该相同
```

### 实验 3: 可视化注意力权重

```python
# 在 attention.py 中,MultiHeadAttention 已经保存了 _attn_weights
model = GPTModel(vocab_size=256, d_model=64, n_heads=4, n_layers=2)
x = torch.randint(0, 256, (1, 32))
logits, _ = model(x)

# 获取第一个 block 的注意力权重
attn = model.blocks[0].attn._attn_weights  # (1, n_heads, S, S)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    axes[i].imshow(attn[0, i].detach().numpy(), cmap='hot')
    axes[i].set_title(f'Head {i+1}')
plt.savefig('attention_heatmap.png')
```

### 实验 4: 对比 Pre-LN vs Post-LN 训练稳定性

```python
# 创建两个相同大小的模型, 一个用 Post-LN, 一个用 Pre-LN
# 观察训练 loss 曲线的平滑程度
# Post-LN 通常需要更小的学习率才能稳定训练
```

### 实验 5: SwiGLU 的门控可视化

```python
swiglu = SwiGLUFFN(d_model=64, d_ff=256)
x = torch.randn(1, 1, 64)

# 观察 gate 的分布
gate = F.silu(swiglu.w_gate(x))
print(f"Gate mean: {gate.mean().item():.4f}")
print(f"Gate std:  {gate.std().item():.4f}")
# gate 值接近 0 的维度 = 被"关闭"的通道
# gate 值大的维度 = 被"放大"的通道
```

---

## 总结: 架构决策速查表

| 决策 | 旧方案 | 新方案 | 为什么改 | 哪些模型在用 |
|------|--------|--------|---------|-------------|
| 位置编码 | Sinusoidal PE | RoPE | 更好外推 | LLaMA, Mistral, Gemma |
| 归一化位置 | Post-LN | Pre-LN | 训练更稳定 | GPT-2+ |
| 归一化类型 | LayerNorm | RMSNorm | 更快 | LLaMA, Mistral |
| 注意力 | MHA | GQA | KV Cache 更省 | LLaMA-2+, Mistral |
| FFN 激活 | ReLU → GELU → SwiGLU | SwiGLU | 表达力更强 | PaLM, LLaMA |
| FFN 结构 | 2 矩阵 | 3 矩阵(门控) | 选择性信息流 | LLaMA, PaLM |
| 模型架构 | Encoder-Decoder | Decoder-only | 简单统一,Scaling 友好 | GPT, LLaMA, Claude |

---

> 🎓 **学习建议**: 每学完一个文件,运行它的 `__main__` 测试,
> 然后做 1-2 个动手实验。理解代码最好的方式是修改它、观察变化。
