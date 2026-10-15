# 注意力线讲义 — MHA → MQA → GQA → MLA

> 生长树总图: [roadmaps/LLM_TREE.md](../../../roadmaps/LLM_TREE.md) 的 **A. 注意力线**。
> 配套代码: [variants.py](variants.py) (四种注意力)、[compare.py](compare.py) (训练对照实验)、[decode.py](decode.py) (推理实验)。
> 讲义里所有数字都来自本机实测或 [解剖台](../../llm-anatomy/ANATOMY.md) 读出的真实模型配置。

---

## 目录

1. [主线问题: KV Cache 为什么成了瓶颈](#1-主线问题-kv-cache-为什么成了瓶颈)
2. [A1 MHA: 出发点](#2-a1-mha-出发点)
3. [A2 MQA: 所有头共用一组 K、V](#3-a2-mqa-所有头共用一组-kv)
4. [A3 GQA: 分组共用](#4-a3-gqa-分组共用)
5. [A4 MLA: 把 K、V 压缩成潜向量](#5-a4-mla-把-kv-压缩成潜向量)
6. [对照实验: 训练质量](#6-对照实验-训练质量)
7. [推理实验: 缓存与速度](#7-推理实验-缓存与速度)
8. [对照真实模型](#8-对照真实模型)
9. [踩坑记录](#9-踩坑记录)
10. [练习](#10-练习)
11. [延伸资源](#11-延伸资源)

**前置**: [llm-minigpt 讲义](../../llm-minigpt/TUTORIAL.md) 第 5 节 (注意力) 和专题 A (KV Cache)。

---

## 1. 主线问题: KV Cache 为什么成了瓶颈

### 1.1 一句话

> 注意力线上的每一次演化, 都是在回答同一个问题: **生成时要缓存的 K、V 太多了, 怎么少存一点又不掉质量?**

### 1.2 回顾: 缓存了什么, 有多大

llm-minigpt 专题 A 讲过: 生成时每来一个新 token, 每一层都要把它的 K、V 存下来, 后面的 token 还要用。对标准的多头注意力 (MHA), 每个 token 的缓存是

```math
\text{KV Cache / token} = \underbrace{2}_{K\text{ 和 }V} \times L \times H \times d_h \times (\text{每个数的字节数})
```

($`L`$ 层数, $`H`$ 头数, $`d_h`$ 每个头的维度。)

### 1.3 真实模型有多大 (解剖台实测)

[解剖台](../../llm-anatomy/ANATOMY.md) 从真实模型的配置里算出的每 token 缓存 (bf16, 只算全注意力层):

| 模型 | 注意力 | 每个 token | 4096 个 token | 32768 个 token |
|---|---|---|---|---|
| OLMo-2-7B | MHA (32 头) | 512 KB | 2.0 GB | 16 GB |
| Qwen3-8B | GQA (32 头 / 8 组) | 144 KB | 0.56 GB | 4.5 GB |
| DeepSeek-V3 | MLA | 68.6 KB | 0.27 GB | 2.1 GB |
| Kimi-Linear-48B-A3B | MLA, 27 层里只有 7 层是全注意力 | 7.0 KB | 0.03 GB | 0.22 GB |

注意两点:

- OLMo-2 和 Qwen3-8B 参数量相近 (都是 70~80 亿), **缓存却差了 3.6 倍**, 差别完全来自注意力的设计。
- DeepSeek-V3 的总参数约 6710 亿 (技术报告数字), 是 OLMo-2 的近百倍, 每个 token 的缓存却只有它的 **13%**。

一张 8GB 的卡, 放下 OLMo-2 的 4 位量化权重 (约 3.5GB) 后, 剩下的显存只够约 9000 个 token 的缓存; 换成 Qwen3-8B 的设计, 同样的显存够 3 万多个 token。**上下文能开多长、能同时服务多少用户, 主要取决于这一项。**

### 1.4 这条线的四个节点

```
MHA ──→ MQA ──→ GQA ──────────────→ MLA
每个头     所有头     分组共用           不按头缓存, 缓存一个
一组 K、V  共用一组   (MHA 与 MQA 之间)  压缩后的潜向量
```

| 节点 | 每层每 token 缓存的数 | 典型取值下 (H=8, d_h=32) |
|---|---|---|
| MHA | $`2 H d_h`$ | 512 |
| GQA (G 组) | $`2 G d_h`$ | G=4: 256, G=2: 128 |
| MQA | $`2 d_h`$ | 64 |
| MLA | $`r + d_{\text{rope}}`$ ($`r`$ 为潜向量维度) | r=128: 144 |

---

## 2. A1 MHA: 出发点

MHA 在 llm-minigpt 讲义第 5.6 节已经逐行讲过, 这里只强调一个对后面很关键的观察:

**每个头都有自己的 $`W_K^h, W_V^h`$, 所以每个头都要单独缓存 K、V。** 头数越多、维度越大, 缓存越大。

在 [variants.py](variants.py) 里, MHA 不是单独的类, 而是 `GroupedQueryAttention` 在 `n_kv_heads == n_heads` 时的特例:

```python
self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)   # K、V 只有 n_kv_heads 组
self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
```

`n_kv_heads = n_heads` 时, 这就是普通的 MHA。

---

## 3. A2 MQA: 所有头共用一组 K、V

### 3.1 一句话

> 查询 Q 仍然每个头一份, 但 K、V 只算一份, 所有头共用。缓存直接缩小到 $`1/H`$。

### 3.2 为什么可行

注意力里, Q 决定"我要找什么", K、V 是"可供查找的内容"。不同的头需要**问不同的问题**, 所以 Q 必须各有一份; 但被查找的内容可以是同一份, 每个头用自己的 Q 去里面找不同的东西。

这个想法来自 Shazeer 2019 年的论文 *Fast Transformer Decoding: One Write-Head is All You Need*。标题里的 "one write-head" 指的就是只写一份 K、V 到缓存里。

### 3.3 代价

所有头只能在**同一个** K、V 空间里查找, 表达能力下降。原论文和后续工作都观察到质量会下降, 而且大规模训练时可能不太稳定。本仓库的小规模实测见第 6 节。

---

## 4. A3 GQA: 分组共用

### 4.1 一句话

> 把 $`H`$ 个查询头分成 $`G`$ 组, 组内的头共用一组 K、V。$`G = H`$ 就是 MHA, $`G = 1`$ 就是 MQA。

出处: Ainslie et al. 2023, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*。

### 4.2 代码

```python
def forward(self, x, cos, sin):
    B, S, _ = x.shape
    q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)      # (B, H,   S, d)
    k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, G,   S, d)
    v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    group = self.n_heads // self.n_kv_heads
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)  # (B, G, S, d) → (B, H, S, d)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.wo(out.transpose(1, 2).reshape(B, S, -1))
```

和 MHA 相比只多了一步 `repeat_interleave`: 把每组的 K、V 复制给组内的每个查询头。

**`repeat_interleave` 而不是 `repeat`**: 对 $`G=2,\ H=8`$, `repeat_interleave(4, dim=1)` 把组排成 `[g0, g0, g0, g0, g1, g1, g1, g1]`, 也就是头 0~3 用第 0 组, 头 4~7 用第 1 组。`repeat(1, 4, 1, 1)` 会排成 `[g0, g1, g0, g1, ...]`。两种排法都能训练, 但必须和你对"哪些头属于同一组"的约定一致; 加载别人训练好的权重时, 排错了就会得到乱码。

**复制只发生在计算时**: 推理时缓存里存的仍然只有 $`G`$ 组。复制出来的 $`H`$ 份只是临时张量 (实际的高性能实现连这个临时复制都省掉了, 直接在内核里按组索引)。

### 4.3 一个等价关系

GQA 等价于**一个 K、V 投影在组内各头之间完全相同**的 MHA。[tests/test_tree.py](../../../tests/test_tree.py) 里的 `test_mqa_with_one_group_equals_mha_with_tied_kv_heads` 验证了这一点: 把 MQA 的 $`W_K`$ 复制 $`H`$ 份填进一个 MHA, 两者输出一致。

这个等价关系还给出了**把训练好的 MHA 改造成 GQA** 的办法 (GQA 论文的做法): 把同一组里各头的 $`W_K^h, W_V^h`$ 取平均作为这一组的投影, 再用少量数据继续训练让模型适应。这留作第 10 节的练习。

### 4.4 为什么 GQA 成了主流

解剖台里的稠密模型, 除了 OLMo-2 还用 MHA, 几乎全部是 GQA: Mistral-7B (32 头 / 8 组)、Qwen2.5-7B (28 / 4)、Qwen3-8B (32 / 8)、phi-4 (40 / 10)、GLM-4.5 (96 / 8)、gpt-oss (64 / 8)。8 组是最常见的选择, 缓存是 MHA 的 $`8/H`$。

---

## 5. A4 MLA: 把 K、V 压缩成潜向量

### 5.1 一句话

> 不再为每个头缓存 K、V, 而是把 $`x`$ 压缩成一个 $`r`$ 维的**潜向量** $`c`$ 缓存起来, 计算注意力时再从 $`c`$ "解压"出每个头的 K、V。

出处: DeepSeek-AI 2024, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*。DeepSeek-V3、Kimi-K2 沿用。

### 5.2 思路: 低秩分解

MHA 里, 所有头的 K 拼起来是 $`k = W_K x`$, $`W_K`$ 的形状是 $`(H d_h) \times C`$。MLA 把它拆成两步:

```math
c = W_{DKV}\, x \ \ (\text{压缩到 } r \text{ 维}), \qquad k^h = W_{UK}^h\, c, \qquad v^h = W_{UV}^h\, c
```

推理时只缓存 $`c`$。DeepSeek-V3 的配置是 $`r = 512`$, 而 128 个头的 K、V 一共有 $`128 \times (192 + 128) = 40960`$ 维 (第 8 节)。

和 GQA 的区别在于: GQA 是让一部分头**完全共用**同一组 K、V; MLA 是让所有头的 K、V **都由同一个低维向量线性生成**, 每个头仍然有自己的 $`W_{UK}^h, W_{UV}^h`$, 所以每个头看到的 K、V 仍然各不相同。

### 5.3 难点: 和 RoPE 冲突

如果直接这样做, RoPE 会出问题。RoPE 要对 K 做一个**依赖位置**的旋转 $`R_j`$:

```math
k_j^h = R_j\, W_{UK}^h\, c_j
```

我们希望推理时能把 $`W_{UK}^h`$ 提前乘到 Q 上 (5.5 节的吸收技巧), 但 $`R_j`$ 夹在中间, 而且每个位置都不同, 矩阵乘法的顺序换不过来。

DeepSeek-V2 的解法是**把每个头的 Q、K 拆成两段**:

| 段 | 维度 | 是否旋转 | K 从哪来 |
|---|---|---|---|
| nope 段 | $`d_{\text{nope}}`$ | 不旋转 | 由 $`c`$ 解压, 可以做吸收 |
| rope 段 | $`d_{\text{rope}}`$ | 旋转 | 直接由 $`x`$ 投影, **所有头共用一份** |

注意力分数是两段点积之和:

```math
\text{score}_{h,j} = \frac{q^{h,\text{nope}} \cdot k_j^{h,\text{nope}} + q^{h,\text{rope}} \cdot k_j^{\text{rope}}}{\sqrt{d_{\text{nope}} + d_{\text{rope}}}}
```

rope 段的 K 所有头共用 (类似 MQA), 所以缓存里只多出 $`d_{\text{rope}}`$ 个数。每层每 token 一共缓存 $`r + d_{\text{rope}}`$ 个数。

### 5.4 代码

```python
c, k_rope = self.w_dkv(x).split([self.kv_rank, self.d_rope], dim=-1)   # 推理时缓存这两个
k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin)                     # (B, 1, S, d_rope) 所有头共用
kv = self.w_ukv(self.kv_norm(c)).view(B, S, H, self.d_nope + self.d_v).transpose(1, 2)
k_nope, v = kv.split([self.d_nope, self.d_v], dim=-1)

q = torch.cat([q_nope, q_rope], dim=-1)
k = torch.cat([k_nope, k_rope.expand(B, H, S, self.d_rope)], dim=-1)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

- `w_dkv` 一次投影同时得到潜向量 $`c`$ 和共享的 rope 键。DeepSeek 的权重里它叫 `kv_a_proj_with_mqa`, 名字里的 "with_mqa" 指的就是那份"所有头共用"的 rope 键。
- `kv_norm` 对潜向量做 RMSNorm (真实权重里叫 `kv_a_layernorm`)。
- `w_ukv` 是解压矩阵, 把 $`c`$ 变成每个头的 $`[k^{\text{nope}}, v]`$ (真实权重里叫 `kv_b_proj`)。
- `expand` 把共享的 rope 键"看成"每个头都有一份, 不复制内存。

DeepSeek 对 Q 也做了低秩压缩 (`q_a_proj` / `q_b_proj`), 那是为了减少训练时的激活显存, 与 KV Cache 无关, 本实现省略。

### 5.5 推理时的吸收技巧

只缓存 $`c`$ 还不够: 如果每生成一个 token 都要把**所有历史位置**的 $`c`$ 解压成每个头的 K、V, 计算量会随上下文长度乘以头数增长。

利用矩阵乘法的结合律, 解压可以省掉。对第 $`h`$ 个头:

```math
\text{分数: } q^h \cdot \big(W_{UK}^h c_j\big) = \big(W_{UK}^{h\top} q^h\big) \cdot c_j
```

```math
\text{输出: } \sum_j a_j \big(W_{UV}^h c_j\big) = W_{UV}^h \Big(\sum_j a_j c_j\Big)
```

- 分数: 先把新 token 的 $`q^h`$ 变换到潜空间 (每步只做一次), 再直接和缓存里的 $`c_j`$ 点积。
- 输出: 先在潜空间里对 $`c_j`$ 加权求和, 最后只乘一次 $`W_{UV}^h`$。

这样, 历史位置的 $`c_j`$ 从头到尾都不需要解压。[decode.py](decode.py) 实现了两种解码方式 (`mla_step_naive` 和 `mla_step_absorbed`), 并验证了它们与整段前向的结果一致 (第 7.1 节)。这正是 5.3 节要把 RoPE 拆出去的原因: rope 段的 $`R_j`$ 夹在中间, 没法吸收。

---

## 6. 对照实验: 训练质量

<!-- RESULTS:COMPARE -->

---

## 7. 推理实验: 缓存与速度

<!-- RESULTS:DECODE -->

---

## 8. 对照真实模型

### 8.1 DeepSeek-V3 的 MLA 参数

从解剖台快照里的 `config.json` 读出:

| 参数 | 值 | 含义 |
|---|---|---|
| `num_attention_heads` | 128 | 查询头数 |
| `qk_nope_head_dim` | 128 | 每个头 Q、K 的 nope 段 |
| `qk_rope_head_dim` | 64 | 每个头 Q、K 的 rope 段 (K 的这一段所有头共用) |
| `v_head_dim` | 128 | 每个头 V 的维度 |
| `kv_lora_rank` | 512 | 潜向量 $`r`$ |
| `q_lora_rank` | 1536 | Q 的低秩压缩 (本实现省略) |

每层每个 token:

```math
\text{MHA 等价: } 128 \times (128 + 64) + 128 \times 128 = 40960, \qquad \text{MLA: } 512 + 64 = 576
```

缩小到 **1/71**。61 层、bf16 下, 每个 token 68.6 KB; 换成同样头数和维度的 MHA 则是 4.77 MB。

### 8.2 权重名与本仓库代码的对应

解剖台从 DeepSeek-V3 的张量名里读出, 每一层的注意力有这些权重 (`N` 代表层号):

```
model.layers.N.self_attn.q_a_proj           Q 的压缩     (本实现省略)
model.layers.N.self_attn.q_a_layernorm
model.layers.N.self_attn.q_b_proj           Q 的解压     (本实现的 wq)
model.layers.N.self_attn.kv_a_proj_with_mqa x → [c, 共享 rope 键]   = w_dkv
model.layers.N.self_attn.kv_a_layernorm     对 c 做 RMSNorm         = kv_norm
model.layers.N.self_attn.kv_b_proj          c → 每个头的 [K_nope, V] = w_ukv
model.layers.N.self_attn.o_proj             输出投影                 = wo
```

DeepSeek-V3.2-Exp 在此之外还多了一组 `self_attn.indexer.*` 权重, 这是它的稀疏注意力索引器, 属于树上的下一个节点 (A6 稀疏注意力)。

### 8.3 MLA 不止 DeepSeek 在用

解剖台核实: Kimi-K2 (与 DeepSeek-V3 相同的注意力参数) 和 Kimi-Linear 都用了 MLA。Kimi-Linear 的配置里还有 `mla_use_nope=True`, 也就是它的 MLA 层**完全不用 RoPE**。这时 5.3 节的拆分就不需要了, 这是树上 B7 (不用位置编码) 节点的一个例子。

---

## 9. 踩坑记录

搭这条线的过程中真实踩过的坑:

### 坑 1: 用 `m(x, x)` 检查初始 loss, 得到了"答案泄漏"的结论

写完四种注意力后, 我用 llm-minigpt 讲义 8.3 节的办法做自检: 初始 loss 应该约等于 $`\ln 65 = 4.17`$。结果三种注意力都只有 3.3~3.6, **明显偏低**, 按排查表应该是"答案泄漏"。

模型本身没有问题, 错在测试脚本: 我写的是 `model(x, targets=x)`, 把**输入本身当成了答案**, 没有错开一位。模型能看到当前位置的字符, 加上输入输出共用一张嵌入表, 它天然倾向于"预测自己", loss 就偏低。改成 `model(seq[:, :-1], seq[:, 1:])` 后, 三种注意力的初始 loss 分别是 4.222、4.241、4.226, 与 $`\ln 65`$ 只差约 0.05。

**教训**: 自检方法本身也要用对。排查表里"x、y 没有错开一位"那一行, 说的不只是训练代码, 也包括你写的测试。[tests/test_tree.py](../../../tests/test_tree.py) 的 `test_initial_loss_near_ln_vocab` 里特意留了注释。

### 坑 2: 过拟合会让参数少的变体"显得更好"

对照实验最初的计划是比较训练 2000 步后的 val_loss。先单独跑了一个 3000 步的 MHA 看曲线:

```
step   250  train 1.5241  val 1.7214
step   750  train 1.2072  val 1.5465   ← 最低点附近
step  1000  train 1.1160  val 1.5491
step  2000  train 0.5635  val 2.1655
step  3000  train 0.1676  val 3.2745   ← 训练 loss 接近 0, 验证 loss 翻倍
```

Tiny Shakespeare 只有约 100 万字符, 300 多万参数的模型约 800 步后就开始背训练集。如果比较第 2000 步, 比的就是谁背得更狠; MQA 参数最少、背得最慢, 会显得"最好", 得出完全错误的结论。

改为每 100 步评估一次, 比较每个模型**自己曲线上的最低点** (早停), 并用 3 个随机种子估计波动。

**教训**: 在小语料上做对照实验, 一定先看一条完整的验证曲线, 再决定比较哪个时刻。

### 坑 3: GQA 的分组顺序

4.2 节讲过: `repeat_interleave` 让相邻的头属于同一组, `repeat` 则是交错分组。两者都能训练, 但**加载别人训练好的权重时必须和对方的约定一致**, 排错了不会报错, 只会得到质量很差的输出。练习 2 的测试 `test_group_convention_is_contiguous` 专门检查这一点。

### 坑 4: GPU 上同时跑两个实验, 速度数据会互相污染

对照实验在记录训练速度 (tokens/s), 推理实验在记录每个 token 的耗时。两者同时占用 GPU 时, 测出来的速度都不可信。本讲义的两个 GPU 实验是先后跑的, 但对照实验运行期间我在 CPU 上跑过几次几秒钟的小测试, 可能让当时那几组的 tokens/s 略微偏低; 所以第 6 节的速度只看量级, 几个百分点的差别不说明问题。你自己复现时也要关掉其它占用 GPU 的程序 (包括开着的 Jupyter 内核)。

---

## 10. 练习

三道练习按 [ROADMAP](../../../roadmaps/ROADMAP.md) 约定的三级递进: 同一条线上逐级拿掉"扶手"。练习文件在 [exercises/](exercises/), 每道都配了验收测试, 测试通过就是做对了。

| 级别 | 练习 | 给你什么 | 验收 |
|---|---|---|---|
| ① 填空 | [ex1_gqa_fill.py](exercises/ex1_gqa_fill.py): 补全 GQA | 完整的类, 三处 `TODO` | 与参考实现逐数值一致 (MHA / GQA-4 / GQA-2 / MQA 四种配置) |
| ② 接口 | [ex2_mha_to_gqa.py](exercises/ex2_mha_to_gqa.py): 把训练好的 MHA 改造成 GQA | 只有函数签名 | 5 条性质: 类型正确、不分组时无损、组内相同时无损、分组顺序正确、不修改输入 |
| ③ 白纸 | [ex3_mla_absorbed.py](exercises/ex3_mla_absorbed.py): MLA 的吸收式解码 | 只有验收标准 | 与整段前向一致; 每个位置只缓存 $`r + d_{\text{rope}}`$ 个数; **不许解压历史位置** |

```bash
cd projects/llm-tree/attention/exercises
python -m pytest test_ex1.py -q     # 一道一道来
```

这些测试本身验证过: 空白版本全部失败, 填入正确答案后全部通过。练习 3 还专门用"朴素解码" (每步都把历史潜向量解压成 K、V) 测过, 它算出的结果是对的、缓存大小也对, 但会被第三条测试抓住。这正是这道题的考点: **结果对, 不代表做法对。**

**卡住时**: 按约定的分级提示来问, 先给方向, 再指出哪一行, 最后才给代码, 每次只升一级。

**练习 2 的进阶实验** (不在测试里): 用 compare.py 的设置训练一个 MHA 到最佳点, 转换成 GQA-2, 分别测 (a) 转换后不训练、(b) 再训练 200 步、(c) 从头训练的 GQA-2 的 val_loss, 想一想三者的差别说明了什么。

---

## 11. 延伸资源

### 论文
- Shazeer 2019, *Fast Transformer Decoding: One Write-Head is All You Need* (MQA)
- Ainslie et al. 2023, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
- DeepSeek-AI 2024, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (MLA, 第 2.1 节)

### 本仓库
- [llm-minigpt 讲义专题 A](../../llm-minigpt/TUTORIAL.md): KV Cache 本身, 以及它在不同规模下的加速比
- [llm-anatomy](../../llm-anatomy/ANATOMY.md): 21 个真实模型的注意力配置与缓存大小
- [LLM_TREE.md](../../../roadmaps/LLM_TREE.md): 注意力线的下一步 (滑动窗口、稀疏注意力、attention sink)
