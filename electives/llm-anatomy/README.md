# 解剖台 (选修)

> 不下载权重, 只读真实开源模型的 `config.json` 和权重张量名, 用数据核对"架构生长树"上的每一条论断。
> 树的总图见 [electives/llm-tree/README.md](../../electives/llm-tree/README.md)。

结果: **[ANATOMY.md](ANATOMY.md)** (21 个模型的统一对比表)。

## 它读什么

| 数据 | 来源 | 大小 | 能看到什么 |
|---|---|---|---|
| 超参数 | `config.json` | 几 KB | 层数、头数、KV 头数、专家数、RoPE 参数…… |
| 张量名 | 多文件模型读 `model.safetensors.index.json`; 单文件模型只读 safetensors 文件头 | 几 KB ~ 几 MB | **配置文件里没写的结构**: QK-Norm、多 token 预测模块、FP8 存储、attention sink…… |
| 仓库创建时间 | Hub API | — | 近似的发布时间 (仅供排序) |

张量名里的层号、专家编号都替换成 `N` 后去重计数。例如 Kimi-K2 的 13.9 万个张量归并成 38 种模式, 整个快照只有几 KB。

## 用法

```bash
cd electives/llm-anatomy
python fetch.py            # 抓取 (需要能访问 huggingface.co); 已有快照的会跳过
python fetch.py --refresh  # 全部重新抓取
python analyze.py          # 由快照生成 ANATOMY.md, 不需要联网
```

新增模型: 在 [models.py](models.py) 的 `MODELS` 里加一行, 再依次运行上面两个脚本。

**受限仓库** (Llama、Gemma、Jamba 等) 需要先在 huggingface.co 同意模型协议并生成访问令牌, 然后设置环境变量 `HF_TOKEN` 再运行 `fetch.py`。

## 配置与权重的互相核对

[analyze.py](analyze.py) 对每个模型同时从配置和张量名推断层的构成, 两者不一致时在表中标 ⚠️。
写规则时它已经抓出过 6 处**分析规则自身的错误**, 例如:

- phi-4 把 Q、K、V 合并成一个 `qkv_proj`, 最初的规则只认 `q_proj`, 误报不一致;
- DeepSeek-V3.2 稀疏索引器里的 `k_norm` 被误认成注意力的 QK-Norm。

所以这张表的每一格都有据可查, 但规则本身也可能出错。发现可疑的格子, 先回到快照里看原始的张量名。
