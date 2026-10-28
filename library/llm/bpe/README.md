# A1-bpe — 从零实现字节级 BPE 分词器

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OGGIoao/roots-car/blob/master/library/llm/bpe/colab.ipynb) 不想在自己电脑上装环境? 点这个按钮, 在浏览器里直接运行本章 ([colab.ipynb](colab.ipynb))。

> LLM 的第一道工序: 把文字变成整数。本项目用不到 400 行纯 Python (无第三方依赖) 实现 GPT-2 式的字节级 BPE,
> 在 TinyShakespeare 上训练、编码、解码, 并与 GPT-2、Qwen2.5、DeepSeek-V3 等 12 个真实分词器对照。

**详细讲义见 [TUTORIAL.md](TUTORIAL.md)** (14 张图, 所有数字均为本机实测)。本文件只是入口。

## GPT 看到 "Hello" 时, 它看到的是什么?

不是字符 H-e-l-l-o, 而是一个或几个整数。用本项目默认训练出的分词器 (莎士比亚, 词表 500):

```python
>>> from tokenizer import BPETokenizer
>>> tok = BPETokenizer.load("tokenizer.json")
>>> tok.encode("Hello world")
[72, 427, 111, 263, 270, 316]          # H  ell  o  ␣w  or  ld
>>> tok.decode([72, 427, 111, 263, 270, 316])
'Hello world'
```

GPT-2 (词表 50257) 把同一句话切成 `Hello` `␣world` 两个 token。差别从哪来、词表多大合适、中文为什么贵, 都在讲义里。

## 文件

```
A1-bpe/
├── tokenizer.py            BPE 核心: train / encode / decode (重点阅读)
├── train.py                运行入口: 训练 → 演示 → 压缩率 → 保存
├── data.py                 语料 (TinyShakespeare, 入库在 data/)
├── tokenizer.json          入库的默认产物 (莎士比亚, 词表 500)
├── modern_tokenizers.py    tiktoken / SentencePiece / HuggingFace 演示
├── information_theory.py   熵、交叉熵、Huffman 与分词效率
├── TUTORIAL.md             讲义
├── figures/                讲义配图 + 生成脚本
├── experiments/            讲义里所有数字的来源 (examples.py、tokenizer_anatomy.py) 与结果
└── exercises/              三道练习 + 验收测试
```

核心 BPE 只用标准库。`modern_tokenizers.py` 和 `experiments/` 用到 `tokenizers` / `sentencepiece` / `regex` / `huggingface_hub`,
都在仓库根目录的 `requirements.txt` 里; 可选的 `tiktoken` 缺失时对应演示自动跳过。

## 快速开始

```bash
cd library/llm/bpe
python train.py                          # 默认: 莎士比亚, 词表 500, 约 15 秒
python train.py --vocab_size 1000        # 更大的词表
python train.py --source sample --vocab_size 300   # 离线用内置小样例 (需显式指定, 下载失败不会自动降级)
```

`tokenizer.json` 是入库的默认产物, **只有默认配置**会写入它; 其它配置保存为 `tokenizer_<来源>_<词表>.json`
(已被 `.gitignore` 忽略), 也可以用 `--output` 指定路径。

## 怎么学

1. 读讲义第 1~3 节, 跑一次 `python train.py`, 对照讲义第 5 节读懂每一行日志;
2. 打开 `tokenizer.py`, 跟着讲义第 4、6、7、8 节逐行读;
3. 做 [exercises/](exercises/) 的三道练习 (讲义第 14 节), 测试通过就是做对了:

| 级别 | 练习 | 验收 |
|---|---|---|
| ① 填空 | `ex1_bpe_fill.py`: 补全 BPE 的四个核心步骤 | merges 与参考实现逐条相同 |
| ② 接口 | `ex2_stream_decode.py`: 流式解码器 | 逐个喂 token 不出 `�`, 拼起来等于整体解码 |
| ③ 白纸 | `ex3_fast_train.py`: 增量更新的快速训练 | merges 逐条相同, 且至少快 3 倍 |

```bash
cd exercises
python -m pytest test_ex1.py -q
```

### 掌握程度自测

| 测试方式 | 达标标准 |
|---|---|
| 看代码讲逻辑 | 能解释 `tokenizer.py` 每一行, 包括并列时怎么选、重叠的 pair 怎么数 |
| 不看参考写 | 练习 ① 的四处 TODO 一次写对 |
| 白纸写 | 20 分钟内从零写出训练 + 编码 + 解码, 并能说出每轮训练的复杂度 |
| 能讲权衡 | 说清词表大小、预分词、字节级 vs 字符级对模型的影响 (讲义第 9、11、17 节) |

---

*下一步: [A2-minigpt](../minigpt/) 把一个 GPT 从零训练起来; LLM 线的全貌见 [LLM 架构生长树 (选修)](../llm-tree/README.md)。*
