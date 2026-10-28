"""
make_notebooks.py — 生成在 Google Colab 上直接运行各章的笔记本 (.ipynb)

为什么要这个: 没有显卡、不想在自己电脑上装环境的读者, 点 README / 讲义里的 "Open in Colab" 就能跑。
笔记本只做三件事: 克隆仓库 → 跑本章的训练/实验 → 跑练习的测试; 讲解仍在 TUTORIAL.md 里, 不重复。

为什么用脚本生成而不手写: .ipynb 是 JSON, 手改容易出错; 改这里的单元格内容后重跑一次即可。
注意: Colab 自带 PyTorch, 所以只装仓库自带的共享包 common/ (pip install -e . --no-deps),
     不装 requirements.txt (里面锁定的版本是为测量平台准备的, 在 Colab 上可能和预装的包冲突)。

运行: python tools/make_notebooks.py   (在仓库根目录)
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "OGGIoao/roots-car"

SETUP = [
    "# 克隆仓库并安装共享包 (Colab 每次新开会话都要重新跑这一格)",
    f"!git clone --depth 1 https://github.com/{REPO}.git",
    "%cd roots-car",
    "!pip install -q -e . --no-deps",
]


def colab_url(path):
    return f"https://colab.research.google.com/github/{REPO}/blob/master/{path}"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "\n".join(lines)}


EXERCISE_NOTE = md(
    "## 练习",
    "",
    "练习文件在左侧\"文件\"面板里 (`roots-car/<本章所在目录>/exercises/`), 双击打开就能编辑, 改完按 Ctrl+S 保存, 再运行下面的测试。",
    "刚克隆下来时测试会失败, 这是正常的: 报 `NotImplementedError` (还没写), 或者 `RuntimeError: Error(s) in loading state_dict` (还没写完, 参数对不上)。等你把 TODO 补完再跑。",
    "",
    "> **注意**: Colab 会话断开后, 你改过的文件会丢失。想保留, 就把练习文件下载下来 (右键 → 下载), 或者 fork 本仓库后把上面的克隆地址换成你自己的。",
)


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def a1():
    path = "library/llm/bpe/colab.ipynb"
    cells = [
        md("# A1 BPE 分词器 · 在 Colab 上运行",
           "",
           "小车 v0.5 \"自己的大脑\" 用到的第一个单元: 把文字变成整数。讲解见 [讲义](https://github.com/OGGIoao/roots-car/blob/master/library/llm/bpe/TUTORIAL.md), 这个笔记本只负责把代码跑起来。",
           "",
           "本章只用 Python 标准库, **不需要 GPU** (菜单 \"代码执行程序 → 更改运行时类型\" 选 CPU 即可)。"),
        code(*SETUP),
        md("## 1. 训练分词器", "", "在莎士比亚文本上训练一个词表大小 500 的字节级 BPE (测量平台上约 15 秒)。日志的含义见讲义第 5 节。"),
        code("!python library/llm/bpe/train.py"),
        md("## 2. 动手试试", "", "编码再解码, 应该一字不差地还原。试试换成你自己的句子, 包括中文。",
           "",
           "你会发现 \"你好, 世界\" 被编成了 14 个 token: 莎士比亚语料里没有中文, 每个汉字只能拆成 3 个 UTF-8 字节。为什么是 3 个见讲义第 16 节 (专题 A, UTF-8), 真实大模型的分词器怎么处理中文见第 11 节。"),
        code("import sys",
             "sys.path.insert(0, \"library/llm/bpe\")",
             "from tokenizer import BPETokenizer",
             "",
             "tok = BPETokenizer.load(\"library/llm/bpe/tokenizer.json\")",
             "for s in [\"Hello\", \"To be, or not to be\", \"你好, 世界\"]:",
             "    ids = tok.encode(s)",
             "    print(f\"{s!r:24} -> {len(ids):2d} 个 token {ids}\")",
             "    assert tok.decode(ids) == s"),
        EXERCISE_NOTE,
        code("!cd library/llm/bpe/exercises && python -m pytest test_ex1.py -q"),
    ]
    return path, cells


def a2():
    path = "library/llm/minigpt/colab.ipynb"
    cells = [
        md("# A2 miniGPT · 在 Colab 上运行",
           "",
           "从零训练一个 GPT-2 结构的小模型, 让它学会写莎士比亚风格的文字。讲解见 [讲义](https://github.com/OGGIoao/roots-car/blob/master/library/llm/minigpt/TUTORIAL.md)。",
           "",
           "建议用 GPU: 菜单 \"代码执行程序 → 更改运行时类型\" 选 GPU (免费的 T4 就够)。CPU 也能跑, 只是慢一些 (测量平台的 CPU 上 nano 配置约 2 分钟)。"),
        code(*SETUP),
        code("import torch",
             "print(\"PyTorch\", torch.__version__)",
             "print(\"GPU:\", torch.cuda.get_device_name() if torch.cuda.is_available() else \"无, 将用 CPU\")"),
        md("## 训练 nano 配置",
           "",
           "5000 步, 训练过程中会定期打印生成的文字: 一开始是乱码, 慢慢出现单词、人名和对话格式。",
           "测量平台 (RTX 5060 Laptop) 上最佳 val_loss 约 1.91; 换一张显卡, 速度不同, loss 应当接近。"),
        code("!python library/llm/minigpt/train.py --config nano"),
        md("## 接下来", "", "- 讲义第二部分的延伸专题 (KV cache、FlashAttention、混合精度、显存账本) 都有对应的实验脚本, 在 `library/llm/minigpt/experiments/` 下;",
           "- 下一章 [A3 注意力与推理延迟](https://github.com/OGGIoao/roots-car/blob/master/library/llm/attention/TUTORIAL.md) 也有 Colab 笔记本。"),
    ]
    return path, cells


def a3():
    path = "library/llm/attention/colab.ipynb"
    cells = [
        md("# A3 注意力与推理延迟 · 在 Colab 上运行",
           "",
           "MHA / MQA / GQA / MLA 四种注意力的对照实验。讲解见 [讲义](https://github.com/OGGIoao/roots-car/blob/master/library/llm/attention/TUTORIAL.md)。",
           "",
           "建议用 GPU (菜单 \"代码执行程序 → 更改运行时类型\" 选 GPU)。"),
        code(*SETUP),
        md("## 快速对照实验",
           "",
           "`--quick` 只跑 1 个随机种子、较少的步数, 用来确认能跑通 (测量平台上约 1 分钟)。",
           "完整实验 (去掉 `--quick`, 3 个种子) 在测量平台上约 25 分钟, 讲义里的数字来自完整实验。"),
        code("!python library/llm/attention/compare.py --quick"),
        EXERCISE_NOTE,
        code("!cd library/llm/attention/exercises && python -m pytest test_ex1.py -q"),
    ]
    return path, cells


def v04():
    path = "projects/car/v0.4-text-commands/colab.ipynb"
    cells = [
        md("# 小车 v0.4 听懂文字 (仿真版) · 在 Colab 上运行",
           "",
           "让大模型通过工具调用开一台仿真小车, 并在 30 个任务上实测。讲解见 [本站讲义](https://github.com/OGGIoao/roots-car/blob/master/projects/car/v0.4-text-commands/TUTORIAL.md);"
           " Agent 循环本身的原理见知识库 [agent-loop](https://github.com/OGGIoao/roots-car/blob/master/library/agent/agent-loop/TUTORIAL.md)。",
           "",
           "前两步只用 CPU; 第 3 步用本地小模型 (Qwen2.5-1.5B-Instruct, 约 3 GB) 跑评测, 需要 GPU",
           "(菜单 \"代码执行程序 → 更改运行时类型\" 选 GPU)。Colab 能直接访问 Hugging Face, 不需要镜像。"),
        code(*SETUP),
        md("## 1. 先看循环长什么样, 再看规则控制器 (都不用大模型)", "",
           "第一格用一个假模型把整段对话打印出来; 第二格是同样的工具和任务、大脑换成几十行 if-else 的对照组。"),
        code("!python library/agent/agent-loop/demo_minimal.py --seed 1",
             "!python projects/car/v0.4-text-commands/baseline.py"),
        md("## 2. 小模型评测 (6 个任务, 快速确认能跑通)", "",
           "讲义里的数字来自每组 30 个任务 (`--n 30`); 这里先各跑 6 个。`loop` 是让模型自己一步步开车, `skill` 是只让它选目标 (讲义第 6 节)。",
           "每局会打印: 成功与否、离目标多远、撞了几次、调用了几次工具。"),
        code("!cd projects/car/v0.4-text-commands && python evaluate.py --backend local:Qwen/Qwen2.5-1.5B-Instruct --mode loop --n 6",
             "!cd projects/car/v0.4-text-commands && python evaluate.py --backend local:Qwen/Qwen2.5-1.5B-Instruct --mode skill --n 6"),
        md("## 3. 看每一局是怎么失败的", "", "把每一局归到一类结局 (讲义第 5 节的表)。"),
        code("!cd projects/car/v0.4-text-commands && python analyze.py"),
        EXERCISE_NOTE,
        code("!cd library/agent/agent-loop/exercises && python -m pytest test_ex1.py -q",
             "!cd projects/car/v0.4-text-commands/exercises && python -m pytest test_ex3.py -q"),
    ]
    return path, cells


ALL = (a1, a2, a3, v04)


if __name__ == "__main__":
    for make in ALL:
        path, cells = make()
        cells[0]["source"] += f"\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(path)})"
        with open(os.path.join(ROOT, path), "w", encoding="utf-8", newline="\n") as f:
            json.dump(notebook(cells), f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("写入", path, "→", colab_url(path))
