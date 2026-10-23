# 第三方内容与致谢

本仓库的代码、讲义和图以 [MIT 许可](LICENSE) 发布。下面列出仓库里用到或参考的外部内容, 以及它们各自的来源与许可。

## 仓库里包含的外部数据

| 内容 | 位置 | 来源 | 许可 / 说明 |
|---|---|---|---|
| Tiny Shakespeare 语料 (约 1.1 MB) | [mainline/A1-bpe/data/tinyshakespeare.txt](mainline/A1-bpe/data/tinyshakespeare.txt) | [karpathy/char-rnn](https://github.com/karpathy/char-rnn) 仓库中的 `data/tinyshakespeare/input.txt` | 文本是莎士比亚的作品, 属于公有领域; 整理版本出自上述仓库 |
| 开源模型的配置与张量名快照 | [electives/llm-anatomy/snapshots/](electives/llm-anatomy/snapshots/) | 各模型在 Hugging Face 上公开的 `config.json` 与权重索引 (只有超参数和张量名, **不含任何权重**) | 归各模型的发布者所有, 适用各自的模型许可; 受限仓库 (如 Llama) 未抓取内容 |

## 参考过的实现

| 项目 | 许可 | 本仓库参考了什么 |
|---|---|---|
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | MIT, Copyright (c) 2022 Andrej Karpathy | [mainline/A2-minigpt](mainline/A2-minigpt) 的整体结构与配置风格 (代码为本仓库重写, 未复制) |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | MIT | [mainline/A1-bpe](mainline/A1-bpe) 讲义中作为对照阅读 |

## 引用的资料 (只引用数字或链接, 未复制内容)

- 乐鑫《ESP32-S3 Series Datasheet》v2.2: [mainline/B1-electronics](mainline/B1-electronics) 引用了第 5 章的电气参数 (标明表号)。
- [Wokwi](https://wokwi.com) 仿真器: B1 的实测数据由 Wokwi 网页版运行本仓库的程序得到; 电路图与程序为本仓库所写。
- Datawhale《动手学具身智能》: 选修 [electives/quadruped](electives/quadruped) 作为对照读物链接, 其内容以 CC BY-NC-SA 4.0 发布, **未复制进本仓库**。
- 各讲义末尾列出的论文、课程 (如 Stanford CS336) 与文章, 只给出链接。

如果你发现某处内容的来源或许可写得不对, 请提 issue。
