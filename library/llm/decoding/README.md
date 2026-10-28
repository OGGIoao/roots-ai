# 解码与采样 (知识库 · 大语言模型 · 第 6 单元)

> 模型每一步给词表里每个 token 打一个分。怎么从这些分数里挑出下一个 token, 就是解码: 贪心、按概率抽签、抽签前划掉不靠谱的候选 (top-k、top-p、min-p)、惩罚重复、束搜索, 以及什么时候停。
> 讲义: [TUTORIAL.md](TUTORIAL.md)。用到它的小车站: [v0.4 听懂文字](../../../projects/car/v0.4-text-commands/TUTORIAL.md) (评测为什么用贪心, 工具调用的 JSON 经不经得起抽签)。
> 温度和 top-k 的基础在 [minigpt 讲义第 11 节](../minigpt/TUTORIAL.md#11-生成-一次一个字符); 本单元从那里接着讲。

**只需要 Python**: 实验对象是一个从莎士比亚语料数出来的字符级三元模型 (`toy_lm.py`), 一秒内建好, 不用下载模型, 不用显卡。

## 实测要点

| 结论 | 数字 | 讲义 |
|---|---|---|
| 贪心会转圈 | 5 个开头全部在第 2~4 个字符掉进 "the " 的循环 | 2 |
| 温度 1.0 写出的文字最接近语料本身 | 平均对数概率 -1.94, 语料本身 -1.89 | 3 |
| top-p、min-p 随把握大小伸缩, top-k 是死数 | 没把握时 top-p 留 14 个、top-k 只留 5 个 | 4 |
| 重复惩罚救不了贪心的转圈 | 惩罚加到 2.0 仍然 100% 转圈 | 5.1 |
| 重复惩罚的力度取决于 logits 的整体平移 | 同一分布同样 1.3 的惩罚, "e" 的概率可以变成 0.40, 也可以变成 0.002 | 5.2 |
| 贪心时模型自带的重复惩罚照样生效 | Qwen2.5 自带 1.1; v0.4 的数字是 "贪心 + 重复惩罚 1.1" | 6.3 |
| 手写的每一步和 transformers 一致 | 2000 组随机打分, 只有并列 token 的取舍不同 | 6.2 |
| 束搜索总分更高, 照样转圈 | 束宽 1→8, 总分 -41.67→-39.79, 全是 "the the" | 7 |

**在真实模型和小车上** (讲义第 8.3 节): Qwen 自带的重复惩罚把 1.5B 回复的第一个 token 从 `<tool_call>` 压成普通文字; 关掉它, v0.4 go_to 从 14/30 升到 18/30, "没动" 从 12 局降到 4 局。采样设置 (3 个种子) 成功 13~15, 和贪心差不多。

## 文件

```
decoding/
├── TUTORIAL.md      讲义
├── sampling.py      解码的每一步 (只用标准库)
├── generate.py      生成循环与停止条件
├── beam.py          束搜索
├── toy_lm.py        字符级三元模型 (实验对象)
├── experiments/     toy_experiments.py (标准库)、check_against_hf.py、greedy_still_penalizes.py; results/
├── figures/         make_charts.py 生成的图
└── exercises/       三道练习 + 验收测试
```

## 运行

```bash
cd library/llm/decoding
python sampling.py                        # 6 个 token 的小例子
python experiments/toy_experiments.py     # 第 2~5 节的全部数字, 约 10 秒
cd exercises && python -m pytest test_ex1.py -q
```

## 练习

| 级别 | 练习 | 验收 |
|---|---|---|
| ① 填空 | `ex1_filters_fill.py`: 补全 top-p、min-p、重复惩罚 | 600 组随机打分上和参考实现逐位相同 |
| ② 接口 | `ex2_generate.py`: 带三种停止条件的生成循环 | 和参考实现在 ToyLM 上逐字一致 |
| ③ 白纸 | `ex3_beam_blank.py`: 束搜索 | 束够宽时等于穷举, 束宽 1 等于贪心 |

### 掌握程度自测

| 测试方式 | 达标标准 |
|---|---|
| 看代码讲逻辑 | 能说清 `sampling.py` 每一步, 包括 top-k 怎么处理并列、重复惩罚为什么按正负号分开 |
| 不看参考写 | 练习 ① 一次写对 |
| 白纸写 | 20 分钟内写出束搜索, 并说出它为什么照样会转圈 |
| 能讲权衡 | 说清评测为什么用贪心、贪心不保证什么、重复惩罚为什么脆 |

---

*上一单元: [chat-template](../chat-template/TUTORIAL.md); 下一单元: structured-output 约束解码 (规划中)。*
