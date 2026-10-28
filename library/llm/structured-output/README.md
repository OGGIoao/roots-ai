# 结构化输出 (知识库 · 大语言模型 · 第 7 单元)

> 约束解码: 每一步把 "接上去就不可能再合法" 的 token 的分数设成 -∞, 模型只能在合法的路里挑。格式由约束保证, 写什么仍由模型决定。
> 讲义: [TUTORIAL.md](TUTORIAL.md)。用到它的小车站: [v0.4 听懂文字](../../../projects/car/v0.4-text-commands/TUTORIAL.md) (go_to 模式 12/30 局 "没动", 都是工具调用写成了普通文字)。

**只需要 Python**: 前缀树、token 级放行、数字自动机和三道练习都只用标准库。第 3 节在 Qwen 上的实验只用 CPU。

## 实测要点

| 结论 | 数字 | 讲义 |
|---|---|---|
| 约束把格式错误降到 0 | 1.5B 在会出问题的设置下: 不加约束 3/6 合法, 加约束 6/6, 目标全对 | 3 |
| 约束保证格式, 不保证内容 | 只懂莎士比亚的字符级模型加约束后选了最短的 look | 2 |
| 约束不能凭空造出模型写不出的字符 | 字符级模型的词表里没有引号和括号 | 2 |
| 一个 token 可能跨过好几个格式边界 | Qwen 的 `"}}\n` 是一个 token | 4.1 |
| 前缀合法不等于写得完 | 词表缺单独的 `:` 时会走进死胡同 | 4.3 |
| 纯 Python 实现的代价 | 每一步把 15 万个 token 试一遍, 41~49 ms | 3 |

**还没有数据**: 把约束接进 v0.4 的完整评测 (30 个任务、多轮), 成功率会怎样。

## 文件

```
structured-output/
├── TUTORIAL.md          讲义
├── constrained.py       前缀树、token 级放行、小车的 4 个合法工具调用
├── number_dfa.py        约束数字格式的自动机
├── experiments/         structured_experiments.py; results/
├── figures/             make_diagrams.py 生成的图
└── exercises/           三道练习 + 验收测试
```

## 运行

```bash
cd library/llm/structured-output
python experiments/structured_experiments.py --only-toy      # 只用标准库, 1 秒
cd exercises && python -m pytest test_ex1.py -q
```

## 练习

| 级别 | 练习 | 验收 |
|---|---|---|
| ① 填空 | `ex1_trie_fill.py`: 前缀树 | 随机字符串集合上和参考实现一致 |
| ② 接口 | `ex2_token_mask.py`: token 级的放行集合 | 跨边界 token、空 token、停止 token; 复现死胡同 |
| ③ 白纸 | `ex3_number_blank.py`: 不用 re, 手写数字格式的自动机 | 约 2.3 万个字符串和正则逐个核对 |

### 掌握程度自测

| 测试方式 | 达标标准 |
|---|---|
| 看代码讲逻辑 | 能说清为什么 token 级不能逐字符判断, 空 token 为什么要拦 |
| 不看参考写 | 练习 ① ② 一次写对 |
| 白纸写 | 15 分钟内画出数字格式的自动机并写成代码 |
| 能讲权衡 | 说清约束管不了什么 (内容、该不该调用工具、模型的理解) |

---

*上一单元: [decoding](../decoding/TUTORIAL.md); 下一单元: local-inference 本地推理 (规划中)。*
