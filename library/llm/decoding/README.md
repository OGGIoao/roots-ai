# 解码与采样 (知识库 · 大语言模型 · 第 6 单元) 🟡 进行中

> 模型每一步给词表里每个 token 打一个分。怎么从这些分数里挑出下一个 token, 就是解码: 贪心 (每次选最高分), 还是按概率抽签; 抽签前要不要先砍掉不靠谱的候选 (top-k、top-p、min-p), 要不要惩罚重复。
> 用到它的小车站: [v0.4 听懂文字](../../../projects/car/v0.4-text-commands/TUTORIAL.md) (评测为什么用贪心, 工具调用的 JSON 经不经得起抽签)。
> 温度和 top-k 的基础 (含字符级小模型上的实测) 已在 [minigpt 讲义第 11 节](../minigpt/TUTORIAL.md#11-生成-一次一个字符); 本单元从那里接着讲。

**只需要 Python**: `sampling.py` 只用标准库。

## 已完成

| 内容 | 文件 | 实测 |
|---|---|---|
| 手写解码的每一步: 重复惩罚 → 温度 → top-k → top-p → min-p → 贪心 / 采样, 顺序和 transformers 的 `generate` 一致 | `sampling.py` | — |
| 和 transformers 的处理器逐一核对 | `experiments/check_against_hf.py` | 2000 组随机打分: 每一步都相同; top-p 有 348 组只差在"几个同分 token 留哪个" (排序时并列的先后), 没有真正的不同 |
| 贪心时模型自带的重复惩罚还在不在 | `experiments/greedy_still_penalizes.py` | **在**。Qwen2.5 自带 `repetition_penalty=1.1`; `generate(do_sample=False)` 只跳过温度、top-k、top-p, 重复惩罚照样执行。v0.4 的评测数字因此是 "贪心 + 重复惩罚 1.1" 下测的 |

```bash
cd library/llm/decoding
python sampling.py                              # 6 个 token 的小例子, 看每一步把概率改成什么样
python experiments/check_against_hf.py          # 需要 transformers, 不需要模型, CPU 几秒
python experiments/greedy_still_penalizes.py    # 需要 transformers 和 Qwen2.5 的生成配置文件 (几百字节)
```

## 待做

- 讲义 `TUTORIAL.md` 与三级练习;
- 在真实模型上测: 小车提示下, 工具调用的每个 token 模型有多大把握; 用模型自带的采样设置时, 工具调用 JSON 写坏的比例; 关掉重复惩罚后 v0.4 的成功率。这几项要加载模型 (其中评测要用显卡), 暂时**没有数据**。

LLM 领域的全部单元见 [library/llm](../README.md)。
