# 对话模板 (知识库 · 大语言模型 · 推理与部署段)

> 模型只会接着一串 token 往下写。对话模板把 "system / user / assistant / 工具结果" 一条条消息拼成那一串文字, 模型才知道谁在说话、什么时候轮到自己、什么时候该停。
> 讲义: [TUTORIAL.md](TUTORIAL.md) (两部分, 22 节, 14 张图, 练习和思考题都附折叠的参考答案)。

**只需要 Python**: 手写模板 (`chat_template.py`) 和三道练习只用标准库; 标准答案 (官方模板的渲染结果) 已经存在 `experiments/results/`。想自己重跑实验, 才需要 `transformers` 和分词器文件 (不下载模型权重, 不用显卡); 只有 `no_template.py` 要下载 0.5B 模型, 在 CPU 上跑。

## 一眼看懂

```bash
cd library/llm/chat-template
python chat_template.py            # 一段 "提问 → 调用工具 → 工具返回 → 回答" 的对话拼成的文字
python chat_template.py --tools    # 再把三个工具的说明书也拼进去
```

```
<|im_start|>system
你是一个乐于助人的助手。需要实时信息或精确计算时, 调用工具。<|im_end|>
<|im_start|>user
台北现在几度?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "台北"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"city": "台北", "temperature": 26, "condition": "多云"}
</tool_response><|im_end|>
<|im_start|>assistant
台北现在 26 度, 多云。<|im_end|>
```

工具结果以 user 的身份送回; 每条消息都夹在 `<|im_start|>` 和 `<|im_end|>` 之间。讲义逐条讲这些规则从哪来、错了会怎样。

## 实测要点

| 结论 | 数字 | 讲义 |
|---|---|---|
| 不套模板, 对话模型也不会对话 | 5 个问题: 不套 0/5 自己停下, 套错 0/5, 套对 4/5 | 2 |
| 手写模板和官方 `apply_chat_template` 逐字节相同 | 500/500 段随机对话, 9/9 个边界用例 | 5 |
| 工具说明是直线增长的成本 | 3 个工具时占一次提示的 89%; 之后每个工具 +98 token | 8 |
| 中文转义成 `\uXXXX` 让工具 JSON 翻倍 | 257 → 511 token | 8.3 |
| 用户文字里的特殊标记会被当成真的结构, `apply_chat_template` 也一样 | 正常 3 个 `<\|im_start\|>`, 注入后 4 个 | 10.1 |
| 三个家族的模板在多轮对话里不保持前缀, 缓存不能整段复用 | Qwen3、gpt-oss 0%; GLM-4.5 带思考时 0% | 专题 B |
| transformers 自带的 assistant 掩码在 Qwen2.5 上全是 0 | 34 个 token 0 个算 loss | 专题 D |
| 生成提示以空格结尾, 分开分词和整段分词不一样 | 5 个回复里 3 个不一样 | 专题 E |

## 文件

```
chat-template/
├── TUTORIAL.md           讲义
├── chat_template.py      手写的 Qwen2.5 模板: render / render_with_spans / assistant_spans
├── examples.py           本单元的例子: 通用小助手的对话和三个工具 (查天气、算数、查资料)
├── mistral_v03.py        手写的 Mistral-v0.3 模板 (练习 3 的参考答案)
├── experiments/          8 个实验脚本 (见讲义 0.3 节); results/ 是输出和标准答案
├── figures/              make_diagrams.py、make_charts.py 生成的 14 张图
└── exercises/            三道练习 + 验收测试
```

## 怎么学

1. 跑 `python chat_template.py`, 读讲义第 1~3 节;
2. 打开 `chat_template.py`, 跟着第 4 节逐行读;
3. 读第 5~13 节, 做第 14 节的练习 (测试通过就是做对了, 做完再展开折叠的参考答案对照):

| 级别 | 练习 | 验收 |
|---|---|---|
| ① 填空 | `ex1_render_fill.py`: 补全 Qwen2.5 模板 | 39 段对话和官方模板逐字节相同 |
| ② 接口 | `ex2_assistant_spans.py`: 找出模型自己说的话的区间 | 和参考实现一致 |
| ③ 白纸 | `ex3_mistral_blank.py`: 从规则说明写出 Mistral-v0.3 模板 | 44 段对话 (含 12 段应报错的) |

```bash
cd exercises
python -m pytest test_ex1.py -q
```

4. 第二部分的专题 (Jinja、多轮缓存、思考模型、loss 掩码、分词边界) 按兴趣读。
5. 第 16 节是 "实践": 小车项目里怎么用到本单元。

### 掌握程度自测

| 测试方式 | 达标标准 |
|---|---|
| 看代码讲逻辑 | 能说清 `chat_template.py` 每一段对应官方 Jinja 模板的哪几行, 包括连续工具结果怎么合并 |
| 不看参考写 | 练习 ① 一次写对, 不靠测试报错来猜换行 |
| 白纸写 | 15 分钟内从规则说明写出练习 ③ |
| 能讲权衡 | 说清工具说明的 token 成本、注入的成因和修法、为什么 loss 区间要包含 `<\|im_end\|>`、思考模型的模板为什么破坏前缀缓存 |

---

*上一单元: [modern-small-model](../modern-small-model/TUTORIAL.md); 下一单元: [decoding 解码与采样](../decoding/TUTORIAL.md)。LLM 领域的全部单元见 [library/llm](../README.md)。*
