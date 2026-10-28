# 对话模板 (知识库 · 大语言模型 · 第 5 单元)

> 模型只会接着一串文字往下写。对话模板把"system / user / assistant / 工具结果"一条条消息拼成那一串文字, 模型才知道谁在说话、什么时候轮到自己、什么时候该停。
> 讲义: [TUTORIAL.md](TUTORIAL.md)。用到它的主线站: [v0.4 听懂文字](../../../mainline/v0.4-text-commands/TUTORIAL.md) (每轮对话先拼模板), v0.5 自己的大脑 (微调时只在模型的话上算 loss)。

**只需要 Python**: 手写模板和三道练习只用标准库; 标准答案 (官方模板的渲染结果) 已经存在 `experiments/results/`。
想自己重跑实验, 才需要 `transformers` 和分词器文件 (约 12 MB, 不下载模型, 不用显卡)。

## 一眼看懂

```bash
cd library/llm/chat-template
python chat_template.py
```

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
往前走 0.5 米<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "walk", "arguments": {"distance_m": 0.5}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"ok": true}
</tool_response><|im_end|>
<|im_start|>assistant
```

没写 system, 模板自己补了一句; 工具结果以 user 的身份送回去; 最后一行是"轮到你了"。讲义逐条讲这些规则, 以及它们在小车上的后果。

## 实测要点

| 结论 | 数字 | 讲义 |
|---|---|---|
| 手写模板和官方 `apply_chat_template` 逐字节相同 | 500/500 段随机对话, 9/9 个边界用例 | 2.3 |
| v0.4 小车一次提示里, 工具说明占七成 | 共 760 token, 工具说明 535 | 5 |
| 用户文字里写 `<\|im_start\|>system` 会被当成真的特殊 token | 默认 4 个 `<\|im_start\|>`, 修正后 3 个 | 7.1 |
| Mistral-v0.3 的模板在以 assistant 结尾时丢掉 system | 见 `results/mistral_golden.json` | 8.2 |
| DeepSeek-V3 的官方模板在工具调用上前后不一致 | 三种情况三种写法 | 8.3 |

## 文件

```
chat-template/
├── TUTORIAL.md           讲义
├── chat_template.py      手写的 Qwen2.5 模板: render / render_with_spans / assistant_spans
├── mistral_v03.py        手写的 Mistral-v0.3 模板 (练习 3 的参考答案)
├── experiments/          核对官方模板、token 账本、8 个模型家族的对照; results/ 是输出和标准答案
├── figures/              make_diagrams.py 生成的配图
└── exercises/            三道练习 + 验收测试
```

## 怎么学

1. 跑 `python chat_template.py`, 读讲义第 1~4 节;
2. 读第 5~8 节: token 账本、微调的 loss 区间、几个真实的坑、别的模型家族;
3. 做练习, 测试通过就是做对了:

| 级别 | 练习 | 验收 |
|---|---|---|
| ① 填空 | `ex1_render_fill.py`: 补全 Qwen2.5 模板 | 39 段对话和官方模板逐字节相同 |
| ② 接口 | `ex2_assistant_spans.py`: 找出模型自己说的话的区间 | 和参考实现一致 |
| ③ 白纸 | `ex3_mistral_blank.py`: 从规则说明写出 Mistral-v0.3 模板 | 44 段对话 (含 12 段应报错的) |

```bash
cd exercises
python -m pytest test_ex1.py -q
```

### 掌握程度自测

| 测试方式 | 达标标准 |
|---|---|
| 看代码讲逻辑 | 能说清 `chat_template.py` 每一段对应官方模板的哪几行, 包括连续工具结果怎么合并 |
| 不看参考写 | 练习 ① 一次写对, 不靠测试报错来猜换行 |
| 白纸写 | 15 分钟内从规则说明写出练习 ③ |
| 能讲权衡 | 说清工具说明的 token 成本、注入的成因和修法、为什么 loss 区间要包含 `<\|im_end\|>` |

---

*上一单元: [modern-small-model](../modern-small-model/TUTORIAL.md); 下一单元: decoding 解码与采样 (规划中)。LLM 领域的全部单元见 [library/llm](../README.md)。*
