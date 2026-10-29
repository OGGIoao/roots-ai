# 对话模板讲义 — 从一条条消息到一串 token

> 知识库 · 大语言模型 · 推理与部署段。所有数字都在作者的笔记本上实测 (只用 CPU), 命令和结果文件见每节末尾。
> 一句话: **模型只会接着一串 token 往下写; 对话模板是把 "一条条消息" 拼成 "一串文字" 的规则。** 拼错一个换行, 模型看到的就是一份它没见过的格式。

## 目录

**第一部分 · 读懂这份代码**

- [0. 阅读指南](#0-阅读指南)
- [1. 全景: 对话模板在链路里做什么](#1-全景-对话模板在链路里做什么)
- [2. 为什么需要模板](#2-为什么需要模板)
- [3. 手算一个最小的例子](#3-手算一个最小的例子)
- [4. chat_template.py 逐行拆解](#4-chat_templatepy-逐行拆解)
- [5. 和官方模板逐字节核对: 逐行读日志](#5-和官方模板逐字节核对-逐行读日志)
- [6. 特殊标记: 词表里专门新增的 token](#6-特殊标记-词表里专门新增的-token)
- [7. 工具调用: 说明书、调用、结果](#7-工具调用-说明书调用结果)
- [8. token 账本: 一次提示里谁占了多少](#8-token-账本-一次提示里谁占了多少)
- [9. 生成提示、预填与 loss 区间](#9-生成提示预填与-loss-区间)
- [10. 四个坑](#10-四个坑)
- [11. 别的模型家族长什么样](#11-别的模型家族长什么样)
- [12. 从教学实现到工业实现](#12-从教学实现到工业实现)
- [13. 本单元踩坑记录汇总](#13-本单元踩坑记录汇总)
- [14. 练习](#14-练习)
- [15. 延伸资源](#15-延伸资源)
- [16. 实践: 在项目里用](#16-实践-在项目里用)

**第二部分 · 延伸专题**

- [17. 专题 A · 读懂 Jinja 模板](#17-专题-a--读懂-jinja-模板)
- [18. 专题 B · 多轮对话: 这一轮的缓存下一轮还能用吗](#18-专题-b--多轮对话-这一轮的缓存下一轮还能用吗)
- [19. 专题 C · 会思考的模型, 模板多了什么](#19-专题-c--会思考的模型-模板多了什么)
- [20. 专题 D · 训练时的模板: 哪些 token 算 loss](#20-专题-d--训练时的模板-哪些-token-算-loss)
- [21. 专题 E · 模板和分词边界](#21-专题-e--模板和分词边界)
- [22. 延伸专题练习](#22-延伸专题练习)

---

# 第一部分 · 读懂这份代码

## 0. 阅读指南

### 0.1 这个单元在知识库里的位置

| | |
|---|---|
| 领域 | 知识库 · 大语言模型 ([目录](../README.md)), 推理与部署段 |
| 上游单元 | [bpe 分词](../bpe/TUTORIAL.md) (文字怎么变成 token)、[minigpt](../minigpt/TUTORIAL.md) (模型怎么 "接着往下写") |
| 下游单元 | [decoding](../decoding/TUTORIAL.md) (模型怎么挑下一个 token)、[structured-output](../structured-output/TUTORIAL.md) (让模型只能写合法的格式)、Agent 领域的 [agent-loop](../../agent/agent-loop/TUTORIAL.md) (工具调用的循环)、后训练领域的 sft (用模板拼训练数据) |
| 本单元的例子 | 一个通用的小助手, 三个工具: 查天气、算数、查资料 ([examples.py](examples.py))。项目里的用法只在第 16 节 |

### 0.2 需要先知道什么

只需要两件事, 不熟的话各读一小段:

1. **token**: 模型不直接看文字, 看的是一串整数, 每个整数对应词表里的一小段文字。读 [bpe 讲义第 1 节](../bpe/TUTORIAL.md#1-全景-分词器在-llm-里做什么)。
2. **模型是 "接着往下写" 的**: 给它一串 token, 它预测下一个, 接上去再预测下一个, 直到写出 "结束" 标记或者到了长度上限。读 [minigpt 讲义第 11 节](../minigpt/TUTORIAL.md#11-生成-一次一个字符)。

第一次出现的词都会先用一句大白话解释。

### 0.3 文件和命令

```
chat-template/
├── chat_template.py              手写的 Qwen2.5 模板 (只用标准库), 第 4 节逐行讲
├── examples.py                   本单元的例子: 通用小助手的对话和三个工具
├── mistral_v03.py                手写的 Mistral-v0.3 模板, 第 11 节的对照 (练习 3 的参考答案)
├── experiments/
│   ├── check_against_hf.py       和官方模板逐字节核对 (第 5 节)
│   ├── measure_tokens.py         特殊标记、注入、token 账本 (第 3、6、8、10 节)
│   ├── no_template.py            不套模板 / 套错 / 套对, 模型写什么 (第 2 节, 要加载 0.5B 模型)
│   ├── template_zoo.py           8 个模型家族的模板对照 (第 11 节)
│   ├── prefix_property.py        多轮对话的前缀性质 (专题 B)
│   ├── thinking_templates.py     会思考的模型的模板 (专题 C)
│   ├── token_boundaries.py       loss 掩码与分词边界 (专题 D、E)
│   ├── practice_car.py           第 16 节实践: 小车项目的提示账本
│   └── results/                  以上脚本的输出; 练习的标准答案也在这里
├── figures/                      make_diagrams.py (结构图)、make_charts.py (数据图)
└── exercises/                    三道练习 + 验收测试
```

**两种跑法:**

| 想做什么 | 需要什么 | 命令 |
|---|---|---|
| 看模板拼出来的样子、做练习 | 只要 Python 3.8+ | `python chat_template.py`; `cd exercises && python -m pytest test_ex1.py -q` |
| 自己重跑实验, 核对讲义里的数字 | `transformers` (见仓库 `requirements.txt`) + 分词器文件 | 见下 |

实验只下载**分词器和模板文件** (Qwen2.5 约 12 MB; 其他模型只有几 KB 到几百 KB), **不下载模型权重, 不需要显卡**。只有 `no_template.py` 要下载 Qwen2.5-0.5B-Instruct (约 1 GB), 在 CPU 上跑约 1~2 分钟。

```bash
cd library/llm/chat-template
export HF_ENDPOINT=https://hf-mirror.com     # 国内网络; Windows PowerShell 写成 $env:HF_ENDPOINT="https://hf-mirror.com"
python experiments/check_against_hf.py       # 约 7 秒
python experiments/measure_tokens.py         # 约 7 秒
python experiments/template_zoo.py           # 第一次要下载 10 个模型的模板文件
python experiments/prefix_property.py        # 几秒, 用上一步下载的模板
python experiments/thinking_templates.py
python experiments/token_boundaries.py
python experiments/no_template.py            # 要下载 0.5B 模型, CPU 约 1~2 分钟
```

作者的环境: Windows 11, Python 3.10, transformers 5.17.0。

### 0.4 怎么读

1. 先跑 `python chat_template.py`, 看一段完整的对话被拼成了什么;
2. 读第 1~3 节: 为什么需要模板, 手算一个最小的例子;
3. 打开 `chat_template.py`, 跟着第 4 节逐行读, 再看第 5 节它是怎么和官方核对的;
4. 第 6~10 节: 特殊标记、工具调用、token 账本、loss 区间、四个坑, 每节都有实测;
5. 第 11~12 节换几个模型家族看, 再看工业界怎么用;
6. 做第 14 节的练习。每道题和每个思考题后面都有折叠的参考答案, 做完再展开对照;
7. 第二部分的专题按兴趣读。

### 0.5 记号约定

| 记号 | 意思 |
|---|---|
| `↵` | 换行符 `\n` (图和表里为了看得见) |
| `<\|im_start\|>` 这类 | 特殊标记 (第 6 节); 表格里竖线要写成 `\|`, 正文和代码里就是 `<|im_start|>` |
| "消息" | 一个 `{"role": ..., "content": ...}` 字典 |
| "提示" (prompt) | 交给模型的那一整串文字, 模型从它的末尾接着写 |
| "生成提示" | 模板在提示末尾加的 "轮到你了" 标记, 第 9 节 |

---

## 1. 全景: 对话模板在链路里做什么

程序这边, 一段对话是一条条**消息**:

```python
messages = [
    {"role": "system", "content": "你是一个乐于助人的助手。需要实时信息或精确计算时, 调用工具。"},
    {"role": "user", "content": "台北现在几度?"},
]
```

`role` (角色) 说明这句话是谁说的:

| role | 是谁 | 例子 |
|---|---|---|
| `system` | 开场设定, 一般由开发者写 | "你是一个乐于助人的助手" |
| `user` | 用户 | "台北现在几度?" |
| `assistant` | 模型自己 | "台北现在 26 度。" 或者一个工具调用 |
| `tool` | 工具执行的结果 | `{"temperature": 26}` |

但模型根本没有 "消息" 这个概念。它只接收**一串 token**, 然后预测下一个。所以去的路上要先把消息拼成一整段文字, 用一些约定好的记号标出 "这里开始是用户说的" "这里结束了" "现在轮到你说"; 回来的路上再把模型写的文字解析回消息。

![对话模板在整条链路里的位置](figures/fig_pipeline.svg)

**怎么读这张图**: 上面一行是去的路, 下面一行是回来的路。本单元只讲上面一行的第二个框: **对话模板** (chat template)。它的输入是消息列表, 输出是一整段文字。文字再交给分词器 (bpe 单元) 变成 token, 模型一次往后写一个 token; 写出来的 token 解码成文字, 再被解析成一条新消息 (Agent 领域的 tool-calling 单元), 接到消息列表末尾, 进入下一轮。

这一整段文字长什么样? 下面是 [examples.py](examples.py) 里一段完整的对话 (提问 → 调用工具 → 工具返回 → 回答) 拼出来的结果, `python chat_template.py` 就能打印出来:

![一段完整对话被模板拼成的文字](figures/fig_render.svg)

**怎么读这张图**: 左边是 5 条消息, 右边是拼出来的文字。橙色是 "特殊标记" (第 6 节), 每个只占 1 个 token; 蓝底是模型自己要写出来的部分 (第 9 节)。注意两件事: 工具调用和普通回答都包在 `<|im_start|>assistant … <|im_end|>` 里; 工具的结果没有自己的角色, 而是包成 `<tool_response>`, 以 **user** 的身份送回去 (第 7 节)。

---

## 2. 为什么需要模板

### 2.1 不套模板, 对话模型会怎样

Qwen2.5-0.5B-Instruct 是一个 "对话模型" (训练时专门学过聊天)。同一个问题, 三种喂法, 看它写什么 (`no_template.py`, CPU, 贪心, 最多 60 个 token):

| 喂法 | 喂进去的文字 |
|---|---|
| 不套模板 | `法国的首都是哪里?` |
| 套错模板 | `[INST] 法国的首都是哪里? [/INST]` (这是 Mistral 家族的格式) |
| 套对模板 | `<|im_start|>system↵You are Qwen…<|im_end|>↵<|im_start|>user↵法国的首都是哪里?<|im_end|>↵<|im_start|>assistant↵` |

![同一个问题, 三种喂法](figures/fig_no_template.svg)

**怎么读这张图**: 三个框是三种喂法下模型写出的文字。**不套模板**时, 模型把问题当成一份文档的开头接着写: 编出一道选择题 "A. 巴黎 B. 里昂 …", 然后跑题去讲 Excel。**套错模板**时, 它模仿 `[INST]` 这种格式, 自己编出 `[IN]` `[OUT]` 一问一答。**套对模板**时, 它只写了 7 个 token "法国的首都是巴黎。", 然后写出 `<|im_end|>` 自己停了。

5 个问题汇总:

| 喂法 | 自己停下来的次数 | 典型表现 |
|---|---|---|
| 不套模板 | 0/5 | 续写成文档: 编选择题; 把问题 "3 乘 9 等于多少?" 原样重复了一遍又一遍 |
| 套错模板 | 0/5 | 模仿格式, 自问自答 |
| 套对模板 | 4/5 | 正常回答; 没停的那一个是在写 Python 代码, 60 个 token 不够写完 |

同一个模型、同一组权重, 区别只在于喂进去的文字长什么样。

### 2.2 模板是后训练时定下来的约定

模型为什么认得 `<|im_start|>`、`<|im_end|>` 这些记号? 因为它在**后训练** (预训练之后, 用大量对话数据再训练一轮, 见 [后训练领域](../../post-training/README.md)) 时, 见过的每一段对话都是这么拼的。它学会了:

- 看到 `<|im_start|>assistant↵`, 接下来该自己说话;
- 说完一段, 写一个 `<|im_end|>`。

所以模板**是模型的一部分**, 不是随便选的。同一个基础模型, 不同团队微调出的对话版本可能用完全不同的模板 ([transformers 文档](https://huggingface.co/docs/transformers/main/en/chat_templating) 举了两个都从 Mistral-7B 微调出来、模板却不同的例子)。

### 2.3 模板随模型一起发布

模板是一段 **Jinja** 程序 (一种模板语言, 有 if、for, 像简化的 Python), 写在模型仓库的 `tokenizer_config.json` 的 `chat_template` 字段里; 较新的模型单独放在 `chat_template.jinja` 文件里 (第 11 节的 8 个家族里, 两种都有)。transformers 里用 `tokenizer.apply_chat_template(messages)` 执行这段程序。

本单元做的事: 把 Qwen2.5 的这段 Jinja 用 Python 重写一遍 (`chat_template.py`, 约 100 行), 让每一条规则都看得见; 再和官方逐字节核对 (第 5 节)。Jinja 本身怎么读, 见专题 A。

---

## 3. 手算一个最小的例子

先不看代码, 用手把两条消息拼出来:

```python
[{"role": "system", "content": "你是助手。"},
 {"role": "user", "content": "你好"}]
```

ChatML 格式 (Qwen 用的格式) 只有一条规则: **一条消息 = `<|im_start|>` + 角色名 + 换行 + 内容 + `<|im_end|>` + 换行**。

| 步骤 | 拼上去的文字 |
|---|---|
| system 消息 | `<|im_start|>system↵你是助手。<|im_end|>↵` |
| user 消息 | `<|im_start|>user↵你好<|im_end|>↵` |
| 生成提示 (轮到模型说话) | `<|im_start|>assistant↵` |

合起来 (`↵` 写成真的换行):

```
<|im_start|>system
你是助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant

```

再数 token。特殊标记各占 1 个 token; 角色名、换行、中文词各自是普通 token。手算是 3 + 3 + 3 + 2 + 3 + ... 不如直接看分词器 (`measure_tokens.py` 的 A 部分):

![两条消息切成 17 个 token](figures/fig_tokens.svg)

**怎么读这张图**: 每个框是一个 token, 下面是它的 id。橙框是特殊标记: `<|im_start|>` 的 id 是 151644, `<|im_end|>` 是 151645, 它们在词表的最末尾 (词表的普通部分只到 151642, 见第 6 节)。"你是" "助手" "你好" 都是一个 token, 换行是 id 198。一共 17 个 token, 其中 5 个是特殊标记, 3 个是角色名, 5 个是换行, 只有 4 个是 "内容"。

**练一练**: 再加一条 `{"role": "assistant", "content": "你好呀"}`, 不加生成提示, 一共多少个 token? (答案在第 14.2 节思考题 1。)

---

## 4. chat_template.py 逐行拆解

整个文件约 130 行, 核心是一个函数 `render_with_spans`: 输入消息列表, 输出拼好的文字, 外加 "模型自己说的话" 在文字里的区间 (第 9 节用)。

### 4.1 常量: 一个字都不能差

```python
# chat_template.py
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# 有工具时, 模板在 system 后面追加的说明书 (原文照抄, 一个字符都不能差)
TOOLS_HEAD = ("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
              "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
TOOLS_TAIL = ("\n</tools>\n\nFor each function call, return a json object with function name and arguments "
              "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
              "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
```

- `DEFAULT_SYSTEM`: 对话里没有 system 消息时, 模板自动补的那句 (第 8.4 节)。
- `TOOLS_HEAD` / `TOOLS_TAIL`: 有工具时, 模板在 system 后面追加的英文说明, 中间夹着每个工具的 JSON (第 7.1 节)。这两段直接从官方模板里抄出来, **多一个空格、少一个换行, 都和模型训练时见过的不一样**。

```python
# chat_template.py
def tojson(x):
    """Jinja 模板里的 `| tojson`。transformers 的实现是 json.dumps(x, ensure_ascii=False): 中文原样输出, 不转成 \\uXXXX。"""
    return json.dumps(x, ensure_ascii=False)
```

官方模板用 Jinja 的 `| tojson` 把工具和参数变成 JSON 字符串。transformers 实现它时用的是 `ensure_ascii=False`: 中文原样输出。如果手写版用了 Python 的默认值 (`ensure_ascii=True`), "台北" 会变成 `台北`, 不但和官方不一致, token 数还会翻倍 (第 8.3 节实测)。

### 4.2 emit: 边拼接边记区间

```python
# chat_template.py
    out = []        # 依次拼接的片段
    spans = []
    length = 0      # 已拼接的总长度, 用来记区间

    def emit(s, trainable=False):
        nonlocal length
        if trainable:
            spans.append((length, length + len(s)))
        out.append(s)
        length += len(s)
```

所有输出都经过 `emit`。它做两件事: 把片段追加到 `out`; 如果这一段是模型自己要写的 (`trainable=True`), 就把它在最终文字里的起止位置记下来。`length` 一直等于目前已经拼了多少个字符, 所以区间是精确的字符下标。最后 `"".join(out)` 得到整段文字。

### 4.3 第一步: 开头的 system 块

```python
# chat_template.py
    # 1) 开头的 system: 用户给了就用用户的, 没给就用默认的那句 "You are Qwen ..."
    if messages[0]["role"] == "system":
        system = messages[0]["content"]
    else:
        system = DEFAULT_SYSTEM
    if tools:                                  # 空列表 [] 和 None 一样, 不加工具说明
        system += TOOLS_HEAD + "".join("\n" + tojson(t) for t in tools) + TOOLS_TAIL
    emit(IM_START + "system\n" + system + IM_END + "\n")
```

三条规则:

1. **只看第一条消息**是不是 system。第一条不是, 就用默认的那句; 对话中途出现的 system 消息不在这里处理 (4.4 节)。
2. **工具说明书追加在 system 后面**: `TOOLS_HEAD` + 每个工具一行 JSON + `TOOLS_TAIL`。所以对模型来说, "有哪些工具" 只是 system 里的一段文字。
3. `if tools:` 对空列表 `[]` 也是假, 所以 "传了空工具列表" 和 "没传工具" 结果一样 (第 5 节的边界用例 "工具列表为空")。

### 4.4 第二步: 普通消息

```python
# chat_template.py
    # 2) 逐条消息
    for i, m in enumerate(messages):
        role = m["role"]
        if role == "system" and i == 0:
            continue                           # 第一条 system 已经在上面写过了
        if role in ("user", "system") or (role == "assistant" and not m.get("tool_calls")):
            if role == "assistant":
                emit(IM_START + "assistant\n")
                emit(m["content"] + IM_END, trainable=True)
                emit("\n")
            else:
                emit(IM_START + role + "\n" + m["content"] + IM_END + "\n")
```

- 第一条 system 跳过 (4.3 节已经写了)。
- user、**中途的** system、不带工具调用的 assistant, 都按第 3 节那条规则写: `<|im_start|>角色↵内容<|im_end|>↵`。
- assistant 的回复拆成三段 emit: 头 `<|im_start|>assistant↵` 不算模型写的; **内容 + `<|im_end|>`** 算 (`trainable=True`); 最后的换行不算。为什么这样切, 见第 9.3 节。

### 4.5 第三步: 带工具调用的回复

```python
# chat_template.py
        elif role == "assistant":              # 带工具调用的回复
            header = IM_START + "assistant\n"
            seg = IM_START + "assistant"
            if m.get("content"):
                seg += "\n" + m["content"]
            for call in m["tool_calls"]:
                f = call.get("function", call)  # OpenAI 格式外面包了一层 "function"
                seg += '\n<tool_call>\n{"name": "' + f["name"] + '", "arguments": ' + tojson(f["arguments"]) + "}\n</tool_call>"
            seg += IM_END
            # seg 一定以 header 开头 (后面紧跟的是 "\n" + 内容, 或 "\n<tool_call>"),
            # header 之后的部分就是模型要自己写出来的
            emit(header)
            emit(seg[len(header):], trainable=True)
            emit("\n")
```

这一段是整个模板里最绕的:

1. 先按官方的写法拼出整段 `seg`: `<|im_start|>assistant`, 有文字就加 `↵文字`, 然后每个工具调用加一段 `↵<tool_call>↵{"name": "…", "arguments": …}↵</tool_call>`, 最后 `<|im_end|>`。
2. 官方模板里 "assistant" 后面的换行不是单独写的, 而是每段内容自带的开头换行。为了让 "模型写的部分" 的起点和第 4.4 节一致 (都在 `<|im_start|>assistant↵` 之后), 这里先拼出整段, 再从 `header` 的长度处切开。切得对不对? `seg` 后面紧跟的一定是 `↵文字` 或 `↵<tool_call>`, 都以换行开头, 所以 `seg` 必然以 `header` 开头。
3. `call.get("function", call)`: OpenAI 格式的工具调用外面包了一层 `{"type": "function", "function": {...}}`, 也有程序直接传 `{"name": ..., "arguments": ...}`; 两种都接受, 和官方模板 `{% if tool_call.function is defined %}` 那一句的行为一样。
4. 参数用 `tojson`: 所以 `arguments` 应该是**对象**。传成字符串会被再编码一次, 第 7.4 节的坑。

### 4.6 第四步: 工具的结果

```python
# chat_template.py
        elif role == "tool":
            # 连续几条工具结果合在同一个 user 块里, 每条包一层 <tool_response>
            first = i == 0 or messages[i - 1]["role"] != "tool"
            last = i == len(messages) - 1 or messages[i + 1]["role"] != "tool"
            if first:
                emit(IM_START + "user")
            emit("\n<tool_response>\n" + m["content"] + "\n</tool_response>")
            if last:
                emit(IM_END + "\n")
        # 其它 role (比如 "observation"): 官方模板一个字也不输出, 这里照做。讲义第 10.2 节的坑。
```

Qwen2.5 的模板**没有 tool 这个角色标记**。工具结果以 user 的身份送回, 包在 `<tool_response>` 里。连续几条工具结果 (一次调用了好几个工具) 合进同一个 user 块: 这一串的第一条前面写 `<|im_start|>user`, 最后一条后面写 `<|im_end|>↵`。`first` / `last` 就是在判断 "我是不是这一串的第一条 / 最后一条"。

不认识的角色 (`if … elif … elif` 都没命中) 什么都不输出, 也不报错, 第 10.2 节的坑。

### 4.7 第五步: 生成提示

```python
# chat_template.py
    # 3) 生成提示: 写好 "<|im_start|>assistant\n", 让模型从这里接着写
    if add_generation_prompt:
        emit(IM_START + "assistant\n")

    return "".join(out), spans
```

推理时 `add_generation_prompt=True`: 在末尾写好 "轮到 assistant 说话了", 模型从这里接着写。训练时一般是 False, 因为对话本身已经以 assistant 的回答结尾 (第 9 节)。

### 4.8 两个外壳

```python
# chat_template.py
def render(messages, tools=None, add_generation_prompt=False):
    """把对话拼成模型看到的那一整段文字。"""
    return render_with_spans(messages, tools, add_generation_prompt)[0]


def assistant_spans(messages, tools=None):
    """只要区间 (练习 2 的参考答案)。"""
    return render_with_spans(messages, tools)[1]
```

`render` 只要文字, 对应 transformers 的 `apply_chat_template(..., tokenize=False)`; `assistant_spans` 只要区间。

---

## 5. 和官方模板逐字节核对: 逐行读日志

手写版对不对, 不靠看, 靠核对。`check_against_hf.py` 生成大量对话, 分别用手写的 `render` 和官方的 `apply_chat_template` 渲染, 逐字节比较。实际输出:

```
随机对话: 500/500 段逐字节相同
  相同  只有一句用户
  相同  自定义 system
  相同  中途的 system
  相同  工具列表为空
  相同  两个连续的工具结果
  相同  回复里既有话又有工具调用
  相同  arguments 是 JSON 字符串 (OpenAI 原样)
  相同  不认识的 role 被丢掉
  相同  用户文字里写了特殊标记
边界用例: 9/9 逐字节相同
assistant 区间: 1505/1505 个起点等于官方生成提示的结尾
写入 experiments\results\qwen25_golden.json
```

逐行读:

| 日志 | 意思 |
|---|---|
| `随机对话: 500/500` | 用固定种子随机生成 500 段对话: 有没有 system、几轮、有没有工具调用、一次调用几个、空字符串、多行、引号和反斜杠、emoji、前后空格, 随机组合。全部逐字节相同 |
| 9 个边界用例 | 故意挑的刁钻情况, 每个都对应第 4 节里一条规则或第 10 节的一个坑 |
| `assistant 区间: 1505/1505` | 第 9.3 节的区间检查: 每个 assistant 回复的起点, 都等于 "前面的消息 + 生成提示" 用**官方**模板渲染出来的长度 |
| `写入 …qwen25_golden.json` | 边界用例和前 30 段随机对话的官方输出存下来, 练习 1 和仓库的测试拿它当标准答案, 不装 transformers 也能核对 |

这种 "随机生成大量输入 + 和权威实现逐个比较" 的做法叫**差分测试**。它比手写几个测试用例可靠得多: 手写的用例只覆盖你想得到的情况, 随机组合会碰到你没想到的 (比如 "回复里既有话又有工具调用" 加上 "连续两条工具结果")。以后改动 `chat_template.py`, 重跑一遍就知道有没有改坏。

---

## 6. 特殊标记: 词表里专门新增的 token

### 6.1 它们在词表的最末尾

`<|im_start|>` 看着像 12 个字符, 但在分词器里它是**一个** token。Qwen2.5 在 BPE 学出来的词表之外, 专门**新增**了 22 个 token (`measure_tokens.py` 的 A 部分):

| 标记 | token id | 作用 | special 属性 |
|---|---|---|---|
| `<\|endoftext\|>` | 151643 | 文档结束; 批量生成时补齐长度 (pad) | True |
| `<\|im_start\|>` | 151644 | 一条消息开始 | True |
| `<\|im_end\|>` | 151645 | 一条消息结束; 也是对话时的停止符 (eos) | True |
| `<tool_call>` | 151657 | 工具调用开始 | **False** |
| `</tool_call>` | 151658 | 工具调用结束 | **False** |

加上新增的 22 个, 词表一共 151,665 个 token。

为什么要单独占一个 token? 如果 `<|im_start|>` 被拆成 `<`、`|`、`im`、`_start` 这些普通 token, 它就和用户碰巧打出的同样字符分不开了。占一个专用的 id, 模型就能确定无疑地认出 "这是结构, 不是内容"。(这个设计也带来了第 10.1 节的注入问题。)

### 6.2 模型怎么知道该停

模型说完一段, 会写出 `<|im_end|>`。分词器的配置里 `eos_token` (end of sequence, 停止符) 就是 `<|im_end|>`, 生成程序看到它就停。所以 "会停下来" 也是后训练时学会的: 每段 assistant 的话后面都跟着 `<|im_end|>`。第 2.1 节里不套模板时模型停不下来, 就是因为它不在 "对话" 里, 不觉得该写 `<|im_end|>`。

### 6.3 special 属性: 解码时去不去掉

解码 (token 变回文字) 时常设 `skip_special_tokens=True`, 去掉特殊 token。但只去掉 `special=True` 的那些:

```
'<tool_call>\n{"name": "get_weather", "arguments": {"city": "台北"}}\n</tool_call><|im_end|>'
    → 解码时去掉特殊 token →
'<tool_call>\n{"name": "get_weather", "arguments": {"city": "台北"}}\n</tool_call>'
```

`<|im_end|>` 去掉了, `<tool_call>` 留下来了。正因为 `<tool_call>` 的 special 是 False, 程序才能在解码后的文字里找到工具调用。

### 6.4 不是所有标记都是新增 token

`<tool_response>` 看起来和 `<tool_call>` 一样, 但它**不在**新增的 22 个里, 会被切成 `<`、`tool`、`_response`、`>↵` 四个普通 token。模型照样认得它 (训练时见过), 只是要多花几个 token。哪些是新增 token, 只能查分词器, 不能看外表。

---

## 7. 工具调用: 说明书、调用、结果

**工具调用** (tool calling / function calling): 模型不直接做任何事, 而是写下 "我要调用某个工具, 参数是什么", 由程序去执行, 再把结果送回给模型。一次完整的来回:

![一次工具调用的来回](figures/fig_tool_flow.svg)

**怎么读这张图**: 从上往下是时间顺序, 左列是模型看到或写出的文字, 右列是程序做的事。模板参与了三个地方: 把说明书拼进 system (第 1 格)、把模型的调用写回历史 (第 2 格)、把工具结果包成 `<tool_response>` 送回 (第 3 格)。

### 7.1 说明书: 追加在 system 后面

调用 `render(messages, tools=TOOLS)` 时 (`python chat_template.py --tools` 可以看到), system 后面多了这样一段:

```
你是一个乐于助人的助手。需要实时信息或精确计算时, 调用工具。

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "查询某个城市现在的天气", "parameters": {...}}}
{"type": "function", "function": {"name": "calculator", ...}}
{"type": "function", "function": {"name": "search_docs", ...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
```

每个工具的说明是一段 **JSON Schema** (用 JSON 描述 "这个工具叫什么、做什么、有哪些参数、参数是什么类型")。对模型来说, "有哪些工具" 只是 system 里的一段文字, 没有任何特殊通道。模型会不会调用工具, 全靠后训练时见过足够多这种格式的例子。

### 7.2 模型写出调用

模型决定调用工具时, 写出:

```
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "台北"}}
</tool_call><|im_end|>
```

一次调用多个工具, 就一段接一段写多个 `<tool_call>` (边界用例 "两个连续的工具结果")。程序用正则表达式找出这些段落、解析 JSON, 再交给工具执行; 解析这一步在 Agent 领域的 [agent-loop](../../agent/agent-loop/TUTORIAL.md) 讲。

### 7.3 结果送回

工具执行完, 结果作为 `role="tool"` 的消息加进对话。如第 4.6 节, Qwen2.5 把它以 user 的身份、包在 `<tool_response>` 里送回:

```
<|im_start|>user
<tool_response>
{"city": "台北", "temperature": 26, "condition": "多云"}
</tool_response><|im_end|>
```

然后模型看到结果, 写出最终回答 "台北现在 26 度, 多云。"。

### 7.4 一个坑: arguments 要给对象, 不要给字符串

OpenAI 的接口格式里, 工具调用的 `arguments` 是 **JSON 字符串** `'{"city": "台北"}'`。但 Qwen 的模板要的是**对象** `{"city": "台北"}`, 它自己会调 `tojson`。如果原样把字符串传进去, 会被再编码一次 (边界用例 "arguments 是 JSON 字符串"):

```
{"name": "get_weather", "arguments": "{\"city\": \"台北\"}"}
```

参数变成了一个带转义引号的字符串, 和模型训练时见过的格式不一样。**不会报错, 只是悄悄错了。** 把 OpenAI 格式的历史消息交给本地模板之前, 要先把 `arguments` 解析成对象。第 11.3 节会看到, DeepSeek-V3 的模板恰好相反, 要的就是字符串: 每个家族都要读它的模板才知道。

---

## 8. token 账本: 一次提示里谁占了多少

模型每次生成之前都要把整段提示读一遍 (为什么提示越长越慢, 见 [attention 讲义](../attention/TUTORIAL.md))。值得算一算: 一次提示里, 谁占了多少 token。

### 8.1 一个带三个工具的小助手

`measure_tokens.py` 的 C 部分, system + "台北现在几度?" + 三个工具:

![一次提示的 token 账本](figures/chart_budget.svg)

**怎么读这张图**: 上面四根横条是一次提示的四个组成部分, 下面一行把 "工具说明" 拆成每个工具。

| 部分 | token 数 |
|---|---|
| 工具说明 (3 个工具) | 337, 其中三个工具的 JSON 共 257, 模板固定的英文说明 80 |
| 系统提示 | 22 |
| 模板标记 (`<\|im_start\|>`、角色名、换行) | 13 |
| 用户的话 | 5 |
| **合计** | **377**, 工具说明占 89% |

用户真正问的只有 5 个 token, 工具说明书却有 337 个。在多轮对话和 Agent 循环里, 每一轮都要把它们连同越来越长的历史重新送进模型。

### 8.2 工具越多越长: 直线增长

把同样结构的工具复制 0~12 个 (`measure_tokens.py` 的 D 部分):

![工具个数和提示长度](figures/chart_tools_scaling.svg)

**怎么读这张图**: 横轴是工具个数, 纵轴是一次提示的 token 数。从 0 个到 1 个跳了 175 个 token, 因为第一个工具要带上模板固定的英文说明 (约 80 个); 之后每加一个工具, 稳定增加约 98 个。**工具说明是直线增长的成本**: 给模型 20 个工具, 光说明书就是两千多个 token, 每一轮都要付一次。这也是为什么 Agent 设计里常常 "按需给工具"、把底层工具合并成少数几个高层工具 (Agent 领域的 harness-engineering)。

### 8.3 中文说明、英文说明、转义

| 写法 (3 个工具) | token 数 |
|---|---|
| 说明用中文, 整段提示 | 377 |
| 说明换成英文, 整段提示 | 379 |
| 三个工具的 JSON, 中文原样 (`ensure_ascii=False`) | 257 |
| 三个工具的 JSON, 中文转义成 `\uXXXX` (Python 的默认 `ensure_ascii=True`) | **511** |

中文和英文说明几乎一样长 (Qwen 的词表对中文很友好, 见 [bpe 讲义第 11 节](../bpe/TUTORIAL.md#11-中文-字节级-bpe-的软肋))。真正的坑是转义: 自己拼提示时用 `json.dumps` 的默认设置, 中文变成 `台北`, JSON 的 token 数翻一倍。

### 8.4 没给 system 时

如果对话里没有 system 消息, Qwen2.5 的模板**自动补一句** `DEFAULT_SYSTEM`。实测: 只发一句 "台北现在几度?" (5 个 token), 模型看到的是 34 个 token, 其中 16 个是这句自动补上的 "You are Qwen, created by Alibaba Cloud…"。这一点只有读模板才知道; 想让模型 "什么设定都没有", 在 Qwen2.5 上做不到。

---

## 9. 生成提示、预填与 loss 区间

### 9.1 生成提示

`add_generation_prompt=True` 时, 模板在最后加上 `<|im_start|>assistant↵`, 意思是 "轮到你了"。不加的话, 最后一条是 user 消息, 模型可能接着用户的话往下写 (第 2.1 节 "不套模板" 就是极端的例子)。transformers 的文档也专门提醒了这一点 (见第 15 节)。

### 9.2 预填 (prefill)

既然模型只是接着写, 我们可以**替它把回答的开头写好**, 叫预填。比如在生成提示后面再加上 `<tool_call>↵`, 模型就只能接着写工具调用的 JSON, 不会写成普通文字。transformers 里对应的参数是 `continue_final_message` (让最后一条 assistant 消息不加结束标记, 模型接着写)。预填只管住开头; 想管住每一个 token, 见 [structured-output](../structured-output/TUTORIAL.md)。

### 9.3 loss 区间: 哪些字是模型自己说的

**loss** (损失) 是训练时衡量 "预测得有多差" 的数, 训练就是让它变小。微调对话模型时, 一段对话里只有 assistant 说的话是要教给模型的 "答案"; system、用户的话、工具结果都是 "题目", 模型不需要学会写它们。所以通常只在 assistant 的部分算 loss。

`render_with_spans` 在拼文字的同时, 记下每条 assistant 回复的区间 `[start, end)` (start 是开始的字符位置, end 是结束位置, 不含 end 本身)。区间怎么定:

- **从 `<|im_start|>assistant↵` 之后开始**: 这个头是生成提示, 是程序写的, 不是模型写的;
- **到 `<|im_end|>` 为止, 包含它**: 模型要学会自己停下来 (第 6.2 节)。

第 1 节那张图里蓝底的部分就是这些区间。

**怎么确认区间没算错?** 有一个简洁的性质: **第 k 条消息是 assistant 时, 它的区间起点正好等于 "前 k 条消息 + 生成提示" 渲染出来的长度。** 因为推理时模型就是从那里开始写的。第 5 节的日志里 `1505/1505` 就是用官方模板逐条核对这个性质。这个性质也是练习 2 的一种解法。

### 9.4 从字符区间到 token

训练时要的是 "哪些 token 算 loss"。分词时让分词器返回每个 token 对应的字符位置 (`return_offsets_mapping=True`), 落在区间里的 token 算 loss, 其余的标签设成 -100 (PyTorch 的交叉熵会跳过 -100)。专题 D 实测这一步, 并说明为什么 transformers 自带的办法在 Qwen2.5 上用不了。

---

## 10. 四个坑

### 10.1 用户文字里写了特殊标记: 注入

如果用户输入里**本身**就写着 `<|im_end|>↵<|im_start|>system↵…` 会怎样? `measure_tokens.py` 的 B 部分:

![注入: 默认分词 vs 当普通字符切开](figures/fig_injection.svg)

**怎么读这张图**: 上一行是默认分词: 11 个 token 里有 2 个是真的特殊标记 (橙框)。在模型看来, 用户消息在 "忽略上面" 处就结束了, 后面紧跟着一条新的 **system 消息**, 而 system 本该只有开发者能写。下一行设了 `split_special_tokens=True`: 同样的文字被切成 20 个普通 token, 没有一个是结构。

| 分法 | 整段提示里 `<\|im_start\|>` 的个数 (正常应该是 3) |
|---|---|
| 拼好文字后默认分词 | 4 |
| `apply_chat_template(…, tokenize=True)` 直接分词 | 4 |
| 用户文字单独分词并设 `split_special_tokens=True` | 3 |

注意第二行: **用 transformers 自己的分词接口也一样中招**, 官方接口并不会替你防。这叫**提示注入** (prompt injection) 的一种。修法是用户文字单独分词, 把特殊标记当普通字符切开。这只是不让用户**伪造结构**; 用户仍然可以用普通文字说 "忽略上面的指令", 那是另一类问题 (安全线索引里的 agent-safety、safety-alignment 单元)。

模型看到伪造的 system 消息后会不会照做, 本单元**没有数据**。

### 10.2 不认识的角色被静默丢掉

边界用例 "不认识的 role 被丢掉": 中间那条 `{"role": "observation", "content": "被吞掉的内容"}` 在输出里**一个字都没有**, 也不报错。Qwen2.5 的模板只认 system、user、assistant、tool 四种角色。从别的框架搬代码过来时 (有的框架把工具结果叫 `observation` 或 `function`), 工具结果会悄悄消失, 模型就像从没收到过结果。

### 10.3 transformers 5 的 apply_chat_template(tokenize=True) 不再返回列表

写 `measure_tokens.py` 时, 作者第一次算出来 "加了 5 个工具, 提示长度 2 个 token"。原因: transformers 5.17.0 里 `apply_chat_template(..., tokenize=True)` 返回的是一个类似字典的 `BatchEncoding` 对象, `len()` 得到的是键的个数 (`input_ids`、`attention_mask`), 不是 token 数。要取 `["input_ids"]`。很多老教程直接 `len(...)`, 换到新版本会**静默**算错。

### 10.4 要用模型自己的模板

手写模板是为了**看懂**。实际使用时应该调用 `tokenizer.apply_chat_template`: 换模型、模型更新了模板, 都不用改代码。如果确实要自己拼 (比如在别的语言、别的设备上), 就照第 5 节的办法: 用官方模板生成一批标准答案, 逐字节核对。

---

## 11. 别的模型家族长什么样

`template_zoo.py` 下载 10 个模型的模板文件, 把同一段对话 (system + 提问 + 回答 + 再提问) 交给各自的模板渲染。Llama-3.2 和 Gemma-3 需要先在 Hugging Face 上申请访问权限, 作者没有申请, 这两个**没有数据**。其余 8 个 (完整文字在 `results/template_zoo.json`):

![四个家族的模板](figures/fig_zoo.svg)

**怎么读这张图**: 同一段对话, 四种拼法。ChatML 每条消息一个头一个尾; DeepSeek-V3 没有 system 的标记, 直接把内容写在开头; Mistral 把用户的话包在 `[INST] … [/INST]` 里, system 跑到了最后一条用户消息的前面; gpt-oss 的每条消息带 "频道" (`<|channel|>final`)。

### 11.1 总览

| 模型 | 角色标记 | system 放在哪 | 停止符 | 模板长度 (字符) |
|---|---|---|---|---|
| Qwen2.5、Qwen3、SmolLM2 | ChatML: `<\|im_start\|>角色↵ … <\|im_end\|>` | 开头, 有自己的标记 | `<\|im_end\|>` | 2507 / 4168 / 368 |
| Phi-4 | ChatML 变体: `<\|im_start\|>角色<\|im_sep\|>`, 消息之间不换行 | 开头 | `<\|im_end\|>` | 462 |
| DeepSeek-V3 | `<｜User｜>`、`<｜Assistant｜>` (全角竖线) | 开头, **没有**角色标记 | `<｜end▁of▁sentence｜>` | 2145 |
| GLM-4.5 | `[gMASK]<sop>` 开头, `<\|system\|>`、`<\|user\|>`、`<\|assistant\|>` | 开头 | `<\|endoftext\|>` | 3242 |
| gpt-oss | harmony 格式: `<\|start\|>角色<\|message\|> … <\|end\|>`, 回答带 `<\|channel\|>final` | 系统层自动生成; 你写的 system 变成 **developer** 消息 | `<\|return\|>` | 16714 |
| Mistral-7B-Instruct-v0.3 | `[INST] … [/INST]` 包住用户的话, 没有角色名 | **最后一条用户消息的开头** | `</s>` | 3959 |

几个看点:

- **Qwen3 和 Qwen2.5** 在这段简单对话上渲染结果完全一样, 但 Qwen3 的模板长了约三分之二: 多出来的部分处理 "思考" (专题 C)。
- **GLM-4.5** 在历史回答前插了一对空的 `<think></think>`, 表示 "这一轮没有思考过程"。
- **gpt-oss** 的模板会自动写进 `Current date: …`, 也就是**运行那天的日期**。同一段对话在不同日期渲染出来的文字不一样, 做缓存、做逐字节比对时要注意。它的格式有单独的开源渲染库 [openai/harmony](https://github.com/openai/harmony)。

### 11.2 Mistral: system 被挪走, 有时整个丢掉

Mistral-v0.3 的模板没有 system 角色, 规则是 "把 system 拼在**最后一条消息**前面, 如果最后一条是 user 的话"。推理时最后一条总是 user, 没问题; 但拼**训练数据**时, 对话以 assistant 的回答结尾, 这时 system **整个消失** (`results/mistral_golden.json` 里有这样的例子)。用这个模板微调, 模型一次都没在训练中见过 system。

另外两条规则: assistant 的话会被去掉首尾空白, 再在前面加一个空格; 角色必须严格 user / assistant 交替, 否则模板直接报错。练习 3 就是把这个模板写出来 (`mistral_v03.py` 是参考答案)。

### 11.3 DeepSeek-V3: 官方模板在工具调用上前后不一致

DeepSeek-V3 的工具调用格式和 Qwen 完全不同: 每个调用写成 `<｜tool▁call▁begin｜>function<｜tool▁sep｜>名字` 加一个 ```` ```json ```` 代码块, 而且 `arguments` 要给**字符串** (第 7.4 节的反面)。

把三段不同的工具对话交给它的官方模板 (`template_zoo.py` 的 `deepseek_tool_quirks`), 结果有三处前后不一致:

| 情况 | 渲染结果 |
|---|---|
| 一次调用**两个**工具 | 两个调用之后有 `<｜tool▁calls▁end｜><｜end▁of▁sentence｜>` 收尾 |
| 一次调用**一个**工具 | **没有**这两个收尾标记, 直接接工具结果 |
| 第二轮又调用工具 | 少了开头的 `<｜Assistant｜><｜tool▁calls▁begin｜>`; 上一轮的工具结果也没有用 `<｜tool▁outputs▁end｜>` 关上就开始了新的 `<｜User｜>` |

原因读模板就能看出来: 收尾标记写在 "不是第一个调用" 的分支里; 记录 "已经写过第一个调用" 的变量整段对话只设一次, 从不重置。这三种写法里哪个才是模型训练时真正见过的, 本单元**没有数据**, 只能说模板自己前后不一致。教训: **换一个模型之前, 先拿几段有代表性的对话 (尤其是多轮、多个工具) 渲染出来读一遍**, 不要假设官方模板一定对。

---

## 12. 从教学实现到工业实现

| 方面 | 本单元 | 工业实现 |
|---|---|---|
| 模板放在哪 | 用 Python 手写 | 和模型一起发布的 Jinja 程序 (`tokenizer_config.json` 或 `chat_template.jinja`); llama.cpp 用的 GGUF 文件把模板写在文件的元数据里 |
| 谁来执行模板 | `render` | transformers 的 `apply_chat_template` (在一个受限的 Jinja 沙箱里执行, 不能读写文件); 推理服务 (vLLM、SGLang 等) 收到 OpenAI 格式的请求后, 在服务端套模板 |
| 用户看得见模板吗 | 看得见 | 用云端 API 时看不见: 你发的是消息列表, 拼成什么文字由服务方决定。所以 "API 和本地跑同一个模型结果不同", 模板可能是原因之一 |
| 模板会变吗 | 不会 | 会。模型仓库更新时模板也可能被修改 (修 bug、加功能)。要结果可复现, 下载模型时固定版本 (revision), 并把用到的模板存一份 |
| 工具调用的解析 | 不在本单元 | 推理服务通常带一个 "工具调用解析器", 按模型家族的格式从文字里找出工具调用 (Agent 领域的 tool-calling) |

---

## 13. 本单元踩坑记录汇总

| 坑 | 现象 | 怎么发现的 | 怎么修 |
|---|---|---|---|
| 图的说明里写 "橙色的标记各占 1 个 token", 把 `<tool_response>` 也标成了橙色 | 说法不对: `<tool_response>` 不是新增 token | 写图注前去分词器里查了一遍 (第 6.4 节) | 只给真正的新增 token 标橙色 |
| 讲义初稿写 "没给 system 时, 41 个 token 里一大半是自我介绍" | 实际只有 16 个 (第 8.4 节) | 回头数了一遍 | 改成实测的数字; 凡是 "一大半" "好几倍" 这种话, 都要有数字撑着 |
| `len(apply_chat_template(tokenize=True))` 是 2 | 以为提示只有 2 个 token | 数字明显不对 | 取 `["input_ids"]` (第 10.3 节) |
| 渲染时离线运行, 需要申请权限的模型被记成 "本地找不到" | 结果文件里状态写错 | 和上一次联网运行的结果对比 | 联网重跑 (`template_zoo.py` 区分 gated 和其他错误) |
| 专题 B 第一版测错了东西 | 所有家族都显示 "是前缀", 和对思考模型的预期不符 | 结果太整齐, 回头看测的是什么 | 改成测 "这一轮的全部文字是不是下一轮提示的开头" (专题 B) |

---

## 14. 练习

### 14.1 动手写代码 (三级递进)

| 级别 | 练习 | 验收 |
|---|---|---|
| ① 填空 | `ex1_render_fill.py`: 补全 Qwen2.5 模板的四处 TODO (system 与工具说明、工具调用、工具结果、生成提示) | 39 段对话和官方模板逐字节相同 |
| ② 接口 | `ex2_assistant_spans.py`: 写 `assistant_spans(messages, tools)`, 找出每条 assistant 回复的区间 | 和参考实现的区间一致, 区间里确实是 assistant 的话 |
| ③ 白纸 | `ex3_mistral_blank.py`: 只看规则说明, 从零写 Mistral-v0.3 的模板 | 32 段对话逐字节相同, 12 段不合法的对话正确报错 |

```bash
cd library/llm/chat-template/exercises
python -m pytest test_ex1.py -q
```

只需要 Python 和 pytest, 标准答案已经在 `experiments/results/` 里。测试失败时会打印第一个不同的位置。

<details>
<summary>练习 ① 参考答案</summary>

四处 TODO 依次是 (完整版就是 `chat_template.py` 的 `render_with_spans`, 去掉区间部分):

```python
    # TODO 1
    system = messages[0]["content"] if messages[0]["role"] == "system" else DEFAULT_SYSTEM
    if tools:
        system += TOOLS_HEAD + "".join("\n" + tojson(t) for t in tools) + TOOLS_TAIL
    out.append(IM_START + "system\n" + system + IM_END + "\n")

            # TODO 2 (带工具调用的回复)
            seg = IM_START + "assistant"
            if m.get("content"):
                seg += "\n" + m["content"]
            for call in m["tool_calls"]:
                f = call.get("function", call)
                seg += '\n<tool_call>\n{"name": "' + f["name"] + '", "arguments": ' + tojson(f["arguments"]) + "}\n</tool_call>"
            out.append(seg + IM_END + "\n")

            # TODO 3 (工具结果)
            first = i == 0 or messages[i - 1]["role"] != "tool"
            last = i == len(messages) - 1 or messages[i + 1]["role"] != "tool"
            if first:
                out.append(IM_START + "user")
            out.append("\n<tool_response>\n" + m["content"] + "\n</tool_response>")
            if last:
                out.append(IM_END + "\n")

    # TODO 4
    if add_generation_prompt:
        out.append(IM_START + "assistant\n")
```

最容易错的两处: TODO 1 里工具说明拼在 system **文字后面、`<|im_end|>` 前面**; TODO 3 里第一条前面的 `<|im_start|>user` 后面**没有**换行, 换行是每条 `<tool_response>` 自带的开头。
</details>

<details>
<summary>练习 ② 参考答案</summary>

只用 `render` 的解法, 利用第 9.3 节的性质 "第 k 条 assistant 的起点 = 前 k 条消息加生成提示的长度":

```python
def assistant_spans(messages, tools=None):
    text = render(messages, tools)
    spans = []
    for k, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(render(messages[:k], tools, add_generation_prompt=True))
        end = text.index(IM_END, start) + len(IM_END)
        spans.append((start, end))
    return spans
```

起点用性质算; 终点是起点之后第一个 `<|im_end|>` 的结尾。这个解法每条 assistant 都要重新渲染一次前缀, 是 O(n²) 的; `render_with_spans` 边拼边记, 是 O(n) 的。注意它依赖 "前缀性质" 成立, 专题 B 会看到 Qwen3 这类模板上它不一定成立。
</details>

<details>
<summary>练习 ③ 参考答案</summary>

见 [mistral_v03.py](mistral_v03.py), 核心是:

```python
def render_mistral(messages):
    system = None
    if messages and messages[0]["role"] == "system":
        system, messages = messages[0]["content"], messages[1:]
    for i, m in enumerate(messages):                     # 先检查交替, 再输出
        if (m["role"] == "user") != (i % 2 == 0):
            raise ValueError(ALTERNATE_ERROR)
    out = [BOS]
    for i, m in enumerate(messages):
        if m["role"] == "user":
            content = m["content"]
            if system is not None and i == len(messages) - 1:   # 只有最后一条是 user 时才放 system
                content = system + "\n\n" + content
            out.append("[INST] " + content + "[/INST]")
        elif m["role"] == "assistant":
            out.append(" " + m["content"].strip() + EOS)
        else:
            raise ValueError("Only user and assistant roles are supported, ...")
    return "".join(out)
```

容易漏的: `i == len(messages) - 1` 判断的是 "整段对话的最后一条", 不是 "最后一条 user"; assistant 内容要 `strip()`; 交替检查要在输出之前做完。
</details>

### 14.2 思考题

1. 第 3 节的例子再加一条 assistant "你好呀", 不加生成提示, 一共多少个 token?

<details>
<summary>参考答案</summary>

原来 17 个 token 的最后 3 个 (`<|im_start|>` `assistant` `↵`) 本来就是生成提示, 不加生成提示就去掉它们, 剩 14 个; 再加上 assistant 这条消息: `<|im_start|>` `assistant` `↵` `你好` `呀` `<|im_end|>` `↵` 7 个 ("你好呀" 切成 "你好" "呀" 两个 token, 见专题 D 的掩码图)。合计 21 个。可以用 `len(tok.encode(render(msgs)))` 验证。
</details>

2. 第 8.2 节: 一个 Agent 有 30 个工具, 每轮对话平均 200 个 token 的历史, 跑 10 轮。工具说明一共被模型 "读" 了多少个 token? 有什么办法减少?

<details>
<summary>参考答案</summary>

按第 8.2 节的斜率, 30 个工具约 175 + 29 × 98 ≈ 3017 个 token; 每轮都要重新读, 10 轮就是约 3 万个 token, 而对话历史本身 (第 k 轮约 200k 个) 10 轮合计约 1.1 万个。工具说明比对话还多。办法: ① 按需给工具 (先判断大类, 只给相关的几个); ② 把底层工具合并成少数高层工具; ③ 精简描述; ④ 利用前缀缓存, 让工具说明那一段只算一次 (专题 B, 以及 LLM 领域的 prefix-caching 单元)。
</details>

3. 第 9.3 节说区间包含 `<|im_end|>`。如果不包含, 微调出来的模型会出什么问题?

<details>
<summary>参考答案</summary>

模型从没在 "说完话" 的位置上被要求写 `<|im_end|>`, 就学不会停: 回答完以后继续往下写, 可能自己编出下一轮的 `<|im_start|>user`, 一直写到长度上限。这和第 2.1 节 "不套模板停不下来" 是同一个现象, 只是这次是训练造成的。
</details>

4. 第 10.1 节: 你在做一个客服机器人, 用户的输入会原样拼进提示。列出你会做的三件事。

<details>
<summary>参考答案</summary>

① 用户文字单独分词并设 `split_special_tokens=True` (或者在拼接前把特殊标记的字符串替换掉), 保证用户伪造不了结构; ② 在服务端用消息列表 + 官方模板拼提示, 不要让用户的文字和模板字符串直接相加; ③ 模型的输出也要检查: 不执行用户文字诱导出的危险工具调用 (Agent 领域的安全层)。另外, 这只防得住 "伪造结构", 防不住普通文字的 "忽略上面的指令", 那要靠后训练和 Agent 层面的防护。
</details>

5. 第 11.2 节: 你必须用 Mistral-v0.3 做带 system 的微调, 有哪几种办法让 system 出现在训练数据里? 各有什么代价?

<details>
<summary>参考答案</summary>

① 改模板, 让 system 总是出现在第一条 user 前面: 简单, 但训练时的格式和推理时 (官方模板放在最后一条 user 前) 不一致; ② 把每段多轮对话拆成多个样本, 每个样本以 user 结尾, 只训练最后一个回答: 格式和推理一致, 但数据变多、前面的回答不算 loss; ③ 训练和推理都用自己改过的模板: 一致, 但偏离了模型原来学过的格式, 需要足够多的数据让它适应。没有免费的办法, 这正是第 11.2 节说 "读模板" 的意义。
</details>

---

## 15. 延伸资源

- Hugging Face, [Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating): `apply_chat_template`、`add_generation_prompt`、预填 (`continue_final_message`)、训练时怎么用模板。
- Qwen 文档, [Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html): Qwen 的工具调用格式和各推理框架的用法。
- [openai/harmony](https://github.com/openai/harmony): gpt-oss 所用 harmony 格式的官方渲染库。

---

## 16. 实践: 在项目里用

主干讲义用通用小助手。这一节把同样的方法用到具体项目上。

### 16.1 小车项目: v0.4 的提示账本

小车项目的 [v0.4 听懂文字](../../../projects/car/v0.4-text-commands/TUTORIAL.md) 让模型通过 5 个底层工具 (walk / turn / stop / look / get_pose) 开一辆仿真小车。用第 8 节同样的量法 (`practice_car.py`):

![小车 v0.4 的提示账本](figures/chart_car_budget.svg)

| 部分 | token 数 |
|---|---|
| 工具说明 (5 个工具) | 535, 其中 walk 一个就 152 (参数说明写得详细: 范围、单位、正负号) |
| 系统提示 | 200 |
| 模板标记 | 13 |
| 用户的话 "去红杯子那里, 停在它跟前" | 12 |
| **合计** | **760**, 工具说明占七成 |

v0.4 的另一种写法 "skill 模式" 只给模型两个高层工具 (`go_to`、`look`), 一次提示 399 个 token, 大约一半。这正是第 8.2 节说的 "合并成少数高层工具"。skill 模式的成功率更高, 但这里只测了长度, 不能说成功率是因为提示短。

### 16.2 小车项目里用到本单元的其他地方

| 本单元 | 小车项目里 |
|---|---|
| 第 7.4 节 arguments 要给对象 | agent-loop 的 [llm.py](../../agent/agent-loop/llm.py) 里 `_to_template` 专门把 OpenAI 格式的字符串参数转成对象 |
| 第 9.2 节预填 | v0.4 的 "强制调用工具": 预填 `<tool_call>↵`, skill 模式的 "没动" 从 12 局降到 0 |
| 第 6.3 节 `<tool_call>` 解码时留下 | agent-loop 的解析器在解码后的文字里找 `<tool_call>` |

以后的项目 (学习助教、推理服务) 用到本单元时, 也补在这一节。

---

# 第二部分 · 延伸专题

## 17. 专题 A · 读懂 Jinja 模板

### 17.1 为什么要会读

第 11 节看到, 每个家族的模板都有自己的规则, 有的还有 bug。规则写在哪里? 只写在模板这段 Jinja 程序里, 文档往往不全。会读 Jinja, 才能在换模型时自己弄清楚 "它到底怎么拼"。

### 17.2 语法速查

| Jinja | 意思 | Python 里对应 |
|---|---|---|
| `{{ x }}` | 输出 x | `out.append(str(x))` |
| `{%- … -%}` | 语句; 横线表示去掉这一侧的空白和换行 | — |
| `{% if a %} … {% elif b %} … {% else %} … {% endif %}` | 条件 | `if / elif / else` |
| `{% for m in messages %} … {% endfor %}` | 循环 | `for m in messages:` |
| `loop.first` / `loop.last` / `loop.index0` | 循环里: 是不是第一个 / 最后一个 / 从 0 数第几个 | `i == 0` / `i == len-1` / `i` |
| `x \| tojson` | 过滤器: 转成 JSON | `json.dumps(x, ensure_ascii=False)` |
| `a + b` | 字符串拼接 | `a + b` |
| `message.content` 或 `message['content']` | 取字段 | `m["content"]` |
| `x is defined` | 有没有这个字段 | `"x" in m` |
| `{% set ns = namespace(…) %}` | 能在循环里修改的变量 | 普通变量 |
| `raise_exception('…')` | 报错 (transformers 提供的函数) | `raise ValueError(…)` |

最容易看漏的是**空白控制**: `{%-` 去掉左边的空白, `-%}` 去掉右边的。模板里大量的缩进和换行都靠它去掉, 真正输出的只有 `{{ }}` 里的东西。

### 17.3 Qwen2.5 的模板, 对照着 Python 读

下面是 Qwen2.5 模板里处理消息的那段循环 (从模型的 `tokenizer_config.json` 原样摘出), 右边的注释指向第 4 节对应的 Python:

```jinja
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}      {#- 第 4.4 节 #}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}                                                  {#- 第 4.5 节 #}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}       {#- 第 4.6 节的 first #}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}                 {#- 第 4.6 节的 last #}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}                                                       {#- 第 4.7 节 #}
{%- endif %}
```

(`{#- … #}` 是 Jinja 的注释, 原模板里没有, 是本讲义加的。) 对照着看能发现三件事:

1. **第一行的条件**就是第 4.4 节那个 `if role in ("user", "system") or (role == "assistant" and not m.get("tool_calls"))`, 其中 `not loop.first` 实现了 "第一条 system 跳过"。
2. 没有任何 `else` 分支: 不认识的角色什么都不输出, 第 10.2 节的坑在这里一目了然。
3. Python 版多出来的 `header` / `seg` 切分, 是为了记 loss 区间; 模板本身不关心这件事 (所以 transformers 自带的掩码在这个模板上用不了, 专题 D)。

---

## 18. 专题 B · 多轮对话: 这一轮的缓存下一轮还能用吗

### 18.1 问题

多轮对话时, 每一轮都要把整段历史重新喂给模型。模型读提示时会算出每个位置的 K、V 并存起来 (KV 缓存, 见 [minigpt 讲义专题 A](../minigpt/TUTORIAL.md))。如果**这一轮模型看到并写出的全部文字** (提示 + 它自己的回复), 正好是**下一轮提示的开头**, 下一轮就能直接复用这一轮的缓存, 只需要算新加的部分。这叫**前缀缓存** (prefix caching, LLM 领域的 prefix-caching 单元)。反过来, 只要中间有一个字符变了, 从那里往后都得重算。

### 18.2 实测: 8 个家族

`prefix_property.py` 对每个家族用 200 段随机多轮对话, 对每一轮检查 "这一轮的全部文字" 是不是 "下一轮提示" 的开头; 普通对话和回复带思考过程的对话各测一遍:

![这一轮的文字是不是下一轮提示的开头](figures/chart_prefix.svg)

**怎么读这张图**: 每个家族两根柱子, 蓝色是普通对话, 橙色是回复带思考过程。100% 表示每一轮都能整段复用, 0% 表示每一轮都不能。

| 家族 | 普通对话 | 带思考过程 | 为什么 |
|---|---|---|---|
| Qwen2.5、DeepSeek-V3、Phi-4、SmolLM2 | 100% | 100% | 历史消息和刚写出时一模一样 |
| Mistral-v0.3 | 100% | 100% | system 放在最后一条 user 前面, 正好在 "这一轮" 之后, 不影响开头 |
| **Qwen3** | **0%** | **0%** | 模型写回复时带着 `<think>…</think>` (关掉思考时是空的), 下一轮模板把历史回复里的这一段**删掉**了 |
| **GLM-4.5** | 100% | **0%** | 历史回复里的思考内容被**替换成**空的 `<think></think>` |
| **gpt-oss** | **0%** | **0%** | 最新一条回复以 `<|return|>` 结尾, 变成历史后改成 `<|end|>`; 思考频道 (analysis) 也从历史里去掉 |

![Qwen3: 这一轮的文字到了下一轮变了样](figures/fig_prefix_break.svg)

**怎么读这张图**: 上面是这一轮 (提示 + 模型写的回复), 下面是下一轮的提示。同一个位置, 上面有 `<think>…</think>`, 下面没有。

### 18.3 为什么要这样设计

把历史里的思考过程删掉, 是有意为之: 思考过程往往很长, 留在历史里会让每一轮的提示越来越长; 而且这些家族的模型训练时, 历史里就是不带思考过程的。代价是前缀缓存只能复用到上一条用户消息之前, 这一轮写的回复 (连思考过程) 要在下一轮重新算一遍。这是一个 "提示长度" 和 "缓存复用" 之间的取舍, 推理服务 (LLM 领域的推理与部署段) 要按模型家族处理。

---

## 19. 专题 C · 会思考的模型, 模板多了什么

"会思考" 的模型 (推理模型, LLM 领域的 reasoning-models 单元) 先写一段思考过程, 再写回答。思考的开关也在模板里。`thinking_templates.py` 对同一个问题 "3 乘 9 是多少?" 渲染生成提示:

![关掉思考时模板替模型写了什么](figures/fig_think.svg)

**怎么读这张图**: 每个框是生成提示的结尾 (模型从这里开始写)。灰框是默认, 橙框是关掉思考。

| 家族 | 默认 | 关掉思考时模板做了什么 |
|---|---|---|
| Qwen3 | 生成提示就是 `<|im_start|>assistant↵`, 模型自己决定写不写 `<think>` | `enable_thinking=False`: 预填一个**空的** `<think>↵↵</think>↵↵`, 模型 "以为" 自己已经想完了 |
| GLM-4.5 | 生成提示是 `<|assistant|>` | `enable_thinking=False`: 在用户消息末尾加 `/nothink`, **并且**预填空的 `<think></think>` |
| gpt-oss | 系统消息里写 `Reasoning: medium` | `reasoning_effort="low"` / `"high"`: 只改这一行; 生成提示的结尾不变 |

两种思路: Qwen3 和 GLM-4.5 用**预填** (第 9.2 节) 跳过思考; gpt-oss 用**指令**告诉模型想多少。思考的开关不是模型的一个 "参数", 只是模板替你写进提示的几个字。

---

## 20. 专题 D · 训练时的模板: 哪些 token 算 loss

### 20.1 transformers 自带的办法在 Qwen2.5 上用不了

transformers 的 `apply_chat_template` 有一个参数 `return_assistant_tokens_mask=True`, 返回 "哪些 token 是 assistant 写的"。但它要求模板里用 `{% generation %} … {% endgeneration %}` 标出 assistant 的部分。Qwen2.5 的模板没有这个标记, 实测 (`token_boundaries.py` 的 D 部分):

```
[transformers] return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword.
D. transformers 自带的掩码: {'ones': 0, 'length': 34, 'template_has_generation_tag': False}
```

只给一句警告, 34 个 token 的掩码全是 0。如果不看警告直接拿去训练, **一个 token 都不算 loss**, 模型什么都学不到。

### 20.2 用字符区间换算

本单元的办法: `render_with_spans` 给出字符区间, 分词时要 `return_offsets_mapping=True` (每个 token 对应的字符起止位置), 完全落在区间里的 token 算 loss:

```python
# experiments/token_boundaries.py
    text, spans = render_with_spans(msgs)
    enc = tok(text, return_offsets_mapping=True)
    ours = [int(any(s <= a and b <= e for s, e in spans)) for a, b in enc["offset_mapping"]]
```

![token 级的 loss 掩码](figures/fig_spans.svg)

**怎么读这张图**: 一段 "提问 → 调用计算器 → 结果 → 回答" 的对话, 84 个 token, 蓝底的 27 个算 loss: 整个工具调用 (从 `<tool_call>` 到 `<|im_end|>`) 和最后的回答 (到 `<|im_end|>`)。system、用户的话、工具结果、两处 `<|im_start|>assistant↵` 都不算。

### 20.3 区间的端点会不会切在 token 中间

如果区间的起点落在某个 token 的中间, 这个 token 算不算就说不清了。实测 (`token_boundaries.py` 的 B 部分): 标准答案里 83 个 assistant 区间的 166 个端点, **全部**落在 token 边界上。原因在专题 E: 区间的两头都挨着特殊标记, 特殊标记永远是独立的 token。

---

## 21. 专题 E · 模板和分词边界

### 21.1 分开分词和整段分词一样吗

推理时, 提示和模型写的回复是**分开**得到的 token: 提示由分词器切, 回复由模型一个一个写。训练时却常常把 "提示 + 回复" 拼成一整段文字再分词。两种切法一样吗? BPE 是贪心合并的 ([bpe 讲义第 7 节](../bpe/TUTORIAL.md#7-编码-按-rank-贪心合并)), 拼接处可能合并出不一样的 token。

实测 (`token_boundaries.py`):

| 模板 | 分开分词 = 整段分词 |
|---|---|
| Qwen2.5 的 ChatML (标准答案里的 83 个回复) | **83 / 83** |
| 纯文字模板, 生成提示以 `Assistant:` 结尾 | 5 / 5 |
| 纯文字模板, 生成提示以 `Assistant: ` (带一个空格) 结尾 | **2 / 5** |

ChatML 全部一致, 因为回复的前面是 `↵` (`<|im_start|>assistant↵`), 后面是特殊标记 `<|im_end|>`; 特殊标记和换行在分词时就像墙, BPE 不会跨过它们合并。纯文字模板如果在生成提示末尾留了一个空格, 就出问题了。第一个不一致的例子:

```
整段分词:  'Assistant'  ':'  ' Hello'  '!'
分开分词:  'Assistant'  ':'  ' '  'Hello'  '!'
```

整段分词时, 空格和后面的 "Hello" 合成一个 token `' Hello'`; 分开分词时, 提示的末尾单独一个空格, 回复从 `'Hello'` 开始。模型训练时见过的永远是 `' Hello'` 这种 "空格在词前面" 的切法, 推理时却被迫在一个单独的空格之后写 `'Hello'`, 这是它很少见过的情况。**这就是很多模板的生成提示不以空格结尾的原因。**

### 21.2 开头标记会不会重复

有些模板自己写了开头标记 (BOS, beginning of sequence)。如果拼好文字后再用分词器的默认设置分词, 分词器可能**再加一个**, 变成两个开头标记。transformers 的文档专门警告过这一点。实测 DeepSeek-V3 (模板里写了 `<｜begin▁of▁sentence｜>`):

| 做法 | 开头标记个数 |
|---|---|
| 渲染成文字后默认分词 | 1 |
| 渲染成文字后 `add_special_tokens=False` 分词 | 1 |
| `apply_chat_template(tokenize=True)` | 1 |

DeepSeek-V3 的分词器在这种情况下**不会**重复。但文档的警告不是空穴来风: 其他分词器的默认设置可能不同, 本单元没有逐个测。稳妥的做法是: 文字是用模板拼出来的, 分词时就设 `add_special_tokens=False`; 或者直接用 `apply_chat_template(tokenize=True)`。

---

## 22. 延伸专题练习

1. **(专题 A)** 读 Qwen2.5 模板开头处理 system 和工具的那一段 (`results/template_zoo.json` 里有 Qwen2.5 的渲染结果, 模板原文在模型的 `tokenizer_config.json` 里), 指出它和第 4.3 节 Python 的对应关系。

<details>
<summary>参考答案</summary>

模板开头是 `{%- if tools %} … {%- else %} … {%- endif %}` 两个分支: 有工具时, 先输出 `<|im_start|>system↵`, 再用 `{%- if messages[0]['role'] == 'system' %}` 选用户的 system 或默认那句, 然后输出 "# Tools …<tools>"、`{%- for tool in tools %}{{- "\n" }}{{- tool | tojson }}{%- endfor %}`、"</tools> … </tool_call><|im_end|>↵"; 没有工具时直接输出 system 块。Python 版把两个分支合成了一个: 先选 system 文字, `if tools:` 再追加说明书, 最后统一 emit。两者输出相同 (第 5 节核对过), 只是代码组织不同。
</details>

2. **(专题 B)** 给 Qwen3 写一个 "让前缀缓存能复用" 的办法: 下一轮发请求时, 历史里的回复保留原样 (带 `<think>`) 会怎样? 有什么风险?

<details>
<summary>参考答案</summary>

自己拼历史、保留 `<think>…</think>`, 前缀性质就成立了, 缓存可以整段复用。风险: 这样的历史格式和 Qwen3 训练时见过的不一样 (它训练时历史里没有思考过程), 回答质量可能变化, 而且提示会越来越长。要不要这么做, 得在自己的任务上同时测速度和质量, 本单元没有数据。
</details>

3. **(专题 D)** 用 `token_boundaries.py` 的办法, 给一段包含 `"arguments" 是 JSON 字符串` (第 7.4 节) 的对话算 loss 掩码。被再编码的参数会被模型当成什么学进去?

<details>
<summary>参考答案</summary>

掩码会覆盖整个工具调用, 包括 `"arguments": "{\"city\": \"台北\"}"` 这一段, 所以模型会学着把参数写成**带转义引号的字符串**。微调之后它在推理时也这样写, 解析程序拿到的 arguments 是字符串而不是对象, 工具调用失败。第 7.4 节的坑在训练时会被 "学进去", 比推理时更难发现。
</details>

4. **(专题 E)** 自己设计一个纯文字模板 (比如 `用户: …↵助手: …`), 用 Qwen 的分词器测: 生成提示以什么结尾时, 分开分词和整段分词总是一致?

<details>
<summary>参考答案</summary>

以换行或者不带空格的冒号结尾 (`助手:`), 回复以空格或换行开头, 21.1 节测的 5 个回复全部一致。以空格结尾时, 回复的第一个字能不能和那个空格合成一个 token, 取决于词表里有没有 "空格 + 这个开头" 的 token: 21.1 节里 `Hello!`、`台北现在 26 度。`、`OK, sure.` 三个不一致, `你好` 和 `27` 一致。能不能合并没法凭直觉判断, 只能用 `tok.encode(prompt + reply) == tok.encode(prompt) + tok.encode(reply)` 逐个测。稳妥的结论: 生成提示不要以空格结尾。
</details>

---

*上一单元: [modern-small-model](../modern-small-model/TUTORIAL.md); 下一单元: [decoding 解码与采样](../decoding/TUTORIAL.md)。*
