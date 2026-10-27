# Agent 循环讲义 — 大模型怎么 "动手"

> 知识库 · Agent 领域的第一个单元 (总目录: [library/README.md](../../README.md))。
> 用到它的主线站: [v0.4 听懂文字](../../../mainline/v0.4-text-commands/TUTORIAL.md) (在仿真小车上实测本单元的每一种写法)。
> 配套代码: [demo_minimal.py](demo_minimal.py) (不用模型的最小演示)、[agent.py](agent.py) (循环本体)、[tools.py](tools.py) (工具说明书与安全层)、[llm.py](llm.py) (本地 / 云端 / 剧本三种模型后端)。

---

## 目录

0. [阅读指南](#0-阅读指南)
1. [全景: Agent 就是一个循环](#1-全景-agent-就是一个循环)
2. [对话接口: messages 与四种角色](#2-对话接口-messages-与四种角色)
3. [工具调用: 模型怎么 "说出" 一个动作](#3-工具调用-模型怎么-说出-一个动作)
4. [最小例子: 不用模型, 看一遍完整的循环](#4-最小例子-不用模型-看一遍完整的循环)
5. [循环的四种写法: chat / plan / loop / step](#5-循环的四种写法-chat--plan--loop--step)
6. [安全层: 模型说要做的事, 不等于应该去做](#6-安全层-模型说要做的事-不等于应该去做)
7. [换一套工具: 高层技能与强制调用](#7-换一套工具-高层技能与强制调用)
8. [怎么评测一个 Agent](#8-怎么评测一个-agent)
9. [在小车上实测的结果 (摘要)](#9-在小车上实测的结果-摘要)
10. [踩坑记录](#10-踩坑记录)
11. [练习](#11-练习)
12. [延伸](#12-延伸)

---

## 0. 阅读指南

### 0.1 这个单元讲什么

"Agent (智能体)" 听起来很玄, 其实就是一个**循环**: 大模型说要做什么 → 程序去做 → 把结果告诉大模型 → 大模型再决定下一步。本单元把这个循环拆开, 每一行代码都看得见: 消息怎么组织、模型怎么表达 "我要调用工具"、程序怎么解析和执行、怎么防止模型乱来、怎么判断它到底做得好不好。

例子始终是同一个: 让大模型开一台**仿真小车** (仓库根目录的 [sim/car2d.py](../../../sim/car2d.py)) 去 "红杯子跟前"。小车只是一个看得见结果的载体, 学到的东西对任何 Agent 都适用。

### 0.2 前置

| 需要 | 到什么程度 |
|---|---|
| Python | 能读懂函数、字典、列表、`for` 循环 |
| JSON | 认识 `{"name": "walk", "arguments": {"distance_m": 0.5}}` 这种键值对写法 |

**不需要**: 深度学习、PyTorch、显卡。本单元的演示和练习**只用 Python 标准库 + pytest**, 用的是假模型, 不下载任何东西。

想知道大模型内部是怎么 "接着往下写字" 的, 可以读 LLM 领域的 [分词 (BPE)](../../llm/bpe/TUTORIAL.md) 和 [miniGPT](../../llm/minigpt/TUTORIAL.md); 但读本单元**不需要**先读它们, 这里只把模型当成 "给一段对话、回一段文字" 的黑盒。

### 0.3 怎么运行

```bash
pip install pytest                      # 只有做练习的测试需要
cd library/agent/agent-loop
python demo_minimal.py --seed 1         # 第 4 节的最小例子, 不到一秒
cd exercises && python -m pytest test_ex1.py -q     # 第 11 节的练习
```

想换成**真的大模型** (本地 Qwen 或云端 API) 跑, 环境配置、模型下载、30 个任务的评测都在主线 [v0.4 这一站](../../../mainline/v0.4-text-commands/TUTORIAL.md)。

---

## 1. 全景: Agent 就是一个循环

![Agent 循环](figures/fig_loop.svg)

[agent.py](agent.py) 的 `run_episode` 就是图上这个循环:

1. 把 `[系统提示, 用户指令, 之前所有的动作和观测]` 交给大模型;
2. 模型要么发出**工具调用** (`walk` / `turn` / `look` ……), 要么只说话;
3. 只说话 → 结束; 有工具调用 → 先过**安全层** → 交给小车执行 → 把结果作为新消息追加进对话 → 回到 1。

三个要点, 后面每一节都在展开它们:

- **模型从来不直接动手**。它只输出一段固定格式的文字, 真正执行的是我们的程序 (第 3 节)。
- **模型不知道结果, 除非你告诉它**。动作做成了什么样, 全靠循环把观测追加回对话 (第 5 节)。
- **模型说的不一定对**。执行前要检查 (第 6 节), 结束后要用独立的判据验收 (第 8 节)。

---

## 2. 对话接口: messages 与四种角色

和大模型的对话是一个**列表**, 每一项是一条消息, 有角色 (`role`) 和内容 (`content`)。本仓库统一用 OpenAI 的格式 (几乎所有厂商和开源模型都兼容):

```python
messages = [
    {"role": "system",    "content": "你在控制一台两轮小车 ……"},        # 说明书: 你是谁、能做什么、怎样算完成
    {"role": "user",      "content": "去红杯子那里, 停在它跟前"},         # 用户说的话
    {"role": "assistant", "content": "", "tool_calls": [                  # 模型的回复: 这里是一个工具调用
        {"id": "call_1", "type": "function",
         "function": {"name": "look", "arguments": "{}"}}]},
    {"role": "tool",      "tool_call_id": "call_1",                       # 工具的执行结果 (观测)
     "content": "{\"ok\": true, \"objects\": [{\"name\": \"red_cup\", \"distance_m\": 0.55, \"bearing_deg\": 24.4}]}"},
]
```

| 角色 | 谁写的 | 作用 |
|---|---|---|
| `system` | 开发者 | 规则和背景: 场地多大、有哪些目标、成功标准 ([agent.py](agent.py) 的 `SYSTEM_PROMPT`) |
| `user` | 用户 | 任务本身 |
| `assistant` | 模型 | 要么是一段话, 要么带着 `tool_calls` |
| `tool` | 程序 | 工具执行的结果, 用 `tool_call_id` 指明是回复哪一个调用 |

**模型是无状态的**: 每次调用都要把整个列表重新发一遍, 它才 "记得" 之前发生了什么。所以对话越长, 每次输入的 token 越多, 也就越慢、越贵 (token 是模型处理文字的单位, 大约一个汉字一个)。

---

## 3. 工具调用: 模型怎么 "说出" 一个动作

### 3.1 先给模型一份说明书

模型怎么知道有哪些工具、参数叫什么? 靠一份 **JSON Schema** 写的说明书, 随对话一起发过去。[tools.py](tools.py) 里 `walk` 的说明书:

```python
{"type": "function", "function": {
    "name": "walk",
    "description": "沿车头方向直走一段距离, 负数表示后退。一次最多走 1.0 米。撞到东西会自动停下并报告。",
    "parameters": {"type": "object", "properties": {
        "distance_m": {"type": "number", "description": "距离, 单位米, 范围 -1.0 到 1.0"},
        "speed": {"type": "number", "description": "速度, 单位米/秒, 可选, 默认 0.3, 最大 0.5"}},
        "required": ["distance_m"]}}}
```

`description` 是写给模型看的, 写得越清楚 (单位、范围、副作用), 模型越不容易用错。

### 3.2 模型怎么回: 两种常见格式

**云端 API (OpenAI 兼容)**: 返回结构化的字段, 不用自己解析:

```python
msg = resp.choices[0].message
msg.tool_calls[0].id                  # "call_abc123"
msg.tool_calls[0].function.name       # "walk"
msg.tool_calls[0].function.arguments  # '{"distance_m": 0.5}'   ← 注意: 是字符串, 要自己 json.loads
```

**本地开源模型**: 模型只会输出文字, 工具调用是按聊天模板约定的**特殊格式文字**。Qwen2.5 的格式 (作者实际拿到的原始输出):

```
<tool_call>
{"name": "go_to", "arguments": {"target": "blue_ball"}}
</tool_call>
```

程序要自己从文字里把它找出来、解析成 JSON ([llm.py](llm.py) 的 `parse_qwen_output`)。这一步是小模型最容易出问题的地方: 忘了标签、JSON 写坏、干脆写成一句普通的话 (第 10 节)。

### 3.3 回复时要对上号

每个工具调用都有一个 `id`。执行完, 回复的 `tool` 消息必须带上同一个 `tool_call_id`。**一轮里有几个调用, 就要有几条 `tool` 回复**, 哪怕某个调用你决定不执行, 也要回一条说明 "没执行, 原因是 ……", 否则云端 API 会直接报错。

---

## 4. 最小例子: 不用模型, 看一遍完整的循环

[demo_minimal.py](demo_minimal.py) 用一个**假的大模型** (`ToyBrain`, 十几行 if-else, 只看最近一次 `look` 的结果) 开车, 但走的是和真模型**完全一样**的循环 (`run_episode`)。运行:

```bash
python demo_minimal.py --seed 1
```

作者实际得到的输出 (系统提示截断):

```
场景 seed=1: 任务 "去红杯子那里, 停在它跟前"
  [system]     你在控制一台两轮小车, 它在一个 2 米 × 2 米、四周有墙的场地里。……
  [user]       去红杯子那里, 停在它跟前
  [assistant]  发出工具调用: look({})
  [tool]       小车返回: {"ok": true, "objects": [{"name": "red_cup", "distance_m": 0.55, "bearing_deg": 24.4}]}
  [assistant]  发出工具调用: turn({"angle_deg": 24.4})
  [tool]       小车返回: {"ok": true}
  [assistant]  发出工具调用: look({})
  [tool]       小车返回: {"ok": true, "objects": [{"name": "red_cup", "distance_m": 0.49, "bearing_deg": -2.8}]}
  [assistant]  发出工具调用: walk({"distance_m": 0.37})
  [tool]       小车返回: {"ok": true, "moved_m": 0.351}
  [assistant]  发出工具调用: look({})
  [tool]       小车返回: {"ok": true, "objects": [{"name": "red_cup", "distance_m": 0.17, "bearing_deg": -3.6}]}
  [assistant]  发出工具调用: turn({"angle_deg": -3.6})
  [tool]       小车返回: {"ok": true}
  [assistant]  发出工具调用: look({})
  [tool]       小车返回: {"ok": true, "objects": [{"name": "red_cup", "distance_m": 0.17, "bearing_deg": 1.3}]}
  [assistant]  发出工具调用: stop({})
  [tool]       小车返回: {"ok": true}
  [assistant]  已经停在目标跟前了。

结果: 成功  (离目标 0.17 m, 碰撞 0 次, 工具调用 8 次, 问了模型 9 轮)
```

逐行读:

1. 对话从 `system` (说明书) 和 `user` (任务) 两条开始。
2. **每一轮**: `assistant` 发出一个调用 → 小车执行 → `tool` 消息把结果放回对话。
3. `look` 说红杯子 "偏左 24.4 度" (`bearing_deg` 左正右负), 于是下一步 `turn(24.4)` 对准; 再看一眼确认 (-2.8 度, 基本对准) 才 `walk`。
4. `walk(0.37)` 实际只走了 0.351 —— 小车有误差, 所以走完要再看, 不能一口气走到底。
5. 最后一轮 `assistant` **只说话、不调用工具**, 循环就结束了。

作者在场景 seed 0~9 上跑过 `--quiet`: 10 个里 9 个成功 (seed 2 失败: 原地转了一圈没找到目标)。

**真的大模型做的事, 就是把 `ToyBrain` 换掉**: 同样的 `messages` 进去, 同样格式的工具调用出来。下面几节讨论的都是: 换成真模型以后, 循环该怎么写才稳。

---

## 5. 循环的四种写法: chat / plan / loop / step

![五种模式](figures/fig_modes.svg)

同一个 `run_episode`, 用 `mode` 参数切换, 每种都只有十来行。它们是一组**对照**: 每去掉或改动循环的一段, 就能看出那一段到底有什么用。

| 模式 | 做法 | 用来说明什么 |
|---|---|---|
| `chat` | 不给工具, 只问一次 | 没有工具, 模型的话说得再对, 车也不会动: "说" 和 "做" 是两件事 |
| `plan` | 先 `look` 一次告诉模型, 要求它**一次写出全部动作**, 照单执行, 中途不再问 | **开环**: 动作有误差、环境有墙, 走偏了也没人告诉它 |
| `loop` | 标准循环: 每轮的结果都喂回去 | **闭环**。但模型可以一轮发好几个调用, 它们会全部执行, 中间不看结果 |
| `step` | 和 loop 一样, 但**每轮只执行第一个调用**, 其余的退回并说明原因 | 强制 "做一步、看一眼" |

循环的核心 (loop / step 共用, 摘自 [agent.py](agent.py)):

```python
for k_turn in range(max_turns):
    reply = llm.chat(messages, tools=tools)
    if not reply.tool_calls:                   # 模型不再调用工具: 它认为做完了 (或放弃了)
        rec["ended_by"] = "answered"
        return rec
    messages.append({"role": "assistant", "content": reply.content, "tool_calls": [...]})
    for k, c in enumerate(reply.tool_calls):
        if mode == "step" and k > 0:           # 一步一停: 只执行第一个, 其余退回 (但每个 id 都要回复)
            obs = {"ok": False, "error": "未执行: 每轮只执行第一个工具调用。请先看上一个工具的结果, 再决定下一步"}
        else:
            obs = run_tool(c["name"], c["arguments"])        # 里面先过安全层, 再交给小车
        messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(obs, ensure_ascii=False)})
rec["ended_by"] = "max_turns"                  # 轮数用完还没结束
```

注意结束条件: **模型一次不调用工具, 循环就结束**。这意味着模型如果说了一句 "好的, 我先走过去" 却没有附带调用, 任务就到此为止了 —— 这是小模型很常见的失败 (第 9 节)。

---

## 6. 安全层: 模型说要做的事, 不等于应该去做

模型会出错: 工具名拼错、少参数、编出不存在的参数、要求一口气走 10 米。[tools.py](tools.py) 的 `validate_call` 在执行前检查, **不合格就不执行**, 把原因作为观测还给模型, 让它改了再试:

| 检查 | 例子 | 返回给模型的话 |
|---|---|---|
| 工具名 | `fly()` | 没有叫 'fly' 的工具, 可用的工具: … |
| JSON 能解析 | `{bad` | 参数不是合法的 JSON |
| 多余参数 | `look(direction="right")` | look 不接受参数 ['direction'], 只接受 无参数 |
| 缺必需参数 | `walk()` | walk 缺少必需的参数 'distance_m' |
| 数字合法 | `"0.5m"`、`"nan"`、`true` | 只给数字, 不要带单位 |
| 范围 | `walk(10)` | 一次最多走 1.0 米 …… 请分几段走, 每走一段先 look 确认 |

两条设计原则:

1. **拦下时要说人话**。错误信息就是模型的下一条观测, 写清楚 "错在哪、该怎么改", 模型才有机会自己修正。
2. **分清两层**。速度上限 (0.5 m/s) 写在小车**固件**里, 任何上层都绕不过去; 安全层管的是 **Agent 层的策略**, 比如 "一次最多走 1 m, 走完先看一眼"。前者保证不出事, 后者让模型更容易做对。

一个容易漏的细节: `float("nan")` 不会报错, 而 `abs(nan) > 1.0` 是 `False`, 所以 `walk("nan")` 会**溜过范围检查**。数字检查必须显式拒绝 nan 和 inf (第 10 节)。

> 练习 1 就是自己补全这个安全层。

---

## 7. 换一套工具: 高层技能与强制调用

### 7.1 工具的粒度决定了模型要做多难的事

给模型五个底层动作 (`walk` / `turn` / `look` ……), 它就得自己规划每一步: 先转着找、对准、走一段、再确认。这对小模型很难 (第 9 节的实测: 几乎全部失败)。

另一种分工: 只给它一个**高层技能** `go_to(target)`, "怎么开过去" 交给一段不用大模型的控制代码。模型只需要做它擅长的事 —— 听懂 "去红杯子那里" 是要去哪个目标、最后汇报结果。

`run_episode` 用 `toolset` 参数换工具集, 循环本身一行不改:

```python
toolset = {"schemas": [go_to 的说明书, look 的说明书],
           "system": "你有一个 go_to 工具, 它会自动把车开到指定目标跟前 ……",
           "execute": lambda car, name, args: ...}      # 调用 go_to 时, 跑控制代码
run_episode(llm, car, task, mode="skill", toolset=toolset)
```

小车上的 `go_to` 实现在主线 [v0.4 的 skills.py](../../../mainline/v0.4-text-commands/skills.py)。这个思路在真实系统里很常见: 大模型负责 "选做什么", 专门的控制器 (规则、传统算法, 或者强化学习训练出的策略) 负责 "怎么做"。

### 7.2 强制调用工具

小模型有时明明 "知道" 要调用工具, 却把它写成了普通的话 (例如回复一句 `go_to blue_ball`, 没有 `<tool_call>` 标签), 程序解析不出调用, 循环就结束了。两种强制手段:

| 后端 | 怎么强制 |
|---|---|
| 云端 API | 请求里加 `tool_choice="required"`, 服务端保证返回工具调用 |
| 本地模型 | **预填**: 在模型回复的开头替它写好 `<tool_call>` 加换行, 它只能接着写 JSON |

[llm.py](llm.py) 的 `chat(..., require_tool=True)` 两种都实现了; `run_episode(..., force_first_tool=True)` 只在第一轮强制 (之后模型要能自由地用一句话结束)。

---

## 8. 怎么评测一个 Agent

### 8.1 "跑通一次" 不可信

只看一局, 结论可能完全是碰巧。作者第一次用 1.5B 模型跑小车, 第一局就成功了, 事后看日志才发现它根本没 `look`, 只是起点恰好朝着目标。所以:

- **固定一批任务** (评测集), 每次比较都在同一批上跑。小车用 30 个随机场景 (seed 0~29), 目标和指令句式轮换。
- **本地模型用贪心解码** (temperature=0), 同样的输入永远得到同样的输出, 结果可以复现。
- **先用 "什么都不做" 跑一遍评测集**, 成功率必须是 0。否则说明任务设计有漏洞 (作者就遇到过: 有一局车一开始就在目标跟前)。

### 8.2 用独立的判据判成功, 不信模型的汇报

模型会报告 "成功到达", 实际没到 (第 9 节的实测里出现过)。成功必须用**模型不知道的真实状态**判: 仿真器知道车的真实位置, `reached()` 用它判; 真车上没有上帝视角, 就要用独立的传感器复核。

### 8.3 不只看成功率, 还要看怎么失败的

成功率只告诉你 "不行", 不告诉你 "为什么不行"。把每一局归到一类结局, 才知道该改哪里:

| 结局 | 含义 | 通常说明 |
|---|---|---|
| 没动 | 一个工具都没执行 | 模型没按格式调用, 或者只说不做 |
| 去错目标 | 停在了别的目标跟前 | 没听懂, 或认错了物体 |
| 撞了 | 有碰撞 | 盲走、不看结果 |
| 没到却说到了 | 没撞、没到, 却声称完成 | **最危险**, 用户会信 |
| 轮数用完 | 一直在动, 没结束 | 在原地打转 |
| 没到, 停了 | 自己停了, 也没声称成功 | 放弃了, 或控制器能力不够 |

分类本身也是代码, **也会错**: 作者的分类器一开始把模型诚实的 "没能直接到达红杯子" 也算成了谎报 (第 10 节)。统计出来以后, 一定要抽几条原始记录对一对。

### 8.4 样本小的时候, 差几个百分点不说明问题

同一个规则控制器, 在 30 个任务上成功率 70%, 在 300 个任务上是 56%。30 个任务的评测, 差 10 个百分点可能只是运气。比较两种方法时, 差距要大, 或者任务要多。

---

## 9. 在小车上实测的结果 (摘要)

完整数据、图和复现方法在主线 [v0.4 这一站](../../../mainline/v0.4-text-commands/TUTORIAL.md)。这里只摘和本单元概念直接相关的几条 (Qwen2.5-1.5B-Instruct, 每组 30 个任务, 作者的 RTX 5060 8GB 实测):

| 写法 | 成功 | 说明了什么 |
|---|---|---|
| chat | 0/30 | 没有工具, 车不动 |
| plan | 0/30, 27 局撞了 | 开环: 走偏了不知道 |
| loop | 0/30, 15 局撞了, 4 局 "没到却说到了" | 闭环也救不了: 模型几乎不会转着找目标 (30 局只有 6 局看到过目标), 还会一轮发十几个调用盲走 |
| step | 0/30, 1 局撞了 | 强制 "做一步看一眼" 消除了盲走, 但模型被退回调用后多半直接结束 |
| skill (只给 go_to) | 14/30, 0 碰撞 | 换一种分工, 小模型就能完成大部分任务 |
| skill + 强制调用 | 21/30, 0 碰撞 | 剩下的 "没动" 全是格式问题, 强制调用后消失 |

---

## 10. 踩坑记录

1. **模型把工具调用写成一行 JSON, 忘了 `<tool_call>` 标签** (小车实测: 1.5B 的 loop 里占调用的 13%)。只按标签解析的话, 这些调用被当成普通文字悄悄丢掉: 模型以为自己 `look` 过了, 其实没有。解决: 也认裸 JSON, 但单独标记、单独统计 (`parse_qwen_output` 的 `bare=True`)。
2. **`tool_call_id` 必须一一对应**: 一轮几个调用就回几条, 没执行的也要回一条错误观测, 否则云端 API 报错。
3. **`float("nan")` 能溜过范围检查**: `abs(nan) > 1.0` 是 False。数字检查要显式拒绝 nan 和 inf。
4. **API 返回的 `arguments` 是字符串**, 不是字典, 要 `json.loads`; 本地模型的聊天模板反过来要求字典 (`LocalLLM._to_template` 做了转换)。
5. **评测分类器的正则太窄**: "没能**直接**到达" 没被认成否定, 诚实的汇报被算成谎报, 数字被夸大。修正后加了回归测试。教训: 统计数字出来后, 抽原始记录核对。

---

## 11. 练习

在 `exercises/` 里, 先看文件头的说明。**只需要 Python + pytest, 不需要模型和显卡** (测试用剧本假模型 `ScriptedLLM`)。

| 级别 | 文件 | 做什么 | 验收 |
|---|---|---|---|
| ① 填空 | [ex1_validate_fill.py](exercises/ex1_validate_fill.py) | 补全安全层 `validate_call` 的四处 TODO | 27 条调用上和参考实现判断一致 (`test_ex1.py`) |
| ② 接口 | [ex2_step_interface.py](exercises/ex2_step_interface.py) | 从签名写出 "一步一停" 的 Agent 循环 | 剧本模型驱动下, 对话与小车动作和参考实现一致 (`test_ex2.py`) |

第三级练习 (自己写 `go_to` 控制代码) 在主线 [v0.4 这一站](../../../mainline/v0.4-text-commands/TUTORIAL.md), 因为它要在小车的 300 个场景上验收。

```bash
cd library/agent/agent-loop/exercises
python -m pytest test_ex1.py -q
```

刚拿到时测试会失败 (`NotImplementedError`), 这是正常的。写完再去对照 [tools.py](tools.py) 和 [agent.py](agent.py)。

---

## 12. 延伸

- **工业界的 Agent 框架**: 选修 [agent-frameworks](../../../electives/agent-frameworks/README.md) 把 LangChain / LangGraph、OpenAI Agents SDK、smolagents、Google ADK、CrewAI 和 MCP 协议, 和本单元的每个部件 (工具说明书、循环、安全层、记忆、记录) 一一对照。
- **用强化学习替换 go_to 里的规则**: 知识库 RL 领域 ([library/rl](../../rl/README.md)), 从 [老虎机](../../rl/bandit/TUTORIAL.md) 开始。
- **让小模型更会调用工具**: 知识库后训练领域 ([library/post-training](../../post-training/README.md))。
