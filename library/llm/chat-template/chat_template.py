"""
chat_template.py — 手写 Qwen2.5 的对话模板 (只用标准库)

大模型只会"接着一串 token 往下写"。对话模板把一段对话 (谁说了什么、有哪些工具、工具返回了什么)
拼成一整段文字, 模型才知道哪段是系统设定、哪段是用户、该轮到谁说话。

本文件逐行复刻 Qwen2.5-Instruct 自带的 Jinja 模板 (tokenizer_config.json 里的 chat_template),
experiments/check_against_hf.py 在 500 段随机对话上核对过: 输出和 transformers 的
tokenizer.apply_chat_template(..., tokenize=False) 逐字节相同。

    render(messages, tools=None, add_generation_prompt=False) -> str
    render_with_spans(...) -> (str, [(start, end), ...])   额外返回"模型自己说的话"所在的字符区间

消息格式和 OpenAI 一致:
    {"role": "system" | "user" | "assistant" | "tool", "content": str}
    assistant 可以带 "tool_calls": [{"type": "function", "function": {"name": ..., "arguments": {...}}}]
"""

import json

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# 有工具时, 模板在 system 后面追加的说明书 (原文照抄, 一个字符都不能差)
TOOLS_HEAD = ("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
              "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
TOOLS_TAIL = ("\n</tools>\n\nFor each function call, return a json object with function name and arguments "
              "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
              "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")


def tojson(x):
    """Jinja 模板里的 `| tojson`。transformers 的实现是 json.dumps(x, ensure_ascii=False): 中文原样输出, 不转成 \\uXXXX。"""
    return json.dumps(x, ensure_ascii=False)


def render(messages, tools=None, add_generation_prompt=False):
    """把对话拼成模型看到的那一整段文字。"""
    return render_with_spans(messages, tools, add_generation_prompt)[0]


def render_with_spans(messages, tools=None, add_generation_prompt=False):
    """
    同 render, 另外返回每条 assistant 回复在文字里的区间 [start, end)。

    区间从 "<|im_start|>assistant\\n" 之后开始, 到 "<|im_end|>" 为止 (含 <|im_end|>):
    正好是模型在 add_generation_prompt 之后要自己写出来的那部分。微调时只在这些位置算 loss
    (见知识库 post-training/tool-finetune)。
    """
    if not messages:
        raise ValueError("messages 不能为空 (官方模板会去读 messages[0])")

    out = []        # 依次拼接的片段
    spans = []
    length = 0      # 已拼接的总长度, 用来记区间

    def emit(s, trainable=False):
        nonlocal length
        if trainable:
            spans.append((length, length + len(s)))
        out.append(s)
        length += len(s)

    # 1) 开头的 system: 用户给了就用用户的, 没给就用默认的那句 "You are Qwen ..."
    if messages[0]["role"] == "system":
        system = messages[0]["content"]
    else:
        system = DEFAULT_SYSTEM
    if tools:                                  # 空列表 [] 和 None 一样, 不加工具说明
        system += TOOLS_HEAD + "".join("\n" + tojson(t) for t in tools) + TOOLS_TAIL
    emit(IM_START + "system\n" + system + IM_END + "\n")

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
        elif role == "tool":
            # 连续几条工具结果合在同一个 user 块里, 每条包一层 <tool_response>
            first = i == 0 or messages[i - 1]["role"] != "tool"
            last = i == len(messages) - 1 or messages[i + 1]["role"] != "tool"
            if first:
                emit(IM_START + "user")
            emit("\n<tool_response>\n" + m["content"] + "\n</tool_response>")
            if last:
                emit(IM_END + "\n")
        # 其它 role (比如 "observation"): 官方模板一个字也不输出, 这里照做。讲义第 7 节的坑。

    # 3) 生成提示: 写好 "<|im_start|>assistant\n", 让模型从这里接着写
    if add_generation_prompt:
        emit(IM_START + "assistant\n")

    return "".join(out), spans


def assistant_spans(messages, tools=None):
    """只要区间 (练习 2 的参考答案)。"""
    return render_with_spans(messages, tools)[1]


if __name__ == "__main__":
    conv = [
        {"role": "user", "content": "往前走 0.5 米"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "walk", "arguments": {"distance_m": 0.5}}}]},
        {"role": "tool", "content": '{"ok": true}'},
    ]
    print(render(conv, add_generation_prompt=True))
