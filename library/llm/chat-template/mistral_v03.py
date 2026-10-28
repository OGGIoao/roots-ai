"""
mistral_v03.py — 手写 Mistral-7B-Instruct-v0.3 的对话模板 (不含工具部分), 练习 3 的参考答案

和 Qwen 的 ChatML 对照着看 (讲义第 8 节):
  - 没有 system 角色标记: system 被塞进**最后一条** user 消息的开头, 中间隔一个空行;
  - user 包在 [INST] ... [/INST] 里, 内容原样;
  - assistant 前面加一个空格, 内容去掉首尾空白 (Jinja 的 trim = Python 的 str.strip), 后面跟 </s>;
  - 开头一个 <s>; 没有"生成提示", 因为 [/INST] 之后就该模型说话了;
  - 角色必须 user / assistant 交替, 第一条 (system 之后) 必须是 user, 否则报错。

已用 experiments/results/mistral_golden.json (官方模板渲染的 44 段对话) 核对。
"""

BOS, EOS = "<s>", "</s>"
ALTERNATE_ERROR = "After the optional system message, conversation roles must alternate user/assistant/user/assistant/..."


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
            # 官方条件是 "loop.last": 只有整段对话以 user 结尾时 system 才会出现;
            # 以 assistant 结尾 (比如拼训练数据) 时 system 整个丢掉。讲义第 8.2 节。
            if system is not None and i == len(messages) - 1:
                content = system + "\n\n" + content
            out.append("[INST] " + content + "[/INST]")
        elif m["role"] == "assistant":
            out.append(" " + m["content"].strip() + EOS)
        else:
            raise ValueError("Only user and assistant roles are supported, with the exception of an initial optional system message!")
    return "".join(out)
