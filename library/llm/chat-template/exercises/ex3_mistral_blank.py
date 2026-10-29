"""
练习 3 (③ 白纸): 给另一个模型家族写模板

只看下面的说明和讲义第 11 节的渲染样例, 从零写出 Mistral-7B-Instruct-v0.3 的对话模板 (不含工具):

    render_mistral(messages) -> str

规则 (官方 Jinja 模板翻译成大白话):
  1. 第一条如果是 system, 先拿出来, 剩下的才是"对话";
  2. 对话里角色必须 user / assistant / user / ... 交替, 从 user 开始; 不交替就抛 ValueError;
     出现 user、assistant 以外的角色也抛 ValueError;
  3. 输出以 "<s>" 开头;
  4. user 消息写成 "[INST] " + 内容 + "[/INST]";
     如果有 system, 并且这条 user 是**整段对话的最后一条消息**, 内容前面加上 system + "\\n\\n";
  5. assistant 消息写成 " " + 去掉首尾空白的内容 + "</s>";
  6. 没有"生成提示": 以 user 结尾时, [/INST] 后面就该模型说话了。

验收 (44 段对话, 其中 12 段应该报错; 标准答案是官方模板渲染的):
    cd library/llm/chat-template/exercises
    python -m pytest test_ex3.py -q

写完想一想: 规则 4 意味着什么? 用这个模板拼训练数据 (以 assistant 结尾) 时, system 去哪了? (讲义第 11.2 节)
"""


def render_mistral(messages):
    raise NotImplementedError
