# 安全线

> 安全不是单独的领域: 每个安全问题都长在某个具体技术上, 离开它讲不清楚。所以安全单元留在各自的领域里, 由这一页串起来。从头读到尾, 就是一条完整的安全路线: 分词层 → 模型层 → Agent 层 → 物理层。

| 层 | 问题 | 单元 | 状态 |
|---|---|---|---|
| 分词与模板 | 用户文字里写特殊标记, 伪造系统消息 (注入) | [llm/chat-template](llm/chat-template/TUTORIAL.md) 第 7.1 节 | ✅ |
| 模型 | 拒答、越狱 | post-training/safety-alignment | ⬜ |
| 模型 | 系统地找出问题: 红队测试 | llm/llm-eval | ⬜ |
| 模型 | 模型内部在 "想" 什么 | llm/interpretability | ⬜ |
| Agent | 模型说要做的事不等于应该去做: 安全层 | [agent/agent-loop](agent/agent-loop/TUTORIAL.md) | 🟡 |
| Agent | 网页和文件里藏的指令、工具权限、沙箱 | agent/agent-safety | ⬜ |
| 机器人 | 状态机、看门狗、急停 | hardware/safety-firmware | ⬜ |
| 机器人 | 仿真里安全的策略到真车上不一定安全 | embodied/sim2real | ⬜ |
| 机器人 | VLA 与具身系统的安全 | embodied/embodied-safety | ⬜ |

等安全单元多到这一页撑不住, 再考虑升格成独立领域。
