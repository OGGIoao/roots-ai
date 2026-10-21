# 选修

> 主线只走小车用得到的内容 (见 [ROADMAP](../ROADMAP.md)); 这里是更大的地图, 全部保留。
> 两种用法: 想把某个领域系统串起来时按图深挖; 不想动手的时候, 当作可以慢慢读的详细文档。

| 选修 | 内容 | 状态 |
|---|---|---|
| [llm-tree/](llm-tree/README.md) | LLM 架构生长树: 从 Transformer 长出的注意力、位置编码、MoE、归一化、替代架构、效率六条分支 | 总图 ✅; 注意力线已在主线 A3 实现, 其余分支 ⬜ |
| [llm-anatomy/](llm-anatomy/README.md) | 解剖台: 读取 20 多个真实开源模型的配置与张量名, 用数据核对生长树 | ✅ 可用 |
| [quadruped/](quadruped/README.md) | 四足机器人 Pupper: 本体与经典控制 → RL 步态 → 模仿学习 → 多峰行为 → 语言与视觉 | 规划 ✅, 实现 ⬜ |
| [post-training/](post-training/README.md) | 后训练: 把语言模型当策略做 RL; DPO; GRPO | ⬜ |
| [scale-up/](scale-up/README.md) | 规模化与上云: 把模型做大、多卡训练、推理服务压测 | ⬜ |

选修的写法和主线一样 (讲义 + 参考实现 + 三级练习, 数字全部实测), 只是不在通往小车 demo 的路上。
