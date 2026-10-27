# 知识库 · 强化学习

> 让程序通过试错学会做决定, 最后用在小车的控制上 (主线 v0.7)。总目录见 [library/README.md](../README.md#强化学习-rl)。

案例一个接一个: 每个新方法都是因为上一个在新问题上失败才出现的。

## 单元

| 顺序 | 单元 | 新问题 | 状态 |
|---|---|---|---|
| 1 | [bandit](bandit/TUTORIAL.md) 老虎机 | 收益未知, 怎么在试和赚之间平衡 | ✅ |
| 2 | q-learning FrozenLake | 这一步会影响以后站在哪里 (状态) | ⬜ |
| 3 | policy-gradient CartPole | 状态是连续的, 表格放不下 | ⬜ |
| 4 | ppo | 策略梯度训练不稳 | ⬜ |

把学到的策略搬到真车上 (sim2real) 在 [具身智能](../embodied/README.md)。

**前置**: 会写 Python 和 numpy 的数组索引。不需要先学深度学习 (第 3 单元起会用到一点 PyTorch)。
