# 知识库 · 强化学习

> 让程序通过试错学会做决定。总目录见 [library/README.md](../README.md#rl)。

**练习环境**: 全领域用 [小车仿真器](../../sim/car2d.py) 当贯穿的练习环境, 从格子版小车一路练到连续控制; 学完 R3 就是小车 v0.7 要做的 "用强化学习训控制器"。每个单元另配一个标准 Gymnasium 环境 (CartPole 等) 做对照, 方便和外面的资料对上。

**已有原稿**: R1 的对照环境用 Gymnasium Taxi。姊妹仓库 roots-modeling 搬家前写过一本 "先亲自开一趟出租车" (状态、动作、即时奖励、回报递推、折扣与终止, 带手写练习和折叠参考解), 保存在它的标签 [`pre-restructure`](https://github.com/OGGIoao/roots-modeling/tree/pre-restructure/强化学习/Taxi接送乘客) 里 (roots-modeling 暂未公开)。写 mdp 单元时改成本仓库的格式收进来, 不在两边各存一份。

**案例驱动**: 每个新方法都是因为上一个在新问题上失败才出现的。

## 单元

<!-- catalog:start -->
| 段 | 单元 | 讲什么 | 状态 | 用到它的地方 |
|---|---|---|---|---|
| R0 总览 | rl-map | 智能体、环境、奖励; 强化学习都用在哪 | ⬜ |  |
| R1 表格方法 | [bandit](bandit/TUTORIAL.md) | 老虎机: 探索与利用, ε-greedy, UCB | ✅ | v0.7 |
|  | mdp | 状态、动作、回报、折扣、贝尔曼方程 | ⬜ | v0.7 |
|  | dynamic-programming | 知道环境规则时: 值迭代、策略迭代 | ⬜ | v0.7 |
|  | mc-td | 从经验里学: 蒙特卡洛、时序差分 | ⬜ | v0.7 |
|  | q-learning | Q-learning 与 SARSA | ⬜ | v0.7 |
| R2 深度价值方法 | dqn | 用神经网络代替 Q 表: 经验回放、目标网络 | ⬜ |  |
|  | dqn-improvements | Double、Dueling、优先回放 | ⬜ |  |
| R3 策略方法 | policy-gradient | REINFORCE 与基线 | ⬜ | v0.7 |
|  | actor-critic | 优势函数、GAE | ⬜ | v0.7 |
|  | ppo | PPO: 裁剪目标; 后训练的 RLHF、GRPO 从这里接上 | ⬜ | v0.7, 基座 |
|  | continuous-control | SAC、TD3: 连续动作与离策略 | ⬜ | v0.7 |
| R4 模型与数据 | model-based-rl | 先学世界模型再规划或训练 (Dreamer、MPC) 🔥 | ⬜ | v0.7 |
|  | offline-rl | 只有历史记录时怎么学 (CQL、IQL) 🔥 | ⬜ |  |
| R5 工程与规模化 | rl-engineering | 为什么难调: 归一化、种子、调试、怎么判断真学会了 | ⬜ | v0.7 |
|  | reward-design | 奖励塑形与奖励被钻空子 | ⬜ | v0.7 |
|  | parallel-simulation | 几千个环境同时在 GPU 上跑 (Windows 能否跑先实测) 🔥 | ⬜ |  |
| R6 延伸 | multi-agent-rl | 自我对弈与多智能体 | ⬜ |  |
|  | exploration | 好奇心、稀疏奖励下的探索 | ⬜ |  |
|  | rl-with-foundation-models | 大模型当奖励或规划器、强化学习微调 VLA | ⬜ |  |
<!-- catalog:end -->

**和其他领域**: 后训练领域的 RLHF、GRPO 从 R3 的 PPO 接上; 世界模型在 R4 和 [具身智能](../embodied/README.md) 的 E4 两边讲; 把策略搬到真车上 (sim2real) 在具身智能。

**前置**: 会写 Python 和 numpy 的数组索引; R2 起会用到 PyTorch (基础领域 pytorch-basics)。
