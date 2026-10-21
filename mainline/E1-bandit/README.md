# E1 老虎机 — 主线 E "让车自己学" 的第一章

> 主线 E 服务于 v0.7: 小车上的策略先在仿真里用强化学习训练, 再部署到真车, 实测仿真和现实差多少。
> 讲义: [TUTORIAL.md](TUTORIAL.md)。四足机器人上的完整具身路线在选修 [electives/quadruped](../../electives/quadruped/README.md)。

E 部分是一串案例, 后一个案例都是因为前一个方法在新问题上失败才出现的:

| 章节 | 新问题 | 朴素办法怎么失败 | 引出的概念 | 状态 |
|---|---|---|---|---|
| E1 老虎机 (本章) | 10 个拉杆, 每个的平均收益未知, 怎么花有限的次数赚最多 | 贪心: 只拉当前估计最高的那个, 实测只有 35% 的局停在最优臂上 | 探索与利用, ε-greedy, 乐观初始值, UCB, 遗憾, 固定步长 | ✅ 讲义 + 4 个实验 + 三级练习 |
| E2 FrozenLake | 这一步怎么走会影响以后站在哪里 | 老虎机的方法没有"状态" | 状态, 折扣回报, 价值, Q-learning | ⬜ |
| E3 CartPole | 状态是连续的 | Q 表放不下 | 策略梯度 REINFORCE, baseline | ⬜ |
| E4 PPO | REINFORCE 训练不稳 | 一步更新太大, 策略直接崩掉 | 裁剪的替代目标, GAE | ⬜ |
| E5 仿真小车 → 真车 | 在仿真里学会的策略, 到真车上还管用吗 | 直接搬过去: 摩擦、延迟、传感器噪声都和仿真不同 | sim2real, 域随机化, 与手写 PID/规则方案对照 | ⬜ |

## 结构

```
E1-bandit/                 只用 numpy
├── TUTORIAL.md            讲义: 案例驱动, 数字全部实测
├── env.py                 10 臂测试台 (R 局并行; 可选非平稳)
├── agents.py              参考实现: 贪心 / ε-greedy (含乐观初值、固定步长) / UCB
├── runner.py              统一跑法与三个指标 (平均收益、最优臂比例、累计遗憾)
├── experiments/           greedy_fails / explore / sweep / nonstationary
├── results/               实验结果 (JSON)
├── figures/               make_charts.py (数据图) 与 make_diagrams.py (结构图)
└── exercises/             三级练习 + 测试
```

## 运行

```bash
cd mainline/E1-bandit
python experiments/explore.py          # 六种方法对照, 测量平台上约 20 秒
python figures/make_charts.py          # 由 results/ 重新生成图
cd exercises && python -m pytest test_ex1.py -q
```

## 几个实测结论

- 贪心一半的局在第 3 步之后就再也不换臂, 30% 的局 1000 步里只试过一个臂;
- 乐观初值和 UCB 最后找到最优臂的比例一样 (约 86%), 但乐观初值的总遗憾是 UCB 的 1.63 倍, 差距几乎全在前 150 步;
- 拉到 10000 步, 探索少的 ε=0.01 在约 1600 步后反超 ε=0.1;
- 本文默认的 UCB c=2 不是最优, 扫参显示 c=1 更好;
- 收益会变时, UCB 前期最好, 后期最优臂比例从 65% 掉到 48%; 固定步长一路升到 76%。
