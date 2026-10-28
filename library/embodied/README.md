# 知识库 · 具身智能

> 让 "大脑" 在真实的身体上工作: 本体与感知、规划、从示范和试错中学、世界模型、具身基础模型、系统与部署。总目录见 [library/README.md](../README.md#embodied)。

**分界**: 运动学、状态估计、规划这些算法在这里; 传感器、电机驱动、单片机上的 PID 在 [硬件](../hardware/README.md) 领域。

**世界模型 (E4) 是重点**, 核心练习是给小车训一个世界模型: 用仿真器的行车数据学会 "这样操作, 下一刻车会在哪", 在里面规划和训练, 再拿到真仿真器里检验。笔记本可做。

**机械臂**: 模仿学习、VLA 的主战场是机械臂抓取, 小车做不了; 这部分先用仿真环境写。真机械臂是否做成项目, 见 [FRAMEWORK.md](../../FRAMEWORK.md) 第 10 节。

## 单元

<!-- catalog:start -->
| 段 | 单元 | 讲什么 | 状态 | 用到它的地方 |
|---|---|---|---|---|
| E0 总览 | embodied-map | 感知 → 决策 → 行动; 大模型在机器人里放在哪一层 | ⬜ |  |
| E1 本体与感知 | kinematics | 坐标系、位姿变换、差速小车运动学、里程计 | ⬜ | v0.2 |
|  | state-estimation | 互补滤波、卡尔曼滤波、编码器 + IMU 融合 | ⬜ | v0.2 |
|  | perception-3d | 相机模型、深度、点云 | ⬜ | v0.6 |
|  | slam | 边走边建图、定位入门 | ⬜ |  |
| E2 规划与经典控制 | motion-planning | A*、RRT: 绕开障碍找路 | ⬜ | v0.3 |
|  | mpc | 模型预测控制 | ⬜ |  |
| E3 从示范和试错中学 | imitation-learning | 行为克隆、DAgger | ⬜ |  |
|  | action-chunking | ACT: 一次预测一小段动作 | ⬜ |  |
|  | diffusion-policy | 用扩散模型生成动作 | ⬜ |  |
|  | sim2real | 仿真到真车: 差距在哪、怎么测、域随机化、系统辨识 | ⬜ | v0.7 |
|  | real-robot-rl | 真机器人上的强化学习 | ⬜ |  |
| E4 世界模型 (重点) | world-model-basics | 世界模型是什么: 预测下一刻、在隐空间里学动力学 🔥 | ⬜ | v0.7 |
|  | car-world-model | 动手: 用小车仿真数据训世界模型, 在里面规划和训练 🔥 | ⬜ | v0.7 |
|  | video-world-models | 生成画面的世界模型 (Genie 3、Cosmos) 🔥 | ⬜ |  |
|  | latent-world-models | 只预测表示的路线 (V-JEPA 2) 🔥 | ⬜ |  |
|  | world-action-models | 画面和动作一起预测 🔥 | ⬜ |  |
|  | world-model-data | 用世界模型生成数据, 让真机器人少用真实数据 🔥 | ⬜ |  |
| E5 具身基础模型 | vla | 视觉-语言-动作模型: 主干 + 动作头, 开源模型对比 🔥 | ⬜ |  |
|  | vla-finetune | 用 LeRobot 微调小号 VLA (8GB 是否够用先实测) 🔥 | ⬜ |  |
|  | robot-data | 遥操作采集与开源数据集 | ⬜ |  |
| E6 系统与部署 | edge-deployment | 机器人上的推理: 延迟预算、哪部分在哪跑 | ⬜ | v0.6 |
|  | robot-software | ROS 2 入门、控制循环频率、实时性 | ⬜ |  |
|  | embodied-eval | 仿真基准与真机评测 | ⬜ |  |
|  | embodied-safety | VLA 的安全、急停、仿真安全不等于真车安全 | ⬜ | v0.7 |
<!-- catalog:end -->

**前置**: 强化学习领域 R1~R3; E5 还需要多模态领域的 vlm。四足机器人的完整路线在选修 [quadruped](../../electives/quadruped/README.md)。
