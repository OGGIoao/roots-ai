# 知识库 · 基础

> 后面七个领域要用的编程工具、数学、机器学习和计算机系统基础。总目录见 [library/README.md](../README.md#foundations)。

**怎么用**: 不用从头读到尾。每个单元的讲义开头都写着 "前置", 缺哪块点过来补哪块; 第一个单元 foundations-map 附一份自测题, 做完就知道从哪开始。

**写法**: 够用为止, 不追求系统的数学课; 每个概念都用代码 (NumPy / PyTorch) 写一遍、画出来。

**门槛**: 会写 Python 函数、会用命令行。不从零教编程。

**和 roots-modeling 的分工**: 这里的数学 (线代、微积分、概率) 和梯度类优化两边共用, 写在这里; 统计学习 (《统计学习方法》深度)、统计推断、约束与离散优化 (规划、动态规划)、微分方程、图与网络、仿真、评价决策归姊妹仓库 [roots-modeling](https://github.com/OGGIoao/roots-modeling) (暂未公开)。ml-basics 写到够用为止, 想手推到底就去那边对应的单元。一个知识点只在一个仓库写, 另一个只放链接。

## 单元

<!-- catalog:start -->
| 段 | 单元 | 讲什么 | 状态 | 用到它的地方 |
|---|---|---|---|---|
| F0 总览 | foundations-map | 怎么用这个领域, 附自测题: 从哪开始、哪些可以跳过 | ⬜ |  |
| F1 编程与工具 | dev-environment | 终端、conda、pip、显卡驱动与 CUDA、Windows 与 WSL; 报错怎么查 | ⬜ | 全部 |
|  | git-github | 版本管理、分支、提 PR | ⬜ |  |
|  | python-for-ai | 本仓库常用的 Python: 类、生成器、类型提示、调试、pytest | ⬜ |  |
|  | numpy | 数组、形状、广播 | ⬜ |  |
| F2 数学 | linear-algebra | 向量、矩阵、矩阵乘法是什么变换、形状怎么对 | ⬜ |  |
|  | calculus | 导数、链式法则、梯度 | ⬜ |  |
|  | probability | 分布、期望、采样、对数概率 | ⬜ |  |
|  | information-theory | 熵、交叉熵、KL 散度 | ⬜ |  |
|  | optimization | 梯度下降、动量、Adam 的来历 | ⬜ |  |
| F3 机器学习与深度学习 | ml-basics | 训练集与验证集、过拟合、正则化; 线性回归与逻辑回归 | ⬜ |  |
|  | autograd | 从零写自动求导, 理解反向传播 | ⬜ | 基座 |
|  | neural-networks | 多层感知机、激活函数、初始化 | ⬜ |  |
|  | pytorch-basics | 张量、模块、数据加载、上 GPU | ⬜ | 基座 |
|  | training-practice | 读 loss 曲线、选学习率和批大小、训练不收敛怎么查 | ⬜ | 基座 |
| F4 计算机系统 | gpu-basics | GPU 怎么算、显存层级、带宽瓶颈与算力瓶颈 | ⬜ | 推理服务 |
|  | profiling | 测时间、测显存、找瓶颈: 本仓库"数字全部实测"的方法 | ⬜ |  |
|  | linux-server | 远程服务器、SSH、后台跑任务 | ⬜ | 基座 |
|  | cloud-gpu | 租 GPU、估成本、传数据、用完记得关 | ⬜ | 基座 |
| F5 研究方法 | experiment-method | 控制变量、对照组、随机种子、置信区间; 结论是不是碰巧 | ⬜ |  |
|  | reading-papers | 读论文和技术报告、核实二手文章里的数字 | ⬜ |  |
<!-- catalog:end -->

F4 (计算机系统) 和 F5 (研究方法) 对应的是本仓库的工作方式: 数字要实测、结论要有对照、信息要核实源头。
