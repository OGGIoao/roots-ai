# 路线图

> 主线只走小车用得到的内容; 其余全部放进选修, 不删除。产品目标与各版本的验收标准见 [PRODUCT.md](PRODUCT.md), 讲义与练习的写法见 [CONTRIBUTING.md](CONTRIBUTING.md)。
> 状态: ✅ 完成 · 🟡 进行中或待改造 · ⬜ 未开始。最后更新: 2026-10-21

## 1. 选内容的规则

> **每个节点先回答: 小车的哪个版本需要它?** 答不上来就进选修, 不进主线。

这样主线是一条通往终点 demo 的路, 选修是一张更大的地图。想动手的时候走主线; 不想动手的时候, 选修就是可以慢慢读的详细文档。

## 2. 主线

```
A 看懂模型                      (笔记本即可, 服务 v0.5)
  A1 BPE 分词器               ✅ mainline/A1-bpe
  A2 miniGPT                  ✅ mainline/A2-minigpt
  A3 注意力与推理延迟          ✅ mainline/A3-attention (待补: "延迟决定小车反应速度"一节)
  A4 现代小模型长什么样        🟡 mainline/A4-modern-small-model (待精简; 用解剖台读懂 Qwen2.5-0.5B)

B 让车听话                      (硬件, v0.1–v0.3)
  B1 电子入门与仿真器          🟡 mainline/B1-electronics: 仿真部分完成 (Wokwi 实测); 真板对照待买板后补
  B2 电机与遥控                ⬜ PWM、电机驱动、WiFi 网页遥控 → v0.1
  B3 编码器与 PID              ⬜ 速度闭环、走直线、走准距离 → v0.2
  B4 传感器、状态机与安全层     ⬜ 测距、避障、看门狗、急停 → v0.3

C 让模型成为小车专家            (软件, v0.4–v0.5)
  C1 Agent 循环与工具调用      ⬜ 先驱动 Python 里的 2D 仿真小车, 再驱动真车; 用 API 模型 → v0.4
  C2 微调会调用工具的小模型    🟡 mainline/C2-tool-finetune (待改造: 手写 LoRA, 0.5B 只输出工具调用)
  C3 本地推理服务              ⬜ 笔记本上跑自己的模型, 实测延迟 → v0.5
  C4 评测                     ⬜ 10 次成功几次; 失败案例分类

D 让车看见                      (v0.6)
  D1 摄像头与图像基础          ⬜
  D2 小模型检测 + int8 量化    ⬜ 芯片上跑 vs 笔记本上跑, 对照实测

E 让车自己学                    (v0.7)
  E1 老虎机                   ✅ mainline/E1-bandit
  E2 FrozenLake               ⬜
  E3 CartPole / REINFORCE     ⬜
  E4 PPO                      ⬜
  E5 仿真小车训练 → 上真车     ⬜ sim2real, 与手写方案对照

F 打通                          (v1.0)
  F1 芯片上的唤醒词            ⬜ 自己训练、量化、手写 C 推理
  F2 全链路 demo 与评测        ⬜
```

**推进顺序按版本, 不按字母**: B1→B4 (v0.1–v0.3) → C1 (v0.4) → A4、C2–C4 (v0.5) → D (v0.6) → E2–E5 (v0.7) → F (v1.0)。A1–A3 和 E1 已经完成。

### 每章的三件套

| 组成 | 作用 |
|---|---|
| 讲义 `TUTORIAL.md` | 讲透原理与代码。样章: [A2-minigpt/TUTORIAL.md](mainline/A2-minigpt/TUTORIAL.md) |
| 参考实现 | 仓库里的代码, 也是练习的标准答案 |
| 手写练习 | 三级递进: 填空 → 接口 → 白纸, 由测试判定对错 |

## 3. 选修

全部在 [electives/](electives/README.md):

| 选修 | 内容 | 和主线的关系 |
|---|---|---|
| [LLM 架构生长树](electives/llm-tree/README.md) | 从 Transformer 长出的整棵架构树: 注意力、位置编码、MoE、归一化、替代架构 (Mamba 等)、效率 | 主线 A1–A4 是它的主干; 想把 LLM 架构系统串起来时读它 |
| [解剖台](electives/llm-anatomy/README.md) | 读取 20 多个真实开源模型的配置与张量名 | A4 用它读懂要微调的小模型 |
| [四足机器人 Pupper](electives/quadruped/README.md) | R1–R5: 本体与经典控制 → RL 步态 → 模仿学习 → 多峰行为 → 语言与视觉 | 工具接口与小车相同, 主线的 Agent 可以直接驱动它 |
| [后训练](electives/post-training/README.md) | 把语言模型当策略做 RL; DPO; GRPO | 可以用 GRPO 提高 C2 小模型的工具调用成功率 |
| [规模化与上云](electives/scale-up/README.md) | 把模型做大、多卡训练、推理服务压测 | 笔记本实测撞墙后再做 |

## 4. 运行环境与上云原则

| 用途 | 环境 |
|---|---|
| 测量平台 (讲义里的数字都出自这里) | Windows 11 笔记本, RTX 5060 Laptop 8GB, Python 3.10 |
| 软件部分 (A、C、E) | 主环境, 安装见根目录 README |
| 往主环境装 gymnasium、mujoco 等新包 | 装之前先 `pip install --dry-run`, 确认不会替换已装好的 torch |
| 会锁定 torch 版本的工具链 | 单独的 conda 环境, 不装进主环境 |
| 硬件部分 (B、D、F) | ESP32-S3 开发板; 每一课先在浏览器仿真器里跑通 |
| 笔记本放不下的模型、Windows 跑不通的工具链 | 云服务器 |

**上云原则**: 只在笔记本**实测**撞墙 (显存溢出、要跑好几天) 之后才上云, 并把这个数字写进讲义。上云的部分也按对照实验来写: 笔记本 vs 云 GPU、单卡 vs 多卡, 让读者看到"为什么需要多卡", 而不只是"换了台机器"。

各工具在 Windows 上能否运行, 动手前逐个实测, 不做假设。

## 5. 降低门槛

目标: 读者从点开仓库到第一次跑出结果, 不需要装环境、不需要买硬件、不需要显卡。

| 事项 | 内容 | 状态 | 依赖 |
|---|---|---|---|
| Colab 笔记本 | A1、A2、A3 各一本 `colab.ipynb`, 各章 README 顶部有 "Open in Colab" 按钮; 由 [tools/make_notebooks.py](tools/make_notebooks.py) 生成 | 🟡 已在全新克隆的仓库里模拟 Colab 逐格运行通过; 在 Colab 上实际运行待首次确认 | — |
| Wokwi 一键链接 | B1 的五个案例存成 Wokwi 公开项目, 讲义第 0 节放链接, 点开就能运行, 不用复制粘贴 | ⬜ | 用维护者的 Wokwi 账号保存为公开项目 |
| 物料清单 | 小车的完整物料清单, 带购买链接与价格, 放进 `hardware/` | ⬜ | B2 开始前定下开发板型号 (先核对 B1 讲义第 8 节的引脚账) |

之后每完成一站, 同时补上这一站的 Colab / Wokwi 入口; 需要显卡的章节 (如 C2 微调) 先实测 Colab 免费 GPU 能否跑完再承诺。
