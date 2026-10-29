"""
make_catalog.py — 知识库单元目录的唯一数据源

总目录 library/README.md、各领域 README 里的单元表、根目录 README 顶部的状态摘要都由这里生成, 不要手改标记之间的内容:
    <!-- catalog:start -->  ...  <!-- catalog:end -->

改了单元的状态、新增单元、单元有了讲义, 只改下面的 DOMAINS, 然后运行:
    python tools/make_catalog.py
tests/test_catalog.py 检查生成的文件和这里一致。

字段: (目录名, 讲什么, 状态, 用到它的地方)
  状态: ✅ 完成 · 🟡 有内容待补 · ⬜ 规划中; 🔥 表示前沿单元 (写在 "讲什么" 末尾)
  用到它的地方: 小车站号 (v0.4)、基座线 (基座)、学习助教 (助教)、推理服务 (推理服务); 空 = 只是知识库的完整性需要
  有内容的单元自动链接到它的 TUTORIAL.md (没有就链接 README.md)。
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "library")

DOMAINS = [
    ("foundations", "基础", "后面七个领域要用的编程、数学、机器学习与计算机系统基础; 够用为止, 每个概念都用代码写一遍", [
        ("F0 总览", [("foundations-map", "怎么用这个领域, 附自测题: 从哪开始、哪些可以跳过", "⬜", "")]),
        ("F1 编程与工具", [
            ("dev-environment", "终端、conda、pip、显卡驱动与 CUDA、Windows 与 WSL; 报错怎么查", "⬜", "全部"),
            ("git-github", "版本管理、分支、提 PR", "⬜", ""),
            ("python-for-ai", "本仓库常用的 Python: 类、生成器、类型提示、调试、pytest", "⬜", ""),
            ("numpy", "数组、形状、广播", "⬜", "")]),
        ("F2 数学", [
            ("linear-algebra", "向量、矩阵、矩阵乘法是什么变换、形状怎么对", "⬜", ""),
            ("calculus", "导数、链式法则、梯度", "⬜", ""),
            ("probability", "分布、期望、采样、对数概率", "⬜", ""),
            ("information-theory", "熵、交叉熵、KL 散度", "⬜", ""),
            ("optimization", "梯度下降、动量、Adam 的来历", "⬜", "")]),
        ("F3 机器学习与深度学习", [
            ("ml-basics", "训练集与验证集、过拟合、正则化; 线性回归与逻辑回归", "⬜", ""),
            ("autograd", "从零写自动求导, 理解反向传播", "⬜", "基座"),
            ("neural-networks", "多层感知机、激活函数、初始化", "⬜", ""),
            ("pytorch-basics", "张量、模块、数据加载、上 GPU", "⬜", "基座"),
            ("training-practice", "读 loss 曲线、选学习率和批大小、训练不收敛怎么查", "⬜", "基座")]),
        ("F4 计算机系统", [
            ("gpu-basics", "GPU 怎么算、显存层级、带宽瓶颈与算力瓶颈", "⬜", "推理服务"),
            ("profiling", "测时间、测显存、找瓶颈: 本仓库\"数字全部实测\"的方法", "⬜", ""),
            ("linux-server", "远程服务器、SSH、后台跑任务", "⬜", "基座"),
            ("cloud-gpu", "租 GPU、估成本、传数据、用完记得关", "⬜", "基座")]),
        ("F5 研究方法", [
            ("experiment-method", "控制变量、对照组、随机种子、置信区间; 结论是不是碰巧", "⬜", ""),
            ("reading-papers", "读论文和技术报告、核实二手文章里的数字", "⬜", "")]),
    ]),
    ("hardware", "硬件", "让东西在物理世界里动起来: 电路、单片机与固件、电机、传感器、连接、安全、端侧 AI。所有单元先在浏览器仿真器里做, 再在真板上对照", [
        ("H0 总览", [("hardware-map", "单片机、单板电脑、电脑与云三层各干什么", "⬜", "")]),
        ("H1 电子基础", [
            ("electronics", "电路与 ESP32 入门: 点灯、按钮消抖、PWM、ADC、非阻塞循环 (浏览器仿真)", "🟡", "v0.1"),
            ("power", "电池、稳压、电流预算; 电机干扰为什么让单片机重启", "⬜", "v0.1"),
            ("datasheets-tools", "读数据手册和原理图; 万用表、逻辑分析仪、示波器", "⬜", "v0.1"),
            ("pcb-basics", "从面包板到自己画电路板 (KiCad)", "⬜", "")]),
        ("H2 单片机与固件", [
            ("mcu-architecture", "GPIO、定时器、中断、DMA、内存", "⬜", "v0.1"),
            ("comm-protocols", "UART、I2C、SPI、I2S、CAN", "⬜", "v0.3"),
            ("rtos", "FreeRTOS 多任务; MicroPython 与 C++ 各适合什么", "⬜", "v0.6"),
            ("firmware-engineering", "编译、烧录、OTA、日志、在电脑上测固件逻辑", "⬜", "v0.1")]),
        ("H3 执行与控制", [
            ("motor-driver", "直流电机与 H 桥驱动", "⬜", "v0.1"),
            ("encoder", "编码器与测速", "⬜", "v0.2"),
            ("pid", "PID 在单片机上的实现与调参", "⬜", "v0.2"),
            ("servo-stepper", "舵机与步进电机", "⬜", "")]),
        ("H4 感知硬件", [
            ("sensors", "测距与 IMU", "⬜", "v0.3"),
            ("camera", "摄像头采集", "⬜", "v0.6"),
            ("audio", "麦克风、I2S、喇叭", "⬜", "v1.0")]),
        ("H5 连接", [
            ("wifi-remote", "WiFi 与网页遥控", "⬜", "v0.1"),
            ("robot-networking", "延迟、断线、MQTT / WebSocket: 电脑上的大模型和地上的车之间怎么连才稳", "⬜", "v0.3")]),
        ("H6 安全", [("safety-firmware", "状态机、看门狗与急停", "⬜", "v0.3")]),
        ("H7 端侧 AI", [
            ("tinyml", "在单片机上跑量化小模型 (例如唤醒词) 🔥", "⬜", "v1.0"),
            ("edge-npu", "单板电脑的 NPU: 模型转换、算子支持 (需要可选的单板电脑) 🔥", "⬜", ""),
            ("edge-gpu", "Jetson 一类带 GPU 的板子与 TensorRT (需要可选的单板电脑) 🔥", "⬜", ""),
            ("on-device-llm", "在板子上跑大模型: INT4、llama.cpp (需要可选的单板电脑) 🔥", "⬜", ""),
            ("edge-system-design", "功能放单片机、板子还是云: 延迟、功耗、成本怎么算", "⬜", "")]),
    ]),
    ("llm", "大语言模型", "一条链: 输入 → 架构 → 预训练 → (后训练, 见后训练领域) → 推理能力 → 推理与部署; 旁边挂评测与可解释性", [
        ("输入", [("bpe", "分词: 把文字变成整数 (字节级 BPE)", "✅", "v0.5, 基座")]),
        ("架构 (总地图: llm-tree)", [
            ("llm-tree", "LLM 架构生长树: 整个架构段的地图, 每个节点写明来历、代价、谁用了", "🟡", ""),
            ("minigpt", "从零写一个 GPT-2 结构的小模型并训练", "✅", "v0.5, 基座"),
            ("attention", "注意力变体 (MHA / MQA / GQA / MLA) 与推理延迟", "✅", "v0.5, 基座"),
            ("modern-small-model", "现代小模型长什么样 (RoPE、RMSNorm、SwiGLU ……)", "🟡", "v0.5, 基座"),
            ("position-encoding", "RoPE 细讲、YaRN 等长度外推、部分维度旋转", "⬜", "基座"),
            ("residual-connections", "Pre-LN、RMSNorm、QK-Norm → 超连接 (DeepSeek mHC、Kimi Attention Residuals) 🔥", "⬜", "基座"),
            ("moe", "混合专家: 路由、细粒度与共享专家、负载均衡 🔥", "⬜", "基座"),
            ("sparse-attention", "滑动窗口 → DeepSeek V3.2 稀疏注意力 → V4 的压缩注意力 🔥", "⬜", "基座"),
            ("linear-attention-ssm", "线性注意力、Mamba、Gated DeltaNet、Kimi Delta Attention 🔥", "⬜", "基座"),
            ("hybrid-architecture", "混合架构: 大部分层线性、少数层全注意力 🔥", "⬜", "基座"),
            ("long-context", "长上下文: 外推、滑动窗口、attention sink、怎么测", "⬜", ""),
            ("diffusion-lm", "扩散语言模型: 整块并行生成 🔥", "⬜", "")]),
        ("预训练", [
            ("pretraining-data", "数据从哪来: 抓取、去重、过滤、配比", "⬜", "基座"),
            ("scaling-laws", "规模定律: 参数、数据、算力怎么分配", "⬜", "基座"),
            ("optimizers", "AdamW → Muon 🔥", "⬜", "基座"),
            ("training-stability", "loss 尖峰、学习率、μP", "⬜", "基座"),
            ("low-precision", "bf16 → FP8 训练 🔥", "⬜", "基座"),
            ("distributed-training", "数据并行、FSDP、张量并行、流水线并行、专家并行", "⬜", "基座"),
            ("mtp", "多 token 预测 🔥", "⬜", "基座"),
            ("mid-training", "退火、长上下文扩展", "⬜", "基座"),
            ("distillation", "大模型教小模型", "⬜", "基座")]),
        ("推理能力", [
            ("chain-of-thought", "思维链: 先想再答为什么有用", "⬜", ""),
            ("test-time-scaling", "推理时多花算力: 多次采样投票、best-of-N、搜索 🔥", "⬜", ""),
            ("reasoning-models", "思考模式与思考预算 🔥", "⬜", "")]),
        ("推理与部署", [
            ("chat-template", "对话模板: 模型怎么分清 system / user / assistant / tool", "✅", "v0.4, v0.5"),
            ("decoding", "解码与采样: 贪心、温度、top-p、重复惩罚、束搜索, 为什么评测用贪心", "✅", "v0.4"),
            ("structured-output", "约束解码: 让模型只能输出合法的 JSON 和工具调用", "✅", "v0.4"),
            ("quantization", "量化: INT4、FP8、MXFP4/NVFP4, 变小变快的代价 🔥", "⬜", "v0.6, 推理服务"),
            ("local-inference", "本地推理: llama.cpp、vLLM、SGLang 的用法与延迟", "⬜", "v0.5, 推理服务"),
            ("paged-kv-cache", "自写推理引擎 ①: 分页 KV 缓存", "⬜", "推理服务"),
            ("continuous-batching", "自写推理引擎 ②: 连续批处理", "⬜", "推理服务"),
            ("prefix-caching", "自写推理引擎 ③: 前缀缓存", "⬜", "推理服务"),
            ("speculative-decoding", "自写推理引擎 ④: 投机解码, 用 MTP 当草稿 🔥", "⬜", "推理服务"),
            ("batch-invariance", "自写推理引擎 ⑤: 批不变与确定性推理 🔥", "⬜", "推理服务"),
            ("serving-at-scale", "专家并行、prefill/decode 分离: 读 DeepInfer 源码 🔥", "⬜", "推理服务")]),
        ("旁支", [
            ("llm-eval", "困惑度、基准、数据污染、红队、用大模型当评委", "⬜", ""),
            ("interpretability", "logit lens、探针、稀疏自编码器、电路追踪", "⬜", "")]),
    ]),
    ("agent", "Agent", "从一个循环到能交付给客户的系统: 工具、上下文与记忆、检索、协议、典型 Agent、多智能体、工程化、评测、安全", [
        ("A0 总览", [("agent-map", "什么时候该用 Agent; 模型 + 外壳 (harness) 的结构", "⬜", "")]),
        ("A1 核心机制", [
            ("agent-loop", "Agent 循环: 消息、工具调用、四种循环写法、安全层、评测方法", "🟡", "v0.4"),
            ("tool-calling", "各家的工具调用格式、并行调用、和约束解码的配合", "⬜", "v0.4"),
            ("planning", "ReAct、先规划再执行、反思与重试", "⬜", ""),
            ("harness-engineering", "把模型包成能干活的系统: 提示、工具设计、循环控制、出错恢复 🔥", "⬜", "助教")]),
        ("A2 上下文与记忆", [
            ("context-engineering", "窗口里放什么、怎么压缩、怎么让前缀命中缓存 🔥", "⬜", "助教"),
            ("memory", "短期与长期记忆, 什么时候写、什么时候读 🔥", "⬜", "助教")]),
        ("A3 检索增强", [
            ("embeddings", "文字怎么变成向量、相似度怎么算", "⬜", "助教"),
            ("rag-basics", "切块、检索、重排、关键词 + 向量混合检索", "⬜", "助教"),
            ("rag-advanced", "Agent 式检索、GraphRAG、长上下文与 RAG 的取舍 🔥", "⬜", "助教"),
            ("rag-eval", "检索召回率、回答是否忠于资料", "⬜", "助教")]),
        ("A4 协议与生态", [
            ("mcp", "MCP: 把工具做成任何应用都能调用的服务", "⬜", "v0.4"),
            ("agent-interop", "A2A、Agent Skills、AGENTS.md 等互通标准 🔥", "⬜", ""),
            ("agent-frameworks", "LangGraph、OpenAI Agents SDK 等框架, 逐个部件对照手写循环", "🟡", "")]),
        ("A5 典型 Agent", [
            ("coding-agent", "编程 Agent: 读代码库、跑测试、在沙箱里改代码", "⬜", ""),
            ("computer-use", "看截图或页面结构操作电脑和网页", "⬜", ""),
            ("deep-research", "搜索、阅读、综合成报告", "⬜", "")]),
        ("A6 多智能体", [("multi-agent", "主管带工人、分工、辩论; 什么时候真有用", "⬜", "")]),
        ("A7 工程化落地", [
            ("data-pipelines", "把文档变成可检索的知识: 解析、OCR、清洗、增量更新", "⬜", "助教"),
            ("agent-deployment", "把 Agent 做成服务: 流式、长任务、重试", "⬜", "助教"),
            ("observability", "追踪每一步调用了什么、花了多少、错在哪", "⬜", "助教"),
            ("cost-latency", "缓存、按难度选模型、预算控制", "⬜", "助教, 推理服务")]),
        ("A8 评测", [("agent-eval", "任务集设计、置信区间、回归测试; 常见公开基准各测什么", "⬜", "v0.5, 助教")]),
        ("A9 安全", [("agent-safety", "网页和文件里藏的指令、工具权限、沙箱", "⬜", "v0.4, 助教")]),
    ]),
    ("rl", "强化学习", "通过试错学会做决定。主干用经典环境 (格子世界、CartPole 等), 小车等项目放在各单元的实践一节", [
        ("R0 总览", [("rl-map", "智能体、环境、奖励; 强化学习都用在哪", "⬜", "")]),
        ("R1 表格方法", [
            ("bandit", "老虎机: 探索与利用, ε-greedy, UCB", "✅", "v0.7"),
            ("mdp", "状态、动作、回报、折扣、贝尔曼方程", "⬜", "v0.7"),
            ("dynamic-programming", "知道环境规则时: 值迭代、策略迭代", "⬜", "v0.7"),
            ("mc-td", "从经验里学: 蒙特卡洛、时序差分", "⬜", "v0.7"),
            ("q-learning", "Q-learning 与 SARSA", "⬜", "v0.7")]),
        ("R2 深度价值方法", [
            ("dqn", "用神经网络代替 Q 表: 经验回放、目标网络", "⬜", ""),
            ("dqn-improvements", "Double、Dueling、优先回放", "⬜", "")]),
        ("R3 策略方法", [
            ("policy-gradient", "REINFORCE 与基线", "⬜", "v0.7"),
            ("actor-critic", "优势函数、GAE", "⬜", "v0.7"),
            ("ppo", "PPO: 裁剪目标; 后训练的 RLHF、GRPO 从这里接上", "⬜", "v0.7, 基座"),
            ("continuous-control", "SAC、TD3: 连续动作与离策略", "⬜", "v0.7")]),
        ("R4 模型与数据", [
            ("model-based-rl", "先学世界模型再规划或训练 (Dreamer、MPC) 🔥", "⬜", "v0.7"),
            ("offline-rl", "只有历史记录时怎么学 (CQL、IQL) 🔥", "⬜", "")]),
        ("R5 工程与规模化", [
            ("rl-engineering", "为什么难调: 归一化、种子、调试、怎么判断真学会了", "⬜", "v0.7"),
            ("reward-design", "奖励塑形与奖励被钻空子", "⬜", "v0.7"),
            ("parallel-simulation", "几千个环境同时在 GPU 上跑 (Windows 能否跑先实测) 🔥", "⬜", "")]),
        ("R6 延伸", [
            ("multi-agent-rl", "自我对弈与多智能体", "⬜", ""),
            ("exploration", "好奇心、稀疏奖励下的探索", "⬜", ""),
            ("rl-with-foundation-models", "大模型当奖励或规划器、强化学习微调 VLA", "⬜", "")]),
    ]),
    ("post-training", "后训练", "预训练好的模型怎么变成能用的助手: 监督微调、偏好对齐、可验证奖励的强化学习、Agent 强化学习、蒸馏、RL 基础设施。强化学习基础在 RL 领域", [
        ("P0 总览", [("post-training-map", "后训练分几个阶段、各家的实际流程", "⬜", "")]),
        ("P1 监督微调", [
            ("sft", "指令微调; loss 只算在模型的回答上", "⬜", "基座"),
            ("instruction-data", "训练数据: 人工、合成、拒绝采样、质量过滤", "⬜", "基座"),
            ("lora-qlora", "只训一小部分参数; 8GB 也能微调", "⬜", "v0.5"),
            ("tool-finetune", "监督微调让小模型稳定输出指定格式的工具调用 (待改造, 现为通用 QLoRA 演示)", "🟡", "v0.5")]),
        ("P2 偏好对齐", [
            ("reward-models", "奖励模型、奖励被钻空子", "⬜", ""),
            ("rlhf-ppo", "经典 RLHF: PPO、KL 约束、价值头", "⬜", ""),
            ("dpo", "不用奖励模型的偏好优化及其变体", "⬜", "基座")]),
        ("P3 可验证奖励的强化学习", [
            ("grpo", "R1 的方法: 对错能自动判的任务直接当奖励", "⬜", "基座"),
            ("grpo-variants", "DAPO、GSPO、CISPO 各修了什么 🔥", "⬜", ""),
            ("process-rewards", "只看结果还是每步打分; 验证器", "⬜", ""),
            ("rl-dynamics", "探索、熵、训练为什么会崩 🔥", "⬜", "")]),
        ("P4 Agent 强化学习", [
            ("agentic-rl", "多轮、带工具、带环境的强化学习; 长轨迹的功劳分配 🔥", "⬜", ""),
            ("tool-use-rl", "用任务成败当奖励, 以强化学习训练模型调用工具", "⬜", "v0.5")]),
        ("P5 蒸馏", [
            ("reasoning-distill", "用大模型的推理过程教小模型", "⬜", ""),
            ("on-policy-distill", "小模型自己生成、大模型逐 token 纠正 🔥", "⬜", "")]),
        ("P6 RL 基础设施", [
            ("rl-infra", "生成与训练两套引擎、异步、权重同步 🔥", "⬜", ""),
            ("mini-rl-loop", "自己写最小的 RL 训练循环", "⬜", "基座"),
            ("framework-reading", "读 slime、OpenRLHF、verl 的源码", "⬜", "")]),
        ("P7 安全对齐", [("safety-alignment", "拒答、越狱、提示注入", "⬜", "")]),
    ]),
    ("embodied", "具身智能", "让大脑在真实的身体上工作: 本体与感知、规划、从示范和试错中学、世界模型、具身基础模型、系统与部署", [
        ("E0 总览", [("embodied-map", "感知 → 决策 → 行动; 大模型在机器人里放在哪一层", "⬜", "")]),
        ("E1 本体与感知", [
            ("kinematics", "坐标系、位姿变换、差速驱动运动学、里程计", "⬜", "v0.2"),
            ("state-estimation", "互补滤波、卡尔曼滤波、编码器 + IMU 融合", "⬜", "v0.2"),
            ("perception-3d", "相机模型、深度、点云", "⬜", "v0.6"),
            ("slam", "边走边建图、定位入门", "⬜", "")]),
        ("E2 规划与经典控制", [
            ("motion-planning", "A*、RRT: 绕开障碍找路", "⬜", "v0.3"),
            ("mpc", "模型预测控制", "⬜", "")]),
        ("E3 从示范和试错中学", [
            ("imitation-learning", "行为克隆、DAgger", "⬜", ""),
            ("action-chunking", "ACT: 一次预测一小段动作", "⬜", ""),
            ("diffusion-policy", "用扩散模型生成动作", "⬜", ""),
            ("sim2real", "仿真到真车: 差距在哪、怎么测、域随机化、系统辨识", "⬜", "v0.7"),
            ("real-robot-rl", "真机器人上的强化学习", "⬜", "")]),
        ("E4 世界模型 (重点)", [
            ("world-model-basics", "世界模型是什么: 预测下一刻、在隐空间里学动力学 🔥", "⬜", "v0.7"),
            ("learned-dynamics", "动手: 用神经网络从数据学世界模型, 在里面规划和训练 🔥", "⬜", "v0.7"),
            ("video-world-models", "生成画面的世界模型 (Genie 3、Cosmos) 🔥", "⬜", ""),
            ("latent-world-models", "只预测表示的路线 (V-JEPA 2) 🔥", "⬜", ""),
            ("world-action-models", "画面和动作一起预测 🔥", "⬜", ""),
            ("world-model-data", "用世界模型生成数据, 让真机器人少用真实数据 🔥", "⬜", "")]),
        ("E5 具身基础模型", [
            ("vla", "视觉-语言-动作模型: 主干 + 动作头, 开源模型对比 🔥", "⬜", ""),
            ("vla-finetune", "用 LeRobot 微调小号 VLA (8GB 是否够用先实测) 🔥", "⬜", ""),
            ("robot-data", "遥操作采集与开源数据集", "⬜", "")]),
        ("E6 系统与部署", [
            ("edge-deployment", "机器人上的推理: 延迟预算、哪部分在哪跑", "⬜", "v0.6"),
            ("robot-software", "ROS 2 入门、控制循环频率、实时性", "⬜", ""),
            ("embodied-eval", "仿真基准与真机评测", "⬜", ""),
            ("embodied-safety", "VLA 的安全、急停、仿真安全不等于真车安全", "⬜", "v0.7")]),
    ]),
    ("multimodal", "多模态", "图像、声音怎么变成模型能处理的 token: 视觉、语音、生成模型基础 (具身与 LLM 领域共用)、多模态大模型", [
        ("M0 总览", [("multimodal-map", "理解与生成两个方向的全景", "⬜", "")]),
        ("M1 视觉", [
            ("image-basics", "像素、颜色、卷积", "⬜", ""),
            ("cnn", "卷积网络、ResNet", "⬜", ""),
            ("detection", "目标检测: 小设备上的轻量检测器", "⬜", "v0.6"),
            ("vit", "Vision Transformer: 图片切块当 token", "⬜", "基座"),
            ("self-supervised-vision", "CLIP、DINO: 不靠标注学图像表示", "⬜", "")]),
        ("M2 音频与语音", [
            ("audio-basics", "采样、频谱、梅尔谱", "⬜", "v1.0"),
            ("keyword-spotting", "唤醒词识别", "⬜", "v1.0"),
            ("asr", "语音识别", "⬜", "v1.0"),
            ("tts", "语音合成", "⬜", "v1.0"),
            ("audio-codecs", "把声音变成离散 token 🔥", "⬜", "基座")]),
        ("M3 生成模型基础", [
            ("vae", "变分自编码器", "⬜", ""),
            ("diffusion", "DDPM、扩散原理、流匹配", "⬜", ""),
            ("image-generation", "潜空间扩散、DiT", "⬜", ""),
            ("video-generation", "视频生成, 通往视频世界模型", "⬜", "")]),
        ("M4 多模态大模型", [
            ("vision-language-alignment", "CLIP 式对齐", "⬜", "基座"),
            ("vlm", "视觉编码器 + 投影层 + 语言模型", "⬜", "基座"),
            ("native-multimodal", "早期融合、不用单独编码器 🔥", "⬜", "基座"),
            ("omni-models", "能听能说、实时全双工的全能模型 🔥", "⬜", "基座"),
            ("multimodal-eval", "多模态怎么评测", "⬜", "")]),
    ]),
]


def unit_link(domain, name):
    d = os.path.join(LIB, domain, name)
    for f in ("TUTORIAL.md", "README.md"):
        if os.path.exists(os.path.join(d, f)):
            return f"{name}/{f}"
    return None


def domain_table(domain, segments, prefix):
    """prefix: 从目标文件到 library/<domain>/ 的相对前缀 ('' 或 'domain/')。"""
    lines = ["| 段 | 单元 | 讲什么 | 状态 | 用到它的地方 |", "|---|---|---|---|---|"]
    for seg, units in segments:
        for i, (name, what, status, used) in enumerate(units):
            link = unit_link(domain, name)
            cell = f"[{name}]({prefix}{link})" if link else name
            lines.append(f"| {seg if i == 0 else ''} | {cell} | {what} | {status} | {used} |")
    return "\n".join(lines)


def counts(segments):
    units = [u for _, us in segments for u in us]
    done = sum(u[2] == "✅" for u in units)
    doing = sum(u[2] == "🟡" for u in units)
    return len(units), done, doing


def render_library():
    total = [0, 0, 0]
    parts = []
    summary = ["| 领域 | 讲什么 | 单元数 | ✅ | 🟡 |", "|---|---|---|---|---|"]
    for key, zh, intro, segs in DOMAINS:
        n, done, doing = counts(segs)
        total = [total[0] + n, total[1] + done, total[2] + doing]
        summary.append(f"| [{zh}](#{key}) | {intro.split('; ')[0].split('。')[0]} | {n} | {done} | {doing} |")
        parts.append(f'<a id="{key}"></a>\n\n### {zh} ([{key}/]({key}/README.md))\n\n{domain_table(key, segs, key + "/")}')
    summary.append(f"| **合计** | | **{total[0]}** | **{total[1]}** | **{total[2]}** |")
    return "\n".join(summary) + "\n\n" + "\n\n".join(parts)


def replace_block(path, body, tag="catalog"):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    start, end = f"<!-- {tag}:start -->", f"<!-- {tag}:end -->"
    if start not in text or end not in text:
        raise SystemExit(f"{path} 里没有 {tag} 标记")
    new = re.sub(re.escape(start) + ".*?" + re.escape(end),
                 lambda m: start + "\n" + body + "\n" + end, text, flags=re.S)
    return text, new


def render_status():
    """根目录 README 顶部的状态: 已完成和进行中的单元 (带链接), 规划中的只给个数。"""
    done, doing, total = [], [], 0
    for key, zh, intro, segs in DOMAINS:
        for _, units in segs:
            for name, what, status, used in units:
                total += 1
                link = unit_link(key, name)
                cell = f"[{name}](library/{key}/{link})" if link else name
                if status == "✅":
                    done.append(cell)
                elif status == "🟡":
                    doing.append(cell)
    return "\n".join([
        "| 知识库单元 | 个数 | 哪些 |", "|---|---|---|",
        f"| ✅ 写完了 | {len(done)} | {' · '.join(done)} |",
        f"| 🟡 有内容, 待补 | {len(doing)} | {' · '.join(doing)} |",
        f"| ⬜ 规划中 | {total - len(done) - len(doing)} | 见 [知识库总目录](library/README.md) |",
    ])


def outputs():
    """[(路径, 生成后的全文)], 供测试比对。"""
    res = []
    p = os.path.join(ROOT, "README.md")
    res.append((p, replace_block(p, render_status(), "status")[1]))
    p = os.path.join(LIB, "README.md")
    res.append((p, replace_block(p, render_library())[1]))
    for key, zh, intro, segs in DOMAINS:
        p = os.path.join(LIB, key, "README.md")
        res.append((p, replace_block(p, domain_table(key, segs, ""))[1]))
    return res


if __name__ == "__main__":
    for path, text in outputs():
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("写入", os.path.relpath(path, ROOT))
    n = sum(counts(s)[0] for *_, s in DOMAINS)
    print(f"共 {n} 个单元")
