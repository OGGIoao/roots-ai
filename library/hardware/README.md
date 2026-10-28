# 知识库 · 硬件

> 让东西在物理世界里动起来: 电路、单片机与固件、电机、传感器、连接、安全、端侧 AI。总目录见 [library/README.md](../README.md#hardware)。

**三层算力**: 单片机 (实时控制, 小车主控就是它) / 单板电脑 (端侧 AI 推理) / 电脑与云 (大模型)。

所有能仿真的单元都**先在浏览器仿真器里做** ([Wokwi](https://wokwi.com), 不用买板子), 再在真板上对照; 仿真和真板哪里不同, 讲义里专门列出。小车的开发板选型、引脚分配和物料清单属于小车项目, 见 [projects/car/hardware](../../projects/car/hardware/README.md)。

**H7 端侧 AI**: tinyml 用小车自带的感知板; 另外三个单元需要单板电脑, 它不进小车预算, 作为可选。没买时讲原理 + 在笔记本上用 CPU 模拟, 真板数字标 "没有数据"。

## 单元

<!-- catalog:start -->
| 段 | 单元 | 讲什么 | 状态 | 用到它的地方 |
|---|---|---|---|---|
| H0 总览 | hardware-map | 单片机、单板电脑、电脑与云三层各干什么 | ⬜ |  |
| H1 电子基础 | [electronics](electronics/TUTORIAL.md) | 电路与 ESP32 入门: 点灯、按钮消抖、PWM、ADC、非阻塞循环 (浏览器仿真) | 🟡 | v0.1 |
|  | power | 电池、稳压、电流预算; 电机干扰为什么让单片机重启 | ⬜ | v0.1 |
|  | datasheets-tools | 读数据手册和原理图; 万用表、逻辑分析仪、示波器 | ⬜ | v0.1 |
|  | pcb-basics | 从面包板到自己画电路板 (KiCad) | ⬜ |  |
| H2 单片机与固件 | mcu-architecture | GPIO、定时器、中断、DMA、内存 | ⬜ | v0.1 |
|  | comm-protocols | UART、I2C、SPI、I2S、CAN | ⬜ | v0.3 |
|  | rtos | FreeRTOS 多任务; MicroPython 与 C++ 各适合什么 | ⬜ | v0.6 |
|  | firmware-engineering | 编译、烧录、OTA、日志、在电脑上测固件逻辑 | ⬜ | v0.1 |
| H3 执行与控制 | motor-driver | 直流电机与 H 桥驱动 | ⬜ | v0.1 |
|  | encoder | 编码器与测速 | ⬜ | v0.2 |
|  | pid | PID 在单片机上的实现与调参 | ⬜ | v0.2 |
|  | servo-stepper | 舵机与步进电机 | ⬜ |  |
| H4 感知硬件 | sensors | 测距与 IMU | ⬜ | v0.3 |
|  | camera | 摄像头采集 | ⬜ | v0.6 |
|  | audio | 麦克风、I2S、喇叭 | ⬜ | v1.0 |
| H5 连接 | wifi-remote | WiFi 与网页遥控 | ⬜ | v0.1 |
|  | robot-networking | 延迟、断线、MQTT / WebSocket: 电脑上的大模型和地上的车之间怎么连才稳 | ⬜ | v0.3 |
| H6 安全 | safety-firmware | 状态机、看门狗与急停 | ⬜ | v0.3 |
| H7 端侧 AI | tinyml | 在单片机上跑量化小模型 (唤醒词), 用小车的感知板 🔥 | ⬜ | v1.0 |
|  | edge-npu | 单板电脑的 NPU: 模型转换、算子支持 (需要可选的单板电脑) 🔥 | ⬜ |  |
|  | edge-gpu | Jetson 一类带 GPU 的板子与 TensorRT (需要可选的单板电脑) 🔥 | ⬜ |  |
|  | on-device-llm | 在板子上跑大模型: INT4、llama.cpp (需要可选的单板电脑) 🔥 | ⬜ |  |
|  | edge-system-design | 功能放单片机、板子还是云: 延迟、功耗、成本怎么算 | ⬜ |  |
<!-- catalog:end -->

**前置**: 会写 Python。不需要电子基础, 从 "电压、电流、电阻是什么" 讲起。
