# 知识库 · 硬件

> 让小车在物理世界里动起来要用到的知识: 电路、芯片、电机、传感器、控制。总目录见 [library/README.md](../README.md#硬件-hardware)。

所有硬件单元都**先在浏览器仿真器里做** ([Wokwi](https://wokwi.com), 不用买板子), 再在真板上对照; 仿真和真板哪里不同, 讲义里专门列出。开发板选型见 [hardware/README.md](../../hardware/README.md)。

## 单元

| 顺序 | 单元 | 状态 |
|---|---|---|
| 1 | [electronics](electronics/TUTORIAL.md) 电路与 ESP32 入门 | 🟡 仿真完成 |
| 2 | motor-driver 直流电机与 H 桥驱动 | ⬜ |
| 3 | wifi-remote WiFi 与网页遥控 | ⬜ |
| 4 | encoder 编码器与测速 | ⬜ |
| 5 | pid PID 控制 | ⬜ |
| 6 | sensors 测距与 IMU | ⬜ |
| 7 | safety-firmware 状态机、看门狗与急停 | ⬜ |
| 8 | camera 摄像头采集 | ⬜ |

**前置**: 会写 Python。不需要电子基础, 从 "电压、电流、电阻是什么" 讲起。
