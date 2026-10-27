# B1 电子入门与仿真器

> 主线第 B1 章, 服务于 **v0.1 (手机遥控小车)**。不需要买硬件: 全部在浏览器里的 [Wokwi](https://wokwi.com) 仿真器上完成, 芯片程序用 MicroPython。
> 讲义: [TUTORIAL.md](TUTORIAL.md)。

五个案例, 每个都是"先用朴素办法, 看它怎么失败":

| 案例 | 朴素办法怎么失败 (仿真实测) | 引出的概念 | 小车上用在哪 |
|---|---|---|---|
| ① 点灯 | LED 直接接引脚: 电流只受内阻限制 | 欧姆定律、限流电阻、E12 标准值 | 所有接线的基础 |
| ② 读按钮 | 按 4 次, 录波里有 644 个下降沿 | 悬空、上拉电阻、电平阈值、弹跳与消抖 | 急停开关、编码器 |
| ③ 调亮度 | 软件 PWM 慢了 18.5%, CPU 一忙周期拉长到 2 ms | 硬件 PWM (LEDC)、占空比、gamma | **电机调速** |
| ④ 读旋钮 | 读数换算成电压有偏差 | ADC、衰减、量程、校准 | 遥控油门、电池电压 |
| ⑤ 同时做两件事 | `sleep` 期间按的按钮, 21 次漏掉 16 次 | 非阻塞、固定频率循环 | **PID 循环、断网自动停车** |

## 结构

```
B1-electronics/
├── TUTORIAL.md          讲义
├── b1core.py            纯逻辑: 芯片和电脑上都能跑 (练习的参考答案)
├── firmware/            每个案例在芯片上跑的程序
├── bundle.py            把 b1core.py 和一个案例拼成单个 main.py, 方便粘贴到 Wokwi 网页版
├── wokwi/diagram.json   电路图 (所有案例共用, 含逻辑分析仪)
├── results/wokwi/       Wokwi 实测: 逻辑分析仪录波 (.cvcd) 与串口记录
├── experiments/         分析脚本: 由录波和串口记录算出讲义里的数字
├── figures/             make_diagrams.py (结构图) 与 make_charts.py (数据图)
└── exercises/           三级练习 + 测试 (在电脑上用 pytest 判对错)
```

## 运行

在 Wokwi 网页版跑一个案例 (详见讲义第 0 节):

```bash
cd library/hardware/electronics
python bundle.py firmware/case3_pwm.py -o main.py   # 粘贴到 wokwi.com 的 main.py; diagram.json 粘贴 wokwi/diagram.json
```

由仓库里的实测数据重新生成讲义的数字和图:

```bash
python experiments/pwm_analysis.py
python experiments/bounce_analysis.py
python experiments/blocking_analysis.py
python experiments/adc_analysis.py
python figures/make_charts.py
python figures/make_diagrams.py
```

练习:

```bash
cd exercises && python -m pytest test_ex1.py -q
```

## 仿真与真板

本章的数字都来自 Wokwi 仿真, 讲义里每处都标明了。仿真器限速 CPU、不模拟悬空引脚的干扰、没有直流电机, 这些差别见讲义第 6 节; 买了开发板之后, 各节的"真板对照 ⏳"会补上实测数字。
