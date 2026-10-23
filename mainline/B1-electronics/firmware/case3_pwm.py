"""
案例 ③ 调亮度: 软件 PWM vs 硬件 PWM, 谁的波形更准?

目标: 1 kHz、占空比 50% 的方波 (周期 1000 微秒, 高电平 500 微秒), 从 GPIO 6 输出。
本程序只负责产生波形, 测量交给 Wokwi 的逻辑分析仪 (D0 接 GPIO 6), 它按仿真时间记录每一次跳变,
不占芯片的 CPU。GPIO 4 是"阶段标记" (D1): 每进入下一个阶段就翻转一次, 分析时据此把波形切成四段。

四个阶段, 各约 300 个周期:
  1. software_quiet: 软件 PWM (朴素办法), 主循环里 "开 → 等 500 µs → 关 → 等 500 µs";
  2. software_busy:  同上, 但每 25 个周期顺手往串口打一行字 (小车上 CPU 还要处理网络、传感器);
  3. hardware_quiet: 硬件 PWM, 交给芯片里的 LEDC 外设, CPU 设置一次就不管了;
  4. hardware_busy:  硬件 PWM 照常输出, CPU 同时不停地打字。

为什么不在芯片上用中断自己测: ESP32 版 MicroPython 的引脚中断不支持 hard=True,
回调要排队等主程序有空才执行, 时间戳会被推迟, 密集的跳变还会丢 (讲义第 3 节)。
"""

import time

from machine import PWM, Pin

out = Pin(6, Pin.OUT, value=0)
mark = Pin(4, Pin.OUT, value=0)


def next_phase(name):
    mark.value(1 - mark.value())
    print("@PHASE,%s;" % name)


def software_pwm(cycles, busy):
    for i in range(cycles):
        out.value(1)
        time.sleep_us(500)
        out.value(0)
        time.sleep_us(500)
        if busy and i % 25 == 0:
            print("@BUSY,%d;" % i)       # 模拟"顺手干点别的"


time.sleep_ms(20)
next_phase("software_quiet")
software_pwm(300, busy=False)

next_phase("software_busy")
software_pwm(300, busy=True)

next_phase("hardware_quiet")
pwm = PWM(out, freq=1000, duty_u16=32768)
time.sleep_ms(300)

next_phase("hardware_busy")
t0 = time.ticks_ms()
i = 0
while time.ticks_diff(time.ticks_ms(), t0) < 300:
    print("@BUSY,%d;" % i)                # CPU 一直在忙, 硬件 PWM 不受影响吗?
    i += 1

next_phase("end")
print("@PWM_FREQ,%d;" % pwm.freq())      # LEDC 实际能产生的频率 (可能和设定值略有差别)
pwm.deinit()
out.value(0)
print("@DONE;")
