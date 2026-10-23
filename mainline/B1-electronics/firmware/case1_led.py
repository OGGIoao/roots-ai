"""
案例 ① 点灯: GPIO 4 → 270 Ω 电阻 → 红色 LED → GND, 每秒闪一次。

先在电脑上把电阻算对 (b1core.led_resistor), 再让芯片去点灯。
运行: 用 bundle.py 打包成单个 main.py, 粘贴到 Wokwi (见 README)。
"""

import time

from machine import Pin

from b1core import led_current_ma, led_resistor, next_e12

V_PIN = 3.3        # 引脚输出高电平时的电压 (伏)
V_LED = 2.0        # 红色 LED 的正向压降 (伏), 不同颜色、不同型号不一样, 以买到的 LED 数据手册为准
I_TARGET = 5.0     # 目标电流 (毫安): 远低于引脚的输出能力, LED 也已经足够亮

r = led_resistor(V_PIN, V_LED, I_TARGET)
r_std = next_e12(r)
print("@CALC,r_exact=%.1f,r_e12=%.0f,i_ma=%.2f;" % (r, r_std, led_current_ma(V_PIN, V_LED, r_std)))

led = Pin(4, Pin.OUT)
while True:
    led.value(1)
    time.sleep_ms(500)
    led.value(0)
    time.sleep_ms(500)
