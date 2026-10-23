"""
案例 ② 后半: 按一下, 被数成几下?

按钮一端接 GPIO 5 (开内部上拉), 一端接地: 松开读 1, 按下读 0。
同时用三种办法数按了几下:
  A. 中断数下降沿 (朴素办法): 每次电平从 1 变 0 都算一下;
  B. 轮询数下降沿 (朴素办法): 主循环里不停地读, 看到 1 → 0 就算一下;
  C. 消抖 (b1core.Debouncer, 20 ms): 电平连续 20 ms 不变才承认。
真实的跳变由 Wokwi 逻辑分析仪 (D2 接 GPIO 5) 按仿真时间记录, 作为对照的"标准答案"。

注意: ESP32 版 MicroPython 的引脚中断回调要排队等主程序有空才执行 (不支持 hard=True),
跳变太密时排不下的会被丢掉, 所以 A 数到的个数可能比真实的跳变少。

每按完一次 (1.5 秒没有新的变化), 打印这一次的三个计数, 然后清零等下一次。
"""

import time
from array import array

from machine import Pin

from b1core import Debouncer

falls = array("i", [0])                   # 中断回调里看到的下降沿个数

btn = Pin(5, Pin.IN, Pin.PULL_UP)


def on_fall(p):
    falls[0] += 1


btn.irq(on_fall, Pin.IRQ_FALLING)

print("@READY;")
press_no = 0
while True:
    deb = Debouncer(stable_ms=20)
    poll_falls = 0
    prev = btn.value()
    active = False
    quiet_since = time.ticks_ms()
    while True:
        raw = btn.value()
        now = time.ticks_ms()
        if raw != prev:                   # 有变化: 记为"活动中", 重新计安静时间
            active = True
            quiet_since = now
            if prev == 1 and raw == 0:
                poll_falls += 1
        prev = raw
        deb.update(raw, now)
        if active and raw == 1 and deb.stable == 1 and time.ticks_diff(now, quiet_since) > 1500:
            break
    press_no += 1
    print("@PRESS,no=%d,irq_falls=%d,poll_falls=%d,debounced=%d;"
          % (press_no, falls[0], poll_falls, deb.presses))
    falls[0] = 0
