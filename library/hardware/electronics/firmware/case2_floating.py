"""
案例 ② 前半: 悬空的引脚读出来是什么?

GPIO 10 什么都不接, 分别在 "不开上下拉" "开上拉" "开下拉" 三种设置下各读 2000 次, 数一数读到 1 的次数。
真板上悬空引脚的读数受手指靠近、旁边导线的干扰影响, 可能乱跳; 仿真器里是什么样, 以实测为准。
"""

import time

from machine import Pin

N = 2000
for name, pull in (("none", None), ("up", Pin.PULL_UP), ("down", Pin.PULL_DOWN)):
    p = Pin(10, Pin.IN, pull)
    time.sleep_ms(5)                      # 改了上下拉, 等电平稳定
    ones = 0
    flips = 0
    prev = p.value()
    for _ in range(N):
        v = p.value()
        ones += v
        flips += v != prev
        prev = v
    print("@FLOAT,pull=%s,n=%d,ones=%d,flips=%d;" % (name, N, ones, flips))
print("@DONE;")
