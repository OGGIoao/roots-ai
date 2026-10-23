"""
案例 ⑤ 同时做两件事: LED 每秒闪一次, 同时不能漏掉按钮。

为了让实验可重复, 不用手去按, 而是让芯片自己"按": GPIO 8 接到 GPIO 9,
一个定时器每 937 毫秒把 GPIO 8 拉低 120 毫秒, 相当于有人按了 120 毫秒的按钮。
GPIO 9 当作按钮输入 (按下读 0)。937 不是 500 的整数倍, 按下的时刻会落在闪烁节拍的各个位置。

  A. 阻塞写法 (朴素办法): 亮 → sleep 500 ms → 灭 → sleep 500 ms, 每次醒来看一眼按钮;
  B. 非阻塞写法: 主循环不睡觉, 用 b1core.Every / Blinker 判断"该闪了吗", 每 1 毫秒看一眼按钮。

各跑 20 秒, 比较: 一共"按"了几次, 程序发现了几次, 发现得有多晚。
"""

import time
from array import array

from machine import Pin, Timer

from b1core import Blinker, Every

PRESS_PERIOD_MS = 937
PRESS_MS = 120
RUN_MS = 20000

gen = Pin(8, Pin.OUT, value=1)            # 1 = 没按
btn = Pin(9, Pin.IN, Pin.PULL_UP)
led = Pin(4, Pin.OUT)

press_t = array("i", [0] * 64)            # 每次"按下"的时刻 (毫秒)
n_press = array("i", [0])


def release(_):
    gen.value(1)


def press(_):
    i = n_press[0]
    if i < 64:
        press_t[i] = time.ticks_ms()
    n_press[0] = i + 1
    gen.value(0)
    Timer(1).init(mode=Timer.ONE_SHOT, period=PRESS_MS, callback=release)


def run(name, loop):
    n_press[0] = 0
    gen.value(1)
    found = []                            # 发现按下的时刻
    t_start = time.ticks_ms()
    Timer(0).init(mode=Timer.PERIODIC, period=PRESS_PERIOD_MS, callback=press)
    loop(t_start, found)
    Timer(0).deinit()
    gen.value(1)
    n = min(n_press[0], 64)
    # 把每次"发现"对应到最近一次在它之前发生的"按下", 算延迟
    lat = []
    for f in found:
        before = [press_t[i] for i in range(n) if time.ticks_diff(f, press_t[i]) >= 0]
        if before:
            lat.append(time.ticks_diff(f, max(before)))
    mean = sum(lat) / len(lat) if lat else -1
    print("@BLOCK,mode=%s,pressed=%d,found=%d,lat_mean=%.1f,lat_max=%d;"
          % (name, n, len(found), mean, max(lat) if lat else -1))
    print("@LAT,%s,%s;" % (name, ",".join(str(x) for x in lat)))


def blocking(t_start, found):
    prev = 1
    while time.ticks_diff(time.ticks_ms(), t_start) < RUN_MS:
        led.value(1)
        time.sleep_ms(500)
        v = btn.value()
        if prev == 1 and v == 0:
            found.append(time.ticks_ms())
        prev = v
        led.value(0)
        time.sleep_ms(500)
        v = btn.value()
        if prev == 1 and v == 0:
            found.append(time.ticks_ms())
        prev = v


def non_blocking(t_start, found):
    blink = Blinker(1000, t_start)
    poll = Every(1, t_start)
    prev = 1
    while True:
        now = time.ticks_ms()
        if time.ticks_diff(now, t_start) >= RUN_MS:
            break
        led.value(blink.update(now))
        if poll.ready(now):
            v = btn.value()
            if prev == 1 and v == 0:
                found.append(now)
            prev = v


run("blocking", blocking)
run("non_blocking", non_blocking)
print("@DONE;")
