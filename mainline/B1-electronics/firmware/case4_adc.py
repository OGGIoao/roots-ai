"""
案例 ④ 读旋钮: ADC 读数和旋钮位置成正比吗?

电位器两端接 3.3 V 和 GND, 中间脚 (SIG) 接 GPIO 1。旋钮转到不同位置, 中间脚的电压在 0 ~ 3.3 V 之间变化。
衰减设为 11 dB (MicroPython 里叫 ATTN_11DB), 这是量程最大的一档。

每秒读 64 次取平均, 打印一条记录, 一边转旋钮一边看:
  - raw:  read_u16(), 0 ~ 65535 的原始读数;
  - mv:   read_uv() / 1000, 芯片用出厂校准数据换算出的电压 (毫伏);
  - *_sd: 64 次读数的标准差, 读数有多抖。

真板上做这个实验时, 用万用表同时量 GPIO 1 的真实电压, 和这里的读数对照 (讲义第 4 节)。
"""

import time

from machine import ADC, Pin

adc = ADC(Pin(1), atten=ADC.ATTN_11DB)


def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


print("@READY;")
n = 0
while True:
    raws, uvs = [], []
    for _ in range(64):
        raws.append(adc.read_u16())
        uvs.append(adc.read_uv())
        time.sleep_ms(1)
    rm, rs = mean_sd(raws)
    um, us = mean_sd(uvs)
    n += 1
    print("@ADC,n=%d,raw_mean=%.1f,raw_sd=%.1f,mv_mean=%.1f,mv_sd=%.2f;" % (n, rm, rs, um / 1000, us / 1000))
    time.sleep_ms(1000)
