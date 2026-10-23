"""
案例 ④ 读旋钮: ADC 读数和旋钮位置成正比吗?

电位器两端接 3.3 V 和 GND, 中间脚 (SIG) 接 GPIO 1。旋钮转到不同位置, 中间脚的电压在 0 ~ 3.3 V 之间变化。
衰减设为 11 dB (MicroPython 里叫 ATTN_11DB), 这是量程最大的一档。
每个位置读 64 次取平均, 同时打印三种读数:
  - read_u16(): 0 ~ 65535 的原始读数;
  - read_uv():  芯片用出厂校准数据换算出的电压 (微伏);
  - 两者的标准差: 读数有多抖。

POT_VALUE 是电位器旋钮的位置 (Wokwi 里 0 ~ 1023), 仅用于打印, 方便和读数对照;
在 Wokwi 里换位置时, 同时改 diagram.json 里 pot1 的 "value" 和这里的数字。
"""

import time

from machine import ADC, Pin

POT_VALUE = 512

adc = ADC(Pin(1), atten=ADC.ATTN_11DB)
time.sleep_ms(10)
raws, uvs = [], []
for _ in range(64):
    raws.append(adc.read_u16())
    uvs.append(adc.read_uv())
    time.sleep_ms(1)


def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


rm, rs = mean_sd(raws)
um, us = mean_sd(uvs)
print("@ADC,pot=%d,raw_mean=%.1f,raw_sd=%.1f,mv_mean=%.1f,mv_sd=%.2f;" % (POT_VALUE, rm, rs, um / 1000, us / 1000))
print("@DONE;")
