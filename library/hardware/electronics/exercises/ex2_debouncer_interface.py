"""
练习 2 (② 接口): 自己写一个按键消抖器

只给接口和测试, 整个类自己写。不要先看 ../b1core.py 里的 Debouncer; 写完、测试通过之后再去对照。

    class Debouncer:
        def __init__(self, stable_ms=20, active_low=True, initial_raw=None): ...
        def update(self, raw, now_ms) -> int      # 喂入一次原始读数 (0/1) 和当前毫秒数, 返回消抖后的电平
        presses: int                              # 一共承认了几次"按下"
        pressed: bool (属性)                       # 此刻是否处于按下状态

约定:
  - active_low=True: 按下读 0, 松开读 1 (一端接引脚并开上拉、一端接地的接法);
  - initial_raw 为 None 时, 初始电平取"没按"的电平;
  - 原始读数连续 stable_ms 毫秒不变, 才承认它是新电平;
  - 承认了一次"从松开变成按下", presses 加 1。

验收 (python -m pytest test_ex2.py -q):
  1. Wokwi 录下的真实弹跳 (results/wokwi/case2_bounce.cvcd) 每 1 ms 喂一次, 数出来的按下次数与真实次数相同;
  2. 一串人造的弹跳 + 不同按住时长, 每次都恰好数一次, 且承认时刻不早于最后一次弹跳 + stable_ms;
  3. 短于 stable_ms 的毛刺不算按下;
  4. active_low=False 的接法同样正确。

讲义对应: 第 2 节。提示规则: 方向 → 哪一行 → 代码, 每次只升一级。
"""


class Debouncer:
    def __init__(self, stable_ms=20, active_low=True, initial_raw=None):
        raise NotImplementedError

    def update(self, raw, now_ms):
        raise NotImplementedError

    @property
    def pressed(self):
        raise NotImplementedError
