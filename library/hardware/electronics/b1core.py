"""
B1 的纯逻辑部分: 不碰任何硬件, 在 MicroPython (芯片 / Wokwi) 和电脑上的 CPython 里都能跑。

为什么要把逻辑和硬件分开:
  - 芯片上的程序很难单步调试, 逻辑错了只能看 LED 猜;
  - 分开以后, 逻辑可以在电脑上用 pytest 测 (练习就是这么判对错的),
    芯片上的 firmware/*.py 只剩"读引脚 → 交给这里 → 写引脚"几行。
  - 所有和时间有关的函数都**把当前时间当参数传进来**, 不在内部读时钟。
    这样测试可以用一个假时钟精确控制时间, 不用真的等。

讲义: TUTORIAL.md。各案例对应的芯片程序在 firmware/ 下。
"""

try:  # MicroPython: 毫秒计数会回绕, 必须用 ticks_diff 求差
    from time import ticks_diff
except ImportError:  # CPython (电脑上测试): 整数不会回绕, 直接相减
    def ticks_diff(a, b):
        return a - b


# ─────────────────────────────── 案例 ① 点灯: 欧姆定律 ───────────────────────────────

def led_resistor(v_supply, v_forward, i_ma):
    """
    给 LED 串多大的电阻, 才能让电流等于 i_ma 毫安。

    电阻两端的电压 = 电源电压 - LED 正向压降, 再由欧姆定律 R = U / I。
    返回欧姆 (浮点数)。例: 3.3 V 引脚、红色 LED 压降 2.0 V、目标 5 mA → 260 Ω。
    """
    if i_ma <= 0:
        raise ValueError("目标电流必须为正")
    if v_supply <= v_forward:
        raise ValueError("电源电压不高于 LED 压降, LED 不会亮")
    return (v_supply - v_forward) / (i_ma / 1000.0)


def led_current_ma(v_supply, v_forward, r_ohm):
    """已知电阻, 反过来算 LED 电流 (毫安)。"""
    if r_ohm <= 0:
        raise ValueError("电阻必须为正")
    return max(0.0, (v_supply - v_forward) / r_ohm * 1000.0)


# E12 系列: 市面上最常见的一组标准电阻值 (每个数量级 12 个)
E12 = (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2)


def next_e12(r_ohm):
    """
    不小于 r_ohm 的最小 E12 标准电阻值。

    算出来的 260 Ω 买不到, 只能买 270 Ω。往大取而不是往小取: 电阻大一点电流就小一点, 更安全。
    """
    if r_ohm <= 0:
        raise ValueError("电阻必须为正")
    decade = 1.0
    while decade * 10 <= r_ohm:
        decade *= 10
    while decade > r_ohm:
        decade /= 10
    for base in E12 + (10.0,):
        value = base * decade
        if value >= r_ohm * (1 - 1e-9):  # 容忍浮点误差: 270 算出来可能是 269.99999
            return round(value, 6)
    return round(10.0 * decade, 6)


# ─────────────────────────────── 案例 ② 读按钮: 消抖 ───────────────────────────────

class Debouncer:
    """
    按键消抖: 原始读数必须**连续 stable_ms 毫秒不变**, 才承认它是新状态。

    按键的金属触点在按下和松开的瞬间会弹跳, 原始读数在约 1 ms 里来回翻几十次。
    直接数下降沿, 按一下会被数成好几下 (Wokwi 实测见讲义第 2 节)。

    active_low=True 对应"一端接引脚 (开上拉), 一端接地"的常见接法: 按下读 0, 松开读 1。
    presses 统计"松开 → 按下"的次数。
    """

    def __init__(self, stable_ms=20, active_low=True, initial_raw=None):
        self.stable_ms = stable_ms
        self.active_low = active_low
        idle = 1 if active_low else 0        # 没按时的原始读数
        raw = idle if initial_raw is None else initial_raw
        self.stable = raw                    # 已经承认的状态 (原始电平)
        self._candidate = raw                # 正在观察、还没承认的电平
        self._since = None                   # candidate 从什么时候开始保持不变
        self.presses = 0

    def update(self, raw, now_ms):
        """喂入一次原始读数和当前时间 (毫秒), 返回消抖后的电平。"""
        if raw != self._candidate:           # 电平变了: 重新开始计时
            self._candidate = raw
            self._since = now_ms
        elif raw != self.stable and self._since is not None \
                and ticks_diff(now_ms, self._since) >= self.stable_ms:
            self.stable = raw                # 新电平保持够久了, 承认它
            if self.pressed:
                self.presses += 1
        return self.stable

    @property
    def pressed(self):
        return (self.stable == 0) if self.active_low else (self.stable == 1)


# ─────────────────────────────── 案例 ③ 调亮度: PWM ───────────────────────────────

DUTY_MAX = 65535  # MicroPython 的 PWM.duty_u16() 用 16 位表示占空比: 0 = 全灭, 65535 = 全亮


def duty_u16(fraction):
    """占空比 (0~1 的小数) → duty_u16 的整数。超出范围的输入截到 [0, 1]。"""
    if fraction <= 0:
        return 0
    if fraction >= 1:
        return DUTY_MAX
    return int(fraction * DUTY_MAX + 0.5)


def perceived_to_duty(level, gamma=2.2):
    """
    "看起来的亮度" (0~1) → 占空比 (0~1)。

    人眼对亮度的感受不是线性的: 占空比从 10% 到 20% 看起来变化很大, 从 80% 到 90% 几乎看不出来。
    用 gamma 曲线 duty = level ** gamma 补偿, 呼吸灯才会"匀速"变亮变暗。2.2 是显示器常用的值。
    """
    if level <= 0:
        return 0.0
    if level >= 1:
        return 1.0
    return level ** gamma


# ─────────────────────────────── 案例 ④ 读旋钮: ADC ───────────────────────────────

def adc_fraction(raw, bits=12):
    """ADC 原始读数 → 0~1 的比例。ESP32-S3 的 ADC 是 12 位: 0 ~ 4095。"""
    full = (1 << bits) - 1
    return min(max(raw, 0), full) / full


class PiecewiseLinear:
    """
    分段线性插值: 用几组实测的 (输入, 输出) 点做校准表。

    ADC 读数和真实电压不是严格成正比的 (数据手册 5.5 节给出了非线性和量程上限),
    实测几个点、中间插值, 是最朴素也最常用的校准办法。超出范围时用两端的值。
    """

    def __init__(self, points):
        pts = sorted(points)
        if len(pts) < 2:
            raise ValueError("至少需要两个点")
        self.xs = [p[0] for p in pts]
        self.ys = [p[1] for p in pts]

    def __call__(self, x):
        xs, ys = self.xs, self.ys
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if x <= xs[i]:
                t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
                return ys[i - 1] + t * (ys[i] - ys[i - 1])
        return ys[-1]


# ─────────────────────────────── 案例 ⑤ 同时做两件事: 非阻塞 ───────────────────────────────

class Every:
    """
    非阻塞定时器: "每隔 period_ms 毫秒该做一次了吗?"

    主循环里反复调用 ready(now), 不睡觉, 所以别的事 (读按钮、看网络) 不会被耽误。
    这是 B3 固定频率 PID 循环、B4 断网自动停车的基础。

    下一次的时刻按 "上一次计划的时刻 + 周期" 推进, 而不是 "实际执行的时刻 + 周期":
    这样即使某一次晚了几毫秒, 误差也不会一圈圈累积。
    如果落后超过一整个周期 (主循环被卡住了), 不补跑错过的那几次, 只记进 overruns。
    """

    def __init__(self, period_ms, start_ms=0):
        if period_ms <= 0:
            raise ValueError("周期必须为正")
        self.period_ms = period_ms
        self._next = start_ms + period_ms
        self.overruns = 0

    def ready(self, now_ms):
        if ticks_diff(now_ms, self._next) < 0:
            return False
        self._next += self.period_ms
        if ticks_diff(now_ms, self._next) >= 0:          # 落后了一个周期以上
            behind = ticks_diff(now_ms, self._next) // self.period_ms + 1
            self.overruns += behind
            self._next += behind * self.period_ms
        return True


class Blinker:
    """非阻塞闪烁: 每过半个周期翻转一次。update(now) 返回 LED 此刻该亮 (True) 还是灭。"""

    def __init__(self, period_ms, start_ms=0):
        self.on = False
        self._tick = Every(period_ms // 2, start_ms)

    def update(self, now_ms):
        if self._tick.ready(now_ms):
            self.on = not self.on
        return self.on


class BreathingLed:
    """
    练习 3 的参考答案: 按住按钮时 LED "呼吸", 松开后渐暗熄灭。

    - 按住: 看起来的亮度按三角波在 0 和 1 之间往返, 一个来回 period_ms 毫秒, 经 gamma 换算成占空比;
    - 松开: 从松开时的占空比线性降到 0, 用时 fade_ms 毫秒 (默认 500, 验收要求 1 秒内熄灭)。
    update(pressed, now) 返回 duty_u16 整数。
    """

    def __init__(self, period_ms=2000, fade_ms=500):
        self.period_ms = period_ms
        self.fade_ms = fade_ms
        self._start = None       # 这一轮按下的时刻
        self._release = None     # 松开的时刻
        self._fade_from = 0      # 松开时的占空比
        self.duty = 0

    def update(self, pressed, now_ms):
        if pressed:
            if self._start is None:
                self._start = now_ms
                self._release = None
            phase = (ticks_diff(now_ms, self._start) % self.period_ms) / self.period_ms
            level = 2 * phase if phase < 0.5 else 2 * (1 - phase)   # 三角波 0 → 1 → 0
            self.duty = duty_u16(perceived_to_duty(level))
        else:
            if self._start is not None:                 # 刚松开
                self._start = None
                self._release = now_ms
                self._fade_from = self.duty
            if self._release is not None:
                t = ticks_diff(now_ms, self._release)
                remain = 1 - t / self.fade_ms
                self.duty = int(self._fade_from * remain) if remain > 0 else 0
        return self.duty
