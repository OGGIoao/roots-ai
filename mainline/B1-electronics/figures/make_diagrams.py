"""
make_diagrams.py — B1 讲义的结构图 (SVG)

  fig_led_circuit.svg   ① GPIO → 电阻 → LED → GND, 每一段的电压与电流
  fig_pullup.svg        ② 按钮 + 内部上拉: 松开读 1, 按下读 0
  fig_thresholds.svg    ② 输入电平的三个区: 低 / 未定义 / 高 (数据手册表 5-4)
  fig_pwm_duty.svg      ③ PWM: 同样的频率, 不同的占空比
  fig_loop.svg          ⑤ 阻塞 vs 非阻塞: 按钮落在睡眠期间会怎样

运行: python figures/make_diagrams.py
"""

import os

from common.svgfig import INK, INK2, KV_FILL, KV_LINE, MUTED, OP_FILL, OP_LINE, Q_FILL, Q_LINE, SERIES, Svg

HERE = os.path.dirname(os.path.abspath(__file__))


def out(name):
    return os.path.join(HERE, name)


def resistor(s, x, y, w=60, h=18, label="", sub=""):
    """水平放置的电阻 (矩形符号, IEC 画法), 左端 (x, y)。"""
    s.rect(x, y - h / 2, w, h, "#fff7e6", "#b06d00", rx=2)
    s.text(x + w / 2, y - h / 2 - 12, label, 13, INK)
    if sub:
        s.text(x + w / 2, y + h / 2 + 13, sub, 11, MUTED)


def wire(s, pts, color=OP_LINE, width=2):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    s.path(d, color, arrow=False, width=width)


def led_circuit():
    s = Svg(760, 300)
    s.text(380, 24, "① 点灯: 引脚 → 电阻 → LED → 地, 电流处处相等", 15, INK, weight="bold")
    # 芯片
    s.box(30, 90, 120, 90, "ESP32-S3", fill=Q_FILL, stroke=Q_LINE, size=14)
    s.text(90, 160, "GPIO 4 = 3.3 V", 11, INK2)
    wire(s, [(150, 135), (230, 135)])
    resistor(s, 230, 135, 80, 22, "R = 270 Ω", "压降 3.3 − 2.0 = 1.3 V")
    wire(s, [(310, 135), (400, 135)])
    # LED 三角形符号
    s.items.append('<polygon points="400,115 400,155 440,135" fill="#fde3d7" stroke="#eb6834" stroke-width="2"/>')
    s.items.append('<line x1="440" y1="115" x2="440" y2="155" stroke="#eb6834" stroke-width="2.5"/>')
    s.items.append('<line x1="425" y1="108" x2="437" y2="96" stroke="#eb6834" stroke-width="1.5"/>')
    s.items.append('<line x1="433" y1="112" x2="445" y2="100" stroke="#eb6834" stroke-width="1.5"/>')
    s.text(420, 175, "红色 LED, 压降约 2.0 V", 11, MUTED)
    wire(s, [(440, 135), (520, 135), (520, 230)])
    s.items.append('<line x1="500" y1="230" x2="540" y2="230" stroke="#52514e" stroke-width="2.5"/>')
    s.items.append('<line x1="507" y1="237" x2="533" y2="237" stroke="#52514e" stroke-width="2"/>')
    s.items.append('<line x1="514" y1="244" x2="526" y2="244" stroke="#52514e" stroke-width="1.5"/>')
    s.text(560, 237, "GND (0 V)", 12, INK2, "start")
    # 电流说明
    s.rect(580, 80, 160, 110, OP_FILL, OP_LINE)
    s.text(660, 102, "I = 1.3 V ÷ 270 Ω", 13, INK)
    s.text(660, 125, "= 4.81 mA", 15, INK, weight="bold")
    s.text(660, 150, "最高驱动档典型 40 mA", 11, MUTED)
    s.text(660, 168, "(数据手册表 5-4)", 11, MUTED)
    s.text(380, 280, "不串电阻: 电流只受引脚内阻和 LED 自身限制, 可能远超额定值, 烧坏 LED 或引脚", 12, INK2)
    s.save(out("fig_led_circuit.svg"))


def pullup():
    s = Svg(760, 330)
    s.text(380, 24, "② 按钮的常见接法: 内部上拉 + 按钮接地", 15, INK, weight="bold")
    for k, (pressed, x0) in enumerate(((False, 40), (True, 400))):
        title = "按下: 引脚被按钮直接接地 → 读 0" if pressed else "松开: 上拉电阻把引脚拉到 3.3 V → 读 1"
        s.text(x0 + 160, 60, title, 13, INK)
        # 3.3V 轨
        s.text(x0 + 60, 92, "3.3 V", 12, INK2)
        wire(s, [(x0 + 60, 102), (x0 + 60, 120)])
        # 上拉电阻 (竖直)
        s.rect(x0 + 50, 120, 20, 60, "#fff7e6", "#b06d00", rx=2)
        s.text(x0 + 40, 150, "45 kΩ", 11, INK2, "end")
        s.text(x0 + 40, 166, "(芯片内部)", 10, MUTED, "end")
        wire(s, [(x0 + 60, 180), (x0 + 60, 210)])
        # 引脚节点
        s.items.append(f'<circle cx="{x0 + 60}" cy="210" r="4" fill="{INK}"/>')
        wire(s, [(x0 + 60, 210), (x0 + 150, 210)])
        s.box(x0 + 150, 190, 90, 40, "GPIO 5", fill=Q_FILL, stroke=Q_LINE, size=13)
        s.text(x0 + 195, 245, "读 0" if pressed else "读 1", 14, KV_LINE if pressed else Q_LINE, weight="bold")
        # 按钮到地
        wire(s, [(x0 + 60, 210), (x0 + 60, 240)])
        if pressed:
            wire(s, [(x0 + 60, 240), (x0 + 60, 270)], color=KV_LINE, width=3)
        else:
            wire(s, [(x0 + 60, 240), (x0 + 82, 262)])
        s.items.append(f'<circle cx="{x0 + 60}" cy="240" r="3" fill="{BG_DOT}"/>')
        wire(s, [(x0 + 60, 270), (x0 + 60, 290)])
        s.items.append(f'<line x1="{x0 + 45}" y1="290" x2="{x0 + 75}" y2="290" stroke="#52514e" stroke-width="2.5"/>')
        s.text(x0 + 90, 290, "GND", 11, INK2, "start")
        s.text(x0 + 100, 252, "按钮", 11, MUTED, "start")
        if pressed:
            s.text(x0 + 270, 270, "电流 3.3 V ÷ 45 kΩ ≈ 0.07 mA", 11, MUTED)
            s.text(x0 + 270, 287, "从上拉电阻流进地, 引脚 ≈ 0 V", 11, MUTED)
    s.text(380, 318, "没有上拉电阻时, 松开的引脚什么都没接 (悬空), 读数不确定: 真板上会乱跳, Wokwi 里固定读 0", 12, INK2)
    s.save(out("fig_pullup.svg"))


BG_DOT = "#0b0b0b"


def thresholds():
    s = Svg(760, 250)
    s.text(380, 24, "② 引脚怎么判断 0 和 1: 中间有一段\"未定义\"", 15, INK, weight="bold")
    x0, x1, y = 80, 680, 120
    vdd = 3.3
    X = lambda v: x0 + v / vdd * (x1 - x0)
    zones = [(0, 0.25 * vdd, Q_FILL, Q_LINE, "低电平 → 读 0", "≤ 0.25 × VDD = 0.83 V"),
             (0.25 * vdd, 0.75 * vdd, KV_FILL, KV_LINE, "未定义: 读 0 还是 1 都有可能", "0.83 ~ 2.48 V"),
             (0.75 * vdd, vdd, "#d8f2e7", SERIES[2], "高电平 → 读 1", "≥ 0.75 × VDD = 2.48 V")]
    for a, b, fill, line, label, sub in zones:
        s.rect(X(a), y - 30, X(b) - X(a), 60, fill, line, rx=0)
        s.text((X(a) + X(b)) / 2, y - 6, label, 13, INK)
        s.text((X(a) + X(b)) / 2, y + 14, sub, 11, INK2)
    for v in (0, 0.83, 1.65, 2.48, 3.3):
        s.line(X(v), y + 30, X(v), y + 38, arrow=False)
        s.text(X(v), y + 52, f"{v:g} V", 11, MUTED)
    s.text(380, 200, "数据手册表 5-4 (VDD = 3.3 V): VIL 最大 0.25 × VDD, VIH 最小 0.75 × VDD", 12, INK2)
    s.text(380, 222, "悬空的引脚电压可能落在中间这一段, 所以读数不可信; 上拉/下拉电阻把它固定在两端之一", 12, INK2)
    s.save(out("fig_thresholds.svg"))


def pwm_duty():
    s = Svg(760, 290)
    s.text(380, 24, "③ PWM: 频率不变, 改变每个周期里高电平占多少", 15, INK, weight="bold")
    rows = [(0.25, "占空比 25%", "LED 较暗 / 电机较慢"), (0.5, "占空比 50%", ""), (0.9, "占空比 90%", "LED 很亮 / 电机较快")]
    x0, period, n = 170, 140, 3
    for i, (duty, label, note) in enumerate(rows):
        yb = 90 + i * 70
        s.text(30, yb - 15, label, 13, INK, "start")
        if note:
            s.text(30, yb + 2, note, 11, MUTED, "start")
        pts = [(x0, yb)]
        for k in range(n):
            xs = x0 + k * period
            pts += [(xs, yb), (xs, yb - 30), (xs + duty * period, yb - 30), (xs + duty * period, yb), (xs + period, yb)]
        wire(s, pts, color=SERIES[0], width=2)
        s.text(x0 + n * period + 20, yb - 15, f"平均电压 {duty * 3.3:.2f} V", 12, INK2, "start")
    s.line(x0, 262, x0 + period, 262, arrow=False)
    s.line(x0, 256, x0, 268, arrow=False)
    s.line(x0 + period, 256, x0 + period, 268, arrow=False)
    s.text(x0 + period / 2, 278, "一个周期 (1 kHz → 1000 µs)", 11, MUTED)
    s.save(out("fig_pwm_duty.svg"))


def loop():
    s = Svg(760, 300)
    s.text(380, 24, "⑤ 阻塞 vs 非阻塞: 按钮只按了 120 ms, 程序在干什么?", 15, INK, weight="bold")
    x0, x1 = 120, 720
    T = 1200
    X = lambda t: x0 + t / T * (x1 - x0)
    # 按钮
    s.text(100, 70, "按钮", 12, INK2, "end")
    wire(s, [(X(0), 70), (X(600), 70), (X(600), 90), (X(720), 90), (X(720), 70), (X(T), 70)], color=SERIES[1])
    s.text(X(660), 104, "按下 120 ms", 11, MUTED)
    # 阻塞
    s.text(100, 150, "阻塞写法", 12, INK2, "end")
    for a, b in ((0, 500), (500, 1000), (1000, 1200)):
        s.rect(X(a) + 1, 135, X(b) - X(a) - 2, 30, OP_FILL, OP_LINE, rx=2)
        s.text((X(a) + X(b)) / 2, 150, "sleep 500 ms", 11, INK2)
    for t in (500, 1000):
        s.items.append(f'<circle cx="{X(t):.1f}" cy="180" r="4" fill="{Q_LINE}"/>')
    s.text(X(500), 196, "看一眼: 没按", 11, MUTED)
    s.text(X(1000), 196, "看一眼: 已经松开了 → 漏掉", 11, KV_LINE)
    # 非阻塞
    s.text(100, 240, "非阻塞写法", 12, INK2, "end")
    for t in range(0, T, 25):
        s.line(X(t), 230, X(t), 250, color=Q_LINE, arrow=False, width=1)
    s.text(X(600), 266, "每 1 ms 看一眼 (图中每 25 ms 画一条): 按下后 ≤ 1 ms 就发现", 11, INK2)
    s.save(out("fig_loop.svg"))


if __name__ == "__main__":
    led_circuit()
    pullup()
    thresholds()
    pwm_duty()
    loop()
