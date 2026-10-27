"""
make_charts.py — 由 Wokwi 实测数据生成 B1 讲义里的数据图 (SVG)

数据来源 (都在 ../results/):
  wokwi/case3_pwm.cvcd       → fig_pwm_wave.svg       软件 PWM 与硬件 PWM 的前几个周期 (真实波形)
                             → fig_pwm_periods.svg    四个阶段每个周期的长度
  wokwi/case2_bounce.cvcd    → fig_bounce.svg         一次按下的弹跳 (放大到微秒)
  bounce_summary.json        → fig_debounce.svg       消抖时长与数到的次数
  blocking_summary.json      → fig_blocking.svg       阻塞 / 非阻塞: 发现了几次、多晚发现
  adc_summary.json           → fig_adc.svg            ADC 读数与旋钮位置

先运行 experiments/ 下的分析脚本生成 *_summary.json, 再运行本脚本:
    python experiments/pwm_analysis.py && python experiments/bounce_analysis.py
    python experiments/blocking_analysis.py && python experiments/adc_analysis.py
    python figures/make_charts.py

画法约定同 library/llm/attention/figures/make_charts.py (dataviz 规范):
  同一个对象在所有图里颜色相同; 线末端直接标注名字; 只有一个 y 轴; 讲义里每张图旁边配数据表。
"""

import json
import os
import sys

from common.svgfig import INK, INK2, MUTED, SERIES, Chart

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from wokwi_data import load_waves  # noqa: E402

SOFT, HARD = SERIES[1], SERIES[0]          # 软件 = 橙, 硬件 = 蓝
PHASE_COLOR = {"software_quiet": SOFT, "software_busy": SERIES[3], "hardware_quiet": HARD, "hardware_busy": SERIES[2]}
PHASE_LABEL = {"software_quiet": "软件 PWM, CPU 空闲", "software_busy": "软件 PWM, CPU 在忙",
               "hardware_quiet": "硬件 PWM, CPU 空闲", "hardware_busy": "硬件 PWM, CPU 在忙"}


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def step(recs, t_from, t_to):
    """录波 → 阶梯线的点 (时间 µs 从 t_from 起算, 电平 0/1)。"""
    v = recs[0][1]
    for t, vv in recs:
        if t > t_from:
            break
        v = vv
    pts = [(0.0, v)]
    for t, vv in recs:
        if t_from < t < t_to:
            x = (t - t_from) / 1000.0
            pts += [(x, v), (x, vv)]
            v = vv
    pts.append(((t_to - t_from) / 1000.0, v))
    return pts


def pwm_wave():
    w = load_waves(os.path.join(RES, "wokwi", "case3_pwm.cvcd"))
    marks = [t for t, _ in w["marker"][1:]]
    span = 5000_000                                     # 5 ms
    c = Chart(760, 320, left=150, right=40, top=60, bottom=55)
    c.scales(0, 5000, -0.2, 3.6)
    c.axes([0, 1000, 2000, 3000, 4000, 5000], [], "时间 (µs, 从该阶段开始后 20 个周期处起算)", "")
    rows = [("software_quiet", marks[0], 2.2), ("hardware_quiet", marks[2], 0.2)]
    for name, t0, base in rows:
        rises = [t for t, v in w["pwm"] if v == 1 and t > t0]
        start = rises[20]                               # 跳过开头的过渡, 从第 20 个周期的上升沿开始
        pts = [(x, base + 1.0 * y) for x, y in step(w["pwm"], start, start + span)]
        c.series(pts, PHASE_COLOR[name], width=2)
        c.text(c.x0 - 10, c.Y(base + 0.5), PHASE_LABEL[name].split(",")[0], 12, INK, "end")
    for k in range(6):                                  # 理想的周期边界: 每 1000 µs
        x = c.X(k * 1000)
        c.add(f'<line x1="{x:.1f}" y1="{c.y1}" x2="{x:.1f}" y2="{c.y0}" stroke="{MUTED}" stroke-dasharray="3 3" stroke-width="1"/>')
    c.save(os.path.join(HERE, "fig_pwm_wave.svg"), "③ 同样要求 1 kHz、50%: 软件 PWM 越走越慢, 硬件 PWM 正好卡在虚线上",
           "Wokwi 逻辑分析仪实测; 虚线是理想的周期边界 (每 1000 µs)")


def pwm_periods():
    r = load("pwm_summary.json")
    c = Chart(760, 380, left=70, right=170, top=60, bottom=60)
    c.scales(0, 300, 900, 2200)
    c.axes([0, 50, 100, 150, 200, 250, 300], [1000, 1200, 1400, 1600, 1800, 2000, 2200], "第几个周期", "周期长度 (µs)")
    ends = []
    for name in ("software_busy", "software_quiet", "hardware_quiet", "hardware_busy"):
        ps = r[name]["periods_us"]
        pts = list(enumerate(ps))
        c.series(pts, PHASE_COLOR[name], width=1.6, dash=name == "hardware_busy")
        ends.append((name, len(ps) - 1, ps[-1]))
    labels = {"software_busy": 1290, "software_quiet": 1227, "hardware_quiet": 1000, "hardware_busy": 1000}
    dy = {"software_busy": -26, "software_quiet": 4, "hardware_quiet": -8, "hardware_busy": 10}
    for name, x, y in ends:
        c.end_label(x, labels[name], PHASE_LABEL[name], dy[name])
    c.save(os.path.join(HERE, "fig_pwm_periods.svg"), "③ 每个周期有多长: CPU 一忙, 软件 PWM 就出现两倍长的周期",
           "Wokwi 实测, 每个阶段约 296 个周期; 目标 1000 µs; 两条硬件 PWM 的线重合在 1000 µs")


def bounce():
    w = load_waves(os.path.join(RES, "wokwi", "case2_bounce.cvcd"))
    s = load("bounce_summary.json")
    first = next(b for b in s["bursts"] if b["kind"] == "press")
    t0 = int(first["at_ms"] * 1e6) - 200_000             # 第一次跳变前 200 µs 起
    span = 3_000_000
    c = Chart(760, 300, left=110, right=40, top=60, bottom=55)
    c.scales(0, 3000, -0.3, 1.4)
    c.axes([0, 500, 1000, 1500, 2000, 2500, 3000], [0, 1], "时间 (µs)", "", yfmt=lambda v: f"读 {v}")
    c.series(step(w["button"], t0, t0 + span), SOFT, width=1.4)
    c.text(c.X(1250), c.Y(1.25), f"约 {first['duration_us'] / 1000:.1f} ms 里跳变 {first['edges']} 次, 其中下降沿 {first['falls']} 个", 12, INK)
    c.save(os.path.join(HERE, "fig_bounce.svg"), "② 按一下按钮, 引脚上实际发生了什么",
           "Wokwi 逻辑分析仪实测 (GPIO 5, 第 1 次按下); Wokwi 的按钮默认模拟触点弹跳")


def debounce():
    s = load("bounce_summary.json")
    sweep = s["debounce_sweep"]
    groups = [f"{d['stable_ms']} ms" for d in sweep]
    c = Chart(760, 340, left=70, right=40, top=70, bottom=60)
    c.scales(-0.5, len(groups) - 0.5, 0, 8)
    c.axes([], [0, 2, 4, 6, 8], "消抖时长 stable_ms (每 1 ms 读一次)", "数到的按下次数")
    for i, g in enumerate(groups):
        c.text(c.X(i), c.y0 + 16, g, 11, MUTED)
    c.bars(groups, ["数到"], [[d["counted"]] for d in sweep], [HARD])
    y = c.Y(s["true_presses"])
    c.add(f'<line x1="{c.x0}" y1="{y:.1f}" x2="{c.x1}" y2="{y:.1f}" stroke="{INK2}" stroke-dasharray="5 4"/>')
    c.text(c.x1 - 4, y - 10, f"真实 {s['true_presses']} 次", 12, INK, "end")
    c.text(c.X(5), c.Y(7.3), f"对照: 数所有下降沿 = {s['naive_interrupt_falls']} 次; 每 1 ms 轮询数 1→0 = {s['naive_poll_1ms']} 次", 12, INK2)
    holds = ", ".join(f"{h:g}" for h in s["hold_ms"])
    c.save(os.path.join(HERE, "fig_debounce.svg"), "② 消抖时长: 0~20 ms 都数对了, 50 ms 把真的按键也滤掉了",
           f"Wokwi 录波回放; 4 次按下分别按住 {holds} ms; stable_ms=0 也要求\"连续两次读数相同\"")


def blocking():
    p = os.path.join(RES, "blocking_summary.json")
    if not os.path.exists(p):
        print("跳过 fig_blocking.svg: 先运行 experiments/blocking_analysis.py")
        return
    s = load("blocking_summary.json")
    modes = [("blocking", "阻塞写法"), ("non_blocking", "非阻塞写法")]
    c = Chart(760, 320, left=70, right=40, top=70, bottom=60)
    c.scales(-0.5, 1.5, 0, s["blocking"]["pressed"] + 4)
    c.axes([], list(range(0, s["blocking"]["pressed"] + 5, 5)), "", "次数")
    for i, (_, label) in enumerate(modes):
        c.text(c.X(i), c.y0 + 16, label, 12, INK2)
    c.bars([m for m, _ in modes], ["按下", "发现"],
           [[s[m]["pressed"], s[m]["found"]] for m, _ in modes], [MUTED, HARD])
    for i, (m, _) in enumerate(modes):
        c.text(c.X(i), c.y0 + 34, f"发现延迟 平均 {s[m]['lat_mean']:.1f} ms, 最长 {s[m]['lat_max']} ms", 11, INK2)
    c.text(c.x1 - 4, c.y1 - 18, "灰 = 一共按了几次, 蓝 = 程序发现了几次", 11, MUTED, "end")
    c.save(os.path.join(HERE, "fig_blocking.svg"), "⑤ 每次按 120 ms, 20 秒按 21 次: 阻塞写法漏掉了大部分",
           "Wokwi 实测; 按键由芯片自己的定时器产生 (GPIO 8 → GPIO 9), 每 937 ms 一次")


def adc():
    p = os.path.join(RES, "adc_summary.json")
    if not os.path.exists(p):
        print("跳过 fig_adc.svg: 先运行 experiments/adc_analysis.py")
        return
    s = load("adc_summary.json")
    pts = [(r["pot"], r["mv_mean"]) for r in s["points"]]
    raw = [(r["pot"], r["raw_mean"] / 65535 * 3300) for r in s["points"]]
    c = Chart(760, 380, left=80, right=170, top=60, bottom=60)
    c.scales(0, 1023, 0, 5200)
    c.axes([0, 256, 512, 768, 1023], [0, 1000, 2000, 3000, 4000, 5000], "旋钮位置 (Wokwi 0~1023)", "读数 (mV)")
    ideal = [(0, 0), (1023, 3300)]
    c.series(ideal, MUTED, dash=True, width=1.4)
    c.end_label(1023, 3300, "理想: 位置 ÷ 1023 × 3.3 V", -10)
    c.series(pts, HARD, mark="circle")
    c.end_label(pts[-1][0], pts[-1][1], "read_uv() 校准后", 8)
    c.series(raw, SOFT, mark="square", dash=True)
    c.end_label(raw[-1][0], raw[-1][1], "read_u16() 换算, 与理想重合", 26)
    c.save(os.path.join(HERE, "fig_adc.svg"), "④ 仿真里: 原始读数严格成正比, \"校准后\"的电压反而不对",
           "Wokwi 实测 (ATTN_11DB); 满量程时 read_uv() 报 5090 mV, 超过了 3.3 V 电源; 真板见讲义第 4 节")


if __name__ == "__main__":
    pwm_wave()
    pwm_periods()
    bounce()
    debounce()
    blocking()
    adc()
