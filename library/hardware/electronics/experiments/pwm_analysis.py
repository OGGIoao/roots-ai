"""
案例 ③ 的分析: 由逻辑分析仪录波算软件 PWM 与硬件 PWM 的周期、高电平时长和抖动。

输入: results/wokwi/case3_pwm.cvcd (或你自己从 Wokwi 下载的 b1-logic.vcd)
输出: results/pwm_summary.json, 以及屏幕上的表格 (讲义第 3 节的数字都来自这里)。

切段: marker 通道每翻转一次进入下一个阶段, 阶段名按 firmware/case3_pwm.py 的顺序。
每段去掉开头 3 个周期 (切换瞬间的过渡), 只统计完整的周期。

运行: python experiments/pwm_analysis.py [录波文件]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from wokwi_data import load_waves  # noqa: E402

PHASES = ["software_quiet", "software_busy", "hardware_quiet", "hardware_busy"]
TARGET_PERIOD_US = 1000.0
TARGET_HIGH_US = 500.0
SKIP = 3


def phase_windows(marker):
    """marker 的每条记录 (第一条是 t=0 的初始电平) 之后是一个新阶段; 返回 [(开始ns, 结束ns), ...]。"""
    toggles = [t for t, _ in marker[1:]]
    return [(toggles[i], toggles[i + 1]) for i in range(len(toggles) - 1)]


def cycles(pwm, start, end):
    """窗口内每个完整周期: (周期 µs, 高电平 µs)。周期 = 上升沿到下一个上升沿。"""
    edges = [(t, v) for t, v in pwm if start <= t < end]
    out = []
    for i in range(len(edges)):
        t_rise, v = edges[i]
        if v != 1:
            continue
        nxt_fall = next((t for t, vv in edges[i + 1:] if vv == 0), None)
        nxt_rise = next((t for t, vv in edges[i + 1:] if vv == 1), None)
        if nxt_fall is None or nxt_rise is None:
            continue
        out.append(((nxt_rise - t_rise) / 1000.0, (nxt_fall - t_rise) / 1000.0))
    return out[SKIP:]


def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / n) ** 0.5
    s = sorted(xs)
    return {"n": n, "mean": m, "sd": sd, "min": s[0], "p50": s[n // 2], "p99": s[min(n - 1, int(n * 0.99))], "max": s[-1]}


def analyze(path):
    w = load_waves(path)
    wins = phase_windows(w["marker"])
    out = {}
    for name, (a, b) in zip(PHASES, wins):
        cyc = cycles(w["pwm"], a, b)
        periods = [p for p, _ in cyc]
        highs = [h for _, h in cyc]
        ps = stats(periods)
        out[name] = {
            "period_us": ps,
            "high_us": stats(highs),
            "duty": sum(highs) / sum(periods),
            "freq_hz": 1e6 / ps["mean"],
            "freq_error_pct": (1e6 / ps["mean"] - 1e6 / TARGET_PERIOD_US) / (1e6 / TARGET_PERIOD_US) * 100,
            "long_periods": sum(1 for p in periods if p > 1.2 * ps["p50"]),  # 比中位数长 20% 以上的周期
            "periods_us": [round(p, 3) for p in periods],
        }
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "wokwi", "case3_pwm.cvcd")
    res = analyze(path)
    print(f"目标: 周期 {TARGET_PERIOD_US:.0f} µs (1 kHz), 高电平 {TARGET_HIGH_US:.0f} µs (占空比 50%)\n")
    print(f"{'阶段':<16}{'周期数':>6}{'平均周期µs':>12}{'标准差':>9}{'最短':>9}{'最长':>9}{'频率Hz':>9}{'频率误差':>9}{'占空比':>8}{'长周期':>7}")
    for name, r in res.items():
        p = r["period_us"]
        print(f"{name:<16}{p['n']:>6}{p['mean']:>12.1f}{p['sd']:>9.1f}{p['min']:>9.1f}{p['max']:>9.1f}"
              f"{r['freq_hz']:>9.1f}{r['freq_error_pct']:>8.1f}%{r['duty']:>8.3f}{r['long_periods']:>7}")
    out = os.path.join(ROOT, "results", "pwm_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n写入 {os.path.relpath(out, ROOT)}")
