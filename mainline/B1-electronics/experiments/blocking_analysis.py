"""
案例 ⑤ 的分析: 阻塞写法与非阻塞写法, 各发现了几次按键、多晚发现。

输入: results/wokwi/case5_blocking_serial.txt (芯片打印的 @BLOCK / @LAT 记录)
输出: results/blocking_summary.json, 以及屏幕上的表格 (讲义第 5 节的数字都来自这里)。

运行: python experiments/blocking_analysis.py [串口记录]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from wokwi_data import load_records  # noqa: E402


def load_lat(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("@LAT,"):
            parts = line[5:].rstrip(";").split(",")
            out[parts[0]] = [int(x) for x in parts[1:] if x]
    return out


def analyze(path):
    lat = load_lat(path)
    res = {}
    for r in load_records(path, "BLOCK"):
        m = r["mode"]
        ls = lat.get(m, [])
        res[m] = {"pressed": r["pressed"], "found": r["found"], "found_pct": r["found"] / r["pressed"] * 100,
                  "lat_mean": r["lat_mean"], "lat_max": r["lat_max"], "latencies_ms": ls}
    return res


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "wokwi", "case5_blocking_serial.txt")
    res = analyze(path)
    print(f"{'写法':<14}{'按下':>5}{'发现':>5}{'发现率':>8}{'平均延迟ms':>11}{'最长延迟ms':>11}")
    for m, r in res.items():
        print(f"{m:<14}{r['pressed']:>5}{r['found']:>5}{r['found_pct']:>7.0f}%{r['lat_mean']:>11.1f}{r['lat_max']:>11}")
        print(f"{'':<14}延迟: {r['latencies_ms']}")
    out = os.path.join(ROOT, "results", "blocking_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n写入 {os.path.relpath(out, ROOT)}")
