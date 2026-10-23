"""
案例 ② 的分析: 按键弹跳长什么样, 消抖时长该取多少。

输入: results/wokwi/case2_bounce.cvcd (Wokwi 逻辑分析仪录下的 GPIO 5 波形)
输出: results/bounce_summary.json, 以及屏幕上的表格 (讲义第 2 节的数字都来自这里)。

1. 切簇: 相邻两次跳变间隔超过 GAP_MS 毫秒, 就算新的一簇。每一簇是一次"按下"或"松开"
   (看簇结束后的电平: 0 = 按下, 1 = 松开), 统计每簇有多少次跳变、持续多久。
2. 回放: 把录波当成"真实的引脚电平", 让 b1core.Debouncer 每 POLL_MS 毫秒读一次,
   扫描不同的 stable_ms, 看数出来的按下次数和真实次数是否一致、承认按下比真实按下晚了多久。
   这一步完全在电脑上完成, 结果可重复; 芯片上的计数受仿真速度影响, 只作参考。

运行: python experiments/bounce_analysis.py [录波文件]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from wokwi_data import load_waves  # noqa: E402

from b1core import Debouncer  # noqa: E402

GAP_MS = 50
POLL_MS = 1
STABLE_SWEEP = [0, 1, 2, 5, 10, 20, 50]


def level_at(recs, t):
    """录波在时刻 t (ns) 的电平。"""
    v = recs[0][1]
    for tt, vv in recs:
        if tt > t:
            break
        v = vv
    return v


def bursts(recs):
    """把跳变切成簇。返回 [{start_ns, end_ns, edges, falls, final}]。第一条记录是 t=0 的初始电平, 不算跳变。"""
    edges = recs[1:]
    out = []
    cur = None
    prev_v = recs[0][1]
    for t, v in edges:
        if cur is None or t - cur["end_ns"] > GAP_MS * 1e6:
            cur = {"start_ns": t, "end_ns": t, "edges": 0, "falls": 0, "final": v}
            out.append(cur)
        cur["end_ns"] = t
        cur["edges"] += 1
        cur["falls"] += prev_v == 1 and v == 0
        cur["final"] = v
        prev_v = v
    # 按簇前后的电平分类: 1→0 按下, 0→1 松开, 1→1 "点按" (按下和松开的弹跳挤在同一簇里, 按住时间几乎为 0)
    for b in out:
        before = level_at(recs, b["start_ns"] - 1)
        b["kind"] = {(1, 0): "press", (0, 1): "release", (1, 1): "tap"}.get((before, b["final"]), "other")
        b["duration_us"] = (b["end_ns"] - b["start_ns"]) / 1000.0
    return out


def replay(recs, stable_ms, t_start_ns, t_end_ns):
    """每 POLL_MS 毫秒读一次录波电平, 交给 Debouncer; 返回 (数到的按下次数, 每次被承认的时刻 ns)。"""
    deb = Debouncer(stable_ms=stable_ms)
    found = []
    i = 0
    v = recs[0][1]
    t = t_start_ns
    while t <= t_end_ns:
        while i < len(recs) and recs[i][0] <= t:
            v = recs[i][1]
            i += 1
        before = deb.presses
        deb.update(v, int(t // 1_000_000))
        if deb.presses > before:
            found.append(t)
        t += POLL_MS * 1_000_000
    return deb.presses, found


def naive_poll(recs, t_start_ns, t_end_ns):
    """朴素办法 B 在"读得足够快"时的表现: 每 POLL_MS 毫秒读一次, 看到 1 → 0 就算一下。"""
    n = 0
    prev = level_at(recs, t_start_ns)
    i = 0
    t = t_start_ns
    while t <= t_end_ns:
        while i < len(recs) and recs[i][0] <= t:
            i += 1
        v = recs[i - 1][1] if i else recs[0][1]
        n += prev == 1 and v == 0
        prev = v
        t += POLL_MS * 1_000_000
    return n


def analyze(path):
    w = load_waves(path)
    recs = w["button"]
    bs = bursts(recs)
    # 开机瞬间的毛刺: 第一簇发生在程序打印 @READY 之前 (几微秒的低电平), 单独报告, 不算按键
    boot = [b for b in bs if b["start_ns"] < 5e9]
    keys = [b for b in bs if b["start_ns"] >= 5e9]
    presses = [b for b in keys if b["kind"] == "press"]
    holds = []
    for p in presses:
        rel = next((b for b in keys if b["kind"] == "release" and b["start_ns"] > p["end_ns"]), None)
        if rel:
            holds.append((rel["start_ns"] - p["start_ns"]) / 1e6)
    t0 = keys[0]["start_ns"] - 10_000_000
    t1 = keys[-1]["end_ns"] + 100_000_000
    sweep = []
    for s in STABLE_SWEEP:
        n, found = replay(recs, s, t0, t1)
        lat = [(f - p["start_ns"]) / 1e6 for f, p in zip(found, presses)] if n == len(presses) else []
        sweep.append({"stable_ms": s, "counted": n, "true": len(presses),
                      "latency_ms_max": max(lat) if lat else None})
    return {
        "boot_glitch": [{"at_ms": b["start_ns"] / 1e6, "edges": b["edges"], "width_us": None if b["edges"] < 2 else b["duration_us"]} for b in boot],
        "bursts": [{"kind": b["kind"], "at_ms": round(b["start_ns"] / 1e6, 3), "edges": b["edges"],
                    "falls": b["falls"], "duration_us": round(b["duration_us"], 1)} for b in keys],
        "true_presses": len(presses),
        "hold_ms": [round(h, 1) for h in holds],
        "naive_interrupt_falls": sum(b["falls"] for b in keys),
        "naive_poll_1ms": naive_poll(recs, t0, t1),
        "debounce_sweep": sweep,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "wokwi", "case2_bounce.cvcd")
    r = analyze(path)
    print("开机毛刺:", r["boot_glitch"])
    print(f"\n{'簇':<4}{'类型':<9}{'时刻ms':>11}{'跳变数':>7}{'下降沿':>7}{'持续µs':>9}")
    for i, b in enumerate(r["bursts"], 1):
        print(f"{i:<4}{b['kind']:<9}{b['at_ms']:>11.1f}{b['edges']:>7}{b['falls']:>7}{b['duration_us']:>9.1f}")
    print(f"\n真实按下 {r['true_presses']} 次, 每次按住 {r['hold_ms']} ms")
    print(f"朴素办法: 数所有下降沿 = {r['naive_interrupt_falls']}; 每 {POLL_MS} ms 轮询一次数 1→0 = {r['naive_poll_1ms']}")
    print(f"\n回放消抖 (每 {POLL_MS} ms 读一次):")
    for s in r["debounce_sweep"]:
        print(f"  stable_ms={s['stable_ms']:>3}: 数到 {s['counted']} 次 (真实 {s['true']}), 最晚承认延迟 {s['latency_ms_max']} ms")
    out = os.path.join(ROOT, "results", "bounce_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(f"\n写入 {os.path.relpath(out, ROOT)}")
