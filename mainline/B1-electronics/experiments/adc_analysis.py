"""
案例 ④ 的分析: 旋钮位置与 ADC 读数的关系。

输入: results/wokwi/case4_adc_serial.txt (每个旋钮位置一条 @ADC 记录)
输出: results/adc_summary.json, 以及屏幕上的表格 (讲义第 4 节的数字都来自这里)。

"理想电压" = 旋钮位置 / 1023 × 3300 mV (电位器两端接 3.3 V 与 GND, Wokwi 的 value 范围 0~1023)。
同时用 b1core.PiecewiseLinear 演示校准: 用 3 个实测点做校准表, 看其余点的误差能降到多少。

运行: python experiments/adc_analysis.py [串口记录]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from wokwi_data import load_records  # noqa: E402

from b1core import PiecewiseLinear  # noqa: E402

VDD_MV = 3300.0


def analyze(path):
    recs = sorted(load_records(path, "ADC"), key=lambda r: r["pot"])
    pts = []
    for r in recs:
        ideal = r["pot"] / 1023 * VDD_MV
        pts.append({**r, "ideal_mv": ideal, "uv_err_mv": r["mv_mean"] - ideal,
                    "raw_scaled_mv": r["raw_mean"] / 65535 * VDD_MV,
                    "raw_err_mv": r["raw_mean"] / 65535 * VDD_MV - ideal})
    # 用两端和中间三个点校准 read_u16(), 看其余点
    if len(pts) >= 3:
        cal_idx = [0, len(pts) // 2, len(pts) - 1]
        table = PiecewiseLinear([(pts[i]["raw_mean"], pts[i]["ideal_mv"]) for i in cal_idx])
        for p in pts:
            p["cal_mv"] = table(p["raw_mean"])
            p["cal_err_mv"] = p["cal_mv"] - p["ideal_mv"]
    return {"points": pts}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "wokwi", "case4_adc_serial.txt")
    res = analyze(path)
    print(f"{'位置':>5}{'理想mV':>9}{'read_uv mV':>12}{'误差':>8}{'read_u16':>10}{'按比例mV':>10}{'误差':>8}{'校准后误差':>11}{'抖动sd':>8}")
    for p in res["points"]:
        print(f"{p['pot']:>5}{p['ideal_mv']:>9.1f}{p['mv_mean']:>12.1f}{p['uv_err_mv']:>8.1f}{p['raw_mean']:>10.1f}"
              f"{p['raw_scaled_mv']:>10.1f}{p['raw_err_mv']:>8.1f}{p.get('cal_err_mv', float('nan')):>11.1f}{p['raw_sd']:>8.1f}")
    out = os.path.join(ROOT, "results", "adc_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n写入 {os.path.relpath(out, ROOT)}")
