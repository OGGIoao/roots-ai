"""主线 B1 电子入门: 纯逻辑部分与分析脚本的回归测试 (CPU, 约 2 秒)。运行: python -m pytest tests -q"""

import os
import subprocess
import sys

import pytest

B1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mainline", "B1-electronics")
sys.path.insert(0, B1)
sys.path.insert(0, os.path.join(B1, "experiments"))

from b1core import Blinker, Debouncer, Every, PiecewiseLinear, duty_u16, led_resistor, next_e12  # noqa: E402
from bundle import bundle  # noqa: E402
from wokwi_data import checksum, expand_cvcd, load_waves  # noqa: E402


def test_ohm_and_e12():
    assert led_resistor(3.3, 2.0, 5) == pytest.approx(260)
    assert next_e12(260) == 270 and next_e12(1000) == 1000 and next_e12(9000) == 10000


def test_duty_u16_bounds():
    assert (duty_u16(-1), duty_u16(0.5), duty_u16(2)) == (0, 32768, 65535)


def test_every_does_not_drift_and_counts_overruns():
    e = Every(10)
    ticks = [t for t in range(0, 1000) if e.ready(t)]
    assert ticks == list(range(10, 1000, 10))     # 不累积误差
    e2 = Every(10)
    assert e2.ready(35) and e2.overruns == 2       # 落后 2 个周期: 不补跑, 只记数


def test_blinker_toggles_every_half_period():
    b = Blinker(1000)
    states = [b.update(t) for t in range(0, 2001)]
    assert states[499] is False and states[500] is True and states[1000] is False


def test_debouncer_ignores_bounce():
    d = Debouncer(stable_ms=20)
    trace = [1, 0, 1, 0, 0, 1, 0] + [0] * 30 + [1, 0, 1] + [1] * 30
    for t, v in enumerate(trace):
        d.update(v, t)
    assert d.presses == 1 and not d.pressed


def test_piecewise_linear():
    f = PiecewiseLinear([(0, 0), (100, 50), (200, 150)])
    assert (f(-5), f(50), f(150), f(999)) == (0, 25, 100, 150)


def test_recordings_match_capture_checksums():
    # 存档与采集时浏览器里算的校验和逐字一致 (见 results/wokwi/*_serial.txt 与讲义附录)
    s3 = open(os.path.join(B1, "results", "wokwi", "case3_pwm.cvcd"), encoding="utf-8", newline="").read()
    assert checksum(s3) == (8609, 1548054901)
    assert checksum(expand_cvcd(s3)) == (19562, 449219333)


def test_pwm_analysis_reproduces_tutorial_numbers():
    from pwm_analysis import analyze
    r = analyze(os.path.join(B1, "results", "wokwi", "case3_pwm.cvcd"))
    for phase in ("hardware_quiet", "hardware_busy"):           # 硬件 PWM: 误差只有 ±1 ns
        p = r[phase]["period_us"]
        assert p["sd"] < 0.001 and abs(p["mean"] - 1000) < 0.001 and p["max"] - p["min"] <= 0.002
    assert r["software_quiet"]["freq_error_pct"] < -10            # 软件 PWM 慢了一成以上
    assert r["software_busy"]["period_us"]["max"] > 1.5 * r["software_quiet"]["period_us"]["max"]


def test_bundle_inlines_core():
    text = bundle(os.path.join(B1, "firmware", "case3_pwm.py"))
    assert "class Debouncer" in text and "from b1core import" not in text


def test_waves_load():
    w = load_waves(os.path.join(B1, "results", "wokwi", "case3_pwm.cvcd"))
    assert set(w) == {"pwm", "marker", "button", "btn5"} and len(w["marker"]) == 6


def test_exercise_tests_pass_against_reference():
    env = dict(os.environ, B1_SOLUTION="reference")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
                       cwd=os.path.join(B1, "exercises"), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
