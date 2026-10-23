"""练习 2 的验收: 用真实录波和人造弹跳检验消抖器。"""

import os
import random
import sys

from _target import HERE, load

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
from wokwi_data import load_waves  # noqa: E402

Debouncer = load("ex2_debouncer_interface", "Debouncer")
WAVE = os.path.join(os.path.dirname(HERE), "results", "wokwi", "case2_bounce.cvcd")


def feed(deb, levels_at, t_end_ms, step_ms=1):
    """每 step_ms 毫秒读一次 levels_at(t) 交给消抖器; 返回每次 presses 增加的时刻。"""
    found = []
    for t in range(0, t_end_ms + 1, step_ms):
        before = deb.presses
        deb.update(levels_at(t), t)
        if deb.presses > before:
            found.append(t)
    return found


def test_real_wokwi_recording():
    recs = load_waves(WAVE)["button"]
    # 录波 (讲义第 2 节): 4 次真实按下 (按住 95/33/33/33 ms), 外加 1 次按住时间几乎为 0 的"点按";
    # 每次动作约 2 ms 的弹跳, 一共 644 个下降沿。正确的消抖器应该数出 4。
    keys = [(t, v) for t, v in recs if t > 5e9]          # 跳过开机阶段 (打开上拉之前读 0)
    t0 = int(keys[0][0] // 1_000_000) - 10
    t1 = int(keys[-1][0] // 1_000_000) + 200
    idx = [0]

    def level(t_ms):
        t = (t0 + t_ms) * 1_000_000
        while idx[0] + 1 < len(recs) and recs[idx[0] + 1][0] <= t:
            idx[0] += 1
        return recs[idx[0]][1]

    deb = Debouncer(stable_ms=20)
    feed(deb, level, t1 - t0)
    assert deb.presses == 4, f"数到 {deb.presses} 次, 真实是 4 次"


def synthetic(press_at, hold_ms, bounce_ms=3, seed=0):
    """人造波形: 在 press_at 时刻按下 hold_ms 毫秒; 按下和松开时各弹跳 bounce_ms 毫秒 (每毫秒随机 0/1)。"""
    rng = random.Random(seed)
    noise = {t: rng.randint(0, 1) for t in range(press_at, press_at + bounce_ms)}
    noise.update({t: rng.randint(0, 1) for t in range(press_at + hold_ms, press_at + hold_ms + bounce_ms)})

    def level(t):
        if t in noise:
            return noise[t]
        return 0 if press_at <= t < press_at + hold_ms else 1
    return level


def test_each_press_counts_once():
    for seed in range(20):
        for hold in (40, 100, 500):
            level = synthetic(100, hold, bounce_ms=4, seed=seed)
            deb = Debouncer(stable_ms=20)
            found = feed(deb, level, 100 + hold + 200)
            assert deb.presses == 1, (seed, hold)
            # 弹跳期间最后一次读到 1 的时刻; 从下一毫秒起一直是 0, 满 20 ms 才该承认
            last_high = max([t for t in range(100, 104) if level(t) == 1], default=99)
            expect = last_high + 1 + 20
            assert found[0] >= expect, f"承认得太早 ({found[0]} < {expect}): 稳定时间还没到就算按下了"
            assert found[0] <= expect + 1, f"承认得太晚 ({found[0]} > {expect + 1})"


def test_glitch_is_ignored():
    deb = Debouncer(stable_ms=20)
    feed(deb, lambda t: 0 if 50 <= t < 60 else 1, 200)   # 10 ms 的低电平毛刺
    assert deb.presses == 0
    assert not deb.pressed


def test_pressed_property_follows_stable_state():
    deb = Debouncer(stable_ms=5)
    for t in range(0, 30):
        deb.update(0, t)
    assert deb.pressed
    for t in range(30, 60):
        deb.update(1, t)
    assert not deb.pressed


def test_active_high_wiring():
    deb = Debouncer(stable_ms=20, active_low=False)
    feed(deb, lambda t: 1 if 100 <= t < 300 else 0, 500)
    assert deb.presses == 1
