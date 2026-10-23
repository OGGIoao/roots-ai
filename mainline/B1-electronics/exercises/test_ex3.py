"""练习 3 的验收: 呼吸灯的行为。"""

from _target import USE_REFERENCE, load

if USE_REFERENCE:
    BreathingLed = load(None, "BreathingLed")

    def make_breathing():
        return BreathingLed()
else:
    make_breathing = load("ex3_breathing_blank", "make_breathing")

GAMMA = 2.2


def perceived(d):
    return (d / 65535) ** (1 / GAMMA)


def hold(obj, t_from, t_to):
    return [obj.update(True, t) for t in range(t_from, t_to)]


def test_values_are_valid_ints():
    obj = make_breathing()
    seq = hold(obj, 0, 5000) + [obj.update(False, t) for t in range(5000, 7000)]
    assert all(isinstance(d, int) and 0 <= d <= 65535 for d in seq)


def test_breathes_with_period_and_full_range():
    obj = make_breathing()
    seq = hold(obj, 0, 10000)
    # 找每个"最亮点" (局部最大且 ≥ 99%), 相邻最亮点的间隔就是一个来回
    peaks = [t for t in range(1, len(seq) - 1)
             if seq[t] >= 64880 and seq[t] >= seq[t - 1] and seq[t] >= seq[t + 1]
             and (t == 1 or seq[t - 1] < 64880 or seq[t] > seq[t - 1])]
    # 合并连续的最亮区间
    merged = []
    for t in peaks:
        if not merged or t - merged[-1] > 100:
            merged.append(t)
    assert len(merged) >= 3, "10 秒里应该至少亮起 3 次"
    gaps = [b - a for a, b in zip(merged, merged[1:])]
    assert all(1500 <= g <= 2500 for g in gaps), f"一个来回的时长 {gaps} ms 不在 1.5~2.5 秒"
    for a, b in zip(merged, merged[1:]):
        assert min(seq[a:b]) <= 655, "每个来回里应该暗到 1% 以下"


def test_smooth():
    obj = make_breathing()
    seq = hold(obj, 0, 6000)
    worst = max(abs(perceived(b) - perceived(a)) for a, b in zip(seq, seq[1:]))
    assert worst <= 0.01, f"相邻毫秒的亮度跳变 {worst:.4f} > 0.01"


def test_release_fades_out_within_1s():
    for release_at in (300, 777, 1000, 1999, 3456):
        obj = make_breathing()
        hold(obj, 0, release_at)
        fade = [obj.update(False, t) for t in range(release_at, release_at + 3000)]
        assert all(b <= a for a, b in zip(fade, fade[1:])), f"松开后 (t={release_at}) 亮度不应回升"
        assert fade[1000] == 0, f"松开 1 秒后 (t={release_at}) 还没熄灭: duty={fade[1000]}"
        assert all(d == 0 for d in fade[1000:])


def test_press_again_restarts():
    obj = make_breathing()
    hold(obj, 0, 1500)
    [obj.update(False, t) for t in range(1500, 3000)]
    again = hold(obj, 3000, 6000)
    assert max(again) >= 64880 and min(again[500:]) <= 655
