"""练习 1 的验收: 在一大批输入上与参考实现逐个比对。"""

import math

import pytest

from _target import load

import b1core  # 参考实现, 只用来算"正确答案"

led_resistor = load("ex1_ohm_pwm_fill", "led_resistor")
next_e12 = load("ex1_ohm_pwm_fill", "next_e12")
duty_u16 = load("ex1_ohm_pwm_fill", "duty_u16")


@pytest.mark.parametrize("v_supply,v_forward,i_ma", [
    (3.3, 2.0, 5), (3.3, 2.0, 10), (3.3, 2.9, 2), (5.0, 2.0, 20), (3.3, 1.8, 1), (12.0, 3.2, 15),
])
def test_led_resistor(v_supply, v_forward, i_ma):
    assert math.isclose(led_resistor(v_supply, v_forward, i_ma),
                        b1core.led_resistor(v_supply, v_forward, i_ma), rel_tol=1e-9)


@pytest.mark.parametrize("args", [(3.3, 2.0, 0), (3.3, 2.0, -1), (2.0, 2.0, 5), (1.8, 2.0, 5)])
def test_led_resistor_rejects_bad_input(args):
    with pytest.raises(ValueError):
        led_resistor(*args)


def test_next_e12_examples():
    assert next_e12(260) == 270
    assert next_e12(270) == 270
    assert next_e12(1000) == 1000
    assert next_e12(0.5) == 0.56
    assert next_e12(9000) == 10000
    assert next_e12(269.99999999) == 270


def test_next_e12_matches_reference_on_grid():
    for k in range(1, 3000):
        r = 0.37 * k ** 1.7                     # 0.37 Ω ~ 300 kΩ, 跨好几个数量级
        assert next_e12(r) == b1core.next_e12(r), r


def test_next_e12_rejects_non_positive():
    with pytest.raises(ValueError):
        next_e12(0)


def test_duty_u16():
    assert duty_u16(0) == 0
    assert duty_u16(-0.3) == 0
    assert duty_u16(1) == 65535
    assert duty_u16(1.7) == 65535
    assert duty_u16(0.5) == 32768
    for k in range(1001):
        f = k / 1000
        assert duty_u16(f) == b1core.duty_u16(f), f
