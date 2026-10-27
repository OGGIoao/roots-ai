"""
练习 1 (① 填空): 电阻怎么选、占空比怎么算

把下面三个函数里的 TODO 补全, 让它们和参考实现 (../b1core.py) 在一大批输入上结果一致。

验收:
    cd library/hardware/electronics/exercises
    python -m pytest test_ex1.py -q

讲义对应: 第 1 节 (欧姆定律、E12 标准电阻) 和第 3 节 (duty_u16)。
提示规则: 卡住了可以要提示, 按"方向 → 哪一行 → 代码"分级给, 每次只升一级。
写完再去对照 ../b1core.py。
"""

E12 = (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2)
DUTY_MAX = 65535


def led_resistor(v_supply, v_forward, i_ma):
    """
    给 LED 串多大的电阻 (欧姆), 才能让电流等于 i_ma 毫安。
    i_ma <= 0 或 v_supply <= v_forward 时抛 ValueError。
    例: led_resistor(3.3, 2.0, 5) ≈ 260
    """
    if i_ma <= 0:
        raise ValueError("目标电流必须为正")
    if v_supply <= v_forward:
        raise ValueError("电源电压不高于 LED 压降, LED 不会亮")
    # TODO: 电阻两端的电压是多少? 电流的单位要先换成安培
    raise NotImplementedError


def next_e12(r_ohm):
    """
    不小于 r_ohm 的最小 E12 标准电阻值。例: 260 → 270, 1000 → 1000, 0.5 → 0.56, 9000 → 10000。
    r_ohm <= 0 时抛 ValueError。浮点误差要容忍: 269.99999999 也应该返回 270。
    返回值用 round(..., 6) 取整, 避免 2.7 * 100 = 270.00000000000006 这类尾巴。
    """
    if r_ohm <= 0:
        raise ValueError("电阻必须为正")
    # TODO: 先找到 r_ohm 所在的数量级 decade (使 decade <= r_ohm < 10 * decade),
    #       再在 E12 (以及下一个数量级的 1.0, 即 10.0) 里找第一个 base * decade >= r_ohm 的值
    raise NotImplementedError


def duty_u16(fraction):
    """
    占空比 (0~1 的小数) → MicroPython PWM.duty_u16() 用的整数 (0~65535), 四舍五入。
    小于等于 0 返回 0, 大于等于 1 返回 65535。例: duty_u16(0.5) == 32768。
    """
    # TODO
    raise NotImplementedError
