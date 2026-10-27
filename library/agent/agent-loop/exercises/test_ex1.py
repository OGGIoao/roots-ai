"""练习 1 的验收测试: 和参考实现的安全层逐条比对。"""

import pytest

from _target import load
from tools import validate_call as ref

mine = load("ex1_validate_fill", "validate_call", "tools")

CASES = [
    ("walk", {"distance_m": 0.5}),
    ("walk", {"distance_m": -0.3, "speed": 0.2}),
    ("walk", '{"distance_m": 0.5}'),                 # 云端 API 给的是 JSON 字符串
    ("walk", {"distance_m": "0.5"}),                  # 数字写成字符串
    ("walk", {"distance_m": 1}),                      # 整数
    ("walk", {"distance_m": 1.0001}),                 # 刚好超过 1 m
    ("walk", {"distance_m": 10}),                     # 讲义第 4 节: 小模型要求一口气走 10 米
    ("walk", {"distance_m": "0.5m"}),                 # 带单位
    ("walk", {"distance_m": "nan"}),                  # NaN 会溜过 abs(x) > 1 的检查
    ("walk", {"distance_m": "inf"}),
    ("walk", {"distance_m": True}),                   # bool 不是数字
    ("walk", {"distance_m": None}),
    ("walk", {"distance_m": 0.5, "speed": 0}),
    ("walk", {"distance_m": 0.5, "direction": "forward"}),   # 多余参数
    ("walk", {}),                                     # 缺参数
    ("walk", ""),                                     # 空字符串 = 没有参数 → 缺参数
    ("walk", "{bad json"),
    ("walk", "[0.5]"),                                # 合法 JSON, 但不是对象
    ("turn", {"angle_deg": 90}),
    ("turn", {"angle_deg": -180}),
    ("turn", {"angle_deg": 270}),
    ("look", {}),
    ("look", ""),
    ("look", {"x": 1}),
    ("get_pose", "{}"),
    ("stop", {}),
    ("fly", {}),                                      # 没有这个工具
]


@pytest.mark.parametrize("name,args", CASES, ids=[f"{n}-{a}" for n, a in CASES])
def test_same_decision_as_reference(name, args):
    ok_ref, res_ref = ref(name, args)
    ok_mine, res_mine = mine(name, args)
    assert ok_mine == ok_ref, f"{name}({args!r}) 参考实现 {'放行' if ok_ref else '拦下: ' + res_ref}, 你的 {'放行' if ok_mine else '拦下'}"
    if ok_ref:
        assert res_mine == res_ref, f"清洗后的参数应为 {res_ref}, 得到 {res_mine}"
        assert all(type(v) is float for v in res_mine.values()), "清洗后的参数值都应是 float"
    else:
        assert isinstance(res_mine, str) and res_mine.strip(), "拦下时要给大模型一句说明 (它靠这句话改正)"
