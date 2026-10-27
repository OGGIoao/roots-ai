"""
练习 3 (③ 白纸): 自己写 go_to 技能, 过一条验收线

讲义第 6 节: 把"怎么开过去"从大模型手里拿走, 交给一段控制代码, 大模型只负责听懂要去哪。
这里你来写这段控制代码。只给验收标准, 方法自己选 (讲义里的、没讲的、自己发明的都行)。
不要照抄 ../baseline.py; 写完、测试通过之后再去对照。

要实现的函数:

    go_to(tool, target, max_calls=30) -> dict, 至少包含 "arrived": bool (你自己认为到没到)

    tool(name, **args) 是你唯一能用的东西, 返回观测 dict, 和大模型拿到的一模一样:
        tool("look")                     → {"ok": True, "objects": [{"name", "distance_m", "bearing_deg"}, ...]}
        tool("turn", angle_deg=30)       → 左正右负
        tool("walk", distance_m=0.4)     → 撞到东西会停下并在结果里带 "event"
        tool("get_pose")                 → 里程计估计的 x, y, heading_deg (会漂)
        tool("stop")
    看不到仿真器内部 (真实位置、地图); 调用超过 max_calls 次会抛 BudgetExceeded, 算这一局没完成。

验收 (python -m pytest test_ex3.py -q), 在 300 个讲义里没用过的场景上 (seed 1000~1299), 每局最多 30 次调用:
    1. 成功率 ≥ 0.55 (成功 = 车身离目标 ≤ 0.25 m 且全程无碰撞);
    2. 有碰撞的局 ≤ 9 (3%)。
参考数据: 参考实现 (baseline.py) 在这 300 个场景上成功率 0.63、有碰撞的局 4。
场地: 2 m × 2 m, 四周是墙 (look 看不到墙); 三个目标物和三个障碍物, 位置每局随机。

想清楚预算花在哪: 原地转一圈找目标就要 16 次调用 (8 次 look + 8 次 turn), 30 次很紧。
"""


def go_to(tool, target, max_calls=30):
    raise NotImplementedError
