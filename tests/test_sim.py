"""仿真小车 sim/car2d.py 的回归测试 (纯标准库, 约 0.1 秒)。"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))

from car2d import Car2D, CarConfig, make_world, reached  # noqa: E402


def test_world_is_reproducible_and_non_overlapping():
    a, b = make_world(7), make_world(7)
    assert [t.as_dict() for t in a["things"]] == [t.as_dict() for t in b["things"]] and a["start"] == b["start"]
    ts = a["things"]
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            assert math.hypot(ts[i].x - ts[j].x, ts[i].y - ts[j].y) > ts[i].r + ts[j].r


def test_same_seed_same_noise():
    w = make_world(3)
    c1, c2 = Car2D(w, seed=5), Car2D(w, seed=5)
    for c in (c1, c2):
        c.turn(45)
        c.walk(0.3)
    assert c1.true_pose() == c2.true_pose()


def test_odometry_drifts_from_truth():
    # 走一大圈, 里程计 (按指令推算) 和真实位姿会分开: 这正是要靠 look() 闭环的原因
    w = {"width": 10.0, "height": 10.0, "things": [], "start": (5.0, 5.0, 0.0), "seed": 0}
    c = Car2D(w, seed=1)
    for _ in range(8):
        c.walk(1.0)
        c.turn(90)
    x, y, _ = c.true_pose()
    est = c.get_pose()
    assert math.hypot(est["x"] - x, est["y"] - y) > 0.01


def test_collision_stops_the_car():
    w = {"width": 2.0, "height": 2.0, "things": [], "start": (1.0, 1.0, 0.0), "seed": 0}
    c = Car2D(w, seed=0)
    r = c.walk(5.0)                                   # 冲向右墙
    assert not r["ok"] and "wall" in r["event"] and c.collisions == 1
    assert c.true_pose()[0] <= 2.0 - CarConfig.radius_m + 1e-9


def test_speed_is_capped_by_firmware():
    w = {"width": 10.0, "height": 10.0, "things": [], "start": (1.0, 5.0, 0.0), "seed": 0}
    c = Car2D(w, seed=0)
    c.walk(1.0, speed=99)
    assert c.log[-1]["args"]["speed"] == CarConfig.max_speed_mps


def test_look_respects_fov_and_occlusion():
    from car2d import Thing
    things = [Thing("red_cup", 1.5, 1.0, 0.04, "target"),        # 正前方
              Thing("blue_ball", 1.0, 1.8, 0.05, "target"),       # 左侧 90°, 在视场外
              Thing("green_box", 1.9, 1.0, 0.06, "target")]       # 被 red_cup 挡住? 杯子太小挡不住
    w = {"width": 2.0, "height": 2.0, "things": things, "start": (1.0, 1.0, 0.0), "seed": 0}
    names = [o["name"] for o in Car2D(w, seed=0).look()["objects"]]
    assert "red_cup" in names and "blue_ball" not in names
    big = [Thing("wall_block", 1.4, 1.0, 0.15, "obstacle"), Thing("red_cup", 1.8, 1.0, 0.04, "target")]
    w2 = {"width": 2.0, "height": 2.0, "things": big, "start": (1.0, 1.0, 0.0), "seed": 0}
    names2 = [o["name"] for o in Car2D(w2, seed=0).look()["objects"]]
    assert "wall_block" in names2 and "red_cup" not in names2   # 被障碍物挡住


def test_reached_requires_no_collision():
    from car2d import Thing
    w = {"width": 2.0, "height": 2.0, "things": [Thing("red_cup", 1.5, 1.0, 0.04, "target")],
         "start": (1.0, 1.0, 0.0), "seed": 0}
    c = Car2D(w, seed=0)
    c.walk(0.25)
    assert reached(c, "red_cup")
    c.walk(1.0)                                       # 撞上杯子
    assert not reached(c, "red_cup")
