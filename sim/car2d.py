"""
car2d.py — 仿真小车: 一台两轮差速小车在 2 m × 2 m 的平面场地里

工具接口和 PRODUCT.md 第 4 节完全一致 (walk / turn / stop / look / get_pose), 以后换成真车, 上层的 Agent 一行不用改。
主线 C1 (Agent 驱动小车) 先用它, 之后 C2 (微调小模型)、E5 (强化学习训练) 也用它。

为什么要模拟误差: 真车走 1 米不会正好 1 米 (轮子打滑、电机不一致), 摄像头测距也不准。
如果仿真是完美的, "一次性把所有动作规划好" 和 "边走边看边调整" 就没有区别, C1 案例 ② 的教训也就看不到了。
误差的大小是本仓库定的假设值 (见 CarConfig), 真车造出来后用实测值替换。

坐标约定:
    场地 x ∈ [0, W], y ∈ [0, H], 单位米; 角度单位度, 0° 指向 +x, 逆时针为正 (左转为正)。
    look() 返回的 bearing_deg: 目标在车头左边为正、右边为负。

依赖: 只用标准库 (random, math), 电脑和以后的芯片端都能跑。
"""

import math
import random

# ─────────────────────────────── 配置 ───────────────────────────────


class CarConfig:
    """小车与传感器的参数。误差是假设值, 真车造出来后用实测替换。"""

    radius_m = 0.08              # 车身半径 (把车近似成圆, 用于碰撞检测)
    max_speed_mps = 0.5          # 速度上限: 写在"固件"里, 上层 (包括大模型) 无法越过 (PRODUCT 第 5 节)
    walk_scale_sd = 0.05         # 走的距离的比例误差 (标准差): 走 1 m 实际约 1 ± 0.05 m
    walk_drift_deg_per_m = 3.0   # 每走 1 m 航向漂移的标准差 (度)
    turn_scale_sd = 0.05         # 转角的比例误差
    turn_offset_sd_deg = 2.0     # 转角的固定误差 (度)
    fov_deg = 60.0               # 摄像头视场角 (左右各 30°)
    view_range_m = 2.0           # 能看清的最远距离
    look_dist_sd = 0.05          # 测距的比例误差
    look_bearing_sd_deg = 3.0    # 方位角误差 (度)
    step_m = 0.01                # 仿真步长: 每走 1 cm 检查一次碰撞


# ─────────────────────────────── 场景 ───────────────────────────────


class Thing:
    """场地里的一个圆形物体: 障碍物或目标物。"""

    def __init__(self, name, x, y, r, kind):
        self.name, self.x, self.y, self.r, self.kind = name, x, y, r, kind   # kind: "obstacle" | "target"

    def as_dict(self):
        return {"name": self.name, "x": round(self.x, 3), "y": round(self.y, 3), "r": self.r, "kind": self.kind}


TARGETS = {"red_cup": 0.04, "blue_ball": 0.05, "green_box": 0.06}   # 目标物名称与半径 (米)


def make_world(seed, width=2.0, height=2.0, n_obstacles=3):
    """
    按随机种子生成一个场景: 小车的起点与朝向、三个目标物、n_obstacles 个障碍物, 互不重叠。
    同一个 seed 永远生成同一个场景 (评测可复现)。
    """
    rng = random.Random(seed)
    things = []

    def free(x, y, r, margin=0.15):
        if not (r + 0.05 <= x <= width - r - 0.05 and r + 0.05 <= y <= height - r - 0.05):
            return False
        return all(math.hypot(x - t.x, y - t.y) > r + t.r + margin for t in things)

    def place(name, r, kind):
        for _ in range(1000):
            x, y = rng.uniform(0, width), rng.uniform(0, height)
            if free(x, y, r):
                things.append(Thing(name, x, y, r, kind))
                return
        raise RuntimeError("场景太挤, 放不下")

    for i in range(n_obstacles):
        place(f"obstacle_{i}", rng.uniform(0.10, 0.18), "obstacle")
    for name, r in TARGETS.items():
        place(name, r, "target")
    # 小车起点: 车身离障碍物至少 0.15 m, 离目标物至少 0.5 m
    # (否则可能一开局就在目标跟前, 一步不动也算"成功"; 本仓库实测 seed 22 出现过)
    rc = CarConfig().radius_m
    for _ in range(1000):
        x, y = rng.uniform(0.2, width - 0.2), rng.uniform(0.2, height - 0.2)
        if all(math.hypot(x - t.x, y - t.y) - t.r - rc > (0.5 if t.kind == "target" else 0.15) for t in things):
            break
    else:
        raise RuntimeError("找不到合适的起点")
    heading = rng.uniform(-180, 180)
    return {"width": width, "height": height, "things": things, "start": (x, y, heading), "seed": seed}


# ─────────────────────────────── 小车 ───────────────────────────────


def _wrap_deg(a):
    """角度归一化到 (-180, 180]。"""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


class Car2D:
    """
    仿真小车。对外只暴露 PRODUCT.md 规定的五个工具, 每个工具返回一个 dict (相当于真车回传的观测)。

    小车内部有两套位姿:
      - 真实位姿 (self._x, self._y, self._h): 仿真器知道, 上层看不到 (评测时用来判断成功);
      - 里程计估计 (self.odom): 按指令推算, 不含误差, 所以会和真实位姿越差越远。get_pose() 返回的是它,
        和真车上"编码器 + IMU 估计出来的位置"一样, 不是上帝视角。
    """

    def __init__(self, world, seed=0, config=None):
        self.world = world
        self.cfg = config or CarConfig()
        self.rng = random.Random(seed)
        self._x, self._y, self._h = world["start"]
        self.odom = [self._x, self._y, self._h]
        self.trace = [(self._x, self._y)]          # 真实轨迹 (画图用)
        self.log = []                              # 每次工具调用的记录
        self.sim_time_s = 0.0
        self.collisions = 0

    # ── 内部 ──

    def _hits(self, x, y):
        """位置 (x, y) 是否撞墙或撞到东西; 返回撞到的名字或 None。"""
        r = self.cfg.radius_m
        w, h = self.world["width"], self.world["height"]
        if x < r or y < r or x > w - r or y > h - r:
            return "wall"
        for t in self.world["things"]:
            if math.hypot(x - t.x, y - t.y) < r + t.r:
                return t.name
        return None

    def _record(self, tool, args, result):
        self.log.append({"tool": tool, "args": args, "result": result})
        return result

    # ── 工具 ──

    def walk(self, distance_m, speed=0.3):
        """直走 distance_m 米 (负数后退)。速度被固件限制在 max_speed_mps 以内。撞到东西会自动停下并报告。"""
        speed = min(abs(float(speed)), self.cfg.max_speed_mps) or self.cfg.max_speed_mps
        d_cmd = float(distance_m)
        d_real = d_cmd * (1 + self.rng.gauss(0, self.cfg.walk_scale_sd))
        drift = self.rng.gauss(0, self.cfg.walk_drift_deg_per_m) * abs(d_real)
        n = max(1, int(abs(d_real) / self.cfg.step_m))
        step = d_real / n
        moved = 0.0
        hit = None
        for i in range(n):
            h = math.radians(self._h + drift * (i + 1) / n)
            nx, ny = self._x + step * math.cos(h), self._y + step * math.sin(h)
            hit = self._hits(nx, ny)
            if hit:
                break
            self._x, self._y = nx, ny
            moved += abs(step)
            self.trace.append((self._x, self._y))
        self._h = _wrap_deg(self._h + drift * (moved / abs(d_real) if d_real else 0))
        self.sim_time_s += moved / speed
        # 里程计按指令推算: 撞了的话, 编码器能知道实际走了多少, 但不知道方向漂了
        sign = 1 if d_cmd >= 0 else -1
        oh = math.radians(self.odom[2])
        self.odom[0] += sign * moved * math.cos(oh)
        self.odom[1] += sign * moved * math.sin(oh)
        result = {"ok": hit is None, "moved_m": round(sign * moved, 3)}
        if hit:
            self.collisions += 1
            result["event"] = f"collision with {hit}"
        return self._record("walk", {"distance_m": d_cmd, "speed": speed}, result)

    def turn(self, angle_deg):
        """原地转 angle_deg 度 (正数左转, 负数右转)。"""
        a_cmd = float(angle_deg)
        a_real = a_cmd * (1 + self.rng.gauss(0, self.cfg.turn_scale_sd)) + self.rng.gauss(0, self.cfg.turn_offset_sd_deg)
        self._h = _wrap_deg(self._h + a_real)
        self.odom[2] = _wrap_deg(self.odom[2] + a_cmd)
        self.sim_time_s += abs(a_real) / 90.0          # 假设原地转 90°/s
        return self._record("turn", {"angle_deg": a_cmd}, {"ok": True})

    def stop(self):
        """立即停车。仿真里车本来就只在 walk 期间运动, 这里只做记录。"""
        return self._record("stop", {}, {"ok": True})

    def look(self):
        """
        拍一张"照片": 返回视场内 (±fov/2)、距离内、没被障碍物挡住的物体。
        每个物体给出名称、距离 (米) 和方位角 (度, 左正右负), 都带噪声。
        """
        seen = []
        for t in self.world["things"]:
            dx, dy = t.x - self._x, t.y - self._y
            dist = math.hypot(dx, dy) - t.r - self.cfg.radius_m   # 车身边缘到物体表面的距离 (和 reached() 的定义一致)
            bearing = _wrap_deg(math.degrees(math.atan2(dy, dx)) - self._h)
            if dist > self.cfg.view_range_m or abs(bearing) > self.cfg.fov_deg / 2:
                continue
            if self._occluded(t):
                continue
            seen.append({
                "name": t.name,
                "distance_m": round(max(0.0, dist * (1 + self.rng.gauss(0, self.cfg.look_dist_sd))), 2),
                "bearing_deg": round(bearing + self.rng.gauss(0, self.cfg.look_bearing_sd_deg), 1),
            })
        seen.sort(key=lambda s: s["distance_m"])
        return self._record("look", {}, {"ok": True, "objects": seen})

    def _occluded(self, target):
        """视线 (车 → 目标中心) 是否穿过别的物体。"""
        x0, y0, x1, y1 = self._x, self._y, target.x, target.y
        L = math.hypot(x1 - x0, y1 - y0)
        for t in self.world["things"]:
            if t is target:
                continue
            # 点到线段的距离
            u = max(0.0, min(1.0, ((t.x - x0) * (x1 - x0) + (t.y - y0) * (y1 - y0)) / (L * L))) if L else 0.0
            px, py = x0 + u * (x1 - x0), y0 + u * (y1 - y0)
            if math.hypot(t.x - px, t.y - py) < t.r and 0 < u < 1:
                return True
        return False

    def get_pose(self):
        """当前位姿的**里程计估计** (不是真实位姿): x, y (米), heading_deg。"""
        x, y, h = self.odom
        return self._record("get_pose", {}, {"ok": True, "x": round(x, 3), "y": round(y, 3), "heading_deg": round(h, 1)})

    # ── 评测用 (上层 Agent 不应调用) ──

    def distance_to(self, name):
        """真实位置下, 车身边缘到物体表面的距离 (米)。"""
        t = next(t for t in self.world["things"] if t.name == name)
        return math.hypot(t.x - self._x, t.y - self._y) - t.r - self.cfg.radius_m

    def true_pose(self):
        return self._x, self._y, self._h


TOOLS = ("walk", "turn", "stop", "look", "get_pose")


def reached(car, target, within_m=0.25):
    """任务"走到 target 跟前"是否完成: 车身边缘离目标不超过 within_m 米, 且全程没有碰撞。"""
    return car.distance_to(target) <= within_m and car.collisions == 0
