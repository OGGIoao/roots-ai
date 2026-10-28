"""练习 3 的验收测试: 300 个新场景上的成功率与碰撞局数 (CPU, 约 5 秒)。"""

from _target import USE_REFERENCE, load
from agent import make_tasks
from car2d import Car2D, make_world, reached
from tools import execute

if USE_REFERENCE:
    go_to = load(None, "drive", "baseline")          # 参考实现里这个函数叫 drive
else:
    go_to = load("ex3_go_to_blank", "go_to", "baseline")

BUDGET = 30


class BudgetExceeded(Exception):
    pass


def run_one(task):
    car = Car2D(make_world(task["seed"]), seed=task["seed"])
    used = 0

    def tool(name, **args):
        nonlocal used
        used += 1
        if used > BUDGET:
            raise BudgetExceeded
        return execute(car, name, args)

    try:
        go_to(tool, task["target"], max_calls=BUDGET)
    except BudgetExceeded:
        pass
    return reached(car, task["target"]), car.collisions > 0


def test_acceptance():
    results = [run_one(t) for t in make_tasks(300, seed0=1000)]
    rate = sum(ok for ok, _ in results) / len(results)
    crashed = sum(c for _, c in results)
    print(f"\n成功率 {rate:.3f} (≥ 0.55), 有碰撞的局 {crashed} (≤ 9)")
    assert rate >= 0.55, f"成功率 {rate:.3f} < 0.55"
    assert crashed <= 9, f"有碰撞的局 {crashed} > 9"
