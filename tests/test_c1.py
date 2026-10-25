"""主线 C1 Agent 循环的回归测试: 用剧本模型 (ScriptedLLM), 不需要真模型和网络, 约 0.1 秒。"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C1 = os.path.join(ROOT, "mainline", "C1-agent-loop")
sys.path.insert(0, C1)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from agent import make_tasks, run_episode  # noqa: E402
from car2d import Car2D, Thing, make_world, reached  # noqa: E402
from llm import ScriptedLLM, parse_qwen_output  # noqa: E402
from tools import validate_call  # noqa: E402

TASK = {"instruction": "去红杯子那里, 停在它跟前", "target": "red_cup"}


def straight_world():
    """车在 (0.5, 1.0) 朝 +x, 红杯子在正前方 1 m 处。"""
    return {"width": 2.0, "height": 2.0, "things": [Thing("red_cup", 1.5, 1.0, 0.04, "target")],
            "start": (0.5, 1.0, 0.0), "seed": 0}


def test_validate_call():
    assert validate_call("walk", {"distance_m": 0.5}) == (True, {"distance_m": 0.5})
    assert validate_call("walk", '{"distance_m": "0.5"}') == (True, {"distance_m": 0.5})   # 字符串数字可以转
    assert not validate_call("walk", {"distance_m": "0.5m"})[0]                            # 带单位的不猜
    assert not validate_call("walk", {"distance_m": 10})[0]                                # 一次最多 1 m
    assert not validate_call("walk", {})[0]                                                # 缺参数
    assert not validate_call("fly", {})[0]                                                 # 没有这个工具
    assert not validate_call("look", {"x": 1})[0]                                          # 多余参数
    assert not validate_call("turn", "{bad")[0]                                            # JSON 写坏


def test_chat_mode_never_moves_the_car():
    car = Car2D(straight_world(), seed=0)
    rec = run_episode(ScriptedLLM([{"content": "好的, 我向前走一米"}]), car, TASK, mode="chat")
    assert rec["ended_by"] == "answered" and car.log == []


def test_loop_feeds_results_back_with_ids():
    llm = ScriptedLLM([
        {"tool_calls": [("look", {})]},
        {"tool_calls": [("walk", {"distance_m": 0.75})]},
        {"tool_calls": [("stop", {})]},
        {"content": "到了"},
    ])
    car = Car2D(straight_world(), seed=0)
    rec = run_episode(llm, car, TASK, mode="loop")
    assert rec["ended_by"] == "answered" and rec["tool_calls"] == 3
    assert reached(car, "red_cup")
    second = llm.calls[1]                                   # 第二次问模型时, 它应该看到 look 的结果
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == second[-2]["tool_calls"][0]["id"]
    assert "red_cup" in second[-1]["content"]


def test_loop_executes_all_calls_but_step_only_the_first():
    script = [{"tool_calls": [("walk", {"distance_m": 0.2}), ("walk", {"distance_m": 0.2})]}, {"content": "完成"}]
    car = Car2D(straight_world(), seed=0)
    rec = run_episode(ScriptedLLM(script), car, TASK, mode="loop")
    assert rec["tool_calls"] == 2 and len(car.log) == 2
    car2 = Car2D(straight_world(), seed=0)
    rec2 = run_episode(ScriptedLLM(script), car2, TASK, mode="step")
    assert rec2["tool_calls"] == 1 and rec2["skipped_calls"] == 1 and len(car2.log) == 1


def test_safety_layer_blocks_and_explains():
    llm = ScriptedLLM([{"tool_calls": [("walk", {"distance_m": 10})]}, {"content": "好"}])
    car = Car2D(straight_world(), seed=0)
    rec = run_episode(llm, car, TASK, mode="loop", safety=True)
    assert rec["invalid_calls"] == 1 and car.log == []
    err = json.loads(llm.calls[1][-1]["content"])["error"]
    assert "最多走" in err


def test_without_safety_the_car_crashes():
    llm = ScriptedLLM([{"tool_calls": [("walk", {"distance_m": 10})]}, {"content": "好"}])
    car = Car2D(straight_world(), seed=0)
    run_episode(llm, car, TASK, mode="loop", safety=False)
    assert car.collisions == 1                              # 一口气冲过去撞上杯子


def test_plan_mode_never_shows_results():
    llm = ScriptedLLM([{"tool_calls": [("walk", {"distance_m": 0.5}), ("look", {})]}])
    car = Car2D(straight_world(), seed=0)
    rec = run_episode(llm, car, TASK, mode="plan")
    assert len(llm.calls) == 1 and rec["tool_calls"] == 2 and rec["ended_by"] == "plan_done"


def test_parse_qwen_output_handles_tags_bare_and_broken():
    _, calls = parse_qwen_output('<tool_call>\n{"name": "walk", "arguments": {"distance_m": 0.5}}\n</tool_call>')
    assert calls[0]["name"] == "walk" and "bare" not in calls[0]
    _, calls = parse_qwen_output('{"name": "look", "arguments": {}} 然后报告')
    assert calls[0]["name"] == "look" and calls[0]["bare"]
    _, calls = parse_qwen_output("<tool_call>{bad</tool_call>")
    assert calls[0]["name"] == "__invalid__"


def test_tasks_are_reproducible():
    assert make_tasks(9) == make_tasks(9)
    assert {t["target"] for t in make_tasks(3)} == {"red_cup", "blue_ball", "green_box"}
    assert make_world(make_tasks(1)[0]["seed"])["seed"] == 0
