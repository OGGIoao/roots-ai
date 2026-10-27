"""主线 v0.4 (仿真小车听懂文字) 的回归测试: go_to 技能、规则对照组、失败分类、练习 3。不需要真模型。"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATION = os.path.join(ROOT, "mainline", "v0.4-text-commands")
sys.path.insert(0, STATION)
sys.path.insert(0, os.path.join(ROOT, "library", "agent", "agent-loop"))
sys.path.insert(0, os.path.join(ROOT, "sim"))

from agent import run_episode  # noqa: E402
from car2d import Car2D, Thing, reached  # noqa: E402
from llm import ScriptedLLM  # noqa: E402
from skills import SKILL_TOOLSET  # noqa: E402

TASK = {"instruction": "去红杯子那里, 停在它跟前", "target": "red_cup"}


def straight_world():
    """车在 (0.5, 1.0) 朝 +x, 红杯子在正前方 1 m 处。"""
    return {"width": 2.0, "height": 2.0, "things": [Thing("red_cup", 1.5, 1.0, 0.04, "target")],
            "start": (0.5, 1.0, 0.0), "seed": 0}


def test_skill_mode_go_to_drives_without_the_llm_steering():
    llm = ScriptedLLM([{"tool_calls": [("go_to", {"target": "red_cup"})]}, {"content": "到了"}])
    car = Car2D(straight_world(), seed=0)
    rec = run_episode(llm, car, TASK, mode="skill", toolset=SKILL_TOOLSET)
    assert rec["tool_calls"] == 1 and reached(car, "red_cup")
    obs = json.loads(llm.calls[1][-1]["content"])
    assert obs["ok"] and obs["low_level_calls"] > 1              # 底层的 look / walk 是控制代码调的
    bad = ScriptedLLM([{"tool_calls": [("go_to", {"target": "purple_cat"})]}, {"content": "好"}])
    run_episode(bad, Car2D(straight_world(), seed=0), TASK, mode="skill", toolset=SKILL_TOOLSET)
    assert "error" in json.loads(bad.calls[1][-1]["content"])


def test_rule_baseline_is_reasonable():
    from baseline import run_set
    summary, _ = run_set(30, max_calls=30)
    assert summary["success_rate"] >= 0.6 and summary["collision_episodes"] <= 2


def test_force_first_tool_only_on_first_turn():
    llm = ScriptedLLM([{"tool_calls": [("go_to", {"target": "red_cup"})]}, {"content": "到了"}])
    run_episode(llm, Car2D(straight_world(), seed=0), TASK, mode="skill", force_first_tool=True, toolset=SKILL_TOOLSET)
    assert llm.required == [True, False]


def test_failure_classifier_does_not_call_honest_reports_false():
    from analyze import classify

    def ep(text, ended="answered"):
        return {"success": False, "collisions": 0, "target": "red_cup", "seed": 0, "ended_by": ended,
                "trace": [(0.5, 1.0)], "turns": [{"content": "", "calls": [{"name": "look", "arguments": "{}", "result": {"ok": True}}]},
                                                  {"content": text, "calls": []}]}
    assert classify(ep("没能直接到达红杯子, 最近一次看到它是 1.68 米远"), "loop") == "stopped_short"
    assert classify(ep("抱歉, 我没有找到蓝色的球"), "loop") == "stopped_short"
    assert classify(ep("找到了红杯子, 现在开到它旁边停下"), "loop") == "stopped_short"      # 说要做, 没说做完了
    assert classify(ep("成功到达红杯子, 停止"), "loop") == "false_report"
    assert classify(ep("已到达, 没有碰撞"), "loop") == "false_report"                      # "没有碰撞" 不是否定




def test_station_exercise_tests_pass_on_reference():
    env = dict(os.environ, C1_SOLUTION="reference", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
                       cwd=os.path.join(STATION, "exercises"), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


def test_skill_mode_requires_a_toolset():
    import pytest
    with pytest.raises(ValueError):
        run_episode(ScriptedLLM([]), Car2D(straight_world(), seed=0), TASK, mode="skill")
