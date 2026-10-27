"""练习 2 的验收测试: 用剧本模型驱动, 和参考实现的 step 模式逐条比对对话。"""

import functools
import json

import pytest

from _target import USE_REFERENCE, load
from agent import run_episode
from car2d import Car2D, Thing
from llm import ScriptedLLM

if USE_REFERENCE:
    mine = functools.partial(run_episode, mode="step")
else:
    mine = load("ex2_step_interface", "run_step_episode", "agent")

TASK = {"instruction": "去红杯子那里, 停在它跟前", "target": "red_cup"}


def world():
    return {"width": 2.0, "height": 2.0, "things": [Thing("red_cup", 1.5, 1.0, 0.04, "target")],
            "start": (0.5, 1.0, 0.0), "seed": 0}


SCRIPTS = {
    "one-call-per-turn": [
        {"tool_calls": [("look", {})]},
        {"tool_calls": [("walk", {"distance_m": 0.5})]},
        {"tool_calls": [("stop", {})]},
        {"content": "到了"}],
    "burst-of-calls": [                               # 小模型一轮发好几个调用: 只能执行第一个
        {"content": "我先看再走", "tool_calls": [("look", {}), ("walk", {"distance_m": 0.5}), ("walk", {"distance_m": 0.3})]},
        {"tool_calls": [("walk", {"distance_m": 0.4}), ("stop", {})]},
        {"content": "完成"}],
    "blocked-by-safety": [                            # 安全层拦下的调用也算"执行了" (结果是错误观测)
        {"tool_calls": [("walk", {"distance_m": 10})]},
        {"tool_calls": [("walk", '{"distance_m": "0.5m"}')]},
        {"content": "放弃"}],
    "never-stops": [{"tool_calls": [("look", {})]}] * 5,
}


def strip_skip_text(messages):
    """退回调用的错误措辞随你写, 比较前统一替换掉。"""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "tool":
            obs = json.loads(m["content"])
            if obs.get("ok") is False and "error" in obs and str(obs["error"]).startswith("未执行"):
                obs["error"] = "<skipped>"
            m["content"] = obs
        out.append(m)
    return out


@pytest.mark.parametrize("name", SCRIPTS)
def test_same_conversation_as_reference(name):
    max_turns = 4
    ref_llm, my_llm = ScriptedLLM(SCRIPTS[name]), ScriptedLLM(SCRIPTS[name])
    ref_car, my_car = Car2D(world(), seed=0), Car2D(world(), seed=0)
    ref = run_episode(ref_llm, ref_car, TASK, mode="step", max_turns=max_turns)
    rec = mine(my_llm, my_car, TASK, max_turns=max_turns)
    for key in ("tool_calls", "skipped_calls", "ended_by"):
        assert rec[key] == ref[key], f"{key}: 应为 {ref[key]!r}, 得到 {rec[key]!r}"
    assert len(my_llm.calls) == len(ref_llm.calls), f"模型应被调用 {len(ref_llm.calls)} 次, 实际 {len(my_llm.calls)} 次"
    for k, (a, b) in enumerate(zip(ref_llm.calls, my_llm.calls)):
        # 你的退回措辞不必以"未执行"开头, 所以只把参考实现的统一替换, 你的按 ok/error 检查
        ra = strip_skip_text(a)
        mb = [dict(m) for m in b]
        assert [m["role"] for m in mb] == [m["role"] for m in ra], f"第 {k + 1} 次调用模型时, 消息的角色顺序不对"
        for x, y in zip(ra, mb):
            if x["role"] == "tool":
                assert y["tool_call_id"] == x["tool_call_id"], "tool_call_id 要和对应的调用对上"
                obs = json.loads(y["content"])
                if x["content"].get("error") == "<skipped>":
                    assert obs.get("ok") is False and obs.get("error"), "没执行的调用要回一条 ok=False 且带 error 的观测"
                else:
                    assert obs == x["content"], f"观测内容不对: 应为 {x['content']}, 得到 {obs}"
            else:
                assert y == x, f"第 {k + 1} 次调用模型时, 有一条 {x['role']} 消息不一样"
    assert my_car.log == ref_car.log, "小车实际执行的动作和参考实现不同 (每轮只能执行第一个调用)"
