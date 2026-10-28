"""练习 2 的验收测试: 停止条件、可复现、重复惩罚看得到提示词, 并和参考实现在 ToyLM 上逐字一致。"""

import random

import pytest

from _target import load  # 先导入它: 它把上一级目录加进搜索路径
import generate as REF  # noqa: E402
from toy_lm import ToyLM  # noqa: E402

generate = load("ex2_generate", "generate", "generate")
LM = ToyLM.from_corpus()


def step(ids):
    return LM.logits(LM.decode(ids[-2:]))


def run(f, prompt, n, **kw):
    return f(step, LM.encode(prompt), n, decode=LM.decode, **kw)


@pytest.mark.parametrize("cfg", [
    dict(do_sample=False), dict(do_sample=False, rep=1.3), dict(do_sample=True, temp=0.8),
    dict(do_sample=True, temp=1.0, p=0.9), dict(do_sample=True, k=5, minp=0.05, rep=1.1),
])
def test_same_as_reference(cfg):
    for prompt in ("ROMEO:\n", "JULIET:\nO "):
        mine = run(generate, prompt, 120, rng=random.Random(5), **cfg)
        ref = run(REF.generate, prompt, 120, rng=random.Random(5), **cfg)
        assert mine == ref


def test_length():
    ids, why = run(generate, "ROMEO:\n", 30, do_sample=False)
    assert len(ids) == 30 and why == "length"


def test_stop_token():
    stop = LM.stoi[" "]
    ids, why = run(generate, "ROMEO:\n", 50, do_sample=False, stop_ids=(stop,))
    assert why == "stop_token" and ids[-1] == stop and stop not in ids[:-1]


def test_stop_string():
    ids, why = run(generate, "ROMEO:\n", 80, do_sample=False, stop_strings=("the",))
    text = LM.decode(ids)
    assert why == "stop_string" and text.endswith("the") and "the" not in text[:-1]


def test_stop_token_wins():
    stop = LM.stoi["e"]
    ids, why = run(generate, "ROMEO:\n", 80, do_sample=False, stop_ids=(stop,), stop_strings=("the",))
    assert why == "stop_token"


def test_default_rng_reproducible():
    a = run(generate, "ROMEO:\n", 60, do_sample=True)
    b = run(generate, "ROMEO:\n", 60, do_sample=True)
    assert a == b


def test_penalty_sees_prompt():
    # 只看提示词也能改变第一步: 提示词里有 'e', 惩罚很大时第一步不该再选 'e'
    calls = []

    def flat(ids):
        calls.append(list(ids))
        x = [0.0] * LM.V
        x[LM.stoi["e"]] = 1.0
        x[LM.stoi["a"]] = 0.9
        return x

    ids, _ = generate(flat, LM.encode("e"), 1, do_sample=False, rep=2.0)
    assert ids == [LM.stoi["a"]]
    assert calls[0] == LM.encode("e")
