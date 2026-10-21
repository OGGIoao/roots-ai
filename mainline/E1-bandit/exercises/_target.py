"""
测试用的小工具: 决定测试检查的是"你的练习文件"还是"参考实现"。

平时 (不设环境变量) 测试导入你写的 ex*.py。
设置 BANDIT_SOLUTION=reference 时改为导入 ../agents.py, 用来确认测试本身是能通过的
(仓库的回归测试 tests/test_bandit.py 就是这样跑的)。你做练习时不需要管它。
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # agents.py, env.py, runner.py
sys.path.insert(0, HERE)

USE_REFERENCE = os.environ.get("BANDIT_SOLUTION") == "reference"


def load(module, name):
    """返回练习模块里的 name; 用参考实现时返回 agents.py 里的同名对象。"""
    if USE_REFERENCE:
        return getattr(importlib.import_module("agents"), name)
    return getattr(importlib.import_module(module), name)
