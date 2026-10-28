"""
测试用的小工具: 决定测试检查的是"你的练习文件"还是"参考实现"。

平时 (不设环境变量) 测试导入你写的 ex*.py。
设置 SO_SOLUTION=reference 时改为导入参考实现 (../constrained.py、../number_dfa.py),
用来确认测试本身能通过 (仓库的回归测试 tests/test_structured_output.py 就是这样跑的)。做练习时不用管它。
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
sys.path.insert(0, UNIT)
sys.path.insert(0, HERE)

USE_REFERENCE = os.environ.get("SO_SOLUTION") == "reference"


def load(module, name, reference_module, reference_name=None):
    if USE_REFERENCE:
        return getattr(importlib.import_module(reference_module), reference_name or name)
    return getattr(importlib.import_module(module), name)
