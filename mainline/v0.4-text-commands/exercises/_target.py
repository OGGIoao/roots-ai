"""
测试用的小工具: 决定测试检查的是"你的练习文件"还是"参考实现"。

平时 (不设环境变量) 测试导入你写的 ex*.py。
设置 C1_SOLUTION=reference 时改为导入参考实现 (../tools.py、../agent.py、../baseline.py),
用来确认测试本身是能通过的 (仓库的回归测试 tests/test_c1.py 就是这样跑的)。你做练习时不需要管它。
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C1 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(C1))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, os.path.join(ROOT, "library", "agent", "agent-loop"))
sys.path.insert(0, C1)
sys.path.insert(0, HERE)

USE_REFERENCE = os.environ.get("C1_SOLUTION") == "reference"


def load(module, name, reference_module):
    """返回练习模块里的 name; 用参考实现时返回 reference_module 里的同名对象。"""
    if USE_REFERENCE:
        return getattr(importlib.import_module(reference_module), name)
    return getattr(importlib.import_module(module), name)
