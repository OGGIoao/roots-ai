"""
测试用的小工具: 决定测试检查的是"你的练习文件"还是"参考实现"。

平时 (不设环境变量) 测试导入你写的 ex*.py。
设置 CT_SOLUTION=reference 时改为导入参考实现 (../chat_template.py、../mistral_v03.py),
用来确认测试本身能通过 (仓库的回归测试 tests/test_chat_template.py 就是这样跑的)。做练习时不用管它。
"""

import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.dirname(HERE)
sys.path.insert(0, UNIT)
sys.path.insert(0, HERE)

USE_REFERENCE = os.environ.get("CT_SOLUTION") == "reference"


def load(module, name, reference_module, reference_name=None):
    """返回练习模块里的 name; 用参考实现时返回 reference_module 里的 reference_name (默认同名)。"""
    if USE_REFERENCE:
        return getattr(importlib.import_module(reference_module), reference_name or name)
    return getattr(importlib.import_module(module), name)


def golden(name):
    """读取 experiments/results/ 下的标准答案 (官方模板渲染的结果)。"""
    with open(os.path.join(UNIT, "experiments", "results", name), encoding="utf-8") as f:
        return json.load(f)["cases"]
