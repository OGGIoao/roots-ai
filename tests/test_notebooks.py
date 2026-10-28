"""Colab 笔记本必须和生成脚本 tools/make_notebooks.py 的输出一致 (改了脚本忘了重新生成, 或手改了笔记本, 都会在这里报出来)。"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import make_notebooks as mk  # noqa: E402


def test_notebooks_match_generator():
    for make in mk.ALL:
        path, cells = make()
        cells[0]["source"] += f"\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({mk.colab_url(path)})"
        expected = mk.notebook(cells)
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            actual = json.load(f)                       # 按 JSON 比较, 不受换行符 (LF/CRLF) 影响
        assert actual == expected, f"{path} 与生成脚本不一致: 运行 python tools/make_notebooks.py"


def test_notebook_commands_point_to_existing_files():
    # 笔记本里 "python projects/..." 和 "python library/..." 运行的脚本都必须存在
    for make in mk.ALL:
        _, cells = make()
        for c in cells:
            if c["cell_type"] != "code":
                continue
            for line in c["source"].splitlines():
                for tok in line.replace("&&", " ").split():
                    if tok.startswith(("projects/", "library/")) and tok.endswith(".py"):
                        assert os.path.exists(os.path.join(ROOT, tok)), tok
