"""知识库目录: library/README.md 和各领域 README 的单元表必须和 tools/make_catalog.py 生成的一致。"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import make_catalog as mc  # noqa: E402


def test_catalog_files_up_to_date():
    for path, expected in mc.outputs():
        with open(path, encoding="utf-8") as f:
            actual = f.read()
        assert actual.replace("\r\n", "\n") == expected.replace("\r\n", "\n"), \
            f"{os.path.relpath(path, ROOT)} 和目录数据不一致: 运行 python tools/make_catalog.py"


def test_unit_names_unique_and_existing_dirs_listed():
    names = [(d, u[0]) for d, _, _, segs in mc.DOMAINS for _, us in segs for u in us]
    assert len(names) == len(set(names))
    listed = set(names)
    for d, *_ in mc.DOMAINS:
        base = os.path.join(mc.LIB, d)
        for entry in os.listdir(base):
            if os.path.isdir(os.path.join(base, entry)) and not entry.startswith((".", "_")):
                assert (d, entry) in listed, f"library/{d}/{entry} 不在目录数据里"


def test_status_marks_valid():
    for _, _, _, segs in mc.DOMAINS:
        for _, us in segs:
            for name, what, status, used in us:
                assert status in ("✅", "🟡", "⬜"), name
