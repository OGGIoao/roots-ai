"""文档里的相对链接必须指向存在的文件或目录 (搬目录、改名之后最容易断的就是它们)。"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", ".local", "node_modules", "__pycache__", "snapshots"}
# [文字](目标) 和 ![图](目标); 目标里不含空格和右括号
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def markdown_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for fn in filenames:
            if fn.endswith(".md"):
                yield os.path.join(dirpath, fn)


def broken_links():
    bad = []
    for path in markdown_files():
        text = open(path, encoding="utf-8").read()
        text = re.sub(r"```.*?```", "", text, flags=re.S)          # 代码块里的不算链接
        for target in LINK_RE.findall(text):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue                                            # http(s):、mailto: 和页内锚点不查
            rel = target.split("#")[0]
            if not rel:
                continue
            if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(path), rel))):
                bad.append(f"{os.path.relpath(path, ROOT)} → {target}")
    return bad


def test_relative_links_exist():
    bad = broken_links()
    assert not bad, f"{len(bad)} 个断链:\n" + "\n".join(bad[:60])


if __name__ == "__main__":
    for b in broken_links():
        print(b)
