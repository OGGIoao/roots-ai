"""文档里的相对链接必须指向存在的文件或目录 (搬目录、改名之后最容易断的就是它们)。"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", ".local", ".claude", "node_modules", "__pycache__", "snapshots"}
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


def slug(heading):
    """GitHub 给标题生成锚点的规则: 小写, 去掉标点 (保留字母、数字、汉字、下划线、连字符), 空格换成连字符。"""
    h = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return h.replace(" ", "-")


def anchors_of(path, cache={}):
    if path not in cache:
        found = set()
        in_code = False
        for line in open(path, encoding="utf-8"):
            if line.startswith("```"):
                in_code = not in_code
            if not in_code and line.startswith("#"):
                found.add(slug(line.lstrip("#")))
            found.update(re.findall(r'<a id="([^"]+)"', line))
        cache[path] = found
    return cache[path]


def broken_anchors():
    """链接里的 #锚点 必须是目标文件里真实存在的标题 (讲义改了节号最容易断的就是它们)。"""
    bad = []
    for path in markdown_files():
        text = re.sub(r"```.*?```", "", open(path, encoding="utf-8").read(), flags=re.S)
        for target in LINK_RE.findall(text):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or "#" not in target:
                continue
            rel, anchor = target.split("#", 1)
            dest = os.path.normpath(os.path.join(os.path.dirname(path), rel)) if rel else path
            if dest.endswith(".md") and os.path.exists(dest) and anchor not in anchors_of(dest):
                bad.append(f"{os.path.relpath(path, ROOT)} → {target}")
    return bad


def test_anchors_exist():
    bad = broken_anchors()
    assert not bad, f"{len(bad)} 个断掉的锚点:\n" + "\n".join(bad[:60])


if __name__ == "__main__":
    for b in broken_links():
        print(b)
