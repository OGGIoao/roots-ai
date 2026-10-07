"""Tiny Shakespeare 语料: 全仓库唯一的定位 / 下载逻辑。

语料只有一份, 入库在 llm-bpe-tokenizer/data/ 下 (该项目是语料的所有者)。
其余项目直接读取这一份, 不再各自复制副本。

获取顺序: 本地入库文件 -> 在线下载到同一路径 -> 带操作指引的明确报错。
先找本地是为了断网 / 代理没开也能训练。
"""

import os
import urllib.error
import urllib.request

# 基于 __file__ 解析, 不依赖当前工作目录 (从任意目录运行脚本都能找到)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHAKESPEARE_PATH = os.path.join(
    REPO_ROOT, "projects", "llm-bpe-tokenizer", "data", "tinyshakespeare.txt"
)
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def ensure_shakespeare(path: str = SHAKESPEARE_PATH) -> str:
    """确保语料文件存在并返回路径; 缺失时下载, 失败则抛出可操作的 RuntimeError。"""
    if os.path.exists(path):
        return path

    print(f"本地无语料, 尝试下载 Tiny Shakespeare -> {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, tmp)
        os.replace(tmp, path)  # 下载完整后才落盘, 避免残缺文件被当成有效语料
    except (urllib.error.URLError, OSError) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"无法获取 Tiny Shakespeare 语料: {e}\n"
            f"  期望路径: {path}\n"
            f"  下载地址: {SHAKESPEARE_URL}\n"
            f"  解决方式: 启动代理后重试, 或手动下载并放到上述路径"
        ) from e
    return path


def load_shakespeare(path: str = SHAKESPEARE_PATH) -> str:
    """读取语料文本 (UTF-8)。"""
    with open(ensure_shakespeare(path), "r", encoding="utf-8") as f:
        return f.read()
