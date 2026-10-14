"""
fetch.py — 抓取模型的"解剖数据" (不下载权重)

对 models.py 里的每个模型, 只读取三样小东西:
  1. config.json                     超参数
  2. 张量名 (权重的"目录")            多文件模型读 model.safetensors.index.json,
                                     单文件模型只读 safetensors 文件头 (HTTP 范围请求)
  3. 仓库创建时间                     作为"发布时间"的近似, 便于按时间排序

张量名里的层号、专家编号统一替换成 N 再去重, 只保留"结构模式"及其出现次数,
所以即使是有 9 万多个张量的 DeepSeek-V3, 快照也只有几 KB。

结果写入 snapshots/<组织>__<模型>.json, 随仓库提交 (配置可能被上游修改, 快照保证可复现)。

运行:
    python fetch.py              # 只抓还没有快照的模型
    python fetch.py --refresh    # 全部重新抓取
受限仓库需要先设置环境变量 HF_TOKEN (在 huggingface.co 同意模型协议后生成)。
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from collections import Counter

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import get_safetensors_metadata, hf_hub_download, model_info  # noqa: E402
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError  # noqa: E402

from models import MODELS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(HERE, "snapshots")


def retry(fn, tries=4, wait=3.0):
    """代理偶尔会断开 TLS 连接 (SSL: UNEXPECTED_EOF), 重试几次通常就好。"""
    for i in range(tries):
        try:
            return fn()
        except (GatedRepoError, RepositoryNotFoundError, EntryNotFoundError):
            raise                                   # 这些重试也没用
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def tensor_names(repo):
    """返回 (张量名列表, 参数总数或 None)。"""
    try:
        path = retry(lambda: hf_hub_download(repo, "model.safetensors.index.json"))
        with open(path, encoding="utf-8") as f:
            index = json.load(f)
        return list(index["weight_map"]), None
    except EntryNotFoundError:
        md = retry(lambda: get_safetensors_metadata(repo))   # 单文件: 只读文件头
        names = [n for fm in md.files_metadata.values() for n in fm.tensors]
        return names, sum(md.parameter_count.values())


def patterns(names):
    """把层号、专家编号等数字替换成 N, 统计每种结构模式出现的次数。"""
    return dict(sorted(Counter(re.sub(r"\.\d+\.", ".N.", re.sub(r"\.\d+\.", ".N.", n)) for n in names).items()))


def fetch_one(repo):
    snap = {"repo": repo, "fetched": dt.date.today().isoformat()}
    try:
        info = retry(lambda: model_info(repo))
        snap["created"] = info.created_at.date().isoformat() if info.created_at else None
        with open(retry(lambda: hf_hub_download(repo, "config.json")), encoding="utf-8") as f:
            snap["config"] = json.load(f)
        names, n_params = tensor_names(repo)
        snap["n_tensors"] = len(names)
        snap["n_params_header"] = n_params
        snap["tensor_patterns"] = patterns(names)
        snap["status"] = "ok"
    except GatedRepoError:
        snap["status"] = "gated"
    except Exception as e:  # noqa: BLE001
        snap["status"] = f"error: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
    return snap


def snap_path(repo):
    return os.path.join(SNAP_DIR, repo.replace("/", "__") + ".json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="已有快照也重新抓取")
    args = ap.parse_args()
    os.makedirs(SNAP_DIR, exist_ok=True)

    for repo, _ in MODELS:
        path = snap_path(repo)
        if os.path.exists(path) and not args.refresh:
            with open(path, encoding="utf-8") as f:
                if json.load(f).get("status") == "ok":
                    print(f"  跳过 (已有快照)  {repo}")
                    continue
        snap = fetch_one(repo)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=1)
        extra = f"{snap['n_tensors']} 个张量 → {len(snap['tensor_patterns'])} 种模式" if snap["status"] == "ok" else ""
        print(f"  {snap['status']:<8} {repo}  {extra}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
