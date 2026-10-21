"""
verify_env.py — 环境健康检查

一次跑完, 明确告诉你: 哪些能用、哪些不能、下一步该敲什么命令。

用法:
    python verify_env.py            # 先 conda activate llm-roots 再跑
    python verify_env.py --full     # 额外做一次真实 forward 前向计算

设计原则: 不抛异常中断, 逐项检查并给出可操作的修复建议。
"""

import argparse
import importlib.metadata as md
import os
import platform
import sys

# Windows 上 libiomp5md.dll 重复加载会刷屏, 先压制 (本仓库历史遗留问题)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RESULTS = []  # (级别, 项目, 结果, 建议)


def _enable_utf8_stdout():
    """
    保证在 Windows GBK 控制台下也能打印本脚本用的符号 (✅ ❌ ℹ️ 等)。

    sys.stdout.reconfigure 只在 TextIOWrapper 上可用, 用 getattr 兜住
    被替换成其它对象的情况 (如 pytest 捕获、IDE 嵌入解释器)。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def record(level, item, ok, detail="", advice=""):
    RESULTS.append((level, item, ok, detail, advice))


def check_python():
    v = sys.version_info
    record(
        "INFO",
        "Python",
        True,
        f"{platform.python_version()} @ {sys.executable}",
        "" if (v.major, v.minor) >= (3, 10) else "需要 3.10+",
    )


def check_torch():
    try:
        import torch
    except ImportError as e:
        record("FAIL", "torch 安装", False, str(e), "见下面 torch 三件套的安装命令")
        return None

    record("OK", "torch 版本", True, torch.__version__)

    # 该构建到底编了哪些 compute capability —— 判断能否跑 sm_120 最硬的证据
    try:
        arch_list = torch.cuda.get_arch_list()
        has_sm120 = any("120" in a for a in arch_list)
        if has_sm120:
            record("OK", "sm_120 支持", True, "构建含 sm_120 kernel")
        elif torch.version.cuda is not None:
            record(
                "FAIL",
                "sm_120 支持",
                False,
                f"构建仅含 {', '.join(arch_list) or '未知'}",
                "RTX 5060 是 Blackwell(sm_120), 需要 cu128 及以上的构建",
            )
        else:
            record("SKIP", "sm_120 支持", False, "CPU 构建, 无 arch list")
    except Exception as e:
        record("WARN", "arch list 查询", False, str(e))

    # CUDA 可用性 —— 这台机器有 RTX 5060 Laptop, 应该为 True
    if torch.cuda.is_available():
        record("OK", "CUDA 可用", True, f"CUDA {torch.version.cuda}")
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            record("OK", "GPU", True, f"{name} | sm_{cap[0]}{cap[1]} | {vram:.1f} GB")
        except Exception as e:  # pragma: no cover
            record("WARN", "GPU 查询", False, str(e))
    else:
        record(
            "FAIL",
            "CUDA 可用",
            False,
            f"torch {torch.__version__} 是 CPU 构建 (torch.version.cuda="
            f"{torch.version.cuda})",
            TORCH_FIX,
        )
    return torch


TORCH_FIX = (
    "RTX 50 系是 Blackwell(sm_120), 需要 CUDA 12.8+ 构建。\n"
    "注意: torch/torchvision/torchaudio 必须**同一条命令一起装**,\n"
    "否则容易出现 ABI 不匹配 (torchaudio 加载失败):\n"
    "    pip install --force-reinstall torch torchvision torchaudio \\\n"
    "        --index-url https://download.pytorch.org/whl/cu128"
)


def check_torch_stack():
    """
    检查 torch 三件套是否版本自洽且都能 import。

    真实事故: 一个旧环境里 conda 装的 torchaudio 2.2.2(cu118) 被 pip 的
    torch 2.11.0+cpu 覆盖后残留, ABI 不匹配 → libtorchaudio.pyd 加载失败。
    """
    vers = {}
    for pkg in ("torch", "torchvision", "torchaudio"):
        try:
            vers[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            vers[pkg] = None

    missing = [p for p, v in vers.items() if v is None]
    if missing:
        record(
            "WARN",
            "torch 三件套完整性",
            False,
            f"缺少: {', '.join(missing)}",
            "QLoRA/训练脚本可能不需要 torchaudio, 但缺了说明环境不干净",
        )
    else:
        record(
            "OK",
            "torch 三件套版本",
            True,
            " | ".join(f"{p} {v}" for p, v in vers.items()),
        )

    for pkg in ("torchvision", "torchaudio"):
        if vers.get(pkg) is None:
            continue
        try:
            __import__(pkg)
            record("OK", f"import {pkg}", True, "OK")
        except Exception as e:
            record(
                "FAIL",
                f"import {pkg}",
                False,
                f"{type(e).__name__}: {str(e)[:90]}",
                "ABI 不匹配的典型症状 —— 用上面那条命令把三件套一起重装",
            )


def check_sm120_forward(torch, full=False):
    """真的在 GPU 上跑一次前向, 确认 sm_120 kernel 能编出来。"""
    if torch is None or not torch.cuda.is_available():
        record("SKIP", "GPU 前向计算", False, "CUDA 不可用")
        return
    if not full:
        record("SKIP", "GPU 前向计算", False, "加 --full 才跑")
        return
    try:
        dev = "cuda"
        a = torch.randn(512, 512, device=dev)
        b = torch.randn(512, 512, device=dev)
        c = (a @ b).relu().sum().item()
        record("OK", "GPU 前向计算", True, f"matmul+relu 成功, sum={c:.3f}")
    except Exception as e:
        record("FAIL", "GPU 前向计算", False, str(e), "sm_120 kernel 未编入该 torch 构建")


def check_packages():
    # QLoRA 链路 (llm-qlora-finetune) 需要的包
    groups = {
        "QLoRA 链路": [
            ("transformers", "4.37+"),
            ("peft", "0.7+"),
            ("trl", "0.7+"),
            ("accelerate", "0.25+"),
            ("bitsandbytes", "0.41+"),
            ("datasets", "2.16+"),
        ],
        "Tokenizer 链路": [("sentencepiece", ""), ("tokenizers", ""), ("regex", "")],
        "LLM Client (模板)": [("openai", "1.0+"), ("python-dotenv", "1.0+")],
        "测试": [("pytest", "8+")],
    }
    for group, pkgs in groups.items():
        for pkg, want in pkgs:
            try:
                ver = md.version(pkg)
                record("OK", f"[{group}] {pkg}", True, ver)
            except md.PackageNotFoundError:
                advice = ""
                if pkg == "bitsandbytes":
                    advice = (
                        "QLoRA 的 4-bit 量化必需。Windows 上用:\n"
                        "    pip install bitsandbytes\n"
                        "  若报错, 改用预编译 wheel (Windows 支持滞后), 或走 WSL2。"
                    )
                if group == "测试":  # 只影响跑测试, 不应让环境整体判红
                    record("WARN", f"[{group}] {pkg}", False, "未安装", "pip install pytest")
                    continue
                record("FAIL", f"[{group}] {pkg}", False, "未安装", advice)

    # 仓库自带的共享包 common/ (llm-transformer / llm-minigpt / llm-bpe-tokenizer 依赖它)
    # 注意: 从仓库根运行本脚本时, 脚本所在目录 (含 common/) 自动在 sys.path[0], 直接 import
    # 永远成功; 所以排除它, 只看"其它目录运行项目脚本时"是否找得到 common。
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    saved = sys.path[:]
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != here]
    try:
        # find_spec 走完整的 import 机制, 可编辑安装 (meta_path 钩子) 也能识别
        spec = importlib.util.find_spec("common")
    except (ImportError, ValueError):
        spec = None
    finally:
        sys.path[:] = saved
    if spec is not None:
        record("OK", "[仓库] 共享包 common/", True, os.path.dirname(spec.origin or ""))
    else:
        record("FAIL", "[仓库] 共享包 common/", False, "不在 Python 搜索路径中",
               "在仓库根目录执行 (必须是可编辑安装):" + chr(10) + "    pip install -e . --no-deps")


def _repo_root() -> str:
    """从本脚本位置向上找到仓库根 (含 projects/ 的那一层)。"""
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, "projects")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return here


def check_corpus():
    """语料是否就位 —— 断网时这是能否训练的关键。"""
    root = _repo_root()
    cands = {
        "BPE 语料": os.path.join(
            root, "projects", "llm-bpe-tokenizer", "data", "tinyshakespeare.txt"
        ),
        "BPE tokenizer": os.path.join(
            root, "projects", "llm-bpe-tokenizer", "tokenizer.json"
        ),
    }
    for label, path in cands.items():
        if os.path.exists(path):
            record("OK", label, True, f"{os.path.getsize(path)/1024:.0f} KB")
        else:
            record("FAIL", label, False, f"缺失: {path}", "语料缺失会导致无法训练")


def _read_system_proxy() -> str | None:
    """读 Windows 系统代理 (WinINET)。返回 host:port 或 None。"""
    if os.name != "nt":
        return None
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        )
        try:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if not enabled:
                return None
            server, _ = winreg.QueryValueEx(key, "ProxyServer")
            # 可能是 "http=host:port;https=host:port" 形式
            if "=" in server:
                for part in server.split(";"):
                    if part.lower().startswith("http="):
                        return part.split("=", 1)[1]
            return server or None
        finally:
            winreg.CloseKey(key)
    except Exception:
        return None


def check_proxy():
    """
    判断**能不能真的出网**, 而不是只看某个端口开没开。

    历史教训: 早期版本硬编码探测 7890, 结果 Clash 已经正常工作、
    端口也确实是 7890 时仍误报 (连接建立时序问题), 反而误导判断。
    """
    import socket

    # 1) 首选: 直连探测是否真能出网 (走系统代理由 urllib 自动处理)
    try:
        import urllib.request

        req = urllib.request.Request(
            "https://pypi.org/simple/", method="HEAD",
            headers={"User-Agent": "verify-env"},
        )
        with urllib.request.urlopen(req, timeout=8):
            record("OK", "出网连通性", True, "可访问 pypi.org")
        return
    except Exception as e:
        direct_err = f"{type(e).__name__}: {str(e)[:70]}"

    # 2) 出网失败 -> 看系统代理端口在不在, 给出针对性建议
    proxy = _read_system_proxy()
    if proxy and ":" in proxy:
        host, _, port = proxy.rpartition(":")
        try:
            with socket.create_connection((host, int(port)), timeout=1.5):
                record(
                    "WARN",
                    "出网连通性",
                    False,
                    f"代理 {proxy} 在监听, 但仍无法出网 ({direct_err})",
                    "代理进程活着但节点不通 —— 检查代理软件是否选到了可用节点,\n"
                    "或代理规则是否把 pypi.org / download.pytorch.org 走了直连。",
                )
                return
        except OSError:
            pass

    record(
        "WARN",
        "出网连通性",
        False,
        direct_err,
        f"系统代理 = {proxy or '(未设置)'}, 但无法出网\n"
        "  结果: pip / huggingface 下载都会失败。\n"
        "  修复: 启动 Clash/V2Ray 等代理并确认节点可用,\n"
        "        或临时取消系统代理: netsh winhttp reset proxy",
    )


def check_quantization(full=False):
    """
    真的用 bitsandbytes 做一次 NF4 4-bit 量化。

    这是 QLoRA 能否跑通的**唯一可信证据** —— 只检查包在不在没有意义,
    bnb 在 Windows 上经常"装了但加载不了 CUDA 后端"。
    """
    try:
        import bitsandbytes as bnb
    except Exception as e:
        record("FAIL", "bitsandbytes 可用", False, f"{type(e).__name__}: {str(e)[:80]}",
               "QLoRA 的 4-bit 量化必需:\n    pip install bitsandbytes")
        return
    if not full:
        record("SKIP", "4-bit 量化实测", False, "加 --full 才跑")
        return
    try:
        import torch

        if not torch.cuda.is_available():
            record("SKIP", "4-bit 量化实测", False, "CUDA 不可用")
            return
        # bnb 4bit 后端自检: 直接建一个 NF4 层并前向, 比只看版本号可信
        from bitsandbytes.nn import Linear4bit

        layer = Linear4bit(64, 64, quant_type="nf4", compute_dtype=torch.float16).cuda()
        out = layer(torch.randn(4, 64, device="cuda", dtype=torch.float16))
        record("OK", "4-bit 量化实测", True,
               f"NF4 Linear4bit 前向成功, 输出 shape={tuple(out.shape)}")
    except Exception as e:
        record("FAIL", "4-bit 量化实测", False, f"{type(e).__name__}: {str(e)[:100]}",
               "bnb 无法在 GPU 上建立 4-bit 层。Windows 支持滞后时走 WSL2。")


def main():
    _enable_utf8_stdout()

    ap = argparse.ArgumentParser(description="环境健康检查")
    ap.add_argument("--full", action="store_true", help="额外跑真实 GPU 前向 + 4-bit 量化")
    args = ap.parse_args()

    print("=" * 68)
    print("  环境健康检查 — llm-roots")
    print("=" * 68)

    check_python()
    torch = check_torch()
    check_torch_stack()
    check_sm120_forward(torch, full=args.full)
    check_quantization(full=args.full)
    check_packages()
    check_corpus()
    check_proxy()

    # ── 汇总输出 ──
    icon = {"OK": "✅", "FAIL": "❌", "WARN": "⚠️ ", "INFO": "ℹ️ ", "SKIP": "⏭️ "}
    print()
    for level, item, ok, detail, advice in RESULTS:
        print(f"{icon[level]} {item:<34} {detail}")
        if advice:
            for line in advice.splitlines():
                print(f"      → {line}")

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print()
    print("-" * 68)
    if fails:
        print(f"结论: {len(fails)} 项失败, {len(warns)} 项警告 —— 见上面 → 的建议")
        print()
        print("重建环境的完整命令 (注意三件套要一起装):")
        print("  conda create -n llm-roots python=3.10 pip -y")
        print("  conda activate llm-roots")
        print("  pip install torch torchvision torchaudio \\")
        print("      --index-url https://download.pytorch.org/whl/cu128")
        print("  pip install bitsandbytes")
        print("  pip install -r requirements.txt")
        print("  pip install -e . --no-deps    # 仓库自带的共享包 common/")
        print("  python verify_env.py --full   # 应为全绿")
        return 1
    print(f"结论: 全绿 ✅  ({len(warns)} 项警告)")
    print("GPU 链路已打通, 可以开始 QLoRA 实操了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
