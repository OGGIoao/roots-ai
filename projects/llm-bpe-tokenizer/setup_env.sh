#!/usr/bin/env bash
# setup_env.sh — llm-bpe-tokenizer 专属虚拟环境一键创建脚本
#
# 用法：
#   cd projects/llm-bpe-tokenizer
#   bash setup_env.sh
#
# 执行后：
#   - 在当前目录创建 .venv/ 虚拟环境
#   - 安装 requirements.txt 中的所有依赖
#   - 打印激活命令提示
#
# 激活环境（每次开始工作时运行）：
#   source .venv/bin/activate       # macOS / Linux
#   .venv\Scripts\activate          # Windows PowerShell

set -e

PROJECT_NAME="llm-bpe-tokenizer"
VENV_DIR=".venv"

echo "========================================"
echo "  正在为 ${PROJECT_NAME} 创建虚拟环境..."
echo "========================================"

if ! command -v python3 &>/dev/null; then
    echo "[错误] 找不到 python3，请先安装 Python 3.9 或以上版本"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "[✓] 检测到 ${PYTHON_VERSION}"

if [ -d "${VENV_DIR}" ]; then
    echo "[!] 目录 ${VENV_DIR} 已存在，跳过创建步骤"
else
    python3 -m venv "${VENV_DIR}"
    echo "[✓] 虚拟环境创建成功：${VENV_DIR}/"
fi

echo "[→] 安装 requirements.txt 中的依赖..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install -r requirements.txt

echo ""
echo "========================================"
echo "  [✓] 环境准备完成！"
echo ""
echo "  现在请运行以下命令激活环境："
echo "    source .venv/bin/activate      # macOS / Linux"
echo "    .venv\\Scripts\\activate         # Windows PowerShell"
echo ""
echo "  然后就可以运行项目了："
echo "    python train.py"
echo "    python train.py --vocab_size 1000"
echo "    python train.py --source sample  # 无网络时使用内置语料"
echo "========================================"
