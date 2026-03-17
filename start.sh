#!/usr/bin/env bash
# 护理教材 AI 系统 - 快速启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  护理教材 AI 问答系统"
echo "=============================================="

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行：uv sync"
    exit 1
fi

# 检查向量数据库
if [ ! -d "data/chroma_db" ]; then
    echo "⚠️  向量数据库不存在，正在导入数据..."
    uv run python src/ingest.py
fi

# 启动应用
echo "🚀 启动 Streamlit 应用..."
echo ""
echo "访问地址：http://localhost:8501"
echo "按 Ctrl+C 停止服务"
echo ""

uv run streamlit run src/app.py --server.address localhost --server.port 8501
