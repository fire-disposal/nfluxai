#!/usr/bin/env python3
"""
护理教材 AI 系统 - 主入口点

功能:
- 检查必要的数据文件是否存在（向量库、索引文件）
- 提供命令行参数支持（--ingest 导入数据，--run 启动应用）
- 作为 Streamlit 应用的启动入口
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 可选：如果安装了 python-dotenv，会自动加载项目根目录下的 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("⚙️ 已加载 .env（如果存在）")
except Exception:
    pass


# 项目根目录
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "chroma_db"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_FILE = INDEX_DIR / "chunks_index.json"
STRUCTURED_INDEX_FILE = INDEX_DIR / "index.json"
LEGACY_INDEX_FILE = INDEX_DIR / "chunks.json"
LEGACY_STRUCTURED_INDEX_FILE = INDEX_DIR / "index_v2.json"


def check_data_files() -> bool:
    """
    检查必要的数据文件是否存在

    Returns:
        bool: 数据文件是否完整
    """
    missing = []

    # 检查向量库目录
    if not DATA_DIR.exists():
        missing.append(f"向量库目录：{DATA_DIR}")
    elif not any(DATA_DIR.glob("*")):
        missing.append(f"向量库目录为空：{DATA_DIR}")

    # 检查索引文件
    if not (INDEX_FILE.exists() or LEGACY_INDEX_FILE.exists()):
        missing.append(f"索引文件：{INDEX_FILE}（或兼容文件 {LEGACY_INDEX_FILE.name}）")
    if not (STRUCTURED_INDEX_FILE.exists() or LEGACY_STRUCTURED_INDEX_FILE.exists()):
        missing.append(
            f"结构化索引文件：{STRUCTURED_INDEX_FILE}（或兼容文件 {LEGACY_STRUCTURED_INDEX_FILE.name}）"
        )

    if missing:
        print("❌ 以下必要数据文件缺失:")
        for item in missing:
            print(f"   - {item}")
        return False

    print("✅ 数据文件检查通过")
    return True


def run_ingest(force: bool = False):
    """运行数据导入脚本"""
    if not force and check_data_files():
        print("ℹ️ 数据已就绪，跳过导入")
        return

    print("=" * 60)
    print("开始导入护理教材数据...")
    print("=" * 60)

    ingest_script = PROJECT_ROOT / "src" / "ingest.py"

    if not ingest_script.exists():
        print(f"❌ 导入脚本不存在：{ingest_script}")
        sys.exit(1)

    # 使用 subprocess 运行导入脚本
    result = subprocess.run(
        [sys.executable, str(ingest_script)],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("❌ 数据导入失败")
        sys.exit(1)

    print("\n✅ 数据导入完成")


def ensure_data_ready(auto_ingest: bool = True) -> bool:
    """确保数据可用。"""
    if check_data_files():
        return True

    if not auto_ingest:
        return False

    print("\n🔄 检测到数据缺失，自动执行数据导入...")
    run_ingest(force=True)
    print("\n🔁 再次检查数据文件...")
    return check_data_files()


def run_app(auto_ingest: bool = True):
    """启动 Streamlit 应用"""
    print("=" * 60)
    print("启动护理教材 AI 问答系统...")
    print("=" * 60)

    # 先检查数据文件（支持自动导入）
    if not ensure_data_ready(auto_ingest=auto_ingest):
        print("\n💡 可手动运行数据导入:")
        print(f"   {sys.argv[0]} --ingest")
        print(f"   或 python {sys.argv[0]} --ingest")
        sys.exit(1)

    app_script = PROJECT_ROOT / "src" / "app.py"

    if not app_script.exists():
        print(f"❌ 应用脚本不存在：{app_script}")
        sys.exit(1)

    # 使用 subprocess 启动 Streamlit
    result = subprocess.run(
        ["streamlit", "run", str(app_script), "--server.address", "0.0.0.0"],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("❌ 应用启动失败")
        sys.exit(1)


def main():
    """主函数 - 解析命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(
        description="护理教材 AI 系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --ingest    导入教材数据到向量库
  %(prog)s --run       启动 Streamlit 应用
  %(prog)s             默认启动应用（如果数据已存在）
        """
    )

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="导入教材数据到向量库"
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="启动 Streamlit 应用"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="仅检查数据文件状态"
    )
    parser.add_argument(
        "--no-auto-ingest",
        action="store_true",
        help="启动应用时不自动导入缺失数据"
    )

    args = parser.parse_args()

    # 根据参数执行相应操作
    if args.ingest:
        run_ingest(force=True)
    elif args.run:
        run_app(auto_ingest=not args.no_auto_ingest)
    elif args.check:
        if check_data_files():
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # 默认行为：启动应用
        run_app(auto_ingest=not args.no_auto_ingest)


if __name__ == "__main__":
    main()
