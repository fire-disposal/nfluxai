#!/usr/bin/env python3
"""入口 - 直接启动 Streamlit"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

app = str((PROJECT_ROOT / "src" / "app.py").resolve())
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", app, "--server.address", "0.0.0.0"],
    cwd=str(PROJECT_ROOT),
)
