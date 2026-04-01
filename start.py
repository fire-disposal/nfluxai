#!/usr/bin/env python3
"""Python 启动入口：默认启动应用（等价于 `python main.py --run`）。"""

from __future__ import annotations

import sys

import main as app_main


def run():
    if len(sys.argv) == 1:
        sys.argv.append("--run")
    app_main.main()


if __name__ == "__main__":
    run()
