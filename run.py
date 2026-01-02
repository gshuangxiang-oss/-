#!/usr/bin/env python3
"""
车牌检测与识别系统 - 启动入口
License Plate Detection and Recognition System - Entry Point

运行方式:
    python run.py
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.api import run_server


if __name__ == '__main__':
    # 启动 Flask 服务器
    run_server(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

