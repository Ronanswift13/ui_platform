"""
UI模块初始化
破夜绘明激光监测平台

提供Web界面功能
"""

from pathlib import Path

# UI模块路径
UI_DIR = Path(__file__).parent
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"

__all__ = [
    "UI_DIR",
    "TEMPLATES_DIR", 
    "STATIC_DIR",
]
