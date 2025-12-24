#!/usr/bin/env python3
"""
打包脚本 - 将平台打包为可执行文件

支持:
- Windows: exe
- macOS: app
- Linux: 可执行文件

使用方法:
    python build.py [--onefile] [--console] [--name NAME]
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def clean():
    """清理构建目录"""
    for d in [DIST_DIR, BUILD_DIR]:
        if d.exists():
            shutil.rmtree(d)
            print(f"已清理: {d}")


def build_with_pyinstaller(args):
    """使用PyInstaller打包"""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", args.name,
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        "--specpath", str(BUILD_DIR),
    ]

    # 单文件模式
    if args.onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # 控制台模式
    if args.console:
        cmd.append("--console")
    else:
        cmd.append("--windowed")

    # 添加数据文件
    cmd.extend([
        "--add-data", f"ui/templates{os.pathsep}ui/templates",
        "--add-data", f"ui/static{os.pathsep}ui/static",
        "--add-data", f"configs{os.pathsep}configs",
        "--add-data", f"plugins{os.pathsep}plugins",
    ])

    # 隐藏导入
    cmd.extend([
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
    ])

    # 图标
    icon_path = PROJECT_ROOT / "ui" / "static" / "images" / "icon.ico"
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])

    # 入口文件
    cmd.append(str(PROJECT_ROOT / "apps" / "main.py"))

    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"\n打包完成! 输出目录: {DIST_DIR}")


def create_spec_file():
    """创建PyInstaller spec文件"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 收集依赖
hiddenimports = collect_submodules('uvicorn') + collect_submodules('fastapi')

a = Analysis(
    ['apps/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('ui/templates', 'ui/templates'),
        ('ui/static', 'ui/static'),
        ('configs', 'configs'),
        ('plugins', 'plugins'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PowerStationMonitor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PowerStationMonitor',
)
'''
    spec_path = PROJECT_ROOT / "PowerStationMonitor.spec"
    spec_path.write_text(spec_content)
    print(f"已创建spec文件: {spec_path}")


def main():
    parser = argparse.ArgumentParser(description="输变电监测平台打包工具")
    parser.add_argument("--clean", action="store_true", help="清理构建目录")
    parser.add_argument("--onefile", action="store_true", help="打包为单个文件")
    parser.add_argument("--console", action="store_true", help="显示控制台窗口")
    parser.add_argument("--name", default="PowerStationMonitor", help="输出文件名")
    parser.add_argument("--spec", action="store_true", help="仅生成spec文件")

    args = parser.parse_args()

    if args.clean:
        clean()
        return

    if args.spec:
        create_spec_file()
        return

    # 检查PyInstaller
    try:
        import PyInstaller
    except ImportError:
        print("错误: 请先安装PyInstaller")
        print("  pip install pyinstaller")
        sys.exit(1)

    build_with_pyinstaller(args)


if __name__ == "__main__":
    main()
