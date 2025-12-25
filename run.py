#!/usr/bin/env python3
"""
快速启动脚本

用法:
    python run.py          # 启动完整平台 (UI + API)
    python run.py --api    # 仅启动API服务
    python run.py --ui     # 仅启动UI服务
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="输变电监测平台启动器")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--api", action="store_true", help="仅启动API服务")
    parser.add_argument("--ui", action="store_true", help="仅启动UI服务")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--reload", action="store_true", help="热重载")

    args = parser.parse_args()

    # 初始化日志
    from platform_core.logging import setup_logging
    setup_logging(log_level="DEBUG" if args.debug else "INFO")

    from platform_core.logging import get_logger
    logger = get_logger(__name__)

    logger.info("=" * 50)
    logger.info("输变电激光星芒破夜绘明监测平台")
    logger.info("=" * 50)

    # 初始化插件管理器并加载所有插件
    from platform_core.plugin_manager import PluginManager
    plugin_manager = PluginManager()
    plugins = plugin_manager.discover_plugins()
    logger.info(f"发现 {len(plugins)} 个插件")

    # 加载所有发现的插件
    loaded_count = 0
    for manifest in plugins:
        try:
            plugin_manager.load_plugin(manifest.id)
            loaded_count += 1
            logger.info(f"  ✓ 已加载: {manifest.id}")
        except Exception as e:
            logger.error(f"  ✗ 加载失败 [{manifest.id}]: {e}")

    logger.info(f"成功加载 {loaded_count}/{len(plugins)} 个插件")

    import uvicorn

    if args.api:
        # 仅启动API
        from apps.api_server import create_api_app
        app = create_api_app()
        logger.info(f"启动API服务: http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

    elif args.ui:
        # 仅启动UI
        from apps.ui_server import create_app
        app = create_app()
        logger.info(f"启动UI服务: http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

    else:
        # 启动完整平台 (UI包含API)
        from apps.ui_server import create_app
        app = create_app()
        logger.info(f"启动平台: http://{args.host}:{args.port}")
        logger.info(f"API文档: http://{args.host}:{args.port}/api/docs")
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
