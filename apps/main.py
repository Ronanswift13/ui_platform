"""
主程序入口

启动完整的输变电监测平台 (包含UI和API)
"""


from __future__ import annotations
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="输变电激光星芒破夜绘明监测平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址 (默认: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="监听端口 (默认: 8080)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/platform.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置配置文件环境变量
    import os
    os.environ["PLATFORM_CONFIG"] = args.config

    # 初始化日志
    from platform_core.logging import setup_logging
    setup_logging(log_level=args.log_level, console=True)

    from platform_core.logging import get_logger
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("输变电激光星芒破夜绘明监测平台")
    logger.info("=" * 60)

    # 初始化各模块
    from platform_core.config import get_config
    config = get_config()
    logger.info(f"配置已加载: {args.config}")

    from platform_core.plugin_manager import PluginManager
    plugin_manager = PluginManager()
    plugins = plugin_manager.discover_plugins()
    logger.info(f"发现 {len(plugins)} 个插件")

    # 启动UI服务器
    from apps.ui_server import create_app
    app = create_app()

    import uvicorn
    logger.info(f"启动UI服务器: http://{args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
