"""
统一日志模块

基于loguru实现:
- 结构化日志
- 日志轮转
- 错误日志分离
- 任务日志追踪
"""


from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from platform_core.config import get_config, PROJECT_ROOT

# 移除默认handler
logger.remove()

# 日志格式
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

LOG_FORMAT_FILE = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    console: bool = True,
) -> None:
    """
    配置日志系统

    Args:
        log_dir: 日志目录
        log_level: 日志级别
        console: 是否输出到控制台
    """
    config = get_config()

    if log_dir is None:
        log_dir = config.get_logs_path()

    log_dir.mkdir(parents=True, exist_ok=True)

    # 控制台输出
    if console:
        logger.add(
            sys.stderr,
            format=LOG_FORMAT,
            level=log_level,
            colorize=True,
        )

    # 主日志文件
    logger.add(
        log_dir / "platform.log",
        format=LOG_FORMAT_FILE,
        level=log_level,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        compression="gz",
        encoding="utf-8",
    )

    # 错误日志单独文件
    logger.add(
        log_dir / "errors.log",
        format=LOG_FORMAT_FILE,
        level="ERROR",
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        compression="gz",
        encoding="utf-8",
    )

    logger.info(f"日志系统初始化完成, 日志目录: {log_dir}")


def get_logger(name: str = __name__) -> Any:
    """获取logger实例"""
    return logger.bind(name=name)


class TaskLogger:
    """
    任务日志记录器

    每个任务有独立的日志文件
    """

    def __init__(self, task_id: str, run_id: str):
        self.task_id = task_id
        self.run_id = run_id
        self._logger = logger.bind(task_id=task_id, run_id=run_id)
        self._log_file: Optional[Path] = None
        self._handler_id: Optional[int] = None

    def setup(self, evidence_dir: Path) -> None:
        """设置任务日志文件"""
        self._log_file = evidence_dir / "task.log"
        self._handler_id = logger.add(
            self._log_file,
            format=LOG_FORMAT_FILE,
            level="DEBUG",
            filter=lambda record: record["extra"].get("task_id") == self.task_id,
        )

    def cleanup(self) -> None:
        """清理任务日志handler"""
        if self._handler_id is not None:
            logger.remove(self._handler_id)

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._logger.error(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.debug(message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        self._logger.exception(message, **kwargs)


# 初始化默认日志配置
def init_default_logging() -> None:
    """初始化默认日志配置"""
    try:
        config = get_config()
        setup_logging(
            log_level=config.logging.level,
            console=True,
        )
    except Exception:
        # 配置加载失败时使用最小化配置
        logger.add(sys.stderr, format=LOG_FORMAT, level="DEBUG")


__all__ = [
    "logger",
    "get_logger",
    "setup_logging",
    "TaskLogger",
    "init_default_logging",
]
