"""
Plugin logging utility.

Provides a standardized logger setup for plugins with consistent
formatting and output.
"""

from __future__ import annotations

import logging
import sys


def setup_plugin_logger(
    plugin_id: str,
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Create and configure a logger for a plugin.

    Args:
        plugin_id: Plugin identifier (used as logger name).
        level: Logging level (default: INFO).
        log_file: Optional file path to write logs to.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(f"darkbreaker.plugin.{plugin_id}")
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
