"""
平台配置中心

统一管理所有配置:
- 站点配置 (sites)
- 设备配置 (devices)
- 规则配置 (rules)
- 任务配置 (tasks)
- 插件配置 (plugins)
"""


from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


class DatabaseConfig(BaseModel):
    """数据库配置"""
    type: str = "sqlite"
    path: str = "data/platform.db"


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    rotation: str = "10 MB"
    retention: str = "30 days"
    log_dir: str = "logs"


class EvidenceConfig(BaseModel):
    """证据链配置"""
    base_dir: str = "evidence"
    save_raw: bool = True
    save_annotated: bool = True
    compression: str = "none"  # none, gzip, lz4
    max_age_days: int = 90


class SchedulerConfig(BaseModel):
    """调度器配置"""
    max_workers: int = 4
    task_timeout: int = 300  # 秒
    retry_count: int = 3
    retry_delay: int = 5  # 秒


class UIConfig(BaseModel):
    """UI配置"""
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False
    title: str = "输变电监测平台"


class APIConfig(BaseModel):
    """API配置"""
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["*"]


class PluginConfig(BaseModel):
    """插件配置"""
    plugins_dir: str = "plugins"
    auto_reload: bool = False
    strict_validation: bool = True


class PlatformConfig(BaseSettings):
    """平台主配置"""

    # 基本信息
    app_name: str = "输变电激光星芒破夜绘明监测平台"
    version: str = "1.0.0"
    env: str = "development"

    # 各模块配置
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    plugin: PluginConfig = Field(default_factory=PluginConfig)

    # 路径配置
    configs_dir: str = "configs"

    model_config = {
        "env_prefix": "PLATFORM_",
        "env_nested_delimiter": "__",
    }

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PlatformConfig":
        """从YAML文件加载配置"""
        config_path = Path(config_path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        return cls(**config_data)

    def get_path(self, relative_path: str) -> Path:
        """获取相对于项目根目录的绝对路径"""
        return PROJECT_ROOT / relative_path

    def get_configs_path(self) -> Path:
        """获取配置目录路径"""
        return self.get_path(self.configs_dir)

    def get_plugins_path(self) -> Path:
        """获取插件目录路径"""
        return self.get_path(self.plugin.plugins_dir)

    def get_evidence_path(self) -> Path:
        """获取证据目录路径"""
        return self.get_path(self.evidence.base_dir)

    def get_logs_path(self) -> Path:
        """获取日志目录路径"""
        return self.get_path(self.logging.log_dir)


@lru_cache()
def get_config() -> PlatformConfig:
    """获取平台配置单例"""
    config_path = os.environ.get("PLATFORM_CONFIG", "configs/platform.yaml")
    return PlatformConfig.from_yaml(PROJECT_ROOT / config_path)


def reload_config() -> PlatformConfig:
    """重新加载配置"""
    get_config.cache_clear()
    return get_config()
