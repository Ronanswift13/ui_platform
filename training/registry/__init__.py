"""
训练注册表

提供 plugin_id ↔ 训练配置的映射能力。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REGISTRY_PATH = Path(__file__).parent / "plugin_training_mapping.json"
_REGISTRY_CACHE: dict | None = None


def get_registry() -> dict[str, Any]:
    """加载训练映射注册表"""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            _REGISTRY_CACHE = json.load(f)
    return _REGISTRY_CACHE


def get_plugin_config(plugin_id: str) -> dict[str, Any] | None:
    """获取指定插件的训练配置"""
    registry = get_registry()
    # 先检查别名
    alias_map = registry.get("legacy_alias_map", {})
    resolved = alias_map.get(plugin_id, plugin_id)
    return registry.get("plugins", {}).get(resolved)


def resolve_alias(name: str) -> str:
    """将旧名称映射到新 plugin_id"""
    registry = get_registry()
    return registry.get("legacy_alias_map", {}).get(name, name)


def list_supported_plugins() -> list[str]:
    """列出所有纳入统一训练库的插件 ID"""
    registry = get_registry()
    return list(registry.get("plugins", {}).keys())


def list_visual_plugins() -> list[str]:
    """兼容旧名称，现返回统一训练库中的全部插件"""

    return list_supported_plugins()


def list_temporal_plugins() -> list[str]:
    """列出时序/数值异常类插件 ID"""

    registry = get_registry()
    result: list[str] = []
    for plugin_id, config in registry.get("plugins", {}).items():
        modalities = str(config.get("input_modality", ""))
        if any(
            token in modalities
            for token in ("audio_", "timeseries", "event_sequence")
        ):
            result.append(plugin_id)
    return result


def list_multimodal_plugins() -> list[str]:
    """列出多模态融合类插件 ID"""

    registry = get_registry()
    result: list[str] = []
    for plugin_id, config in registry.get("plugins", {}).items():
        if str(config.get("input_modality", "")) == "multimodal":
            result.append(plugin_id)
    return result


def get_task_type_info(task_type: str) -> dict[str, Any] | None:
    """获取 task_type 的元信息"""
    registry = get_registry()
    return registry.get("task_types", {}).get(task_type)
