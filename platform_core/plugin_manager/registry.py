"""
插件注册表

维护插件元信息,支持:
- 按ID/能力/设备类型查询
- 插件依赖关系
- 插件状态统计
"""


from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from platform_core.plugin_manager.base import BasePlugin, PluginCapability, PluginStatus


@dataclass
class PluginInfo:
    """插件信息"""
    id: str
    name: str
    version: str
    code_hash: str
    status: PluginStatus
    capabilities: list[PluginCapability]
    device_types: list[str]
    loaded_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    use_count: int = 0
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "code_hash": self.code_hash,
            "status": self.status.value,
            "capabilities": [c.value for c in self.capabilities],
            "device_types": self.device_types,
            "loaded_at": self.loaded_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "use_count": self.use_count,
            "error_count": self.error_count,
        }


class PluginRegistry:
    """
    插件注册表

    维护所有已加载插件的元信息
    """

    def __init__(self):
        self._plugins: dict[str, PluginInfo] = {}
        self._capability_index: dict[PluginCapability, set[str]] = {}
        self._device_type_index: dict[str, set[str]] = {}

    def register(self, plugin: BasePlugin) -> PluginInfo:
        """注册插件"""
        info = PluginInfo(
            id=plugin.id,
            name=plugin.name,
            version=plugin.version,
            code_hash=plugin.code_hash,
            status=plugin.status,
            capabilities=plugin.manifest.capabilities,
            device_types=plugin.manifest.device_types,
        )

        self._plugins[plugin.id] = info

        # 更新索引
        for cap in plugin.manifest.capabilities:
            if cap not in self._capability_index:
                self._capability_index[cap] = set()
            self._capability_index[cap].add(plugin.id)

        for device_type in plugin.manifest.device_types:
            if device_type not in self._device_type_index:
                self._device_type_index[device_type] = set()
            self._device_type_index[device_type].add(plugin.id)

        return info

    def unregister(self, plugin_id: str) -> bool:
        """注销插件"""
        if plugin_id not in self._plugins:
            return False

        info = self._plugins[plugin_id]

        # 清理索引
        for cap in info.capabilities:
            if cap in self._capability_index:
                self._capability_index[cap].discard(plugin_id)

        for device_type in info.device_types:
            if device_type in self._device_type_index:
                self._device_type_index[device_type].discard(plugin_id)

        del self._plugins[plugin_id]
        return True

    def get(self, plugin_id: str) -> Optional[PluginInfo]:
        """获取插件信息"""
        return self._plugins.get(plugin_id)

    def list_all(self) -> list[PluginInfo]:
        """列出所有插件"""
        return list(self._plugins.values())

    def find_by_capability(self, capability: PluginCapability) -> list[str]:
        """按能力查找插件"""
        return list(self._capability_index.get(capability, set()))

    def find_by_device_type(self, device_type: str) -> list[str]:
        """按设备类型查找插件"""
        return list(self._device_type_index.get(device_type, set()))

    def update_status(self, plugin_id: str, status: PluginStatus) -> None:
        """更新插件状态"""
        if plugin_id in self._plugins:
            self._plugins[plugin_id].status = status

    def record_usage(self, plugin_id: str, success: bool = True) -> None:
        """记录插件使用"""
        if plugin_id in self._plugins:
            info = self._plugins[plugin_id]
            info.last_used = datetime.now()
            info.use_count += 1
            if not success:
                info.error_count += 1

    def get_statistics(self) -> dict[str, Any]:
        """获取统计信息"""
        total = len(self._plugins)
        by_status = {}
        for info in self._plugins.values():
            status = info.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total": total,
            "by_status": by_status,
            "capabilities": {
                cap.value: len(ids)
                for cap, ids in self._capability_index.items()
            },
            "device_types": {
                dt: len(ids)
                for dt, ids in self._device_type_index.items()
            },
        }
