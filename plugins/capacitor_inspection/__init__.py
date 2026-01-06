"""
电容器自主巡视插件 (D组)

功能范围:
- 结构完整性检测: 倾斜、倒塌、部件缺失
- 区域入侵检测: 人员、车辆、动物入侵告警
- 电容器组状态监控: 三相电容器组排列检测
"""

from plugins.capacitor_inspection.plugin import CapacitorInspectionPlugin

__all__ = ["CapacitorInspectionPlugin"]
