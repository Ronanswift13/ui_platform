"""
开关间隔自主巡视插件 (B组)

功能范围:
- 分合位状态识别: 断路器、隔离开关、接地开关
- 五防逻辑校验: 防止带负荷拉刀闸、防止带电合接地刀
- 图像清晰度评价: 识别结果可信度评估
"""

from plugins.switch_inspection.plugin import SwitchInspectionPlugin

__all__ = ["SwitchInspectionPlugin"]
