"""
母线自主巡视插件 (C组)

功能范围:
- 远距小目标检测: 4K大视场精准识别微小销钉缺失或裂纹
- 多目标并发处理: 绝缘子串、金具、导线连接
- 环境干扰过滤: 鸟类、飞虫等干扰过滤
"""

from plugins.busbar_inspection.plugin import BusbarInspectionPlugin

__all__ = ["BusbarInspectionPlugin"]
