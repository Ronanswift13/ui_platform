"""
表计无建模增强读数插件 (E组)

功能范围:
- 任意角度读数: 关键点检测进行透视矫正
- 自动量程识别: 无需预设表盘参数
- 失败兜底策略: 多次采样、人工复核阈值
"""

from plugins.meter_reading.plugin import MeterReadingPlugin

__all__ = ["MeterReadingPlugin"]
