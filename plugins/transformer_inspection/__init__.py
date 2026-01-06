"""
主变自主巡视插件 (A组)

功能范围:
- 外观缺陷识别: 破损、锈蚀、渗漏油、异物悬挂
- 状态识别: 呼吸器硅胶变色、阀门开闭状态
- 热成像集成: 红外图像温度提取
"""

from plugins.transformer_inspection.plugin import TransformerInspectionPlugin

__all__ = ["TransformerInspectionPlugin"]
