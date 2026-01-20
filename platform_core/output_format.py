"""
统一插件输出格式规范
输变电激光星芒破夜绘明监测平台

基于UPDATE.md规范:
- 统一的UI组件格式
- 标准化的结果标签
- 规范化的元数据结构
- 兼容多种显示模式

版本: 1.0.0
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# 标准标签定义
# =============================================================================

class StandardLabel(str, Enum):
    """
    标准识别标签

    所有插件应使用这些标准标签，确保UI组件可以正确解析和显示
    """

    # 通用状态
    NORMAL = "normal"               # 正常
    ABNORMAL = "abnormal"           # 异常
    WARNING = "warning"             # 警告
    ERROR = "error"                 # 错误
    UNKNOWN = "unknown"             # 未知

    # 缺陷类型 (A组-主变/C组-母线)
    OIL_LEAK = "oil_leak"           # 渗漏油
    RUST = "rust"                   # 锈蚀
    DAMAGE = "damage"               # 破损
    CRACK = "crack"                 # 裂纹
    FOREIGN_OBJECT = "foreign_object"  # 异物悬挂
    DEFORMATION = "deformation"     # 变形
    DISCOLORATION = "discoloration" # 变色
    CORROSION = "corrosion"         # 腐蚀
    OVERHEATING = "overheating"     # 过热

    # 开关状态 (B组)
    BREAKER_OPEN = "breaker_open"           # 断路器分闸
    BREAKER_CLOSED = "breaker_closed"       # 断路器合闸
    BREAKER_INTERMEDIATE = "breaker_intermediate"  # 断路器中间态
    ISOLATOR_OPEN = "isolator_open"         # 隔离开关分闸
    ISOLATOR_CLOSED = "isolator_closed"     # 隔离开关合闸
    GROUNDING_OPEN = "grounding_open"       # 接地开关分闸
    GROUNDING_CLOSED = "grounding_closed"   # 接地开关合闸

    # 表计读数 (E组)
    METER_READING = "meter_reading"         # 表计读数
    METER_OVERRANGE = "meter_overrange"     # 表计超量程
    METER_UNDERRANGE = "meter_underrange"   # 表计欠量程

    # 硅胶状态 (A组)
    SILICA_GEL_NORMAL = "silica_gel_normal"     # 硅胶正常(蓝色)
    SILICA_GEL_WARNING = "silica_gel_warning"   # 硅胶警告(淡蓝)
    SILICA_GEL_ALARM = "silica_gel_alarm"       # 硅胶告警(粉红/白)

    # 油位状态 (A组)
    OIL_LEVEL_NORMAL = "oil_level_normal"       # 油位正常
    OIL_LEVEL_LOW = "oil_level_low"             # 油位偏低
    OIL_LEVEL_HIGH = "oil_level_high"           # 油位偏高

    # 气体检测 (F组)
    GAS_NORMAL = "gas_normal"                   # 气体正常
    GAS_LEAK = "gas_leak"                       # 气体泄漏
    GAS_CONCENTRATION_HIGH = "gas_concentration_high"  # 浓度过高

    # 安全监控 (G组)
    SAFE = "safe"                               # 安全
    LINE_CROSS = "line_cross"                   # 越线
    ON_LINE = "on_line"                         # 压线
    UNAUTHORIZED = "unauthorized"               # 未授权
    MULTI_PERSON = "multi_person"               # 多人
    HIGH_RISK = "high_risk"                     # 高风险

    # 声学监测 (H组)
    ACOUSTIC_NORMAL = "acoustic_normal"         # 声学正常
    PARTIAL_DISCHARGE = "partial_discharge"     # 局部放电
    MECHANICAL_FAULT = "mechanical_fault"       # 机械故障

    # 鸟害监测 (I组)
    NO_BIRD = "no_bird"                         # 无鸟
    BIRD_DETECTED = "bird_detected"             # 检测到鸟
    BIRD_NESTING = "bird_nesting"               # 筑巢

    # 图像质量
    CLARITY_HIGH = "clarity_high"               # 清晰度高
    CLARITY_LOW = "clarity_low"                 # 清晰度低
    RESHOOT_REQUIRED = "reshoot_required"       # 需要复拍


# 标签中文名称映射
LABEL_NAMES: Dict[str, str] = {
    # 通用
    "normal": "正常",
    "abnormal": "异常",
    "warning": "警告",
    "error": "错误",
    "unknown": "未知",

    # 缺陷
    "oil_leak": "渗漏油",
    "rust": "锈蚀",
    "damage": "破损",
    "crack": "裂纹",
    "foreign_object": "异物悬挂",
    "deformation": "变形",
    "discoloration": "变色",
    "corrosion": "腐蚀",
    "overheating": "过热",

    # 开关
    "breaker_open": "断路器分闸",
    "breaker_closed": "断路器合闸",
    "breaker_intermediate": "断路器中间态",
    "isolator_open": "隔离开关分闸",
    "isolator_closed": "隔离开关合闸",
    "grounding_open": "接地开关分闸",
    "grounding_closed": "接地开关合闸",

    # 表计
    "meter_reading": "表计读数",
    "meter_overrange": "超量程",
    "meter_underrange": "欠量程",

    # 硅胶
    "silica_gel_normal": "硅胶正常",
    "silica_gel_warning": "硅胶变色警告",
    "silica_gel_alarm": "硅胶变色告警",

    # 油位
    "oil_level_normal": "油位正常",
    "oil_level_low": "油位偏低",
    "oil_level_high": "油位偏高",

    # 气体
    "gas_normal": "气体正常",
    "gas_leak": "气体泄漏",
    "gas_concentration_high": "浓度过高",

    # 安全
    "safe": "安全",
    "line_cross": "越线",
    "on_line": "压线",
    "unauthorized": "未授权操作",
    "multi_person": "同柜多人",
    "high_risk": "高风险",

    # 声学
    "acoustic_normal": "声学正常",
    "partial_discharge": "局部放电",
    "mechanical_fault": "机械故障",

    # 鸟害
    "no_bird": "无鸟",
    "bird_detected": "检测到鸟类",
    "bird_nesting": "鸟类筑巢",

    # 图像质量
    "clarity_high": "清晰度高",
    "clarity_low": "清晰度低",
    "reshoot_required": "需要复拍",
}


# =============================================================================
# UI显示配置
# =============================================================================

class LabelSeverity(str, Enum):
    """标签严重程度 (用于UI颜色)"""
    SUCCESS = "success"     # 绿色
    INFO = "info"           # 蓝色
    WARNING = "warning"     # 黄色
    DANGER = "danger"       # 红色
    NEUTRAL = "neutral"     # 灰色


# 标签严重程度映射
LABEL_SEVERITY: Dict[str, LabelSeverity] = {
    # 正常状态 - 绿色
    "normal": LabelSeverity.SUCCESS,
    "safe": LabelSeverity.SUCCESS,
    "silica_gel_normal": LabelSeverity.SUCCESS,
    "oil_level_normal": LabelSeverity.SUCCESS,
    "gas_normal": LabelSeverity.SUCCESS,
    "acoustic_normal": LabelSeverity.SUCCESS,
    "no_bird": LabelSeverity.SUCCESS,
    "clarity_high": LabelSeverity.SUCCESS,

    # 信息状态 - 蓝色
    "meter_reading": LabelSeverity.INFO,
    "breaker_open": LabelSeverity.INFO,
    "breaker_closed": LabelSeverity.INFO,
    "isolator_open": LabelSeverity.INFO,
    "isolator_closed": LabelSeverity.INFO,
    "grounding_open": LabelSeverity.INFO,
    "grounding_closed": LabelSeverity.INFO,

    # 警告状态 - 黄色
    "warning": LabelSeverity.WARNING,
    "on_line": LabelSeverity.WARNING,
    "silica_gel_warning": LabelSeverity.WARNING,
    "oil_level_low": LabelSeverity.WARNING,
    "oil_level_high": LabelSeverity.WARNING,
    "bird_detected": LabelSeverity.WARNING,
    "clarity_low": LabelSeverity.WARNING,
    "meter_overrange": LabelSeverity.WARNING,
    "meter_underrange": LabelSeverity.WARNING,
    "breaker_intermediate": LabelSeverity.WARNING,

    # 危险状态 - 红色
    "abnormal": LabelSeverity.DANGER,
    "error": LabelSeverity.DANGER,
    "oil_leak": LabelSeverity.DANGER,
    "rust": LabelSeverity.DANGER,
    "damage": LabelSeverity.DANGER,
    "crack": LabelSeverity.DANGER,
    "foreign_object": LabelSeverity.DANGER,
    "deformation": LabelSeverity.DANGER,
    "overheating": LabelSeverity.DANGER,
    "line_cross": LabelSeverity.DANGER,
    "unauthorized": LabelSeverity.DANGER,
    "high_risk": LabelSeverity.DANGER,
    "gas_leak": LabelSeverity.DANGER,
    "gas_concentration_high": LabelSeverity.DANGER,
    "silica_gel_alarm": LabelSeverity.DANGER,
    "partial_discharge": LabelSeverity.DANGER,
    "mechanical_fault": LabelSeverity.DANGER,
    "bird_nesting": LabelSeverity.DANGER,
    "reshoot_required": LabelSeverity.WARNING,

    # 中性状态 - 灰色
    "unknown": LabelSeverity.NEUTRAL,
}


# =============================================================================
# 标准化输出模型
# =============================================================================

class StandardBoundingBox(BaseModel):
    """标准边界框 (归一化坐标)"""
    x: float = Field(ge=0, le=1, description="左上角X坐标")
    y: float = Field(ge=0, le=1, description="左上角Y坐标")
    width: float = Field(ge=0, le=1, description="宽度")
    height: float = Field(ge=0, le=1, description="高度")


class StandardDetection(BaseModel):
    """标准检测结果"""
    label: str = Field(description="标准标签")
    label_name: str = Field(default="", description="标签中文名称")
    confidence: float = Field(ge=0, le=1, description="置信度")
    bbox: StandardBoundingBox = Field(description="边界框")
    value: Optional[Any] = Field(default=None, description="检测值")
    severity: str = Field(default="neutral", description="严重程度")


class StandardResult(BaseModel):
    """
    标准识别结果

    所有插件输出都应转换为此格式，确保UI组件可以统一处理
    """
    # 基础信息
    task_id: str = Field(description="任务ID")
    site_id: str = Field(description="站点ID")
    device_id: str = Field(description="设备ID")
    component_id: str = Field(default="", description="部件ID")
    roi_id: str = Field(default="", description="ROI ID")

    # 检测结果
    label: str = Field(description="标准标签")
    label_name: str = Field(default="", description="标签中文名称")
    confidence: float = Field(ge=0, le=1, description="置信度")
    value: Optional[Any] = Field(default=None, description="检测值(读数/温度等)")
    bbox: StandardBoundingBox = Field(description="边界框")

    # UI显示
    severity: str = Field(default="neutral", description="严重程度")
    display_text: str = Field(default="", description="显示文本")
    icon: str = Field(default="", description="图标名称")

    # 来源信息
    plugin_id: str = Field(default="", description="插件ID")
    model_version: str = Field(default="", description="模型版本")
    code_version: str = Field(default="", description="代码版本")

    # 时间戳
    timestamp: str = Field(default="", description="时间戳")

    # 扩展元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    # 失败信息
    failure_reason: Optional[str] = Field(default=None, description="失败原因码")
    error_message: Optional[str] = Field(default=None, description="错误消息")


class StandardPluginOutput(BaseModel):
    """
    标准插件输出

    所有插件的返回值应转换为此格式
    """
    # 基础信息
    plugin_id: str = Field(description="插件ID")
    plugin_name: str = Field(default="", description="插件名称")
    plugin_version: str = Field(default="", description="插件版本")
    code_hash: str = Field(default="", description="代码哈希")

    # 任务信息
    task_id: str = Field(description="任务ID")
    site_id: str = Field(default="", description="站点ID")
    device_id: str = Field(default="", description="设备ID")

    # 执行状态
    success: bool = Field(default=True, description="是否成功")
    error_code: Optional[str] = Field(default=None, description="错误码")
    error_message: Optional[str] = Field(default=None, description="错误消息")

    # 性能指标
    processing_time_ms: float = Field(default=0, description="处理时间(毫秒)")
    inference_count: int = Field(default=0, description="推理次数")

    # 结果列表
    results: List[StandardResult] = Field(default_factory=list, description="识别结果列表")
    detections: List[StandardDetection] = Field(default_factory=list, description="检测结果列表")

    # 告警列表
    alarm_count: int = Field(default=0, description="告警数量")

    # 时间戳
    timestamp: str = Field(default="", description="时间戳")

    # 扩展元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")


# =============================================================================
# 输出格式转换工具
# =============================================================================

class OutputFormatter:
    """
    输出格式转换工具

    将各种插件的原始输出转换为标准格式
    """

    @staticmethod
    def get_label_name(label: str) -> str:
        """获取标签中文名称"""
        return LABEL_NAMES.get(label, label)

    @staticmethod
    def get_label_severity(label: str) -> str:
        """获取标签严重程度"""
        severity = LABEL_SEVERITY.get(label, LabelSeverity.NEUTRAL)
        return severity.value

    @staticmethod
    def format_result(
        task_id: str,
        site_id: str,
        device_id: str,
        label: str,
        confidence: float,
        bbox: Dict[str, float],
        value: Any = None,
        component_id: str = "",
        roi_id: str = "",
        plugin_id: str = "",
        model_version: str = "",
        code_version: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
    ) -> StandardResult:
        """
        格式化单个识别结果

        Args:
            task_id: 任务ID
            site_id: 站点ID
            device_id: 设备ID
            label: 标准标签
            confidence: 置信度
            bbox: 边界框字典
            value: 检测值
            component_id: 部件ID
            roi_id: ROI ID
            plugin_id: 插件ID
            model_version: 模型版本
            code_version: 代码版本
            metadata: 扩展元数据
            failure_reason: 失败原因码

        Returns:
            标准化的识别结果
        """
        label_name = OutputFormatter.get_label_name(label)
        severity = OutputFormatter.get_label_severity(label)

        # 生成显示文本
        display_text = label_name
        if value is not None:
            if isinstance(value, (int, float)):
                display_text = f"{label_name}: {value}"
            else:
                display_text = f"{label_name}: {value}"

        return StandardResult(
            task_id=task_id,
            site_id=site_id,
            device_id=device_id,
            component_id=component_id,
            roi_id=roi_id,
            label=label,
            label_name=label_name,
            confidence=confidence,
            value=value,
            bbox=StandardBoundingBox(**bbox),
            severity=severity,
            display_text=display_text,
            plugin_id=plugin_id,
            model_version=model_version,
            code_version=code_version,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
            failure_reason=failure_reason,
        )

    @staticmethod
    def format_plugin_output(
        plugin_id: str,
        plugin_name: str,
        plugin_version: str,
        code_hash: str,
        task_id: str,
        results: List[StandardResult],
        success: bool = True,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        processing_time_ms: float = 0,
        site_id: str = "",
        device_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StandardPluginOutput:
        """
        格式化插件输出

        Args:
            plugin_id: 插件ID
            plugin_name: 插件名称
            plugin_version: 插件版本
            code_hash: 代码哈希
            task_id: 任务ID
            results: 识别结果列表
            success: 是否成功
            error_code: 错误码
            error_message: 错误消息
            processing_time_ms: 处理时间
            site_id: 站点ID
            device_id: 设备ID
            metadata: 扩展元数据

        Returns:
            标准化的插件输出
        """
        # 转换为检测结果列表
        detections = []
        for r in results:
            detections.append(StandardDetection(
                label=r.label,
                label_name=r.label_name,
                confidence=r.confidence,
                bbox=r.bbox,
                value=r.value,
                severity=r.severity,
            ))

        # 计算告警数量 (danger级别)
        alarm_count = sum(1 for r in results if r.severity == "danger")

        return StandardPluginOutput(
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            code_hash=code_hash,
            task_id=task_id,
            site_id=site_id,
            device_id=device_id,
            success=success,
            error_code=error_code,
            error_message=error_message,
            processing_time_ms=processing_time_ms,
            inference_count=len(results),
            results=results,
            detections=detections,
            alarm_count=alarm_count,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )

    @staticmethod
    def convert_legacy_result(legacy_result: Dict[str, Any]) -> StandardResult:
        """
        转换旧版结果格式为标准格式

        Args:
            legacy_result: 旧版结果字典

        Returns:
            标准化的识别结果
        """
        # 提取基础字段
        task_id = legacy_result.get("task_id", "")
        site_id = legacy_result.get("site_id", "")
        device_id = legacy_result.get("device_id", "")
        component_id = legacy_result.get("component_id", "")
        roi_id = legacy_result.get("roi_id", "")

        # 提取标签和置信度
        label = legacy_result.get("label", "unknown")
        confidence = legacy_result.get("confidence", 0.0)
        value = legacy_result.get("value")

        # 提取边界框
        bbox = legacy_result.get("bbox", {})
        if isinstance(bbox, dict):
            bbox = {
                "x": bbox.get("x", 0),
                "y": bbox.get("y", 0),
                "width": bbox.get("width", 1),
                "height": bbox.get("height", 1),
            }
        else:
            bbox = {"x": 0, "y": 0, "width": 1, "height": 1}

        # 提取元数据
        metadata = legacy_result.get("metadata", {})
        model_version = legacy_result.get("model_version", "")
        code_version = legacy_result.get("code_version", "")
        failure_reason = legacy_result.get("failure_reason")

        return OutputFormatter.format_result(
            task_id=task_id,
            site_id=site_id,
            device_id=device_id,
            label=label,
            confidence=confidence,
            bbox=bbox,
            value=value,
            component_id=component_id,
            roi_id=roi_id,
            model_version=model_version,
            code_version=code_version,
            metadata=metadata,
            failure_reason=failure_reason,
        )


# =============================================================================
# 便捷函数
# =============================================================================

def format_result(**kwargs) -> StandardResult:
    """格式化识别结果的便捷函数"""
    return OutputFormatter.format_result(**kwargs)


def format_plugin_output(**kwargs) -> StandardPluginOutput:
    """格式化插件输出的便捷函数"""
    return OutputFormatter.format_plugin_output(**kwargs)


def get_label_name(label: str) -> str:
    """获取标签中文名称"""
    return OutputFormatter.get_label_name(label)


def get_label_severity(label: str) -> str:
    """获取标签严重程度"""
    return OutputFormatter.get_label_severity(label)
