"""
数据验证器
===========

提供传感器数据验证功能:
- 范围检查
- 变化率检查
- 数据连续性检查
- 异常值检测

作者: 破夜绘明团队
版本: 1.0.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """验证级别"""
    PASS = "pass"           # 通过
    INFO = "info"           # 信息
    WARNING = "warning"     # 警告
    ERROR = "error"         # 错误
    CRITICAL = "critical"   # 严重


@dataclass
class ValidationRule:
    """验证规则"""
    rule_id: str
    name: str
    description: str = ""
    enabled: bool = True
    level: ValidationLevel = ValidationLevel.WARNING

    # 规则参数
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_change_rate: Optional[float] = None  # 最大变化率 (单位/秒)
    max_null_count: int = 3                   # 最大连续空值数
    outlier_sigma: float = 3.0               # 异常值标准差倍数

    # 自定义验证函数
    custom_validator: Optional[Callable[[Any, Dict], bool]] = None


@dataclass
class ValidationResult:
    """验证结果"""
    rule_id: str
    rule_name: str
    passed: bool
    level: ValidationLevel
    message: str
    value: Any = None
    expected_range: Optional[Tuple[float, float]] = None
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "passed": self.passed,
            "level": self.level.value,
            "message": self.message,
            "value": self.value,
            "expected_range": self.expected_range,
            "timestamp": self.timestamp,
        }


class DataValidator:
    """
    数据验证器

    用于验证传感器数据的有效性和质量
    """

    # 预定义规则集
    PRESETS = {
        "gas_sf6": [
            ValidationRule(
                rule_id="sf6_range",
                name="SF6浓度范围",
                min_value=0,
                max_value=3000,
                level=ValidationLevel.ERROR
            ),
            ValidationRule(
                rule_id="sf6_change",
                name="SF6变化率",
                max_change_rate=50,  # ppm/秒
                level=ValidationLevel.WARNING
            ),
        ],
        "temperature": [
            ValidationRule(
                rule_id="temp_range",
                name="温度范围",
                min_value=-40,
                max_value=150,
                level=ValidationLevel.ERROR
            ),
            ValidationRule(
                rule_id="temp_change",
                name="温度变化率",
                max_change_rate=5,  # 度/秒
                level=ValidationLevel.WARNING
            ),
        ],
        "thermal": [
            ValidationRule(
                rule_id="thermal_range",
                name="热成像温度范围",
                min_value=-20,
                max_value=200,
                level=ValidationLevel.ERROR
            ),
        ],
    }

    def __init__(self, rules: List[ValidationRule] = None):
        """
        初始化验证器

        Args:
            rules: 验证规则列表
        """
        self.rules: Dict[str, ValidationRule] = {}
        self._history: Dict[str, deque] = {}  # 历史数据用于变化率检查
        self._history_size = 100

        if rules:
            for rule in rules:
                self.add_rule(rule)

    def add_rule(self, rule: ValidationRule) -> None:
        """添加规则"""
        self.rules[rule.rule_id] = rule
        logger.debug(f"添加验证规则: {rule.rule_id}")

    def remove_rule(self, rule_id: str) -> None:
        """移除规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]

    def load_preset(self, preset_name: str) -> None:
        """加载预定义规则集"""
        if preset_name in self.PRESETS:
            for rule in self.PRESETS[preset_name]:
                self.add_rule(rule)
            logger.info(f"加载规则集: {preset_name}")
        else:
            logger.warning(f"未知规则集: {preset_name}")

    def validate(self, value: Any, sensor_id: str = None,
                 timestamp: float = None, context: Dict = None) -> List[ValidationResult]:
        """
        验证数据

        Args:
            value: 数据值
            sensor_id: 传感器ID (用于历史跟踪)
            timestamp: 时间戳
            context: 上下文信息

        Returns:
            验证结果列表
        """
        results = []
        context = context or {}
        timestamp = timestamp or datetime.now().timestamp()

        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue

            result = self._check_rule(rule, value, sensor_id, timestamp, context)
            results.append(result)

        # 更新历史
        if sensor_id:
            self._update_history(sensor_id, value, timestamp)

        return results

    def validate_batch(self, values: List[Tuple[Any, str, float]]) -> Dict[str, List[ValidationResult]]:
        """
        批量验证

        Args:
            values: [(value, sensor_id, timestamp), ...]

        Returns:
            {sensor_id: [ValidationResult, ...], ...}
        """
        results = {}

        for value, sensor_id, timestamp in values:
            validation_results = self.validate(value, sensor_id, timestamp)
            if sensor_id not in results:
                results[sensor_id] = []
            results[sensor_id].extend(validation_results)

        return results

    def _check_rule(self, rule: ValidationRule, value: Any, sensor_id: str,
                    timestamp: float, context: Dict) -> ValidationResult:
        """检查单个规则"""

        # 1. 自定义验证
        if rule.custom_validator:
            try:
                passed = rule.custom_validator(value, context)
                return ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    passed=passed,
                    level=ValidationLevel.PASS if passed else rule.level,
                    message="自定义验证" + ("通过" if passed else "失败"),
                    value=value,
                    timestamp=timestamp
                )
            except Exception as e:
                return ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"自定义验证异常: {e}",
                    value=value,
                    timestamp=timestamp
                )

        # 2. 空值检查
        if value is None:
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=False,
                level=ValidationLevel.WARNING,
                message="数据为空",
                value=value,
                timestamp=timestamp
            )

        # 3. 数值型检查
        if isinstance(value, (int, float)):
            # 范围检查
            if rule.min_value is not None and value < rule.min_value:
                return ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    passed=False,
                    level=rule.level,
                    message=f"值 {value} 低于最小值 {rule.min_value}",
                    value=value,
                    expected_range=(rule.min_value, rule.max_value),
                    timestamp=timestamp
                )

            if rule.max_value is not None and value > rule.max_value:
                return ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    passed=False,
                    level=rule.level,
                    message=f"值 {value} 超过最大值 {rule.max_value}",
                    value=value,
                    expected_range=(rule.min_value, rule.max_value),
                    timestamp=timestamp
                )

            # 变化率检查
            if rule.max_change_rate is not None and sensor_id:
                change_rate = self._calculate_change_rate(sensor_id, value, timestamp)
                if change_rate is not None and abs(change_rate) > rule.max_change_rate:
                    return ValidationResult(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        passed=False,
                        level=rule.level,
                        message=f"变化率 {change_rate:.2f}/s 超过限制 {rule.max_change_rate}/s",
                        value=value,
                        timestamp=timestamp
                    )

        # 4. 数组型检查 (如热成像)
        elif isinstance(value, np.ndarray):
            # 范围检查
            if rule.min_value is not None and np.min(value) < rule.min_value:
                return ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    passed=False,
                    level=rule.level,
                    message=f"最小值 {np.min(value):.1f} 低于限制 {rule.min_value}",
                    value=float(np.min(value)),
                    expected_range=(rule.min_value, rule.max_value),
                    timestamp=timestamp
                )

            if rule.max_value is not None and np.max(value) > rule.max_value:
                return ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    passed=False,
                    level=rule.level,
                    message=f"最大值 {np.max(value):.1f} 超过限制 {rule.max_value}",
                    value=float(np.max(value)),
                    expected_range=(rule.min_value, rule.max_value),
                    timestamp=timestamp
                )

            # 异常值检测
            if rule.outlier_sigma > 0:
                mean = np.mean(value)
                std = np.std(value)
                outlier_mask = np.abs(value - mean) > rule.outlier_sigma * std
                outlier_count = np.sum(outlier_mask)
                outlier_ratio = outlier_count / value.size

                if outlier_ratio > 0.01:  # 超过1%的点为异常
                    return ValidationResult(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        passed=False,
                        level=ValidationLevel.WARNING,
                        message=f"异常值比例 {outlier_ratio*100:.1f}% 过高",
                        value=outlier_ratio,
                        timestamp=timestamp
                    )

        # 通过验证
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            passed=True,
            level=ValidationLevel.PASS,
            message="验证通过",
            value=value if not isinstance(value, np.ndarray) else float(np.mean(value)),
            expected_range=(rule.min_value, rule.max_value) if rule.min_value is not None else None,
            timestamp=timestamp
        )

    def _update_history(self, sensor_id: str, value: Any, timestamp: float) -> None:
        """更新历史记录"""
        if sensor_id not in self._history:
            self._history[sensor_id] = deque(maxlen=self._history_size)

        if isinstance(value, (int, float)):
            self._history[sensor_id].append((timestamp, value))

    def _calculate_change_rate(self, sensor_id: str, value: float, timestamp: float) -> Optional[float]:
        """计算变化率"""
        if sensor_id not in self._history or len(self._history[sensor_id]) == 0:
            return None

        last_ts, last_value = self._history[sensor_id][-1]
        dt = timestamp - last_ts

        if dt <= 0:
            return None

        return (value - last_value) / dt

    def get_statistics(self, sensor_id: str) -> Dict[str, Any]:
        """获取传感器统计信息"""
        if sensor_id not in self._history:
            return {}

        history = list(self._history[sensor_id])
        if not history:
            return {}

        values = [v for _, v in history]

        return {
            "sensor_id": sensor_id,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "std": np.std(values) if len(values) > 1 else 0,
            "first_timestamp": history[0][0],
            "last_timestamp": history[-1][0],
        }


# =============================================================================
# 质量评估器
# =============================================================================
class DataQualityAssessor:
    """
    数据质量评估器

    计算数据质量分数
    """

    def __init__(self):
        self._metrics = {
            "completeness": 0.3,     # 完整性权重
            "accuracy": 0.3,         # 准确性权重
            "consistency": 0.2,      # 一致性权重
            "timeliness": 0.2,       # 时效性权重
        }

    def assess(self, validation_results: List[ValidationResult],
               expected_count: int = None,
               max_age_seconds: float = 60.0) -> Dict[str, float]:
        """
        评估数据质量

        Args:
            validation_results: 验证结果列表
            expected_count: 期望数据数量
            max_age_seconds: 最大数据年龄

        Returns:
            质量评估结果
        """
        if not validation_results:
            return {"overall_score": 0.0, "completeness": 0.0, "accuracy": 0.0,
                    "consistency": 0.0, "timeliness": 0.0}

        # 1. 完整性评估
        if expected_count:
            completeness = len(validation_results) / expected_count
        else:
            null_count = sum(1 for r in validation_results if r.value is None)
            completeness = 1 - (null_count / len(validation_results))

        # 2. 准确性评估 (基于验证通过率)
        passed_count = sum(1 for r in validation_results if r.passed)
        accuracy = passed_count / len(validation_results)

        # 3. 一致性评估 (基于错误级别分布)
        error_weights = {
            ValidationLevel.PASS: 1.0,
            ValidationLevel.INFO: 0.9,
            ValidationLevel.WARNING: 0.7,
            ValidationLevel.ERROR: 0.3,
            ValidationLevel.CRITICAL: 0.0,
        }
        consistency_scores = [error_weights.get(r.level, 0.5) for r in validation_results]
        consistency = sum(consistency_scores) / len(consistency_scores)

        # 4. 时效性评估
        current_time = datetime.now().timestamp()
        if any(r.timestamp for r in validation_results):
            ages = [current_time - r.timestamp for r in validation_results if r.timestamp]
            avg_age = sum(ages) / len(ages) if ages else 0
            timeliness = max(0, 1 - (avg_age / max_age_seconds))
        else:
            timeliness = 1.0

        # 综合评分
        overall_score = (
            self._metrics["completeness"] * completeness +
            self._metrics["accuracy"] * accuracy +
            self._metrics["consistency"] * consistency +
            self._metrics["timeliness"] * timeliness
        )

        return {
            "overall_score": overall_score,
            "completeness": completeness,
            "accuracy": accuracy,
            "consistency": consistency,
            "timeliness": timeliness,
            "total_count": len(validation_results),
            "passed_count": passed_count,
            "failed_count": len(validation_results) - passed_count,
        }
