"""
模型评估器

对训练完成的模型进行统一评估，生成评估报告。
支持按 plugin_id + task_type 选择对应的评估指标。

使用方式:
    evaluator = BaseEvaluator("busbar_inspection", "detection")
    result = evaluator.evaluate(model_path, test_data_dir)

两种数据来源:
1. 训练器直接传入的训练指标 (metrics_from_training)
   → 适用于 ultralytics 等已自带评估的框架
2. 评估器自行加载模型跑推理
   → 适用于 PyTorch 原生训练、需要独立验证的场景
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """评估结果"""

    plugin_id: str = ""
    task_type: str = ""
    model_path: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    inference_speed_ms: float = 0.0
    qualified: bool = False
    qualification_reason: str = ""
    eval_time_seconds: float = 0.0

    def save(self, path: Path) -> None:
        """保存评估报告"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2, default=str)


# 各插件的最低达标要求
QUALIFICATION_CRITERIA: dict[str, dict[str, float]] = {
    "busbar_inspection": {"mAP@0.5": 0.65, "recall": 0.70},
    "capacitor_inspection": {"mAP@0.5": 0.60, "recall": 0.65},
    "switch_inspection": {"mAP@0.5": 0.60, "state_accuracy": 0.90},
    "transformer_inspection": {"mAP@0.5": 0.65, "recall": 0.70},
    "animal_detection": {"mAP@0.5": 0.55, "recall": 0.75},
    "bird_monitoring": {"mAP@0.5": 0.50, "recall": 0.70},
    "fire_detection": {"mAP@0.5": 0.60, "recall": 0.85, "false_alarm_rate": 0.05},
    "meter_reading": {"reading_accuracy": 0.90, "digit_accuracy": 0.95},
    "temperature_monitoring": {"anomaly_accuracy": 0.85, "false_alarm_rate": 0.10},
    "thermal": {"anomaly_accuracy": 0.80},
    "hyperspectral_detection": {"overall_accuracy": 0.85, "kappa": 0.80},
    "acoustic_monitoring": {"f1": 0.80, "false_alarm_rate": 0.10},
    "gas_detection": {"anomaly_auc": 0.80, "lead_time_hours": 4.0},
    "device_monitoring": {"anomaly_auc": 0.80, "failure_recall": 0.75},
    "action_event_monitoring": {"sequence_accuracy": 0.85, "event_recall": 0.85},
    "multimodal_fusion": {"overall_accuracy": 0.85, "modality_dropout_robustness": 0.75},
}

# task_type → 指标计算方法的映射
_TASK_METRICS_MAP = {
    "detection": "detection",
    "segmentation": "detection",
    "classification": "classification",
    "ocr": "ocr",
    "thermal_anomaly": "thermal",
    "hyperspectral_classification": "hyperspectral",
    "hyperspectral_anomaly": "hyperspectral",
    "acoustic_time_frequency_anomaly": "temporal_anomaly",
    "multivariate_sensor_anomaly": "temporal_anomaly",
    "equipment_health_prediction": "regression",
    "health_index_calibration": "regression",
    "action_event_sequence_recognition": "sequence",
    "multimodal_feature_fusion": "multimodal",
    "multimodal_decision_fusion": "multimodal",
}


class BaseEvaluator:
    """统一模型评估器"""

    def __init__(self, plugin_id: str, task_type: str):
        self.plugin_id = plugin_id
        self.task_type = task_type
        self.metrics_calc = MetricsCalculator()

    def evaluate(
        self,
        model_path: Path,
        test_data_dir: Path,
        metrics_from_training: dict[str, float] | None = None,
    ) -> EvalResult:
        """
        评估模型。

        优先使用 metrics_from_training (训练器已计算好的指标)；
        若为空则尝试自行加载模型进行推理评估。

        Args:
            model_path: 训练产出的模型文件
            test_data_dir: 测试集目录
            metrics_from_training: 训练过程中已产出的指标 (可选)

        Returns:
            EvalResult
        """
        start_time = time.time()
        logger.info(f"开始评估: plugin={self.plugin_id}, task={self.task_type}")

        metrics: dict[str, float] = {}
        per_class: dict[str, dict[str, float]] = {}
        inference_speed = 0.0

        # 策略 1: 使用训练器已产出的指标
        if metrics_from_training:
            metrics = dict(metrics_from_training)
            logger.info(f"使用训练器产出指标: {list(metrics.keys())}")

        # 策略 2: 尝试 ultralytics val (仅 detection)
        if not metrics and self.task_type in ("detection", "segmentation"):
            yolo_metrics = self._eval_ultralytics(model_path, test_data_dir)
            if yolo_metrics:
                metrics = yolo_metrics

        # 策略 3: 扫描测试集做轻量推理 (适用于分类/OCR 等)
        if not metrics:
            inferred = self._eval_from_test_data(model_path, test_data_dir)
            if inferred:
                metrics = inferred

        # 策略 4: 从训练日志中恢复
        if not metrics:
            log_metrics = self._recover_from_logs(model_path)
            if log_metrics:
                metrics = log_metrics

        # 达标判定
        qualified, reasons = self._check_qualification(metrics)

        result = EvalResult(
            plugin_id=self.plugin_id,
            task_type=self.task_type,
            model_path=str(model_path),
            metrics=metrics,
            per_class_metrics=per_class,
            inference_speed_ms=inference_speed,
            qualified=qualified,
            qualification_reason="; ".join(reasons) if reasons else "达标",
            eval_time_seconds=time.time() - start_time,
        )

        # 保存评估报告
        parent = model_path if model_path.is_dir() else model_path.parent
        report_path = parent / "eval_report.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(report_path)
            logger.info(f"评估报告已保存: {report_path}")
        except Exception as e:
            logger.warning(f"保存评估报告失败: {e}")

        return result

    # ================================================================
    # 评估策略
    # ================================================================

    def _eval_ultralytics(
        self, model_path: Path, test_data_dir: Path
    ) -> dict[str, float] | None:
        """使用 ultralytics YOLO val 进行评估"""
        try:
            from ultralytics import YOLO

            model_file = model_path
            if model_file.is_dir():
                for candidate in (
                    model_file / "weights" / "best.pt",
                    model_file / "best.pt",
                ):
                    if candidate.exists():
                        model_file = candidate
                        break

            if not model_file.exists() or model_file.is_dir():
                return None

            # 查找 data.yaml
            yaml_path = None
            for p in (test_data_dir, test_data_dir.parent):
                for pattern in ("*.yaml", "*.yml"):
                    yamls = list(p.glob(pattern))
                    if yamls:
                        yaml_path = yamls[0]
                        break
                if yaml_path:
                    break

            if not yaml_path:
                return None

            model = YOLO(str(model_file))
            results = model.val(data=str(yaml_path), verbose=False)

            if hasattr(results, "results_dict"):
                return {k: float(v) for k, v in results.results_dict.items()}
            return None

        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"ultralytics 评估失败: {e}")
            return None

    def _eval_from_test_data(
        self, model_path: Path, test_data_dir: Path
    ) -> dict[str, float] | None:
        """从测试集预生成的 predictions.json 获取指标"""
        metrics_type = _TASK_METRICS_MAP.get(self.task_type)
        if not metrics_type:
            return None

        pred_file = test_data_dir / "predictions.json"
        if pred_file.exists():
            try:
                data = json.loads(pred_file.read_text(encoding="utf-8"))
                preds = data.get("predictions", [])
                gts = data.get("ground_truths", [])
                return self._compute_by_type(metrics_type, preds, gts)
            except Exception as e:
                logger.warning(f"解析 predictions.json 失败: {e}")

        return None

    def _recover_from_logs(self, model_path: Path) -> dict[str, float] | None:
        """从训练日志或结果文件恢复指标"""
        parent = model_path if model_path.is_dir() else model_path.parent

        for candidate in (parent / "results.json", parent / "results.csv"):
            if candidate.exists():
                try:
                    if candidate.suffix == ".json":
                        data = json.loads(candidate.read_text(encoding="utf-8"))
                        if isinstance(data, dict):
                            return {
                                k: float(v)
                                for k, v in data.items()
                                if isinstance(v, (int, float))
                            }
                except Exception:
                    pass
        return None

    def _compute_by_type(
        self, metrics_type: str, preds: list, gts: list
    ) -> dict[str, float]:
        """根据 metrics_type 选择对应的计算器"""
        dispatch = {
            "detection": self.metrics_calc.compute_detection_metrics,
            "classification": self.metrics_calc.compute_classification_metrics,
            "ocr": self.metrics_calc.compute_ocr_metrics,
            "thermal": self.metrics_calc.compute_thermal_metrics,
            "hyperspectral": self.metrics_calc.compute_hyperspectral_metrics,
            "temporal_anomaly": self.metrics_calc.compute_temporal_anomaly_metrics,
            "regression": lambda p, g: self.metrics_calc.compute_regression_metrics(
                p, g, prefix="health"
            ),
            "sequence": self.metrics_calc.compute_sequence_metrics,
            "multimodal": self.metrics_calc.compute_multimodal_fusion_metrics,
        }
        fn = dispatch.get(metrics_type)
        if fn:
            return fn(preds, gts)
        return {}

    def _check_qualification(
        self, metrics: dict[str, float]
    ) -> tuple[bool, list[str]]:
        """达标判定"""
        criteria = QUALIFICATION_CRITERIA.get(self.plugin_id, {})
        if not criteria:
            return True, []

        qualified = True
        reasons: list[str] = []
        for metric_name, threshold in criteria.items():
            actual = metrics.get(metric_name)
            if actual is None:
                continue

            # false_alarm_rate / ece 越低越好
            if "false_alarm" in metric_name or "ece" in metric_name:
                if actual > threshold:
                    qualified = False
                    reasons.append(f"{metric_name}={actual:.3f} > {threshold}")
            else:
                if actual < threshold:
                    qualified = False
                    reasons.append(f"{metric_name}={actual:.3f} < {threshold}")

        return qualified, reasons
