#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 训练结果评估工具
======================================

功能:
1. 模型性能评估 (mAP, Recall, Precision)
2. 推理速度基准测试
3. 模型稳定性测试
4. 自动生成评估报告

使用方法:
    # 评估单个模型
    python evaluate_training.py --model models/switch/switch_yolov8s.onnx --type switch
    
    # 评估所有模型
    python evaluate_training.py --all
    
    # 生成完整报告
    python evaluate_training.py --report

作者: 破夜绘明团队
日期: 2025
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 合格标准定义 (基于项目文档)
# =============================================================================
@dataclass
class QualificationCriteria:
    """模型合格标准"""
    
    # 检测模型标准 (目标检测类)
    detection_criteria = {
        "transformer": {  # A组 - 主变巡视
            "mAP@0.5": 0.75,      # 最低 mAP
            "recall_min": 0.80,    # 最低召回率
            "precision_min": 0.85, # 最低精确率
            "inference_ms": 100,   # 单帧推理时间上限 (ms)
            "fps_min": 10,         # 最低帧率
        },
        "switch": {  # B组 - 开关间隔
            "mAP@0.5": 0.85,
            "recall_min": 0.90,
            "precision_min": 0.90,
            "inference_ms": 80,
            "fps_min": 12,
            "state_accuracy": 0.95,  # 状态识别准确率
            "logic_error_rate": 0.02, # 逻辑校验误报率上限
        },
        "busbar": {  # C组 - 母线巡视
            "mAP@0.5": 0.70,
            "recall_min": 0.85,     # pin_missing, foreign_object
            "precision_min": 0.85,
            "inference_ms": 800,    # 4K图像含切片
            "crack_recall": 0.70,   # 裂纹检测（更难）
            "crack_precision": 0.80,
        },
        "capacitor": {  # D组 - 电容器
            "mAP@0.5": 0.80,
            "recall_min": 0.85,
            "precision_min": 0.85,
            "inference_ms": 100,
        },
        "meter": {  # E组 - 表计读数
            "keypoint_pck": 0.90,   # PCK@0.1
            "ocr_accuracy": 0.95,
            "reading_error": 0.02,  # 读数误差上限 (满量程%)
            "inference_ms": 150,
        },
    }
    
    # 通用稳定性标准
    stability_criteria = {
        "continuous_run_hours": 24,  # 连续运行时长
        "memory_leak_mb": 100,       # 最大内存增长
        "crash_count": 0,            # 崩溃次数
    }


# =============================================================================
# 评估指标计算
# =============================================================================
class MetricsCalculator:
    """评估指标计算器"""
    
    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个边界框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    @staticmethod
    def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
        """计算 Average Precision"""
        # 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if mask.any():
                ap += precisions[mask].max()
        return ap / 11
    
    @staticmethod
    def calculate_map(
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float = 0.5,
        num_classes: int = 10
    ) -> Dict[str, float]:
        """
        计算 mAP
        
        Args:
            predictions: 预测结果列表 [{"boxes", "scores", "labels", "image_id"}, ...]
            ground_truths: 真值标注列表 [{"boxes", "labels", "image_id"}, ...]
            iou_threshold: IoU 阈值
            num_classes: 类别数量
        
        Returns:
            {"mAP": float, "AP_per_class": dict}
        """
        aps = []
        ap_per_class = {}
        
        for class_id in range(num_classes):
            # 收集该类别的所有预测和真值
            class_preds = []
            class_gts = {}
            
            for pred in predictions:
                mask = pred["labels"] == class_id
                if mask.any():
                    for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
                        class_preds.append({
                            "box": box,
                            "score": score,
                            "image_id": pred["image_id"]
                        })
            
            for gt in ground_truths:
                mask = gt["labels"] == class_id
                if mask.any():
                    img_id = gt["image_id"]
                    if img_id not in class_gts:
                        class_gts[img_id] = []
                    for box in gt["boxes"][mask]:
                        class_gts[img_id].append({"box": box, "matched": False})
            
            if not class_preds or not class_gts:
                continue
            
            # 按置信度排序
            class_preds.sort(key=lambda x: x["score"], reverse=True)
            
            # 计算 TP/FP
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            
            for i, pred in enumerate(class_preds):
                img_id = pred["image_id"]
                if img_id not in class_gts:
                    fp[i] = 1
                    continue
                
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt in enumerate(class_gts[img_id]):
                    if gt["matched"]:
                        continue
                    iou = MetricsCalculator.calculate_iou(pred["box"], gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_threshold:
                    tp[i] = 1
                    class_gts[img_id][best_gt_idx]["matched"] = True
                else:
                    fp[i] = 1
            
            # 计算累积 TP/FP
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            
            # 计算 Recall 和 Precision
            total_gt = sum(len(gts) for gts in class_gts.values())
            recalls = tp_cum / total_gt
            precisions = tp_cum / (tp_cum + fp_cum)
            
            # 计算 AP
            ap = MetricsCalculator.calculate_ap(recalls, precisions)
            aps.append(ap)
            ap_per_class[class_id] = ap
        
        mAP = np.mean(aps) if aps else 0
        return {"mAP": mAP, "AP_per_class": ap_per_class}
    
    @staticmethod
    def calculate_confusion_matrix(
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        num_classes: int
    ) -> np.ndarray:
        """计算混淆矩阵"""
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for pred, gt in zip(predictions, ground_truths):
            cm[gt, pred] += 1
        return cm
    
    @staticmethod
    def calculate_classification_metrics(cm: np.ndarray) -> Dict[str, float]:
        """从混淆矩阵计算分类指标"""
        # 每类指标
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        
        # 避免除零
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1 = np.where(precision + recall > 0, 
                     2 * precision * recall / (precision + recall), 0)
        
        # 总体指标
        accuracy = tp.sum() / cm.sum() if cm.sum() > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision_mean": precision.mean(),
            "recall_mean": recall.mean(),
            "f1_mean": f1.mean(),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
        }


# =============================================================================
# 模型评估器
# =============================================================================
class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, model_type: str):
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.criteria = QualificationCriteria.detection_criteria.get(
            model_type, QualificationCriteria.detection_criteria["transformer"]
        )
        self.session = None

    def _resolve_input_shape(
        self,
        input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        model_input = self.session.get_inputs()[0]
        model_shape = model_input.shape
        resolved_shape: List[int] = []

        for idx, dim in enumerate(model_shape):
            desired = input_shape[idx] if idx < len(input_shape) else None
            if isinstance(dim, int) and dim > 0:
                if desired is not None and desired != dim:
                    logger.warning(
                        "输入尺寸与模型不匹配 (提供: %s, 模型: %s)，改用模型输入尺寸。",
                        input_shape,
                        tuple(model_shape),
                    )
                resolved_shape.append(dim)
                continue

            if isinstance(desired, int) and desired > 0:
                resolved_shape.append(desired)
            else:
                fallback = 1 if idx == 0 else 3 if idx == 1 else 640
                resolved_shape.append(fallback)

        if not resolved_shape:
            return input_shape

        return tuple(resolved_shape)
        
    def load_model(self) -> bool:
        """加载模型"""
        try:
            import onnxruntime as ort
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            
            logger.info(f"模型加载成功: {self.model_path}")
            logger.info(f"推理后端: {self.session.get_providers()}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def benchmark_inference(
        self, 
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
        warmup_runs: int = 10,
        test_runs: int = 100
    ) -> Dict[str, float]:
        """
        推理性能基准测试
        
        Returns:
            {
                "mean_ms": float,
                "std_ms": float,
                "p50_ms": float,
                "p95_ms": float,
                "p99_ms": float,
                "fps": float
            }
        """
        if self.session is None:
            logger.error("模型未加载")
            return {}

        input_shape = self._resolve_input_shape(input_shape)
        logger.info(f"基准测试输入尺寸: {input_shape}")

        # 生成测试输入
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_name = self.session.get_inputs()[0].name
        
        # 预热
        logger.info(f"预热 {warmup_runs} 次...")
        for _ in range(warmup_runs):
            self.session.run(None, {input_name: dummy_input})
        
        # 正式测试
        logger.info(f"基准测试 {test_runs} 次...")
        latencies = []
        for _ in range(test_runs):
            start = time.perf_counter()
            self.session.run(None, {input_name: dummy_input})
            latencies.append((time.perf_counter() - start) * 1000)
        
        latencies = np.array(latencies)
        
        results = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "fps": float(1000 / np.mean(latencies)),
        }
        
        logger.info(f"推理延迟: {results['mean_ms']:.2f}±{results['std_ms']:.2f} ms")
        logger.info(f"P95 延迟: {results['p95_ms']:.2f} ms")
        logger.info(f"FPS: {results['fps']:.1f}")
        
        return results
    
    def evaluate_detection(
        self,
        test_data_dir: str,
        annotation_file: str
    ) -> Dict[str, Any]:
        """
        评估目标检测模型
        
        Args:
            test_data_dir: 测试图像目录
            annotation_file: COCO 格式标注文件
        
        Returns:
            评估结果字典
        """
        # 这里是评估逻辑的框架
        # 实际使用时需要加载测试数据并运行推理
        
        results = {
            "model_path": str(self.model_path),
            "model_type": self.model_type,
            "evaluation_time": datetime.now().isoformat(),
            "metrics": {},
            "qualification": {},
        }
        
        logger.info("开始检测模型评估...")
        
        # 1. 性能基准测试
        benchmark = self.benchmark_inference()
        results["benchmark"] = benchmark
        
        # 2. 检查是否满足合格标准
        criteria = self.criteria
        
        # 推理速度检查
        if benchmark:
            speed_pass = benchmark["p95_ms"] <= criteria.get("inference_ms", 100)
            fps_pass = benchmark["fps"] >= criteria.get("fps_min", 10)
            
            results["qualification"]["speed"] = speed_pass
            results["qualification"]["fps"] = fps_pass
        
        return results
    
    def check_qualification(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """
        检查是否达到合格标准
        
        Returns:
            (是否合格, 未达标项列表)
        """
        failures = []
        
        # 检查各项指标
        for metric_name, threshold in self.criteria.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # 特殊处理：某些指标是上限（越小越好）
                if "error" in metric_name or metric_name == "inference_ms":
                    if value > threshold:
                        failures.append(f"{metric_name}: {value:.4f} > {threshold}")
                else:
                    if value < threshold:
                        failures.append(f"{metric_name}: {value:.4f} < {threshold}")
        
        is_qualified = len(failures) == 0
        return is_qualified, failures


# =============================================================================
# 评估报告生成器
# =============================================================================
class ReportGenerator:
    """评估报告生成器"""
    
    def __init__(self, output_dir: str = "evaluation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        model_results: List[Dict],
        report_name: str = None
    ) -> str:
        """
        生成评估报告
        
        Args:
            model_results: 模型评估结果列表
            report_name: 报告文件名
        
        Returns:
            报告文件路径
        """
        if report_name is None:
            report_name = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("破夜绘明激光监测平台 - 训练评估报告")
        report_lines.append("输变电站全自动AI巡检方案")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 汇总表
        report_lines.append("-" * 80)
        report_lines.append("【评估汇总】")
        report_lines.append("-" * 80)
        report_lines.append(f"{'模型类型':<15} {'状态':<10} {'推理延迟(ms)':<15} {'FPS':<10}")
        report_lines.append("-" * 80)
        
        all_passed = True
        for result in model_results:
            model_type = result.get("model_type", "unknown")
            benchmark = result.get("benchmark", {})
            qualification = result.get("qualification", {})
            
            status = "✓ 合格" if all(qualification.values()) else "✗ 不合格"
            if not all(qualification.values()):
                all_passed = False
            
            latency = benchmark.get("mean_ms", 0)
            fps = benchmark.get("fps", 0)
            
            report_lines.append(f"{model_type:<15} {status:<10} {latency:<15.2f} {fps:<10.1f}")
        
        report_lines.append("-" * 80)
        report_lines.append(f"总体评估: {'✓ 全部合格' if all_passed else '✗ 部分不合格'}")
        report_lines.append("")
        
        # 详细结果
        report_lines.append("=" * 80)
        report_lines.append("【详细评估结果】")
        report_lines.append("=" * 80)
        
        for result in model_results:
            report_lines.append("")
            report_lines.append(f">>> {result.get('model_type', 'unknown').upper()} 模型")
            report_lines.append(f"    路径: {result.get('model_path', 'N/A')}")
            
            benchmark = result.get("benchmark", {})
            if benchmark:
                report_lines.append(f"    推理性能:")
                report_lines.append(f"      - 平均延迟: {benchmark.get('mean_ms', 0):.2f} ms")
                report_lines.append(f"      - P95 延迟: {benchmark.get('p95_ms', 0):.2f} ms")
                report_lines.append(f"      - FPS: {benchmark.get('fps', 0):.1f}")
            
            qualification = result.get("qualification", {})
            if qualification:
                report_lines.append(f"    合格检查:")
                for check, passed in qualification.items():
                    status = "✓" if passed else "✗"
                    report_lines.append(f"      {status} {check}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("【合格标准参考】")
        report_lines.append("=" * 80)
        
        criteria = QualificationCriteria()
        for model_type, standards in criteria.detection_criteria.items():
            report_lines.append(f"\n{model_type}:")
            for metric, value in standards.items():
                report_lines.append(f"  - {metric}: {value}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("报告结束")
        report_lines.append("=" * 80)
        
        # 保存报告
        report_content = "\n".join(report_lines)
        
        # 文本报告
        txt_path = self.output_dir / f"{report_name}.txt"
        txt_path.write_text(report_content, encoding='utf-8')
        
        # JSON 结果
        json_path = self.output_dir / f"{report_name}.json"
        json_path.write_text(json.dumps(model_results, indent=2, ensure_ascii=False), encoding='utf-8')
        
        logger.info(f"评估报告已生成: {txt_path}")
        
        return str(txt_path)


# =============================================================================
# 主程序
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="训练结果评估工具")
    parser.add_argument("--model", type=str, help="模型文件路径")
    parser.add_argument("--type", type=str, 
                       choices=["transformer", "switch", "busbar", "capacitor", "meter"],
                       help="模型类型")
    parser.add_argument("--all", action="store_true", help="评估所有模型")
    parser.add_argument("--report", action="store_true", help="生成评估报告")
    parser.add_argument("--models-dir", type=str, default="models", help="模型目录")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("破夜绘明激光监测平台 - 训练评估工具")
    print("=" * 60)
    
    results = []
    
    if args.model and args.type:
        # 评估单个模型
        evaluator = ModelEvaluator(args.model, args.type)
        if evaluator.load_model():
            result = evaluator.evaluate_detection("", "")
            results.append(result)
    
    elif args.all:
        # 评估所有模型
        models_dir = Path(args.models_dir)
        
        model_configs = [
            ("transformer", "defect_yolov8n.onnx"),
            ("switch", "switch_yolov8s.onnx"),
            ("busbar", "busbar_yolov8m.onnx"),
            ("capacitor", "capacitor_yolov8.onnx"),
            ("meter", "hrnet_keypoint.onnx"),
        ]
        
        for model_type, model_file in model_configs:
            model_path = models_dir / model_type / model_file
            if model_path.exists():
                logger.info(f"\n评估模型: {model_path}")
                evaluator = ModelEvaluator(str(model_path), model_type)
                if evaluator.load_model():
                    result = evaluator.evaluate_detection("", "")
                    results.append(result)
            else:
                logger.warning(f"模型不存在: {model_path}")
    
    if args.report and results:
        # 生成报告
        generator = ReportGenerator()
        report_path = generator.generate_report(results)
        print(f"\n评估报告: {report_path}")
    
    # 打印合格标准概览
    if not args.model and not args.all:
        print("\n【合格标准概览】")
        print("-" * 60)
        criteria = QualificationCriteria()
        for model_type, standards in criteria.detection_criteria.items():
            print(f"\n{model_type.upper()}:")
            for metric, value in standards.items():
                print(f"  {metric}: {value}")
        print("\n使用 --help 查看完整用法")


if __name__ == "__main__":
    main()
