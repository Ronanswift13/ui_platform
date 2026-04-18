"""
统一指标计算器

为不同 task_type 提供标准化的评估指标计算。
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

logger = logging.getLogger(__name__)


# ================================================================
# 检测指标辅助函数
# ================================================================

def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """
    计算两个 bbox 的 IoU。

    box 格式: [x_center, y_center, width, height] (归一化坐标)
    """
    # 转换为 [x1, y1, x2, y2]
    ax1, ay1 = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2
    ax2, ay2 = box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2
    bx1, by1 = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2
    bx2, by2 = box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _compute_ap(recalls: list[float], precisions: list[float]) -> float:
    """
    计算单类 AP (11 点插值法)。

    recalls / precisions 按置信度降序排列。
    """
    if not recalls:
        return 0.0

    # 11 点插值
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        p_interp = 0.0
        for r, p in zip(recalls, precisions):
            if r >= t:
                p_interp = max(p_interp, p)
        ap += p_interp
    return ap / 11.0


def _match_detections(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float,
) -> tuple[dict[str, list], dict[str, int]]:
    """
    按 IoU 阈值匹配预测与真值，按类分组。

    predictions 格式:  [{"class": int/str, "confidence": float, "bbox": [x,y,w,h]}, ...]
    ground_truths 格式: [{"class": int/str, "bbox": [x,y,w,h]}, ...]

    返回:
        per_class_matches: {class: [(confidence, is_tp), ...]}
        per_class_gt_count: {class: int}
    """
    per_class_matches: dict[str, list] = defaultdict(list)
    per_class_gt_count: dict[str, int] = Counter()

    # 按类别分组真值
    gt_by_class: dict[str, list[dict]] = defaultdict(list)
    for gt in ground_truths:
        cls = str(gt.get("class", 0))
        per_class_gt_count[cls] += 1
        gt_by_class[cls].append({**gt, "_matched": False})

    # 按置信度降序排列预测
    sorted_preds = sorted(predictions, key=lambda p: p.get("confidence", 0.0), reverse=True)

    for pred in sorted_preds:
        cls = str(pred.get("class", 0))
        pred_bbox = pred.get("bbox", [0, 0, 0, 0])
        conf = pred.get("confidence", 0.0)

        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt in enumerate(gt_by_class.get(cls, [])):
            if gt["_matched"]:
                continue
            iou = _compute_iou(pred_bbox, gt.get("bbox", [0, 0, 0, 0]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_by_class[cls][best_gt_idx]["_matched"] = True
            per_class_matches[cls].append((conf, True))
        else:
            per_class_matches[cls].append((conf, False))

    return per_class_matches, dict(per_class_gt_count)


def _compute_auc_roc(scores: list[float], labels: list[int]) -> float:
    """计算 AUC-ROC (梯形积分)"""
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp, fp = 0, 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.0

    points: list[tuple[float, float]] = [(0.0, 0.0)]
    prev_score = None
    for score, label in paired:
        if score != prev_score and prev_score is not None:
            fpr_val = fp / total_neg
            tpr_val = tp / total_pos
            points.append((fpr_val, tpr_val))
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    points.append((fp / total_neg, tp / total_pos))

    # 梯形求和
    auc = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        auc += (x1 - x0) * (y0 + y1) / 2
    return auc


def _compute_auc_pr(scores: list[float], labels: list[int]) -> float:
    """计算 AUC-PR (precision-recall 梯形积分)"""
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    total_pos = sum(labels)
    if total_pos == 0:
        return 0.0

    tp, fp = 0, 0
    points: list[tuple[float, float]] = []
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        rec = tp / total_pos
        prec = tp / (tp + fp)
        points.append((rec, prec))

    # 梯形求和
    auc = 0.0
    for i in range(1, len(points)):
        r0, p0 = points[i - 1]
        r1, p1 = points[i]
        auc += (r1 - r0) * (p0 + p1) / 2
    return auc


class MetricsCalculator:
    """指标计算器"""

    @staticmethod
    def compute_detection_metrics(
        predictions: list[dict],
        ground_truths: list[dict],
        iou_threshold: float = 0.5,
    ) -> dict[str, float]:
        """
        计算目标检测指标 (IoU 匹配 + AP 计算)。

        predictions:  [{"class": 0, "confidence": 0.92, "bbox": [cx,cy,w,h]}, ...]
        ground_truths: [{"class": 0, "bbox": [cx,cy,w,h]}, ...]

        Returns:
            {"mAP@0.5": ..., "mAP@0.5:0.95": ..., "precision": ..., "recall": ...}
        """
        if not predictions and not ground_truths:
            return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0, "precision": 0.0, "recall": 0.0}

        if not predictions:
            return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0, "precision": 0.0, "recall": 0.0}

        if not ground_truths:
            return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0, "precision": 1.0, "recall": 0.0}

        # ── mAP@0.5 ───────────────────────────────────────────
        per_class_matches, gt_counts = _match_detections(predictions, ground_truths, iou_threshold)

        all_classes = set(gt_counts.keys()) | set(per_class_matches.keys())
        aps: list[float] = []
        total_tp, total_fp, total_gt = 0, 0, 0

        for cls in all_classes:
            matches = per_class_matches.get(cls, [])
            n_gt = gt_counts.get(cls, 0)
            total_gt += n_gt

            if n_gt == 0:
                total_fp += len(matches)
                continue

            # 按置信度降序累计
            tp_cumsum = 0
            fp_cumsum = 0
            recalls_list: list[float] = []
            precs_list: list[float] = []

            for _, is_tp in matches:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                recalls_list.append(tp_cumsum / n_gt)
                precs_list.append(tp_cumsum / (tp_cumsum + fp_cumsum))

            total_tp += tp_cumsum
            total_fp += fp_cumsum
            aps.append(_compute_ap(recalls_list, precs_list))

        map50 = sum(aps) / len(aps) if aps else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0

        # ── mAP@0.5:0.95 (多阈值平均) ────────────────────────
        map_sum = 0.0
        thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.50 ~ 0.95
        for thr in thresholds:
            pcm, gc = _match_detections(predictions, ground_truths, thr)
            thr_aps: list[float] = []
            for cls in set(gc.keys()) | set(pcm.keys()):
                matches = pcm.get(cls, [])
                n_gt = gc.get(cls, 0)
                if n_gt == 0:
                    continue
                tp_cum, fp_cum = 0, 0
                rr, pp = [], []
                for _, is_tp in matches:
                    if is_tp:
                        tp_cum += 1
                    else:
                        fp_cum += 1
                    rr.append(tp_cum / n_gt)
                    pp.append(tp_cum / (tp_cum + fp_cum))
                thr_aps.append(_compute_ap(rr, pp))
            map_sum += (sum(thr_aps) / len(thr_aps)) if thr_aps else 0.0

        map50_95 = map_sum / len(thresholds)

        return {
            "mAP@0.5": round(map50, 4),
            "mAP@0.5:0.95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    @staticmethod
    def compute_classification_metrics(
        predictions: list[int], ground_truths: list[int]
    ) -> dict[str, float]:
        """
        计算分类指标 (macro 平均)。

        Returns:
            {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
        """
        if not predictions:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        accuracy = correct / len(predictions)

        # macro 平均
        all_classes = set(predictions) | set(ground_truths)
        precs, recs = [], []
        for cls in all_classes:
            tp = sum(p == cls and g == cls for p, g in zip(predictions, ground_truths))
            fp = sum(p == cls and g != cls for p, g in zip(predictions, ground_truths))
            fn = sum(p != cls and g == cls for p, g in zip(predictions, ground_truths))
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            precs.append(p)
            recs.append(r)

        macro_prec = sum(precs) / len(precs) if precs else 0.0
        macro_rec = sum(recs) / len(recs) if recs else 0.0
        macro_f1 = (
            2 * macro_prec * macro_rec / (macro_prec + macro_rec)
            if (macro_prec + macro_rec) > 0
            else 0.0
        )

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "f1": round(macro_f1, 4),
        }

    @staticmethod
    def compute_ocr_metrics(predictions: list[str], ground_truths: list[str]) -> dict[str, float]:
        """
        计算 OCR 指标

        Returns:
            {"reading_accuracy": ..., "digit_accuracy": ..., "mae": ..., "rmse": ...}
        """
        if not predictions:
            return {"reading_accuracy": 0.0, "digit_accuracy": 0.0, "mae": 0.0, "rmse": 0.0}

        exact_match = sum(p == g for p, g in zip(predictions, ground_truths))
        reading_accuracy = exact_match / len(predictions)

        # MAE / RMSE (将读数转为数值)
        mae_sum, rmse_sum, count = 0.0, 0.0, 0
        for p, g in zip(predictions, ground_truths):
            try:
                pv, gv = float(p), float(g)
                mae_sum += abs(pv - gv)
                rmse_sum += (pv - gv) ** 2
                count += 1
            except ValueError:
                continue

        mae = mae_sum / count if count else 0.0
        rmse = (rmse_sum / count) ** 0.5 if count else 0.0

        # 逐字符准确率
        total_chars, correct_chars = 0, 0
        for p, g in zip(predictions, ground_truths):
            for pc, gc in zip(p, g):
                total_chars += 1
                if pc == gc:
                    correct_chars += 1
            total_chars += abs(len(p) - len(g))
        digit_accuracy = correct_chars / total_chars if total_chars else 0.0

        return {
            "reading_accuracy": round(reading_accuracy, 4),
            "digit_accuracy": round(digit_accuracy, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        }

    @staticmethod
    def compute_thermal_metrics(predictions: list[str], ground_truths: list[str]) -> dict[str, float]:
        """
        计算热像异常检测指标

        Returns:
            {"anomaly_accuracy": ..., "false_alarm_rate": ..., "detection_latency_ms": ..., "mae_celsius": ...}
        """
        if not predictions:
            return {"anomaly_accuracy": 0.0, "false_alarm_rate": 0.0}

        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        accuracy = correct / len(predictions)

        # 误报率: 预测异常但实际正常
        false_alarms = sum(
            p != "normal" and g == "normal" for p, g in zip(predictions, ground_truths)
        )
        total_normal = sum(g == "normal" for g in ground_truths)
        far = false_alarms / total_normal if total_normal else 0.0

        return {"anomaly_accuracy": accuracy, "false_alarm_rate": far}

    @staticmethod
    def compute_hyperspectral_metrics(predictions: list[int], ground_truths: list[int], num_classes: int = 0) -> dict[str, float]:
        """
        计算高光谱分类指标

        Returns:
            {"overall_accuracy": ..., "kappa": ..., "per_class_f1": ...}
        """
        if not predictions:
            return {"overall_accuracy": 0.0, "kappa": 0.0}

        n = len(predictions)
        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        oa = correct / n

        # Cohen's Kappa (简化计算)
        # kappa = (p_o - p_e) / (1 - p_e)
        from collections import Counter
        pred_counts = Counter(predictions)
        gt_counts = Counter(ground_truths)
        p_e = sum(pred_counts.get(c, 0) * gt_counts.get(c, 0) for c in set(predictions) | set(ground_truths)) / (n * n)
        kappa = (oa - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

        return {"overall_accuracy": oa, "kappa": kappa}

    @staticmethod
    def compute_temporal_anomaly_metrics(
        anomaly_scores: list[float],
        ground_truths: list[int],
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """
        计算时序异常检测指标

        Returns:
            {"auc_roc": ..., "auc_pr": ..., "f1": ..., "false_positive_rate": ...}
        """
        if not anomaly_scores or not ground_truths or len(anomaly_scores) != len(ground_truths):
            return {
                "auc_roc": 0.0,
                "auc_pr": 0.0,
                "f1": 0.0,
                "false_positive_rate": 0.0,
            }

        preds = [1 if score >= threshold else 0 for score in anomaly_scores]
        tp = sum(p == 1 and g == 1 for p, g in zip(preds, ground_truths))
        fp = sum(p == 1 and g == 0 for p, g in zip(preds, ground_truths))
        fn = sum(p == 0 and g == 1 for p, g in zip(preds, ground_truths))
        tn = sum(p == 0 and g == 0 for p, g in zip(preds, ground_truths))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        # AUC-ROC: 梯形积分
        auc_roc = _compute_auc_roc(anomaly_scores, ground_truths)
        # AUC-PR: precision-recall 积分
        auc_pr = _compute_auc_pr(anomaly_scores, ground_truths)

        return {
            "auc_roc": round(auc_roc, 4),
            "auc_pr": round(auc_pr, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
        }

    @staticmethod
    def compute_regression_metrics(
        predictions: list[float],
        ground_truths: list[float],
        prefix: str = "regression",
    ) -> dict[str, float]:
        """
        计算回归指标

        prefix="health" 时返回 health_mae / health_rmse。
        """
        if not predictions or not ground_truths or len(predictions) != len(ground_truths):
            return {f"{prefix}_mae": 0.0, f"{prefix}_rmse": 0.0}

        n = len(predictions)
        mae = sum(abs(p - g) for p, g in zip(predictions, ground_truths)) / n
        rmse = (
            sum((p - g) ** 2 for p, g in zip(predictions, ground_truths)) / n
        ) ** 0.5
        return {f"{prefix}_mae": mae, f"{prefix}_rmse": rmse}

    @staticmethod
    def compute_sequence_metrics(
        predictions: list[str],
        ground_truths: list[str],
    ) -> dict[str, float]:
        """
        计算事件序列识别指标

        Returns:
            {"sequence_accuracy": ..., "macro_f1": ..., "event_recall": ..., "edit_distance": ...}
        """
        if not predictions or not ground_truths or len(predictions) != len(ground_truths):
            return {
                "sequence_accuracy": 0.0,
                "macro_f1": 0.0,
                "event_recall": 0.0,
                "edit_distance": 0.0,
            }

        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        accuracy = correct / len(predictions)

        # macro F1
        all_labels = set(predictions) | set(ground_truths)
        f1s = []
        recalls = []
        for lbl in all_labels:
            tp = sum(p == lbl and g == lbl for p, g in zip(predictions, ground_truths))
            fp = sum(p == lbl and g != lbl for p, g in zip(predictions, ground_truths))
            fn = sum(p != lbl and g == lbl for p, g in zip(predictions, ground_truths))
            pr = tp / (tp + fp) if (tp + fp) else 0.0
            rc = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
            f1s.append(f1)
            recalls.append(rc)

        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        event_recall = sum(recalls) / len(recalls) if recalls else 0.0

        # 归一化编辑距离 (Levenshtein)
        def _edit_dist(a: str, b: str) -> int:
            m, n = len(a), len(b)
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev = dp[0]
                dp[0] = i
                for j in range(1, n + 1):
                    temp = dp[j]
                    dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
                    prev = temp
            return dp[n]

        total_ed = sum(_edit_dist(str(p), str(g)) for p, g in zip(predictions, ground_truths))
        max_len = sum(max(len(str(p)), len(str(g))) for p, g in zip(predictions, ground_truths))
        norm_ed = total_ed / max_len if max_len else 0.0

        return {
            "sequence_accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "event_recall": round(event_recall, 4),
            "edit_distance": round(norm_ed, 4),
        }

    @staticmethod
    def compute_multimodal_fusion_metrics(
        predictions: list[str],
        ground_truths: list[str],
        confidence_scores: list[float] | None = None,
    ) -> dict[str, float]:
        """
        计算多模态融合指标

        Returns:
            {"overall_accuracy": ..., "macro_f1": ..., "ece": ..., "diagnosis_hit_rate": ...}
        """
        if not predictions or not ground_truths or len(predictions) != len(ground_truths):
            return {
                "overall_accuracy": 0.0,
                "macro_f1": 0.0,
                "ece": 0.0,
                "diagnosis_hit_rate": 0.0,
                "modality_dropout_robustness": 0.0,
                "rule_consistency": 0.0,
            }

        n = len(predictions)
        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        accuracy = correct / n

        # macro F1
        all_labels = set(predictions) | set(ground_truths)
        f1s = []
        for lbl in all_labels:
            tp = sum(p == lbl and g == lbl for p, g in zip(predictions, ground_truths))
            fp = sum(p == lbl and g != lbl for p, g in zip(predictions, ground_truths))
            fn = sum(p != lbl and g == lbl for p, g in zip(predictions, ground_truths))
            pr = tp / (tp + fp) if (tp + fp) else 0.0
            rc = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
            f1s.append(f1)
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        # ECE (Expected Calibration Error) — 10 bins
        ece = 0.0
        if confidence_scores and len(confidence_scores) == n:
            n_bins = 10
            bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
            for conf, p, g in zip(confidence_scores, predictions, ground_truths):
                idx = min(int(conf * n_bins), n_bins - 1)
                bins[idx].append((conf, p == g))
            for b in bins:
                if not b:
                    continue
                avg_conf = sum(c for c, _ in b) / len(b)
                avg_acc = sum(int(ok) for _, ok in b) / len(b)
                ece += abs(avg_acc - avg_conf) * len(b) / n

        return {
            "overall_accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "ece": round(ece, 4),
            "diagnosis_hit_rate": round(accuracy, 4),
            "modality_dropout_robustness": 0.0,
            "rule_consistency": 0.0,
        }
