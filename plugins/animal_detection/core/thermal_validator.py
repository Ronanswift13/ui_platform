#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测 - 热成像融合校验
==================================

双光摄像头场景下:
- 获取可见光和红外图像
- 检测框区域内温差计算
- 活体判定 vs 阴影/光斑过滤
- 背景温度自适应更新

作者: G组 | 版本: 1.0.0
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .event_schema import AnimalDetectionResult, BoundingBox

logger = logging.getLogger(__name__)


class ThermalValidator:
    """
    热成像活体校验器

    使用红外图像的温差来判定检测目标是否为活体:
    - 计算检测框内平均温度
    - 与背景区域温度比较
    - 温差 >= 阈值 => 活体（通过）
    - 温差 < 阈值 => 阴影/反光（过滤）
    """

    def __init__(
        self,
        heat_diff_threshold_c: float = 2.0,
        background_update_rate: float = 0.01,
        min_roi_pixels: int = 16,
        calibration_offset_c: float = 0.0,
        enabled: bool = True,
    ):
        """
        Args:
            heat_diff_threshold_c: 活体判定温差阈值 (摄氏度)
            background_update_rate: 背景温度指数移动平均更新率
            min_roi_pixels: ROI最小像素数（小于此值跳过校验）
            calibration_offset_c: 温度校准偏移量
            enabled: 是否启用
        """
        self.heat_diff_threshold_c = heat_diff_threshold_c
        self.background_update_rate = background_update_rate
        self.min_roi_pixels = min_roi_pixels
        self.calibration_offset_c = calibration_offset_c
        self.enabled = enabled

        self._background_temp: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._validate_count: int = 0
        self._reject_count: int = 0

    def validate_detections(
        self,
        detections: List[AnimalDetectionResult],
        thermal_frame: Optional[np.ndarray],
    ) -> List[AnimalDetectionResult]:
        """
        对检测结果进行热成像校验

        Args:
            detections: 可见光检测结果列表
            thermal_frame: 红外热成像帧 (单通道float32温度图或uint16原始值)
                          如果为None则跳过校验,全部通过

        Returns:
            校验后的结果列表（已标注 is_thermal_validated 和 thermal_diff_c）
        """
        if not self.enabled or thermal_frame is None:
            # 无热成像数据时，全部标记为未校验但通过
            for d in detections:
                d.is_thermal_validated = False
                d.thermal_diff_c = None
            return detections

        self._frame_count += 1
        th, tw = thermal_frame.shape[:2]

        # 转换为温度图 (float32)
        temp_map = self._to_temperature_map(thermal_frame)

        # 更新背景温度
        self._update_background(temp_map)

        validated = []
        for det in detections:
            if det.bbox is None:
                det.is_thermal_validated = False
                validated.append(det)
                continue

            # 计算ROI区域温度
            roi_temp = self._get_roi_temperature(temp_map, det.bbox, tw, th)
            bg_temp = self._get_background_temperature(temp_map, det.bbox, tw, th)

            if roi_temp is None or bg_temp is None:
                det.is_thermal_validated = False
                validated.append(det)
                continue

            temp_diff = roi_temp - bg_temp + self.calibration_offset_c
            det.thermal_diff_c = float(temp_diff)
            self._validate_count += 1

            if temp_diff >= self.heat_diff_threshold_c:
                # 温差显著 -> 活体
                det.is_thermal_validated = True
                logger.debug(
                    f"热成像校验通过: {det.animal_class} "
                    f"温差={temp_diff:.1f}°C >= {self.heat_diff_threshold_c}°C"
                )
            else:
                # 温差不显著 -> 阴影/反光
                det.is_thermal_validated = False
                self._reject_count += 1
                logger.info(
                    f"热成像校验拒绝(疑似阴影/光斑): {det.animal_class} "
                    f"温差={temp_diff:.1f}°C < {self.heat_diff_threshold_c}°C | "
                    f"confidence={det.confidence:.2f}"
                )
            # 保留在列表中（标记状态），由上层 filter_non_thermal 决定是否移除
            validated.append(det)

        return validated

    def filter_non_thermal(
        self, detections: List[AnimalDetectionResult]
    ) -> List[AnimalDetectionResult]:
        """
        过滤掉热成像校验不通过的检测

        调用此方法前需先调用 validate_detections
        """
        return [d for d in detections if d.is_thermal_validated or d.thermal_diff_c is None]

    def _to_temperature_map(self, frame: np.ndarray) -> np.ndarray:
        """将原始热成像帧转换为温度图 (float32 摄氏度)"""
        if frame.dtype == np.float32:
            return frame
        elif frame.dtype == np.float64:
            return frame.astype(np.float32)
        elif frame.dtype == np.uint16:
            # 常见红外相机格式: raw_value / 100.0 = 摄氏度
            return frame.astype(np.float32) / 100.0
        elif frame.dtype == np.uint8:
            # 8bit灰度热图 -> 近似温度映射 (需要校准)
            # 假设 0=15°C, 255=45°C (室内典型范围)
            return frame.astype(np.float32) / 255.0 * 30.0 + 15.0
        else:
            return frame.astype(np.float32)

    def _update_background(self, temp_map: np.ndarray):
        """指数移动平均更新背景温度"""
        if self._background_temp is None:
            self._background_temp = temp_map.copy()
        else:
            alpha = self.background_update_rate
            self._background_temp = (
                (1 - alpha) * self._background_temp + alpha * temp_map
            )

    def _get_roi_temperature(
        self,
        temp_map: np.ndarray,
        bbox: BoundingBox,
        tw: int,
        th: int,
    ) -> Optional[float]:
        """获取检测框内平均温度"""
        x1 = int(bbox.x * tw)
        y1 = int(bbox.y * th)
        x2 = int((bbox.x + bbox.width) * tw)
        y2 = int((bbox.y + bbox.height) * th)

        # 边界裁剪
        x1 = max(0, min(tw - 1, x1))
        y1 = max(0, min(th - 1, y1))
        x2 = max(x1 + 1, min(tw, x2))
        y2 = max(y1 + 1, min(th, y2))

        roi = temp_map[y1:y2, x1:x2]
        if roi.size < self.min_roi_pixels:
            return None

        return float(np.mean(roi))

    def _get_background_temperature(
        self,
        temp_map: np.ndarray,
        bbox: BoundingBox,
        tw: int,
        th: int,
    ) -> Optional[float]:
        """获取检测框周围背景温度"""
        # 扩展检测框作为背景区域（排除检测框本身）
        expand = 0.5  # 扩展50%
        cx = bbox.x + bbox.width / 2
        cy = bbox.y + bbox.height / 2
        bw = bbox.width * (1 + expand)
        bh = bbox.height * (1 + expand)

        bx1 = int(max(0, (cx - bw / 2)) * tw)
        by1 = int(max(0, (cy - bh / 2)) * th)
        bx2 = int(min(1, (cx + bw / 2)) * tw)
        by2 = int(min(1, (cy + bh / 2)) * th)

        # 内部框
        ix1 = int(bbox.x * tw)
        iy1 = int(bbox.y * th)
        ix2 = int((bbox.x + bbox.width) * tw)
        iy2 = int((bbox.y + bbox.height) * th)

        # 创建背景掩码
        mask = np.ones((by2 - by1, bx2 - bx1), dtype=bool)
        # 排除内框
        inner_y1 = max(0, iy1 - by1)
        inner_y2 = min(mask.shape[0], iy2 - by1)
        inner_x1 = max(0, ix1 - bx1)
        inner_x2 = min(mask.shape[1], ix2 - bx1)
        if inner_y2 > inner_y1 and inner_x2 > inner_x1:
            mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

        bg_region = temp_map[by1:by2, bx1:bx2]
        if bg_region.size == 0 or not np.any(mask):
            # 回退到使用背景温度模型
            if self._background_temp is not None:
                return float(np.mean(self._background_temp[by1:by2, bx1:bx2]))
            return None

        bg_values = bg_region[mask]
        if bg_values.size < self.min_roi_pixels:
            return None

        return float(np.mean(bg_values))

    def get_stats(self) -> Dict[str, Any]:
        """获取校验统计"""
        return {
            "enabled": self.enabled,
            "heat_diff_threshold_c": self.heat_diff_threshold_c,
            "frame_count": self._frame_count,
            "validate_count": self._validate_count,
            "reject_count": self._reject_count,
            "reject_rate": (
                round(self._reject_count / self._validate_count, 4)
                if self._validate_count > 0
                else 0.0
            ),
            "has_background_model": self._background_temp is not None,
        }

    def reset(self):
        """重置背景模型"""
        self._background_temp = None
        self._frame_count = 0
        self._validate_count = 0
        self._reject_count = 0
