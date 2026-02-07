#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测 - 多目标跟踪器
=================================

基于 IoU + 卡尔曼滤波的轻量级跟踪器 (ByteTrack风格):
- 跨帧目标关联
- 全局唯一 track_id
- 停留时间统计
- 轨迹历史记录

作者: G组 | 版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from .event_schema import AnimalDetectionResult, BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class AnimalTrack:
    """动物跟踪轨迹"""
    track_id: str
    animal_class: str
    bbox: BoundingBox
    confidence: float
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    hits: int = 1
    age: int = 0              # 自最后一次检测到的帧数
    state: str = "tentative"  # tentative -> confirmed -> lost -> deleted
    trajectory: List[Tuple[float, float]] = field(default_factory=list)  # (cx, cy) 历史
    stay_duration_s: float = 0.0
    deterrent_count: int = 0
    is_deterred: bool = False

    def center(self) -> Tuple[float, float]:
        return (self.bbox.x + self.bbox.width / 2, self.bbox.y + self.bbox.height / 2)

    def update_stay(self):
        self.stay_duration_s = self.last_seen - self.first_seen


class AnimalTracker:
    """
    多目标动物跟踪器

    使用 IoU 匹配进行跨帧关联
    """

    def __init__(
        self,
        max_age: int = 30,          # 最大丢失帧数
        min_hits: int = 3,           # 确认所需最少命中数
        iou_threshold: float = 0.3,  # IoU匹配阈值
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self._tracks: Dict[str, AnimalTrack] = {}
        self._next_id: int = 1
        self._frame_count: int = 0

        # 统计
        self._total_tracks_created: int = 0
        self._total_tracks_deleted: int = 0

    def update(
        self, detections: List[AnimalDetectionResult]
    ) -> List[AnimalDetectionResult]:
        """
        更新跟踪器

        Args:
            detections: 当前帧检测结果

        Returns:
            带有track_id和stay_duration的检测结果
        """
        self._frame_count += 1
        now = time.time()

        # 1. 预测阶段 - 增加所有轨迹age
        for track in self._tracks.values():
            track.age += 1

        # 2. 匹配阶段
        if detections and self._tracks:
            matched, unmatched_dets, unmatched_tracks = self._match(detections)

            # 更新匹配的轨迹
            for det_idx, track_id in matched:
                det = detections[det_idx]
                track = self._tracks[track_id]
                track.bbox = det.bbox
                track.confidence = det.confidence
                track.animal_class = det.animal_class
                track.last_seen = now
                track.hits += 1
                track.age = 0
                track.update_stay()
                cx, cy = track.center()
                track.trajectory.append((cx, cy))
                # 限制轨迹长度
                if len(track.trajectory) > 1000:
                    track.trajectory = track.trajectory[-500:]

                if track.state == "tentative" and track.hits >= self.min_hits:
                    track.state = "confirmed"

                # 赋值回检测结果
                det.track_id = track_id
                det.stay_duration_s = track.stay_duration_s

        elif detections:
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = []
        else:
            unmatched_dets = []
            unmatched_tracks = list(self._tracks.keys())

        # 3. 创建新轨迹 (未匹配的检测)
        for idx in unmatched_dets:
            det = detections[idx]
            track_id = self._create_track(det, now)
            det.track_id = track_id
            det.stay_duration_s = 0.0

        # 4. 处理丢失轨迹
        for track_id in unmatched_tracks:
            track = self._tracks.get(track_id)
            if track and track.age > self.max_age:
                track.state = "deleted"

        # 5. 删除过期轨迹
        to_delete = [tid for tid, t in self._tracks.items() if t.state == "deleted"]
        for tid in to_delete:
            del self._tracks[tid]
            self._total_tracks_deleted += 1

        return detections

    def _match(
        self, detections: List[AnimalDetectionResult]
    ) -> Tuple[List[Tuple[int, str]], List[int], List[str]]:
        """IoU匹配"""
        det_boxes = []
        for d in detections:
            if d.bbox:
                det_boxes.append([d.bbox.x, d.bbox.y,
                                  d.bbox.x + d.bbox.width,
                                  d.bbox.y + d.bbox.height])
            else:
                det_boxes.append([0, 0, 0, 0])

        track_ids = list(self._tracks.keys())
        track_boxes = []
        for tid in track_ids:
            t = self._tracks[tid]
            track_boxes.append([t.bbox.x, t.bbox.y,
                                t.bbox.x + t.bbox.width,
                                t.bbox.y + t.bbox.height])

        det_arr = np.array(det_boxes)
        trk_arr = np.array(track_boxes)

        # 计算IoU矩阵
        iou_matrix = self._compute_iou_matrix(det_arr, trk_arr)

        # 贪心匹配
        matched = []
        used_dets = set()
        used_trks = set()

        # 按IoU从大到小匹配
        if iou_matrix.size > 0:
            flat_indices = np.argsort(-iou_matrix.flatten())
            for flat_idx in flat_indices:
                det_idx = flat_idx // iou_matrix.shape[1]
                trk_idx = flat_idx % iou_matrix.shape[1]

                if det_idx in used_dets or trk_idx in used_trks:
                    continue

                if iou_matrix[det_idx, trk_idx] >= self.iou_threshold:
                    matched.append((int(det_idx), track_ids[trk_idx]))
                    used_dets.add(det_idx)
                    used_trks.add(trk_idx)

        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_trks = [track_ids[i] for i in range(len(track_ids)) if i not in used_trks]

        return matched, unmatched_dets, unmatched_trks

    @staticmethod
    def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """计算IoU矩阵"""
        na = len(boxes_a)
        nb = len(boxes_b)
        iou = np.zeros((na, nb), dtype=np.float32)

        for i in range(na):
            for j in range(nb):
                xx1 = max(boxes_a[i, 0], boxes_b[j, 0])
                yy1 = max(boxes_a[i, 1], boxes_b[j, 1])
                xx2 = min(boxes_a[i, 2], boxes_b[j, 2])
                yy2 = min(boxes_a[i, 3], boxes_b[j, 3])

                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                area_a = (boxes_a[i, 2] - boxes_a[i, 0]) * (boxes_a[i, 3] - boxes_a[i, 1])
                area_b = (boxes_b[j, 2] - boxes_b[j, 0]) * (boxes_b[j, 3] - boxes_b[j, 1])
                union = area_a + area_b - inter

                iou[i, j] = inter / (union + 1e-8)

        return iou

    def _create_track(self, det: AnimalDetectionResult, now: float) -> str:
        """创建新轨迹"""
        track_id = f"AT-{self._next_id:06d}"
        self._next_id += 1

        track = AnimalTrack(
            track_id=track_id,
            animal_class=det.animal_class,
            bbox=det.bbox or BoundingBox(0, 0, 0, 0),
            confidence=det.confidence,
            first_seen=now,
            last_seen=now,
        )
        if self.min_hits <= 1:
            track.state = "confirmed"
        cx, cy = track.center()
        track.trajectory.append((cx, cy))

        self._tracks[track_id] = track
        self._total_tracks_created += 1
        return track_id

    def get_active_tracks(self) -> List[AnimalTrack]:
        """获取所有活跃（已确认）轨迹"""
        return [t for t in self._tracks.values() if t.state == "confirmed"]

    def get_all_tracks(self) -> List[AnimalTrack]:
        """获取所有轨迹"""
        return list(self._tracks.values())

    def get_track(self, track_id: str) -> Optional[AnimalTrack]:
        """获取指定轨迹"""
        return self._tracks.get(track_id)

    def get_overstayed_tracks(self, threshold_s: float) -> List[AnimalTrack]:
        """获取停留时间超过阈值的轨迹"""
        return [
            t for t in self._tracks.values()
            if t.state == "confirmed" and t.stay_duration_s >= threshold_s
        ]

    def mark_deterred(self, track_id: str):
        """标记轨迹已被驱离"""
        track = self._tracks.get(track_id)
        if track:
            track.deterrent_count += 1
            track.is_deterred = True

    def get_stats(self) -> Dict[str, Any]:
        """获取跟踪统计"""
        active = [t for t in self._tracks.values() if t.state in ("tentative", "confirmed")]
        confirmed = [t for t in active if t.state == "confirmed"]
        return {
            "total_tracks": len(self._tracks),
            "active_tracks": len(active),
            "confirmed_tracks": len(confirmed),
            "total_created": self._total_tracks_created,
            "total_deleted": self._total_tracks_deleted,
            "frame_count": self._frame_count,
        }

    def reset(self):
        """重置跟踪器"""
        self._tracks.clear()
        self._frame_count = 0
        self._next_id = 1
