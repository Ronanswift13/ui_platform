#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测 - 统计与报告
================================

- 每日/每周入侵次数统计
- 各类动物比例分析
- 平均停留时间
- 驱离成功率
- 趋势图数据输出
- CSV/JSON报告导出

作者: G组 | 版本: 1.0.0
"""

from __future__ import annotations
import csv
import io
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .event_schema import AnimalDetectionResult

logger = logging.getLogger(__name__)


class InvasionRecord:
    """单次入侵记录"""
    __slots__ = [
        "record_id", "timestamp", "animal_class", "confidence",
        "stay_duration_s", "track_id", "camera_id", "location",
        "is_thermal_validated", "was_deterred", "deterrent_result",
    ]

    def __init__(
        self,
        record_id: str,
        timestamp: float,
        animal_class: str,
        confidence: float,
        stay_duration_s: float = 0.0,
        track_id: str = "",
        camera_id: str = "",
        location: str = "",
        is_thermal_validated: bool = False,
        was_deterred: bool = False,
        deterrent_result: str = "",
    ):
        self.record_id = record_id
        self.timestamp = timestamp
        self.animal_class = animal_class
        self.confidence = confidence
        self.stay_duration_s = stay_duration_s
        self.track_id = track_id
        self.camera_id = camera_id
        self.location = location
        self.is_thermal_validated = is_thermal_validated
        self.was_deterred = was_deterred
        self.deterrent_result = deterrent_result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "animal_class": self.animal_class,
            "confidence": round(self.confidence, 4),
            "stay_duration_s": round(self.stay_duration_s, 1),
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "location": self.location,
            "is_thermal_validated": self.is_thermal_validated,
            "was_deterred": self.was_deterred,
            "deterrent_result": self.deterrent_result,
        }


class StatisticsCollector:
    """
    入侵统计收集器

    维护入侵历史记录并提供查询/聚合/报告功能
    """

    def __init__(
        self,
        history_retention_days: int = 90,
        report_interval_hours: int = 24,
    ):
        self.history_retention_days = history_retention_days
        self.report_interval_hours = report_interval_hours

        self._records: List[InvasionRecord] = []
        self._record_counter: int = 0

        # 聚合缓存
        self._daily_cache: Dict[str, Dict] = {}
        self._last_cleanup: float = time.time()

    def record_invasion(
        self,
        detection: AnimalDetectionResult,
        camera_id: str = "",
        location: str = "",
        was_deterred: bool = False,
        deterrent_result: str = "",
    ):
        """记录一次入侵事件"""
        self._record_counter += 1
        record = InvasionRecord(
            record_id=f"IR-{self._record_counter:08d}",
            timestamp=time.time(),
            animal_class=detection.animal_class,
            confidence=detection.confidence,
            stay_duration_s=detection.stay_duration_s,
            track_id=detection.track_id or "",
            camera_id=camera_id,
            location=location,
            is_thermal_validated=detection.is_thermal_validated,
            was_deterred=was_deterred,
            deterrent_result=deterrent_result,
        )
        self._records.append(record)

        # 定期清理
        now = time.time()
        if now - self._last_cleanup > 3600:  # 每小时清理一次
            self._cleanup_old_records()
            self._last_cleanup = now

    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """获取日报数据"""
        if date is None:
            date = datetime.now()

        day_start = datetime(date.year, date.month, date.day).timestamp()
        day_end = day_start + 86400

        day_records = [r for r in self._records if day_start <= r.timestamp < day_end]

        return self._aggregate(day_records, "daily", date.strftime("%Y-%m-%d"))

    def get_weekly_summary(self, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """获取周报数据"""
        if end_date is None:
            end_date = datetime.now()

        week_end = datetime(end_date.year, end_date.month, end_date.day).timestamp() + 86400
        week_start = week_end - 7 * 86400

        week_records = [r for r in self._records if week_start <= r.timestamp < week_end]

        return self._aggregate(
            week_records, "weekly",
            f"{datetime.fromtimestamp(week_start).strftime('%Y-%m-%d')} ~ "
            f"{end_date.strftime('%Y-%m-%d')}"
        )

    def get_trend_data(self, days: int = 30) -> Dict[str, Any]:
        """获取趋势数据（每日入侵次数）"""
        now = datetime.now()
        daily_counts = []
        class_daily = defaultdict(lambda: defaultdict(int))

        for i in range(days - 1, -1, -1):
            date = now - timedelta(days=i)
            day_str = date.strftime("%Y-%m-%d")
            day_start = datetime(date.year, date.month, date.day).timestamp()
            day_end = day_start + 86400

            day_records = [r for r in self._records if day_start <= r.timestamp < day_end]
            daily_counts.append({
                "date": day_str,
                "count": len(day_records),
            })

            for r in day_records:
                class_daily[r.animal_class][day_str] += 1

        return {
            "period_days": days,
            "daily_counts": daily_counts,
            "class_trends": {
                cls: [{"date": d["date"], "count": counts.get(d["date"], 0)}
                      for d in daily_counts]
                for cls, counts in class_daily.items()
            },
            "total": sum(d["count"] for d in daily_counts),
        }

    def export_csv(self, start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> str:
        """导出CSV格式报告"""
        records = self._filter_by_time(start_time, end_time)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "record_id", "datetime", "animal_class", "confidence",
            "stay_duration_s", "track_id", "camera_id", "location",
            "is_thermal_validated", "was_deterred", "deterrent_result",
        ])
        writer.writeheader()
        for r in records:
            d = r.to_dict()
            writer.writerow({k: d.get(k, "") for k in writer.fieldnames})

        return output.getvalue()

    def export_json(self, start_time: Optional[float] = None,
                    end_time: Optional[float] = None) -> str:
        """导出JSON格式报告"""
        records = self._filter_by_time(start_time, end_time)
        return json.dumps(
            {"records": [r.to_dict() for r in records], "count": len(records)},
            ensure_ascii=False,
            indent=2,
        )

    def get_today_count(self) -> int:
        """获取今日入侵次数"""
        today = datetime.now()
        day_start = datetime(today.year, today.month, today.day).timestamp()
        return sum(1 for r in self._records if r.timestamp >= day_start)

    def _aggregate(self, records: List[InvasionRecord],
                   period_type: str, period_label: str) -> Dict[str, Any]:
        """聚合统计"""
        if not records:
            return {
                "period_type": period_type,
                "period_label": period_label,
                "total_invasions": 0,
                "class_distribution": {},
                "avg_confidence": 0.0,
                "avg_stay_duration_s": 0.0,
                "deterrent_stats": {"total": 0, "success": 0, "rate": 0.0},
                "thermal_validated_count": 0,
                "hourly_distribution": {},
            }

        # 类别分布
        class_counts = defaultdict(int)
        for r in records:
            class_counts[r.animal_class] += 1

        total = len(records)

        # 驱离统计
        deterred = [r for r in records if r.was_deterred]
        deterred_success = [r for r in deterred if r.deterrent_result == "success"]

        # 小时分布
        hourly = defaultdict(int)
        for r in records:
            hour = datetime.fromtimestamp(r.timestamp).hour
            hourly[hour] += 1

        return {
            "period_type": period_type,
            "period_label": period_label,
            "total_invasions": total,
            "class_distribution": dict(class_counts),
            "class_percentages": {
                cls: round(cnt / total * 100, 1) for cls, cnt in class_counts.items()
            },
            "avg_confidence": round(sum(r.confidence for r in records) / total, 4),
            "avg_stay_duration_s": round(
                sum(r.stay_duration_s for r in records) / total, 1
            ),
            "max_stay_duration_s": round(max(r.stay_duration_s for r in records), 1),
            "deterrent_stats": {
                "total": len(deterred),
                "success": len(deterred_success),
                "rate": round(len(deterred_success) / len(deterred), 4) if deterred else 0.0,
            },
            "thermal_validated_count": sum(1 for r in records if r.is_thermal_validated),
            "hourly_distribution": dict(sorted(hourly.items())),
        }

    def _filter_by_time(
        self, start_time=None, end_time=None
    ) -> List[InvasionRecord]:
        """按时间过滤记录 (支持 datetime 或 float 时间戳)"""
        def _to_ts(t):
            if t is None:
                return None
            if isinstance(t, datetime):
                return t.timestamp()
            return float(t)

        st = _to_ts(start_time)
        et = _to_ts(end_time)
        records = self._records
        if st is not None:
            records = [r for r in records if r.timestamp >= st]
        if et is not None:
            records = [r for r in records if r.timestamp <= et]
        return records

    def _cleanup_old_records(self):
        """清理过期记录"""
        cutoff = time.time() - self.history_retention_days * 86400
        before = len(self._records)
        self._records = [r for r in self._records if r.timestamp >= cutoff]
        after = len(self._records)
        if before > after:
            logger.info(f"清理过期入侵记录: {before - after} 条")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_records": len(self._records),
            "today_count": self.get_today_count(),
            "retention_days": self.history_retention_days,
        }
