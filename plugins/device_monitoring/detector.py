"""
设备状态监测检测器 - 核心引擎
===============================

功能: 健康指数计算、异常检测、故障预测、工单生成
版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeviceHealth:
    device_id: str
    device_name: str
    device_type: str
    status: str                      # online | standby | warning | error | offline
    health_index: float              # 0-100
    anomaly_score: float             # 0-1
    metrics: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]
    predicted_failure: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceSummary:
    total_devices: int
    online_count: int
    warning_count: int
    error_count: int
    offline_count: int
    avg_health: float
    critical_devices: List[str]


@dataclass
class MaintenanceTicket:
    ticket_id: str
    device_id: str
    device_name: str
    priority: str                    # P1 | P2 | P3
    title: str
    description: str
    timestamp: float
    auto_generated: bool = True


class DeviceHealthCalculator:
    """设备健康指数计算器"""

    def __init__(self, weights: Dict[str, float], thresholds: Dict[str, float]):
        self.weights = weights
        self.thresholds = thresholds

    def calculate(self, metrics: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """
        计算健康指数

        Returns:
            (health_index, issues, recommendations)
        """
        score = 100.0
        issues = []
        recs = []

        # CPU温度
        cpu_temp = metrics.get("cpu_temp", 40)
        w = self.weights.get("cpu_temp", 0.15)
        if cpu_temp > self.thresholds.get("cpu_temp_alarm", 85):
            deduction = w * 100
            score -= deduction
            issues.append(f"CPU温度过高: {cpu_temp}°C")
            recs.append("检查设备散热，清理风扇或更换散热片")
        elif cpu_temp > self.thresholds.get("cpu_temp_warning", 70):
            deduction = w * 50
            score -= deduction
            issues.append(f"CPU温度偏高: {cpu_temp}°C")
            recs.append("关注设备散热情况")

        # CPU使用率
        cpu_usage = metrics.get("cpu_usage", 30)
        w = self.weights.get("cpu_usage", 0.15)
        if cpu_usage > 90:
            score -= w * 100
            issues.append(f"CPU使用率过高: {cpu_usage}%")
            recs.append("检查是否有异常进程，考虑升级硬件")
        elif cpu_usage > 70:
            score -= w * 40
            issues.append(f"CPU使用率偏高: {cpu_usage}%")

        # 内存使用
        mem = metrics.get("memory_usage", 40)
        w = self.weights.get("memory_usage", 0.10)
        mem_warn = self.thresholds.get("memory_warning_percent", 85)
        if mem > mem_warn:
            score -= w * 80
            issues.append(f"内存使用率过高: {mem}%")
            recs.append("考虑增加内存或优化程序")

        # 网络质量
        net = metrics.get("network_quality", 90)
        w = self.weights.get("network_quality", 0.20)
        if net < 50:
            score -= w * 100
            issues.append(f"网络质量差: {net}%")
            recs.append("检查网线连接和交换机状态")
        elif net < 70:
            score -= w * 50
            issues.append(f"网络质量一般: {net}%")

        # 错误率
        errors = metrics.get("error_count", 0)
        w = self.weights.get("error_rate", 0.20)
        err_warn = self.thresholds.get("error_rate_warning", 5)
        if errors > err_warn * 3:
            score -= w * 100
            issues.append(f"错误频繁: {errors}次")
            recs.append("立即排查设备错误日志")
        elif errors > err_warn:
            score -= w * 50
            issues.append(f"错误数偏多: {errors}次")

        # 心跳超时
        last_hb = metrics.get("last_heartbeat", time.time())
        timeout = self.thresholds.get("heartbeat_timeout_sec", 60)
        elapsed = time.time() - last_hb
        if elapsed > timeout * 3:
            score -= 30
            issues.append("设备长时间无心跳")
            recs.append("检查设备供电和网络连接")
        elif elapsed > timeout:
            score -= 15
            issues.append("设备心跳超时")

        return max(0, min(100, round(score, 1))), issues, recs


class DeviceMonitorDetector:
    """设备状态监测检测器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        weights = config.get("health_weights", {})
        thresholds = config.get("thresholds", {})
        self._calculator = DeviceHealthCalculator(weights, thresholds)

        self.th_warning = thresholds.get("health_warning", 70)
        self.th_alarm = thresholds.get("health_alarm", 50)
        self.th_critical = thresholds.get("health_critical", 30)

        pred_cfg = config.get("prediction", {})
        self.pred_enabled = pred_cfg.get("enabled", True)
        self.pred_method = pred_cfg.get("method", "statistical")

        maint_cfg = config.get("maintenance", {})
        self.auto_ticket = maint_cfg.get("auto_ticket", True)
        self.priority_map = maint_cfg.get("ticket_priority_map", {"critical": "P1", "alarm": "P2", "warning": "P3"})

        self.managed_devices = config.get("managed_devices", [])

        # 历史
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2880))
        self._tickets: List[MaintenanceTicket] = []
        self._ticket_counter = 0
        self._inference_count = 0
        self._initialized = False

    def initialize(self) -> bool:
        self._initialized = True
        logger.info("[DeviceMonitorDetector] 初始化完成")
        return True

    def detect(self, device_readings: List[Dict]) -> Dict[str, Any]:
        start = time.time()
        self._inference_count += 1

        device_results = []
        new_tickets = []

        for reading in device_readings:
            did = reading.get("device_id", "unknown")
            metrics = reading.get("metrics", {})

            # 健康计算
            health_idx, issues, recs = self._calculator.calculate(metrics)

            # 异常评分
            anomaly = self._calc_anomaly(did, metrics)

            # 状态判定
            status = self._determine_status(health_idx, reading.get("status", "online"))

            # 故障预测
            predicted = self._predict_failure(did) if self.pred_enabled else None

            # 记录历史
            self._history[did].append({"timestamp": time.time(), "health": health_idx, "metrics": metrics})

            dh = DeviceHealth(
                device_id=did,
                device_name=reading.get("device_name", did),
                device_type=reading.get("device_type", "unknown"),
                status=status,
                health_index=health_idx,
                anomaly_score=round(anomaly, 3),
                metrics=metrics,
                issues=issues,
                recommendations=recs,
                predicted_failure=predicted,
            )
            device_results.append(dh)

            # 工单
            if self.auto_ticket and health_idx < self.th_alarm:
                ticket = self._create_ticket(dh)
                if ticket:
                    new_tickets.append(ticket)

        # 汇总
        summary = self._summarize(device_results)

        elapsed = (time.time() - start) * 1000

        return {
            "devices": device_results,
            "summary": summary,
            "new_tickets": new_tickets,
            "inference_time_ms": round(elapsed, 2),
        }

    def _determine_status(self, health: float, raw_status: str) -> str:
        if raw_status == "offline":
            return "offline"
        if health < self.th_critical:
            return "error"
        if health < self.th_alarm:
            return "error"
        if health < self.th_warning:
            return "warning"
        return raw_status if raw_status in ("online", "standby") else "online"

    def _calc_anomaly(self, device_id: str, metrics: Dict) -> float:
        hist = list(self._history.get(device_id, []))
        if len(hist) < 10:
            return 0.0

        # 简单统计异常
        health_vals = [h["health"] for h in hist[-60:]]
        mean = np.mean(health_vals)
        std = np.std(health_vals) + 1e-6
        current = hist[-1]["health"] if hist else mean

        z = abs(current - mean) / std
        return min(1.0, z / 3.0)

    def _predict_failure(self, device_id: str) -> Optional[Dict]:
        hist = list(self._history.get(device_id, []))
        if len(hist) < 30:
            return None

        health_vals = [h["health"] for h in hist[-60:]]
        if len(health_vals) < 10:
            return None

        x = np.arange(len(health_vals))
        try:
            coeffs = np.polyfit(x, health_vals, 1)
            slope = coeffs[0]

            if slope < -0.01:
                current = health_vals[-1]
                steps_to_alarm = (current - self.th_alarm) / abs(slope) if slope != 0 else float("inf")
                hours = steps_to_alarm * (30 / 3600)  # 30秒间隔

                if hours < 168:
                    return {
                        "predicted": True,
                        "hours_to_failure": round(max(0, hours), 1),
                        "trend_slope": round(slope, 4),
                        "current_health": round(current, 1),
                    }
        except Exception:
            pass

        return None

    def _create_ticket(self, dh: DeviceHealth) -> Optional[MaintenanceTicket]:
        # 避免重复工单
        recent = [t for t in self._tickets if t.device_id == dh.device_id and (time.time() - t.timestamp) < 3600]
        if recent:
            return None

        severity = "critical" if dh.health_index < self.th_critical else "alarm"
        priority = self.priority_map.get(severity, "P2")

        self._ticket_counter += 1
        ticket = MaintenanceTicket(
            ticket_id=f"MT-{self._ticket_counter:04d}",
            device_id=dh.device_id,
            device_name=dh.device_name,
            priority=priority,
            title=f"设备健康度异常: {dh.device_name}",
            description=f"健康指数: {dh.health_index}%, 问题: {'; '.join(dh.issues)}",
            timestamp=time.time(),
        )
        self._tickets.append(ticket)
        return ticket

    def _summarize(self, devices: List[DeviceHealth]) -> DeviceSummary:
        total = len(devices)
        online = sum(1 for d in devices if d.status in ("online", "standby"))
        warning = sum(1 for d in devices if d.status == "warning")
        error = sum(1 for d in devices if d.status == "error")
        offline = sum(1 for d in devices if d.status == "offline")
        avg_h = float(np.mean([d.health_index for d in devices])) if devices else 100.0
        critical = [d.device_id for d in devices if d.health_index < self.th_critical]

        return DeviceSummary(total_devices=total, online_count=online, warning_count=warning,
                             error_count=error, offline_count=offline, avg_health=round(avg_h, 1),
                             critical_devices=critical)

    def get_device_history(self, device_id: str, limit: int = 100) -> List[Dict]:
        hist = list(self._history.get(device_id, []))
        return hist[-limit:]

    def get_all_tickets(self) -> List[Dict]:
        return [{"ticket_id": t.ticket_id, "device_id": t.device_id, "device_name": t.device_name,
                 "priority": t.priority, "title": t.title, "description": t.description,
                 "timestamp": t.timestamp, "auto": t.auto_generated} for t in self._tickets]

    def reset(self):
        self._history.clear()
        self._tickets.clear()

    @property
    def is_initialized(self): return self._initialized

    @property
    def stats(self):
        return {"inference_count": self._inference_count, "tracked_devices": len(self._history),
                "open_tickets": len(self._tickets)}
