#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内电子围栏配置管理
==========================================

职责:
- 统一加载默认配置与运行时覆盖配置
- 兼容历史字段并规范为统一结构
- 执行基础参数校验，避免错误配置直接生效
- 统一解析区域配置文件路径
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

from darkbreaker_sdk.utils import load_plugin_config


class ConfigValidationError(ValueError):
    """配置校验错误"""


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典，override 优先。"""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class IndoorFenceConfigManager:
    """电子围栏配置管理器。"""

    def __init__(self, plugin_dir: Path, config_schema: Optional[Dict[str, Any]] = None):
        self.plugin_dir = Path(plugin_dir)
        self.config_schema = config_schema or {}
        self.default_config_path = self.plugin_dir / "configs" / "default.yaml"
        self.default_zone_config_path = self.plugin_dir / "standalone" / "configs" / "zone.yaml"
        self._cached_defaults: Optional[Dict[str, Any]] = None

    def load_defaults(self) -> Dict[str, Any]:
        """加载默认配置。"""
        if self._cached_defaults is None:
            defaults = load_plugin_config(self.default_config_path)
            self._cached_defaults = self.normalize(defaults)
        return copy.deepcopy(self._cached_defaults)

    def build(
        self,
        update_config: Optional[Dict[str, Any]] = None,
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """基于默认配置构建完整运行配置。"""
        merged = self.load_defaults()
        if base_config:
            merged = deep_merge(merged, base_config)
        if update_config:
            merged = deep_merge(merged, update_config)

        normalized = self.normalize(merged)
        self.validate(normalized)
        return normalized

    def normalize(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """兼容历史配置并补全默认字段。"""
        cfg = copy.deepcopy(config) if isinstance(config, dict) else {}

        # 兼容旧字段: zone_config_path -> zone_config.path
        if "zone_config_path" in cfg and "zone_config" not in cfg:
            cfg["zone_config"] = {"path": cfg.get("zone_config_path")}

        model_cfg = cfg.setdefault("model", {})
        model_cfg.setdefault("path", "models/indoor/person_yolov8n.onnx")
        model_cfg.setdefault("device", "cpu")
        model_cfg.setdefault("input_size", [640, 640])

        infer_cfg = cfg.setdefault("inference", {})
        infer_cfg.setdefault("confidence_threshold", 0.5)
        infer_cfg.setdefault("nms_threshold", 0.45)
        infer_cfg.setdefault("tracking_enabled", True)
        infer_cfg.setdefault("max_track_age", 30)

        lidar_cfg = cfg.setdefault("lidar", {})
        lidar_cfg.setdefault("enabled", True)
        lidar_cfg.setdefault("device_ip", "192.168.1.100")
        lidar_cfg.setdefault("device_port", 2368)
        lidar_cfg.setdefault("scan_rate_hz", 10)
        lidar_cfg.setdefault("angle_min_deg", -45)
        lidar_cfg.setdefault("angle_max_deg", 45)
        lidar_cfg.setdefault("angle_offset_deg", 0.0)

        camera_cfg = cfg.setdefault("camera", {})
        camera_cfg.setdefault("enabled", True)
        camera_cfg.setdefault("source", "0")
        camera_cfg.setdefault("resolution", [640, 480])
        camera_cfg.setdefault("fps", 30)
        camera_cfg.setdefault("confidence_threshold", infer_cfg.get("confidence_threshold", 0.5))
        camera_cfg.setdefault("nms_threshold", infer_cfg.get("nms_threshold", 0.45))
        camera_cfg.setdefault("model_path", model_cfg.get("path", ""))
        camera_cfg.setdefault("height_m", 3.0)
        camera_cfg.setdefault("tilt_deg", 40.0)
        camera_cfg.setdefault("fx", 500.0)
        camera_cfg.setdefault("fy", 500.0)
        camera_cfg.setdefault("cx", 320.0)
        camera_cfg.setdefault("cy", 240.0)

        safety_cfg = cfg.setdefault("safety_zone", {})
        safety_cfg.setdefault("yellow_line_distance_m", 0.5)
        safety_cfg.setdefault("warning_distance_m", 0.3)
        safety_cfg.setdefault("danger_distance_m", 0.1)
        safety_cfg.setdefault("yellow_line_position_m", 2.0)

        auth_cfg = cfg.setdefault("cabinet_authorization", {})
        auth_cfg.setdefault("enabled", True)
        auth_cfg.setdefault("allow_list", [3, 4, 5])
        auth_cfg.setdefault("check_multi_person", True)
        auth_cfg.setdefault("max_persons_per_cabinet", 1)

        fusion_cfg = cfg.setdefault("fusion", {})
        fusion_cfg.setdefault("enabled", True)
        fusion_cfg.setdefault("max_time_diff_ms", 100)
        fusion_cfg.setdefault("angle_match_threshold_deg", 5.0)
        fusion_cfg.setdefault("distance_match_threshold_m", 0.5)

        tracking_cfg = cfg.setdefault("tracking", {})
        tracking_cfg.setdefault("enabled", infer_cfg.get("tracking_enabled", True))
        tracking_cfg.setdefault("max_age", infer_cfg.get("max_track_age", 30))
        tracking_cfg.setdefault("min_hits", 3)
        tracking_cfg.setdefault("distance_threshold", 1.0)
        tracking_cfg.setdefault("use_kalman", True)

        light_cfg = cfg.setdefault("light", {})
        light_cfg.setdefault("enabled", True)
        light_cfg.setdefault("output_type", "simulate")

        audit_cfg = cfg.setdefault("audit", {})
        audit_cfg.setdefault("enabled", True)
        audit_cfg.setdefault("log_level", "event")
        audit_cfg.setdefault("log_dir", "logs/indoor_fence")

        zone_cfg = cfg.setdefault("zone_config", {})
        if not zone_cfg.get("path") and self.default_zone_config_path.exists():
            zone_cfg["path"] = str(self.default_zone_config_path.relative_to(self.plugin_dir))
        zone_cfg.setdefault("persist_on_update", True)

        return cfg

    def validate(self, config: Dict[str, Any]) -> None:
        """执行关键参数校验。"""
        errors: list[str] = []

        def _as_float(value: Any, field_name: str) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                errors.append(f"{field_name} 必须是数字")
                return None

        safety = config.get("safety_zone", {})
        warning = _as_float(safety.get("warning_distance_m"), "safety_zone.warning_distance_m")
        danger = _as_float(safety.get("danger_distance_m"), "safety_zone.danger_distance_m")
        yellow = _as_float(safety.get("yellow_line_distance_m"), "safety_zone.yellow_line_distance_m")
        if warning is not None and warning < 0:
            errors.append("safety_zone.warning_distance_m 不能小于 0")
        if danger is not None and danger < 0:
            errors.append("safety_zone.danger_distance_m 不能小于 0")
        if yellow is not None and yellow < 0:
            errors.append("safety_zone.yellow_line_distance_m 不能小于 0")
        if warning is not None and danger is not None and warning < danger:
            errors.append("warning_distance_m 不能小于 danger_distance_m")

        lidar = config.get("lidar", {})
        if lidar.get("enabled", True):
            min_angle = _as_float(lidar.get("angle_min_deg"), "lidar.angle_min_deg")
            max_angle = _as_float(lidar.get("angle_max_deg"), "lidar.angle_max_deg")
            scan_rate = _as_float(lidar.get("scan_rate_hz"), "lidar.scan_rate_hz")
            if min_angle is not None and max_angle is not None and min_angle >= max_angle:
                errors.append("lidar.angle_min_deg 必须小于 lidar.angle_max_deg")
            if scan_rate is not None and scan_rate <= 0:
                errors.append("lidar.scan_rate_hz 必须大于 0")

        camera = config.get("camera", {})
        resolution = camera.get("resolution", [640, 480])
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            errors.append("camera.resolution 必须是长度为 2 的数组")
        else:
            try:
                w = int(resolution[0])
                h = int(resolution[1])
                if w <= 0 or h <= 0:
                    errors.append("camera.resolution 必须为正整数")
            except (TypeError, ValueError):
                errors.append("camera.resolution 必须为整数")

        fusion = config.get("fusion", {})
        max_diff = _as_float(fusion.get("max_time_diff_ms"), "fusion.max_time_diff_ms")
        if max_diff is not None and max_diff <= 0:
            errors.append("fusion.max_time_diff_ms 必须大于 0")
        dist_match = _as_float(fusion.get("distance_match_threshold_m"), "fusion.distance_match_threshold_m")
        if dist_match is not None and dist_match <= 0:
            errors.append("fusion.distance_match_threshold_m 必须大于 0")

        tracking = config.get("tracking", {})
        for field_name in ("max_age", "min_hits"):
            value = tracking.get(field_name)
            try:
                if int(value) <= 0:
                    errors.append(f"tracking.{field_name} 必须大于 0")
            except (TypeError, ValueError):
                errors.append(f"tracking.{field_name} 必须是整数")

        auth = config.get("cabinet_authorization", {})
        allow_list = auth.get("allow_list", [])
        if not isinstance(allow_list, list):
            errors.append("cabinet_authorization.allow_list 必须是数组")
        else:
            for cabinet_id in allow_list:
                if not isinstance(cabinet_id, int) or cabinet_id <= 0:
                    errors.append("cabinet_authorization.allow_list 中机柜ID必须是正整数")
                    break

        zone_path = self.resolve_zone_config_path(config)
        if zone_path is not None and zone_path.suffix.lower() not in (".yaml", ".yml"):
            errors.append("zone_config.path 必须是 YAML 文件")

        if errors:
            raise ConfigValidationError("; ".join(errors))

    def resolve_zone_config_path(self, config: Dict[str, Any]) -> Optional[Path]:
        """解析区域配置路径。"""
        zone_cfg = config.get("zone_config", {})
        raw_path = zone_cfg.get("path") or config.get("zone_config_path")
        if not raw_path:
            return None
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = (self.plugin_dir / path).resolve()
        return path

    @staticmethod
    def patch_by_key(config: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
        """按点路径更新配置字段。"""
        if not key:
            raise ConfigValidationError("配置键不能为空")

        patched = copy.deepcopy(config)
        cursor: Dict[str, Any] = patched
        parts = key.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = value
        return patched

