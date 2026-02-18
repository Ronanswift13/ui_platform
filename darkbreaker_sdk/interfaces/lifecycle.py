"""
Plugin lifecycle types.

Contains PluginCapability, PluginStatus, and HealthStatus - extracted from
platform_core/plugin_manager/base.py with zero platform dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PluginCapability(str, Enum):
    """Plugin capability enumeration."""
    DEFECT_DETECTION = "defect_detection"
    STATE_RECOGNITION = "state_recognition"
    METER_READING = "meter_reading"
    THERMAL_ANALYSIS = "thermal_analysis"
    INTRUSION_DETECTION = "intrusion_detection"
    IMAGE_QUALITY = "image_quality"
    FOCUS_SUGGESTION = "focus_suggestion"
    BIRD_DETECTION = "bird_detection"
    SPECIES_IDENTIFICATION = "species_identification"
    RISK_ASSESSMENT = "risk_assessment"
    DETERRENT_CONTROL = "deterrent_control"
    PERSON_DETECTION = "person_detection"
    MULTI_TARGET_TRACKING = "multi_target_tracking"
    ZONE_INTRUSION = "zone_intrusion"
    AUTHORIZATION_CHECK = "authorization_check"
    LIDAR_FENCE = "lidar_fence"
    PARTIAL_DISCHARGE_DETECTION = "partial_discharge_detection"
    ACOUSTIC_MONITORING = "acoustic_monitoring"
    GAS_CONCENTRATION_MONITORING = "gas_concentration_monitoring"
    LEAKAGE_DETECTION = "leakage_detection"
    HYPERSPECTRAL_ANALYSIS = "hyperspectral_analysis"
    POINT_CLOUD_PROCESSING = "point_cloud_processing"
    PATH_PLANNING = "path_planning"
    MULTIMODAL_DATA_FUSION = "multimodal_data_fusion"
    ANIMAL_DETECTION = "animal_detection"
    SPECIES_CLASSIFICATION = "species_classification"
    THERMAL_FUSION_DETECTION = "thermal_fusion_detection"
    BEHAVIOR_TRACKING = "behavior_tracking"
    INTRUSION_STATISTICS = "intrusion_statistics"
    THERMAL_IMAGING = "thermal_imaging"
    HOTSPOT_DETECTION = "hotspot_detection"
    TEMPERATURE_TREND_ANALYSIS = "temperature_trend_analysis"
    HEATMAP_GENERATION = "heatmap_generation"
    TEMPERATURE_PREDICTION = "temperature_prediction"
    CROSS_MODULE_LINKAGE = "cross_module_linkage"
    DATA_ARCHIVING = "data_archiving"
    DEVICE_STATUS_MONITORING = "device_status_monitoring"
    HEALTH_INDEX_CALCULATION = "health_index_calculation"
    FAULT_PREDICTION = "fault_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    MAINTENANCE_SCHEDULING = "maintenance_scheduling"
    STATISTICS_REPORTING = "statistics_reporting"
    PROTOCOL_INTEGRATION = "protocol_integration"
    FIRE_DETECTION = "fire_detection"
    SMOKE_DETECTION = "smoke_detection"
    THERMAL_ANOMALY_DETECTION = "thermal_anomaly_detection"
    MULTI_SENSOR_FUSION = "multi_sensor_fusion"
    ACTIVE_SUPPRESSION_CONTROL = "active_suppression_control"
    EVACUATION_GUIDANCE = "evacuation_guidance"
    DRILL_SIMULATION = "drill_simulation"


class PluginStatus(str, Enum):
    """Plugin status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class HealthStatus:
    """Health status report."""
    healthy: bool
    message: str = ""
    last_check: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)
