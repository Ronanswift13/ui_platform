"""Indoor Fence V3.0 - Fusion module.

Re-exports legacy fusion classes for backward compatibility,
plus new V3 fusion components.
"""

# Legacy V2 fusion (backward compatibility)
from .legacy_fusion import (
    VisualDetection,
    LidarDetection,
    FusedDetection,
    SensorFusion,
    FusedTracker,
)

# V3 fusion components
from .ekf_fusion import EKF6DOF
from .nlos_detector import NLOSDetector
from .weight_manager import WeightManager, SensorWeight
from .sensor_fusion_v3 import SensorFusionV3

__all__ = [
    # Legacy
    "VisualDetection",
    "LidarDetection",
    "FusedDetection",
    "SensorFusion",
    "FusedTracker",
    # V3
    "EKF6DOF",
    "NLOSDetector",
    "WeightManager",
    "SensorWeight",
    "SensorFusionV3",
]
