"""
平台核心增强模块
输变电激光监测平台 - 全自动AI巡检改造

模块列表:
- inference_engine: 深度学习推理引擎
- extended_inference_engine: 扩展推理引擎（点云/声学/时序/多模态）
- model_registry_manager: 模型注册管理器
- extended_model_registry_manager: 扩展模型注册管理器
- auto_roi_detector: 自动ROI检测器
- fusion_engine: 多证据融合引擎
- ptz_controller: 云台联动控制器
- api_routes: 增强版API路由
"""

from __future__ import annotations

# 版本
__version__ = "2.0.0"

# 显式导入 - 推理引擎
from platform_core.inference_engine import (
    InferenceBackend,
    ModelType,
    ModelConfig,
    InferenceResult,
    ONNXInferenceEngine,
    ModelRegistry,
    get_model_registry,
    register_model,
    infer,
)

# 显式导入 - 自动ROI
from platform_core.auto_roi_detector import (
    DeviceType,
    ROIType,
    AutoROI,
    DeviceTopology,
    AutoROIDetector,
    ROITracker,
    get_auto_roi_detector,
    detect_rois,
)

# 显式导入 - 融合引擎
from platform_core.fusion_engine import (
    EvidenceType,
    ConflictStrategy,
    Evidence,
    FusionResult,
    FusionRule,
    WeightedVoteFusion,
    BayesianFusion,
    EvidenceFusionEngine,
    SwitchStateFusionEngine,
    get_fusion_engine,
    fuse_evidences,
)

# 显式导入 - 云台控制
from platform_core.ptz_controller import (
    PTZCommand,
    PTZPosition,
    PresetPosition,
    PatrolRoute,
    ReshootStrategy,
    BasePTZAdapter,
    SimulatedPTZAdapter,
    ONVIFPTZAdapter,
    PTZController,
    get_ptz_controller,
    goto_preset,
    smart_reshoot,
)

# 显式导入 - API路由
from platform_core.api_routes import (
    create_enhanced_router,
    integrate_enhanced_routes,
)

# 显式导入 - 扩展推理引擎（点云/声学/时序/多模态）
from platform_core.extended_inference_engine import (
    PointCloudInferenceEngine,
    AudioInferenceEngine,
    TimeSeriesInferenceEngine,
    MultimodalFusionEngine,
    ExtendedModelType,
    ExtendedInferenceBackend,
)

# 显式导入 - 扩展模型注册管理器
from platform_core.extended_model_registry_manager import (
    ExtendedModelRegistry,
    ExtendedModelRegistryManager,
    get_extended_model_registry_manager,
    get_extended_registry,
    initialize_extended_models,
)

# 导出列表
__all__ = [
    # 推理引擎
    "InferenceBackend",
    "ModelType",
    "ModelConfig",
    "InferenceResult",
    "ONNXInferenceEngine",
    "ModelRegistry",
    "get_model_registry",
    "register_model",
    "infer",

    # 自动ROI
    "DeviceType",
    "ROIType",
    "AutoROI",
    "DeviceTopology",
    "AutoROIDetector",
    "ROITracker",
    "get_auto_roi_detector",
    "detect_rois",

    # 融合引擎
    "EvidenceType",
    "ConflictStrategy",
    "Evidence",
    "FusionResult",
    "FusionRule",
    "WeightedVoteFusion",
    "BayesianFusion",
    "EvidenceFusionEngine",
    "SwitchStateFusionEngine",
    "get_fusion_engine",
    "fuse_evidences",

    # 云台控制
    "PTZCommand",
    "PTZPosition",
    "PresetPosition",
    "PatrolRoute",
    "ReshootStrategy",
    "BasePTZAdapter",
    "SimulatedPTZAdapter",
    "ONVIFPTZAdapter",
    "PTZController",
    "get_ptz_controller",
    "goto_preset",
    "smart_reshoot",

    # API路由
    "create_enhanced_router",
    "integrate_enhanced_routes",

    # 扩展推理引擎
    "PointCloudInferenceEngine",
    "AudioInferenceEngine",
    "TimeSeriesInferenceEngine",
    "MultimodalFusionEngine",
    "ExtendedModelType",
    "ExtendedInferenceBackend",

    # 扩展模型注册管理器
    "ExtendedModelRegistry",
    "ExtendedModelRegistryManager",
    "get_extended_model_registry_manager",
    "get_extended_registry",
    "initialize_extended_models",
]
