"""
平台核心增强模块
输变电激光监测平台 - 全自动AI巡检改造

模块列表:
- inference_engine: 深度学习推理引擎
- auto_roi_detector: 自动ROI检测器
- fusion_engine: 多证据融合引擎
- ptz_controller: 云台联动控制器
- api_routes: 增强版API路由
"""

from __future__ import annotations

# 版本
__version__ = "2.0.0"

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
]

# 延迟导入
def __getattr__(name):
    """延迟导入以减少启动时间"""
    
    # 推理引擎
    if name in ["InferenceBackend", "ModelType", "ModelConfig", "InferenceResult",
                "ONNXInferenceEngine", "ModelRegistry", "get_model_registry", 
                "register_model", "infer"]:
        from platform_core.inference_engine import (
            InferenceBackend, ModelType, ModelConfig, InferenceResult,
            ONNXInferenceEngine, ModelRegistry, get_model_registry,
            register_model, infer
        )
        return locals()[name]
    
    # 自动ROI
    if name in ["DeviceType", "ROIType", "AutoROI", "DeviceTopology",
                "AutoROIDetector", "ROITracker", "get_auto_roi_detector", "detect_rois"]:
        from platform_core.auto_roi_detector import (
            DeviceType, ROIType, AutoROI, DeviceTopology,
            AutoROIDetector, ROITracker, get_auto_roi_detector, detect_rois
        )
        return locals()[name]
    
    # 融合引擎
    if name in ["EvidenceType", "ConflictStrategy", "Evidence", "FusionResult",
                "FusionRule", "WeightedVoteFusion", "BayesianFusion",
                "EvidenceFusionEngine", "SwitchStateFusionEngine",
                "get_fusion_engine", "fuse_evidences"]:
        from platform_core.fusion_engine import (
            EvidenceType, ConflictStrategy, Evidence, FusionResult,
            FusionRule, WeightedVoteFusion, BayesianFusion,
            EvidenceFusionEngine, SwitchStateFusionEngine,
            get_fusion_engine, fuse_evidences
        )
        return locals()[name]
    
    # 云台控制
    if name in ["PTZCommand", "PTZPosition", "PresetPosition", "PatrolRoute",
                "ReshootStrategy", "BasePTZAdapter", "SimulatedPTZAdapter",
                "ONVIFPTZAdapter", "PTZController", "get_ptz_controller",
                "goto_preset", "smart_reshoot"]:
        from platform_core.ptz_controller import (
            PTZCommand, PTZPosition, PresetPosition, PatrolRoute,
            ReshootStrategy, BasePTZAdapter, SimulatedPTZAdapter,
            ONVIFPTZAdapter, PTZController, get_ptz_controller,
            goto_preset, smart_reshoot
        )
        return locals()[name]
    
    # API路由
    if name in ["create_enhanced_router", "integrate_enhanced_routes"]:
        from platform_core.api_routes import (
            create_enhanced_router, integrate_enhanced_routes
        )
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
