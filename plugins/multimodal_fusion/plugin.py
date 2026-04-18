#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合插件 V3.5 - 室外巡视增强版
=====================================

实现功能:
1. 添加 id 属性 (兼容 platform_core.plugin_manager)
2. 添加 set_status 方法
3. 支持多种构造函数签名
4. V3.5新增: 决策总线集成
5. V3.5新增: 闭环控制支持
6. V3.5新增: 证据链管理

作者: AI巡检系统
版本: 3.5.0
"""

from __future__ import annotations
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse as StarletteJSONResponse

# 导入增强版融合引擎
try:
    from .fusion_engine_enhanced import (
        EnhancedFusionEngine,
        FusionStrategyManager,
        ModalityData,
        FusionResult,
        BayesianDecisionNetwork,
        FusionStrategy as EnhancedFusionStrategy
    )
    ENHANCED_ENGINE_AVAILABLE = True
except ImportError as e:
    ENHANCED_ENGINE_AVAILABLE = False
    EnhancedFusionEngine = None
    FusionStrategyManager = None
    logging.getLogger(__name__).warning(f"增强融合引擎不可用: {e}")

from darkbreaker_sdk.interfaces import HealthStatus

logger = logging.getLogger(__name__)


class PluginStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


class FusionStrategy(str, Enum):
    EARLY = "early"
    LATE = "late"
    ATTENTION = "attention"
    HYBRID = "hybrid"


@dataclass
class MultimodalConfig:
    fusion_strategy: str = "hybrid"
    max_history_length: int = 100
    alarm_threshold: int = 3
    modalities: List[str] = field(default_factory=lambda: [
        "visual",
        "thermal",
        "acoustic",
        "gas",
        "hyperspectral",
    ])
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "visual": 0.25,
        "acoustic": 0.20,
        "gas": 0.20,
        "hyperspectral": 0.20,
        "thermal": 0.15
    })
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "warning_confidence": 0.55,
        "abnormal_confidence": 0.70,
        "critical_confidence": 0.85,
        "consensus_min_modalities": 2,
        "conflict_severity_gap": 2,
        "missing_modality_penalty": 0.03,
        "conflict_confidence_penalty": 0.10,
        "default_missing_confidence": 0.50,
    })
    evidence: Dict[str, Any] = field(default_factory=lambda: {
        "preserve_source_plugin_id": True,
        "preserve_evidence_path": True,
        "include_degradation_reasons": True,
    })
    fallback: Dict[str, Any] = field(default_factory=lambda: {
        "missing_modality_policy": "degrade",
        "missing_field_policy": "default_with_reason",
        "enhanced_engine_failure": "rule_fusion",
    })
    upgrade_placeholders: Dict[str, Any] = field(default_factory=lambda: {
        "bayesian_fusion": {
            "status": "placeholder",
            "interface": "fuse_bayesian(observations, priors=None)",
        },
        "attention_fusion": {
            "status": "placeholder",
            "interface": "fuse_attention(observations, embeddings=None)",
        },
        "temporal_fusion": {
            "status": "placeholder",
            "interface": "fuse_temporal(observations, history_window=None)",
        },
        "evidence_chain_enhancement": {
            "status": "placeholder",
            "interface": "enhance_evidence_chain(observations, context)",
        },
    })
    runtime_mode: str = "standalone"


@dataclass
class FusionObservation:
    modality: str
    label: str
    confidence: float
    severity: str
    value: Any
    evidence_path: str
    source_plugin_id: str
    component_id: str
    timestamp: Any
    task_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    degradation_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "label": self.label,
            "confidence": self.confidence,
            "severity": self.severity,
            "value": self.value,
            "evidence_path": self.evidence_path,
            "source_plugin_id": self.source_plugin_id,
            "component_id": self.component_id,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "metadata": self.metadata,
            "degradation_reasons": list(self.degradation_reasons),
        }


class MultimodalFusionPlugin:
    """多模态融合插件 - 完整修复版"""

    @classmethod
    def create_standalone(cls, config=None):
        """Create plugin instance for standalone operation."""
        plugin_dir = Path(__file__).resolve().parent
        instance = cls()
        if config is None:
            from darkbreaker_sdk.utils import load_plugin_config
            config = load_plugin_config(plugin_dir / "configs" / "default.yaml")
        if hasattr(instance, 'init'):
            instance.init(config)
        elif hasattr(instance, 'initialize'):
            instance.initialize(config)
        return instance

    PLUGIN_ID = "multimodal_fusion"
    PLUGIN_NAME = "多模态融合诊断"
    PLUGIN_VERSION = "1.0.1"

    def __init__(self, manifest=None, plugin_dir=None, config=None):
        self.manifest = manifest
        self.plugin_dir = plugin_dir if plugin_dir else Path(__file__).parent
        
        self._status: PluginStatus = PluginStatus.UNLOADED
        self._last_error: str = ""
        
        if isinstance(config, MultimodalConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = self._parse_config(config)
        elif manifest and hasattr(manifest, 'config_schema'):
            self.config = self._parse_config(manifest.config_schema or {})
        else:
            self.config = MultimodalConfig()
        
        self._model_registry = None
        self._is_initialized = False
        self._history: List[Dict] = []
        self._modality_plugins: Dict[str, Any] = {}

        # 增强融合引擎
        self.enhanced_engine: Optional[EnhancedFusionEngine] = None
        self.strategy_manager: Optional[FusionStrategyManager] = None
        self._use_enhanced_engine: bool = False

        logger.info(f"[{self.PLUGIN_NAME}] 实例已创建")
    
    def _parse_config(self, config_dict: Dict) -> MultimodalConfig:
        defaults = MultimodalConfig()
        return MultimodalConfig(
            fusion_strategy=config_dict.get("fusion_strategy", defaults.fusion_strategy),
            max_history_length=int(config_dict.get("max_history_length", defaults.max_history_length)),
            alarm_threshold=int(config_dict.get("alarm_threshold", defaults.alarm_threshold)),
            modalities=list(config_dict.get("modalities", defaults.modalities)),
            modality_weights=dict(config_dict.get("modality_weights", defaults.modality_weights)),
            thresholds=dict(config_dict.get("thresholds", defaults.thresholds)),
            evidence=dict(config_dict.get("evidence", defaults.evidence)),
            fallback=dict(config_dict.get("fallback", defaults.fallback)),
            upgrade_placeholders=dict(
                config_dict.get("upgrade_placeholders", defaults.upgrade_placeholders)
            ),
            runtime_mode=str(config_dict.get("runtime_mode", defaults.runtime_mode)),
        )
    
    # =========================================================================
    # 关键属性
    # =========================================================================
    
    @property
    def id(self) -> str:
        if self.manifest and hasattr(self.manifest, 'id'):
            return self.manifest.id
        return self.PLUGIN_ID
    
    @property
    def name(self) -> str:
        if self.manifest and hasattr(self.manifest, 'name'):
            return self.manifest.name
        return self.PLUGIN_NAME
    
    @property
    def version(self) -> str:
        if self.manifest and hasattr(self.manifest, 'version'):
            return self.manifest.version
        return self.PLUGIN_VERSION
    
    @property
    def code_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        plugin_file = self.plugin_dir / "plugin.py"
        if plugin_file.exists():
            h.update(plugin_file.read_bytes())
        return f"sha256:{h.hexdigest()[:12]}"
    
    @property
    def status(self) -> PluginStatus:
        return self._status
    
    @status.setter
    def status(self, value: PluginStatus):
        self._status = value
    
    def set_status(self, status, error: str = "") -> None:
        if isinstance(status, str):
            try:
                status = PluginStatus(status)
            except ValueError:
                status = PluginStatus.ERROR
        self._status = status
        if error:
            self._last_error = error
    
    def get_plugin_status(self) -> Dict[str, Any]:
        return {
            'plugin_id': self.id,
            'name': self.name,
            'version': self.version,
            'status': self._status.value,
            'initialized': self._is_initialized,
            'last_error': self._last_error,
            'capabilities': ["multimodal_data_fusion", "comprehensive_diagnosis", "fault_correlation"],
            'registered_modalities': list(self._modality_plugins.keys()),
            'configured_modalities': list(self.config.modalities),
            'fusion_strategy': self.config.fusion_strategy,
            'runtime_mode': "enhanced_engine" if self.is_enhanced_engine_active() else "rule_fusion",
        }
    
    # =========================================================================
    # 初始化和关闭
    # =========================================================================
    
    def init(self, config_or_registry=None) -> bool:
        try:
            self.set_status(PluginStatus.LOADING)

            if isinstance(config_or_registry, dict):
                self.config = self._parse_config(config_or_registry)
            else:
                self._model_registry = config_or_registry

            # 初始化增强融合引擎
            if ENHANCED_ENGINE_AVAILABLE and EnhancedFusionEngine is not None:
                try:
                    engine_config = {
                        'fusion_strategy': self.config.fusion_strategy,
                        'modality_weights': self.config.modality_weights,
                        'd_model': 256,
                        'n_heads': 8,
                        'dropout': 0.1
                    }
                    self.enhanced_engine = EnhancedFusionEngine(engine_config)
                    self.strategy_manager = self.enhanced_engine.strategy_manager
                    self._use_enhanced_engine = True
                    logger.info(f"[{self.PLUGIN_NAME}] 增强融合引擎初始化成功")
                except Exception as e:
                    logger.warning(f"[{self.PLUGIN_NAME}] 增强融合引擎初始化失败，使用基础引擎: {e}")
                    self._use_enhanced_engine = False

            self._is_initialized = True
            self.set_status(PluginStatus.READY)
            logger.info(f"[{self.PLUGIN_NAME}] 初始化成功")
            return True
        except Exception as e:
            self.set_status(PluginStatus.ERROR, str(e))
            logger.error(f"[{self.PLUGIN_NAME}] 初始化失败: {e}")
            return False
    
    def shutdown(self) -> bool:
        try:
            self._history.clear()
            self._modality_plugins.clear()
            self._is_initialized = False
            self.set_status(PluginStatus.UNLOADED)
            logger.info(f"[{self.PLUGIN_NAME}] 已关闭")
            return True
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 关闭失败: {e}")
            return False
    
    def cleanup(self) -> None:
        self.shutdown()
    
    def register_modality_plugin(self, modality: str, plugin: Any) -> None:
        """注册模态插件"""
        self._modality_plugins[modality] = plugin
        logger.info(f"[{self.PLUGIN_NAME}] 注册模态: {modality}")
    
    # =========================================================================
    # 核心处理方法
    # =========================================================================
    
    def process(self, inputs: Any) -> Dict[str, Any]:
        if not self._is_initialized:
            return {"success": False, "error": "插件未初始化"}

        started_at = time.time()
        try:
            context = self._build_fusion_context(inputs)
            fusion_strategy = context["fusion_strategy"]
            upstream_outputs = self._extract_upstream_outputs(inputs, context)

            if not upstream_outputs:
                return {"success": False, "error": "缺少模态数据"}

            observations = self._normalize_upstream_outputs(upstream_outputs, context)
            fusion_result = self._rule_fuse_observations(observations)
            enhanced_summary = self._try_enhanced_observation_fusion(observations, fusion_strategy)
            output = self._build_standard_fusion_output(
                context=context,
                observations=observations,
                fusion_result=fusion_result,
                enhanced_summary=enhanced_summary,
                processing_time_ms=(time.time() - started_at) * 1000,
            )

            modality_results = {
                observation.modality: {
                    "success": True,
                    "status": observation.severity,
                    "overall_status": observation.severity,
                    "confidence": observation.confidence,
                    "data": observation.to_dict(),
                }
                for observation in observations
            }
            self._update_history(context["device_id"], modality_results, {
                "status": output["fused_status"],
                "confidence": output["fusion_confidence"],
            })

            return output
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 处理失败: {e}")
            return {"success": False, "error": str(e)}

    def fuse(self, inputs: Any) -> Dict[str, Any]:
        """兼容融合调用入口。"""
        return self.process(inputs)

    def _build_fusion_context(self, inputs: Any) -> Dict[str, Any]:
        if isinstance(inputs, list) and inputs:
            payload = self._to_plain_dict(inputs[0])
        else:
            payload = inputs if isinstance(inputs, Mapping) else {}
        context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        first_result = {}
        if isinstance(payload.get("results"), list) and payload["results"]:
            first_result = self._to_plain_dict(payload["results"][0])
        timestamp = payload.get("timestamp", time.time())
        return {
            "task_id": str(context.get("task_id") or payload.get("task_id") or metadata.get("task_id") or "multimodal-fusion-task"),
            "site_id": str(context.get("site_id") or payload.get("site_id") or metadata.get("site_id") or first_result.get("site_id") or "virtual-site"),
            "device_id": str(context.get("device_id") or payload.get("device_id") or metadata.get("device_id") or first_result.get("device_id") or "unknown"),
            "timestamp": timestamp,
            "fusion_strategy": str(payload.get("fusion_strategy", self.config.fusion_strategy)),
        }

    def _extract_upstream_outputs(self, inputs: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if inputs is None:
            return []
        if isinstance(inputs, list):
            return [self._to_plain_dict(item) for item in inputs]
        if hasattr(inputs, "model_dump"):
            return [self._to_plain_dict(inputs)]
        if not isinstance(inputs, Mapping):
            return []

        for key in ("plugin_outputs", "outputs", "upstream_outputs"):
            value = inputs.get(key)
            if isinstance(value, list):
                return [self._to_plain_dict(item) for item in value]

        modality_results = inputs.get("modality_results")
        if isinstance(modality_results, list):
            return [self._to_plain_dict(item) for item in modality_results]
        if isinstance(modality_results, Mapping):
            extracted = []
            for modality, value in modality_results.items():
                output = self._to_plain_dict(value)
                output.setdefault("modality", modality)
                extracted.append(output)
            return extracted

        modalities = inputs.get("modalities")
        if isinstance(modalities, Mapping):
            processed = self._process_modalities(dict(modalities))
            return [
                self._legacy_modality_result_to_output(modality, result, context)
                for modality, result in processed.items()
            ]

        if inputs.get("modality") or inputs.get("results") or inputs.get("plugin_id"):
            return [self._to_plain_dict(inputs)]

        return []

    def _legacy_modality_result_to_output(
        self,
        modality: str,
        result: Mapping[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = self._to_plain_dict(result)
        data = result.get("data") if isinstance(result.get("data"), Mapping) else result
        data = self._to_plain_dict(data)
        metadata = dict(data.get("metadata", {}) if isinstance(data.get("metadata"), Mapping) else {})
        metadata.update(result.get("metadata", {}) if isinstance(result.get("metadata"), Mapping) else {})
        if "simulated" in data:
            metadata["simulated"] = data.get("simulated")

        return {
            "modality": modality,
            "plugin_id": (
                result.get("plugin_id")
                or data.get("plugin_id")
                or metadata.get("source_plugin_id")
                or f"{modality}_direct_input"
            ),
            "task_id": result.get("task_id") or data.get("task_id") or context["task_id"],
            "timestamp": result.get("timestamp") or data.get("timestamp") or context["timestamp"],
            "success": result.get("success", True),
            "status": result.get("status") or result.get("overall_status") or data.get("status") or data.get("overall_status"),
            "label": result.get("label") or data.get("label"),
            "confidence": result.get("confidence", data.get("confidence")),
            "severity": result.get("severity", data.get("severity")),
            "value": result.get("value", data.get("value")),
            "component_id": result.get("component_id") or data.get("component_id") or data.get("channel_id"),
            "evidence_path": result.get("evidence_path") or data.get("evidence_path", ""),
            "results": result.get("results", data.get("results", [])),
            "alarms": result.get("alarms", data.get("alarms", [])),
            "metadata": metadata,
            "error_message": result.get("error_message") or result.get("error") or data.get("error_message"),
        }

    def _normalize_upstream_outputs(
        self,
        upstream_outputs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[FusionObservation]:
        observations: List[FusionObservation] = []
        for output in upstream_outputs:
            observations.extend(self._normalize_single_output(output, context))
        return observations

    def _normalize_single_output(
        self,
        output: Mapping[str, Any],
        context: Dict[str, Any],
    ) -> List[FusionObservation]:
        data = self._to_plain_dict(output)
        metadata = dict(data.get("metadata", {}) if isinstance(data.get("metadata"), Mapping) else {})
        modality = (
            data.get("modality")
            or metadata.get("modality")
            or self._infer_modality_from_plugin_id(data.get("plugin_id", ""))
            or "unknown"
        )
        source_plugin_id = (
            data.get("plugin_id")
            or data.get("source_plugin_id")
            or metadata.get("source_plugin_id")
            or f"{modality}_unknown"
        )
        task_id = str(data.get("task_id") or context["task_id"])
        timestamp = data.get("timestamp") or metadata.get("timestamp") or context["timestamp"]

        root_reasons: List[str] = []
        if modality == "unknown":
            root_reasons.append("missing_modality")
        if not data.get("plugin_id"):
            root_reasons.append("missing_plugin_id")
        if not data.get("task_id"):
            root_reasons.append("missing_task_id_defaulted")
        if not data.get("timestamp"):
            root_reasons.append("missing_timestamp_defaulted")
        if data.get("success") is False:
            root_reasons.append(data.get("error_message") or data.get("error") or "upstream_success_false")

        results = data.get("results")
        if isinstance(results, list) and results:
            return [
                self._observation_from_result(
                    result=item,
                    root=data,
                    modality=str(modality),
                    source_plugin_id=str(source_plugin_id),
                    task_id=task_id,
                    timestamp=timestamp,
                    inherited_reasons=root_reasons,
                )
                for item in results
            ]

        return [
            self._observation_from_result(
                result=data,
                root=data,
                modality=str(modality),
                source_plugin_id=str(source_plugin_id),
                task_id=task_id,
                timestamp=timestamp,
                inherited_reasons=root_reasons + ["missing_results_defaulted_to_root"],
            )
        ]

    def _observation_from_result(
        self,
        result: Any,
        root: Mapping[str, Any],
        modality: str,
        source_plugin_id: str,
        task_id: str,
        timestamp: Any,
        inherited_reasons: List[str],
    ) -> FusionObservation:
        item = self._to_plain_dict(result)
        metadata = dict(root.get("metadata", {}) if isinstance(root.get("metadata"), Mapping) else {})
        metadata.update(item.get("metadata", {}) if isinstance(item.get("metadata"), Mapping) else {})
        reasons = list(inherited_reasons)

        label = item.get("label") or root.get("label")
        status = (
            item.get("severity")
            or item.get("status")
            or item.get("overall_status")
            or root.get("severity")
            or root.get("status")
            or root.get("overall_status")
            or metadata.get("severity")
        )
        if not label:
            label = self._label_from_status(status)
            reasons.append("missing_label_defaulted")

        confidence_raw = item.get("confidence", root.get("confidence"))
        if confidence_raw is None:
            confidence_raw = self.config.thresholds.get("default_missing_confidence", 0.5)
            reasons.append("missing_confidence_defaulted")
        confidence = self._clamp_confidence(confidence_raw)

        severity = self._normalize_fused_status(status, label, confidence)
        value = item.get("value", root.get("value"))

        evidence_path = item.get("evidence_path") or root.get("evidence_path") or metadata.get("evidence_path") or ""
        if not evidence_path:
            reasons.append("missing_evidence_path")

        component_id = item.get("component_id") or root.get("component_id") or metadata.get("component_id") or ""
        if not component_id:
            component_id = "unknown_component"
            reasons.append("missing_component_id_defaulted")

        if item.get("failure_reason"):
            reasons.append(str(item["failure_reason"]))

        return FusionObservation(
            modality=modality,
            label=str(label),
            confidence=confidence,
            severity=severity,
            value=self._safe_json_value(value),
            evidence_path=str(evidence_path),
            source_plugin_id=source_plugin_id,
            component_id=str(component_id),
            timestamp=timestamp,
            task_id=task_id,
            metadata=self._safe_json_value(metadata),
            degradation_reasons=reasons,
        )

    def _rule_fuse_observations(self, observations: List[FusionObservation]) -> Dict[str, Any]:
        expected_modalities = list(self.config.modalities)
        contributing_modalities = sorted({obs.modality for obs in observations if obs.modality != "unknown"})
        missing_modalities = [m for m in expected_modalities if m not in contributing_modalities]
        if not observations:
            return {
                "fused_label": "unknown",
                "fused_status": "unknown",
                "fusion_confidence": 0.0,
                "contributing_modalities": contributing_modalities,
                "missing_modalities": missing_modalities,
                "conflict_status": "insufficient_evidence",
                "evidence_chain": [],
                "recommended_actions": ["融合输入为空，建议检查上游插件输出链路"],
                "modality_contributions": {},
            }

        status_scores = [self._status_score(obs.severity) for obs in observations]
        max_score = max(status_scores)
        min_score = min(status_scores)
        conflict_gap = int(self.config.thresholds.get("conflict_severity_gap", 2))
        conflict_status = "conflict_detected" if max_score - min_score >= conflict_gap else "none"

        abnormal_observations = [obs for obs in observations if self._status_score(obs.severity) >= 3]
        label_counts: Dict[str, int] = {}
        for obs in abnormal_observations:
            label_counts[obs.label] = label_counts.get(obs.label, 0) + 1
        consensus_min = int(self.config.thresholds.get("consensus_min_modalities", 2))
        consensus_label = max(label_counts, key=label_counts.get) if label_counts else ""
        has_abnormal_consensus = bool(consensus_label and label_counts[consensus_label] >= consensus_min)

        top_observation = max(observations, key=lambda obs: (self._status_score(obs.severity), obs.confidence))
        if has_abnormal_consensus:
            fused_status = "critical"
            fused_label = consensus_label
        elif max_score >= 4:
            fused_status = "critical"
            fused_label = top_observation.label
        elif max_score >= 3:
            fused_status = "abnormal"
            fused_label = top_observation.label
        elif max_score >= 2:
            fused_status = "warning"
            fused_label = top_observation.label
        elif max_score <= 0:
            fused_status = "normal"
            fused_label = "normal"
        else:
            fused_status = "unknown"
            fused_label = top_observation.label or "unknown"

        modality_contributions = self._calculate_observation_contributions(observations)
        weighted_confidence = self._weighted_confidence(observations)
        missing_penalty = float(self.config.thresholds.get("missing_modality_penalty", 0.03))
        if expected_modalities:
            weighted_confidence -= missing_penalty * (len(missing_modalities) / len(expected_modalities))
        if conflict_status == "conflict_detected":
            weighted_confidence -= float(self.config.thresholds.get("conflict_confidence_penalty", 0.10))
        fusion_confidence = self._clamp_confidence(weighted_confidence)

        evidence_chain = [self._build_evidence_item(obs) for obs in observations]
        recommended_actions = self._recommended_actions(fused_status, conflict_status, missing_modalities)

        return {
            "fused_label": fused_label,
            "fused_status": fused_status,
            "fusion_confidence": fusion_confidence,
            "contributing_modalities": contributing_modalities,
            "missing_modalities": missing_modalities,
            "conflict_status": conflict_status,
            "evidence_chain": evidence_chain,
            "recommended_actions": recommended_actions,
            "modality_contributions": modality_contributions,
        }

    def _try_enhanced_observation_fusion(
        self,
        observations: List[FusionObservation],
        fusion_strategy: str,
    ) -> Dict[str, Any]:
        if not (self._use_enhanced_engine and self.enhanced_engine is not None and observations):
            return {
                "attempted": False,
                "runtime_mode": "rule_fusion",
                "reason": "enhanced_engine_unavailable_or_disabled",
            }

        try:
            modality_data_dict: Dict[str, Any] = {}
            for observation in observations:
                features = np.array(
                    [self._status_score(observation.severity) / 4.0, observation.confidence],
                    dtype=np.float32,
                )
                modality_data_dict[observation.modality] = ModalityData(
                    modality_type=observation.modality,
                    features=features,
                    confidence=observation.confidence,
                    detections=[observation.to_dict()],
                    metadata=observation.metadata,
                    timestamp=time.time(),
                    quality_score=1.0 if not observation.degradation_reasons else 0.7,
                )

            self.enhanced_engine.set_strategy(fusion_strategy)
            enhanced = self.enhanced_engine.fuse(modality_data_dict)
            enhanced_success = bool(getattr(enhanced, "success", False))
            if not enhanced_success:
                self._use_enhanced_engine = False
            return {
                "attempted": True,
                "success": enhanced_success,
                "runtime_mode": "enhanced_engine" if enhanced_success else "rule_fusion",
                "overall_status": getattr(enhanced, "overall_status", "unknown"),
                "confidence": self._clamp_confidence(getattr(enhanced, "confidence", 0.0)),
                "modality_contributions": getattr(enhanced, "modality_contributions", {}),
                "fault_chain": getattr(enhanced, "fault_chain", []),
                "bayesian_inference": getattr(enhanced, "bayesian_inference", {}),
                "reason": "" if enhanced_success else "enhanced_engine_returned_unsuccessful_result",
            }
        except Exception as e:
            logger.warning(f"增强引擎处理失败，回退到规则融合: {e}")
            self._use_enhanced_engine = False
            return {
                "attempted": True,
                "success": False,
                "runtime_mode": "rule_fusion",
                "reason": str(e),
            }

    def _build_standard_fusion_output(
        self,
        context: Dict[str, Any],
        observations: List[FusionObservation],
        fusion_result: Dict[str, Any],
        enhanced_summary: Dict[str, Any],
        processing_time_ms: float,
    ) -> Dict[str, Any]:
        metadata = {
            "runtime_mode": enhanced_summary.get("runtime_mode", "rule_fusion"),
            "fusion_strategy": context["fusion_strategy"],
            "modality_weights": dict(self.config.modality_weights),
            "algorithm_stage": "stage_1_rule_fusion_contract",
            "upgrade_placeholders": dict(self.config.upgrade_placeholders),
            "fallback": dict(self.config.fallback),
            "thresholds": dict(self.config.thresholds),
            "enhanced_engine_available": bool(self.enhanced_engine is not None),
            "enhanced_engine_attempted": bool(enhanced_summary.get("attempted", False)),
            "enhanced_engine": enhanced_summary,
            "degradation_reasons": self._collect_degradation_reasons(observations),
        }
        result = self._build_recognition_result(context, fusion_result, metadata)
        alarms = self._build_alarm_outputs(context, fusion_result)
        diagnostic_report = {
            "timestamp": time.time(),
            "fusion_method": context["fusion_strategy"],
            "overall_status": fusion_result["fused_status"],
            "overall_confidence": fusion_result["fusion_confidence"],
            "modality_summary": {
                obs.modality: {
                    "status": obs.severity,
                    "label": obs.label,
                    "confidence": obs.confidence,
                    "source_plugin_id": obs.source_plugin_id,
                    "evidence_path": obs.evidence_path,
                }
                for obs in observations
            },
            "fault_analysis": fusion_result["evidence_chain"],
            "correlation_analysis": [{
                "conflict_status": fusion_result["conflict_status"],
                "missing_modalities": fusion_result["missing_modalities"],
            }],
        }

        return {
            "success": True,
            "plugin_id": self.id,
            "plugin_version": self.version,
            "code_hash": self.code_hash,
            "task_id": context["task_id"],
            "site_id": context["site_id"],
            "device_id": context["device_id"],
            "timestamp": time.time(),
            "fused_label": fusion_result["fused_label"],
            "fused_status": fusion_result["fused_status"],
            "fusion_confidence": fusion_result["fusion_confidence"],
            "contributing_modalities": fusion_result["contributing_modalities"],
            "missing_modalities": fusion_result["missing_modalities"],
            "conflict_status": fusion_result["conflict_status"],
            "evidence_chain": fusion_result["evidence_chain"],
            "recommended_actions": fusion_result["recommended_actions"],
            "metadata": metadata,
            "results": [result],
            "alarms": alarms,
            "processing_time_ms": processing_time_ms,
            "overall_status": fusion_result["fused_status"],
            "confidence": fusion_result["fusion_confidence"],
            "modality_contributions": fusion_result["modality_contributions"],
            "detections": [result],
            "fused_detections": [result],
            "diagnostic_report": diagnostic_report,
            "recommendations": fusion_result["recommended_actions"],
            "engine": metadata["runtime_mode"],
        }

    def _to_plain_dict(self, value: Any) -> Dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, Mapping):
            return {str(k): self._safe_json_value(v) for k, v in value.items()}
        return {}

    def _safe_json_value(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, Mapping):
            return {str(k): self._safe_json_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._safe_json_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._safe_json_value(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "value") and not isinstance(value, (str, bytes)):
            return value.value
        return value

    def _infer_modality_from_plugin_id(self, plugin_id: Any) -> str:
        text = str(plugin_id or "").lower()
        aliases = {
            "visual": ("visual", "transformer", "busbar", "switch", "meter", "camera"),
            "thermal": ("thermal", "temperature"),
            "acoustic": ("acoustic", "audio", "ultrasonic"),
            "gas": ("gas", "dga", "sf6"),
            "hyperspectral": ("hyperspectral", "spectrum", "spectral"),
        }
        for modality, tokens in aliases.items():
            if any(token in text for token in tokens):
                return modality
        return ""

    def _label_from_status(self, status: Any) -> str:
        normalized = str(status or "unknown").lower()
        if normalized in ("normal", "ok", "healthy"):
            return "normal"
        if normalized in ("warning", "attention"):
            return "warning"
        if normalized in ("critical", "alarm", "abnormal", "error"):
            return "abnormal"
        return "unknown"

    def _normalize_fused_status(self, status: Any, label: Any, confidence: float) -> str:
        raw = str(status or "").lower()
        if raw in ("normal", "ok", "healthy"):
            return "normal"
        if raw in ("warning", "attention"):
            return "warning"
        if raw in ("critical", "fatal"):
            return "critical"
        if raw in ("alarm", "abnormal", "error", "fault"):
            return "abnormal"
        if raw in ("unknown", "none"):
            return "unknown"

        label_text = str(label or "").lower()
        if label_text in ("normal", "ok", "healthy"):
            return "normal"
        if label_text in ("unknown", ""):
            return "unknown"
        if confidence >= float(self.config.thresholds.get("critical_confidence", 0.85)):
            return "abnormal"
        if confidence >= float(self.config.thresholds.get("warning_confidence", 0.55)):
            return "warning"
        return "unknown"

    def _status_score(self, status: str) -> int:
        return {
            "unknown": 0,
            "normal": 0,
            "warning": 2,
            "abnormal": 3,
            "critical": 4,
        }.get(str(status or "unknown").lower(), 0)

    def _clamp_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = float(self.config.thresholds.get("default_missing_confidence", 0.5))
        return round(max(0.0, min(1.0, confidence)), 4)

    def _calculate_observation_contributions(
        self,
        observations: List[FusionObservation],
    ) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        for observation in observations:
            weight = float(self.config.modality_weights.get(observation.modality, 0.1))
            raw[observation.modality] = raw.get(observation.modality, 0.0) + weight * observation.confidence
        total = sum(raw.values())
        if total <= 0:
            return {key: 0.0 for key in raw}
        return {key: round(value / total * 100, 1) for key, value in raw.items()}

    def _weighted_confidence(self, observations: List[FusionObservation]) -> float:
        weighted_total = 0.0
        weight_total = 0.0
        for observation in observations:
            weight = float(self.config.modality_weights.get(observation.modality, 0.1))
            severity_factor = max(1.0, self._status_score(observation.severity)) / 4.0
            weighted_total += weight * observation.confidence * severity_factor
            weight_total += weight * severity_factor
        if weight_total <= 0:
            return 0.0
        return weighted_total / weight_total

    def _build_evidence_item(self, observation: FusionObservation) -> Dict[str, Any]:
        return {
            "modality": observation.modality,
            "label": observation.label,
            "confidence": observation.confidence,
            "severity": observation.severity,
            "value": observation.value,
            "evidence_path": observation.evidence_path,
            "source_plugin_id": observation.source_plugin_id,
            "component_id": observation.component_id,
            "timestamp": observation.timestamp,
            "simulated": bool(observation.metadata.get("simulated", False)),
            "degradation_reasons": list(observation.degradation_reasons),
        }

    def _recommended_actions(
        self,
        fused_status: str,
        conflict_status: str,
        missing_modalities: List[str],
    ) -> List[str]:
        actions = {
            "critical": ["严重多模态异常，建议立即复核现场并升级处置"],
            "abnormal": ["检测到融合异常，建议安排人工复核并补充证据"],
            "warning": ["存在预警信号，建议缩短巡检间隔并关注趋势"],
            "normal": ["融合结果正常，建议继续按计划监测"],
            "unknown": ["融合证据不足，建议检查上游模态输出"],
        }.get(fused_status, ["融合状态未知，建议检查输入数据"])
        result = list(actions)
        if conflict_status == "conflict_detected":
            result.append("模态间存在冲突，建议优先复核高置信度异常模态证据")
        if missing_modalities:
            result.append(f"缺失模态已降级处理: {', '.join(missing_modalities)}")
        return result

    def _build_recognition_result(
        self,
        context: Dict[str, Any],
        fusion_result: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        evidence_path = ""
        component_id = "multimodal_fusion"
        for item in fusion_result["evidence_chain"]:
            if item.get("evidence_path") and not evidence_path:
                evidence_path = item["evidence_path"]
            if item.get("component_id") and item.get("component_id") != "unknown_component":
                component_id = item["component_id"]
                break

        return {
            "task_id": context["task_id"],
            "site_id": context["site_id"],
            "device_id": context["device_id"],
            "component_id": component_id,
            "roi_id": context["device_id"],
            "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
            "label": fusion_result["fused_label"],
            "value": {
                "fused_status": fusion_result["fused_status"],
                "contributing_modalities": fusion_result["contributing_modalities"],
                "missing_modalities": fusion_result["missing_modalities"],
                "conflict_status": fusion_result["conflict_status"],
            },
            "confidence": fusion_result["fusion_confidence"],
            "evidence_path": evidence_path,
            "model_version": metadata["algorithm_stage"],
            "code_version": self.code_hash,
            "timestamp": self._timestamp_to_iso(context["timestamp"]),
            "metadata": metadata,
            "failure_reason": None,
        }

    def _build_alarm_outputs(
        self,
        context: Dict[str, Any],
        fusion_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        fused_status = fusion_result["fused_status"]
        if fused_status not in ("warning", "abnormal", "critical"):
            return []
        evidence_path = ""
        for item in fusion_result["evidence_chain"]:
            if item.get("evidence_path"):
                evidence_path = item["evidence_path"]
                break
        return [{
            "task_id": context["task_id"],
            "result_id": None,
            "rule_id": "multimodal_fusion_stage_1",
            "level": self._status_to_alarm_level(fused_status),
            "status": "active",
            "title": f"多模态融合诊断: {fused_status}",
            "message": (
                f"融合标签 {fusion_result['fused_label']}，"
                f"置信度 {fusion_result['fusion_confidence']:.2f}"
            ),
            "site_id": context["site_id"],
            "device_id": context["device_id"],
            "component_id": "multimodal_fusion",
            "evidence_path": evidence_path,
            "created_at": self._timestamp_to_iso(time.time()),
            "acknowledged_at": None,
            "resolved_at": None,
            "acknowledged_by": "",
            "resolved_by": "",
            "notes": "",
        }]

    def _status_to_alarm_level(self, status: str) -> str:
        if status == "critical":
            return "critical"
        if status == "abnormal":
            return "error"
        return "warning"

    def _collect_degradation_reasons(
        self,
        observations: List[FusionObservation],
    ) -> List[Dict[str, Any]]:
        reasons = []
        for observation in observations:
            if observation.degradation_reasons:
                reasons.append({
                    "modality": observation.modality,
                    "source_plugin_id": observation.source_plugin_id,
                    "reasons": list(observation.degradation_reasons),
                })
        return reasons

    def _timestamp_to_iso(self, value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value)).isoformat()
            except (OSError, OverflowError, ValueError):
                return datetime.now().isoformat()
        if isinstance(value, str) and value:
            return value
        return datetime.now().isoformat()

    def _process_with_enhanced_engine(self, device_id: str, modality_results: Dict, fusion_strategy: str) -> Dict[str, Any]:
        """使用增强融合引擎处理数据"""
        try:
            # 转换模态数据格式 - 使用 Dict[str, ModalityData] 格式
            modality_data_dict: Dict[str, Any] = {}
            for modality_type, result in modality_results.items():
                if result.get("success", False):
                    features = result.get("features", result.get("data", {}))
                    if isinstance(features, dict):
                        features = np.array(list(features.values()) if features else [0.0], dtype=np.float32)
                    elif isinstance(features, list):
                        features = np.array(features, dtype=np.float32)
                    elif not isinstance(features, np.ndarray):
                        features = np.array([float(features)] if features else [0.0], dtype=np.float32)
                    else:
                        features = features.astype(np.float32)

                    modality_data_dict[modality_type] = ModalityData(
                        modality_type=modality_type,
                        features=features,
                        confidence=result.get("confidence", 0.8),
                        detections=result.get("detections", []),
                        metadata=result.get("metadata", {}),
                        timestamp=time.time(),
                        quality_score=result.get("quality_score", 1.0)
                    )

            # 使用增强引擎进行融合
            if self.enhanced_engine is not None:
                # 先切换策略
                self.enhanced_engine.set_strategy(fusion_strategy)

                # 执行融合
                fusion_result = self.enhanced_engine.fuse(modality_data_dict)

                # 更新历史
                self._update_history(device_id, modality_results, {
                    "status": fusion_result.overall_status,
                    "confidence": fusion_result.confidence
                })

                # 生成告警
                alarms = self._generate_alarms(device_id, fusion_result.diagnostic_report)

                return {
                    "success": fusion_result.success,
                    "device_id": device_id,
                    "timestamp": time.time(),
                    "overall_status": fusion_result.overall_status,
                    "confidence": fusion_result.confidence,
                    "modality_contributions": fusion_result.modality_contributions,
                    "detections": fusion_result.fused_detections,
                    "fused_detections": fusion_result.fused_detections,
                    "diagnostic_report": fusion_result.diagnostic_report,
                    "recommendations": fusion_result.recommendations,
                    "fault_chain": fusion_result.fault_chain,
                    "bayesian_inference": fusion_result.bayesian_inference,
                    "alarms": alarms,
                    "engine": "enhanced"
                }

            return {"success": False, "error": "增强引擎不可用"}
        except Exception as e:
            logger.warning(f"增强引擎处理失败，回退到基础引擎: {e}")
            # 回退到基础融合
            self._use_enhanced_engine = False
            return self.process({
                "device_id": device_id,
                "modalities": {k: v.get("data", v) for k, v in modality_results.items()},
                "fusion_strategy": fusion_strategy
            })
    
    def _process_modalities(self, modalities: Dict) -> Dict[str, Dict]:
        """处理各模态数据"""
        results = {}
        
        for modality, data in modalities.items():
            if modality in self._modality_plugins:
                plugin = self._modality_plugins[modality]
                try:
                    result = plugin.process(data)
                    results[modality] = result
                except Exception as e:
                    logger.warning(f"模态 {modality} 处理失败: {e}")
                    results[modality] = {"success": False, "error": str(e)}
            else:
                # 直接使用输入数据作为结果
                results[modality] = {
                    "success": True,
                    "data": data,
                    "confidence": data.get("confidence", 0.8) if isinstance(data, dict) else 0.8
                }
        
        return results
    
    def _early_fusion(self, modality_results: Dict) -> Dict:
        """早期融合（特征级）"""
        features = []
        for modality, result in modality_results.items():
            if result.get("success", False):
                feat = result.get("features", result.get("data", {}))
                if isinstance(feat, (list, np.ndarray)):
                    features.extend(feat if isinstance(feat, list) else feat.tolist())
        
        return {
            "status": "normal",
            "confidence": 0.85,
            "detections": [],
            "method": "early_fusion"
        }
    
    def _late_fusion(self, modality_results: Dict) -> Dict:
        """晚期融合（决策级）"""
        decisions = []
        for modality, result in modality_results.items():
            if result.get("success", False):
                status = result.get("status", result.get("overall_status", "normal"))
                conf = result.get("confidence", 0.8)
                decisions.append({"modality": modality, "status": status, "confidence": conf})
        
        # 投票决策
        status_counts = {}
        for d in decisions:
            s = d["status"]
            status_counts[s] = status_counts.get(s, 0) + d["confidence"]
        
        final_status = max(status_counts, key=status_counts.get) if status_counts else "normal"
        
        return {
            "status": final_status,
            "confidence": max(status_counts.values()) / len(decisions) if decisions else 0.5,
            "detections": [],
            "method": "late_fusion",
            "decisions": decisions
        }
    
    def _attention_fusion(self, modality_results: Dict) -> Dict:
        """注意力融合"""
        weighted_scores = {}
        total_weight = 0
        
        for modality, result in modality_results.items():
            if result.get("success", False):
                weight = self.config.modality_weights.get(modality, 0.2)
                conf = result.get("confidence", 0.8)
                weighted_scores[modality] = weight * conf
                total_weight += weight
        
        # 归一化
        if total_weight > 0:
            for m in weighted_scores:
                weighted_scores[m] /= total_weight
        
        avg_confidence = sum(weighted_scores.values()) if weighted_scores else 0.5
        
        return {
            "status": "normal" if avg_confidence > 0.7 else "warning",
            "confidence": avg_confidence,
            "detections": [],
            "method": "attention_fusion",
            "attention_weights": weighted_scores
        }
    
    def _hybrid_fusion(self, modality_results: Dict) -> Dict:
        """混合融合"""
        early_result = self._early_fusion(modality_results)
        late_result = self._late_fusion(modality_results)
        attention_result = self._attention_fusion(modality_results)
        
        # 组合结果
        confidences = [
            early_result["confidence"],
            late_result["confidence"],
            attention_result["confidence"]
        ]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 确定最终状态
        statuses = [early_result["status"], late_result["status"], attention_result["status"]]
        severity_order = ["critical", "alarm", "warning", "attention", "normal"]
        
        final_status = "normal"
        for severity in severity_order:
            if severity in statuses:
                final_status = severity
                break
        
        return {
            "status": final_status,
            "confidence": avg_confidence,
            "detections": [],
            "method": "hybrid_fusion",
            "sub_results": {
                "early": early_result,
                "late": late_result,
                "attention": attention_result
            }
        }
    
    def _generate_diagnostic_report(self, modality_results: Dict, fusion_result: Dict) -> Dict:
        """生成诊断报告"""
        report = {
            "timestamp": time.time(),
            "fusion_method": fusion_result.get("method", "unknown"),
            "overall_status": fusion_result.get("status", "normal"),
            "overall_confidence": fusion_result.get("confidence", 0.0),
            "modality_summary": {},
            "fault_analysis": [],
            "correlation_analysis": []
        }
        
        for modality, result in modality_results.items():
            report["modality_summary"][modality] = {
                "success": result.get("success", False),
                "status": result.get("status", result.get("overall_status", "unknown")),
                "confidence": result.get("confidence", 0.0)
            }
        
        return report
    
    def _calculate_contributions(self, modality_results: Dict) -> Dict[str, float]:
        """计算各模态贡献度"""
        contributions = {}
        total = 0
        
        for modality, result in modality_results.items():
            if result.get("success", False):
                weight = self.config.modality_weights.get(modality, 0.2)
                conf = result.get("confidence", 0.8)
                contributions[modality] = weight * conf
                total += contributions[modality]
        
        # 归一化
        if total > 0:
            for m in contributions:
                contributions[m] = round(contributions[m] / total * 100, 1)
        
        return contributions
    
    def _generate_recommendations(self, diagnostic_report: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        status = diagnostic_report.get("overall_status", "normal")
        
        status_recommendations = {
            "critical": "紧急: 发现严重异常，建议立即停机检查",
            "alarm": "告警: 检测到多模态异常，建议尽快安排检修",
            "warning": "注意: 部分指标异常，建议加强监测"
        }
        
        if status in status_recommendations:
            recommendations.append(status_recommendations[status])
        
        if not recommendations:
            recommendations.append("设备运行状态良好，建议继续定期监测")
        
        return recommendations
    
    def _generate_alarms(self, device_id: str, diagnostic_report: Dict) -> List[Dict]:
        """生成告警"""
        alarms = []
        status = diagnostic_report.get("overall_status", "normal")
        
        if status in ["critical", "alarm", "warning"]:
            alarms.append({
                "type": "multimodal_fusion",
                "level": status,
                "device_id": device_id,
                "timestamp": time.time(),
                "message": f"多模态融合诊断: {status}",
                "details": diagnostic_report.get("modality_summary", {})
            })
        
        return alarms
    
    def _update_history(self, device_id: str, modality_results: Dict, fusion_result: Dict) -> None:
        """更新历史记录"""
        self._history.append({
            "device_id": device_id,
            "timestamp": time.time(),
            "status": fusion_result.get("status"),
            "confidence": fusion_result.get("confidence")
        })
        
        # 限制历史长度
        if len(self._history) > self.config.max_history_length:
            self._history = self._history[-self.config.max_history_length:]
    
    # =========================================================================
    # BasePlugin 兼容方法
    # =========================================================================
    
    def infer(self, frame, rois, context):
        return []

    def postprocess(self, results, rules):
        return []

    def healthcheck(self):
        return HealthStatus(
            healthy=self._is_initialized,
            message="OK" if self._is_initialized else "未初始化",
            details={
                "fusion_strategy": self.config.fusion_strategy,
                "modalities": list(self.config.modalities),
                "runtime_mode": "enhanced_engine" if self.is_enhanced_engine_active() else "rule_fusion",
            },
        )

    def _build_smoke_fixture(self) -> Dict[str, Any]:
        return {
            "device_id": "fusion_smoke_device",
            "task_id": "fusion-smoke",
            "site_id": "standalone",
            "fusion_strategy": self.config.fusion_strategy,
            "plugin_outputs": [
                {
                    "modality": "visual",
                    "plugin_id": "visual_fixture_simulated",
                    "task_id": "fusion-smoke",
                    "timestamp": time.time(),
                    "results": [{
                        "label": "normal",
                        "confidence": 0.86,
                        "severity": "normal",
                        "value": {"defects": 0},
                        "evidence_path": "fixtures/simulated/visual_normal.jpg",
                        "component_id": "bushing_a",
                        "metadata": {"simulated": True},
                    }],
                    "alarms": [],
                    "metadata": {"simulated": True},
                },
                {
                    "modality": "gas",
                    "plugin_id": "gas_detection_fixture_simulated",
                    "task_id": "fusion-smoke",
                    "timestamp": time.time(),
                    "results": [{
                        "label": "gas_threshold_warning",
                        "confidence": 0.72,
                        "severity": "warning",
                        "value": {"H2": 80.0, "unit": "ppm"},
                        "evidence_path": "fixtures/simulated/gas_warning.json",
                        "component_id": "gas_sensor",
                        "metadata": {"simulated": True},
                    }],
                    "alarms": [],
                    "metadata": {"simulated": True},
                },
            ],
        }

    def get_standalone_routes(self) -> list:
        plugin = self

        async def fusion_smoke(request: StarletteRequest):
            import json as _json

            body = {}
            try:
                body = await request.json()
            except Exception as exc:
                logger.debug("融合 smoke 请求未提供 JSON body: %s", exc)
            sample = plugin._build_smoke_fixture()
            if isinstance(body, dict) and body:
                sample.update(body)
            result = plugin.process(sample)
            return StarletteJSONResponse(_json.loads(_json.dumps(result, default=str)))

        async def fuse_demo(request: StarletteRequest):
            import json as _json

            body = {}
            try:
                body = await request.json()
            except Exception as exc:
                logger.debug("融合 demo 请求未提供 JSON body: %s", exc)
            sample = plugin._build_smoke_fixture()
            if isinstance(body, dict) and body:
                sample.update(body)
            result = plugin.fuse(sample)
            return StarletteJSONResponse(_json.loads(_json.dumps(result, default=str)))

        return [
            {"path": "/smoke", "endpoint": fusion_smoke, "methods": ["GET", "POST"], "summary": "Run multimodal fusion smoke"},
            {"path": "/fuse-demo", "endpoint": fuse_demo, "methods": ["GET", "POST"], "summary": "Run multimodal fusion demo"},
            {"path": "/api/multimodal/smoke", "endpoint": fusion_smoke, "methods": ["GET", "POST"], "summary": "Run multimodal fusion smoke"},
            {"path": "/api/multimodal/fuse-demo", "endpoint": fuse_demo, "methods": ["GET", "POST"], "summary": "Run multimodal fusion demo"},
            {"path": "/api/fusion/smoke", "endpoint": fusion_smoke, "methods": ["GET", "POST"], "summary": "Run multimodal fusion smoke"},
            {"path": "/api/fusion/fuse-demo", "endpoint": fuse_demo, "methods": ["GET", "POST"], "summary": "Run multimodal fusion demo"},
        ]

    @property
    def plugin_info(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": "多模态融合诊断",
            "capabilities": ["multimodal_data_fusion", "comprehensive_diagnosis", "fault_correlation"],
            "input_modalities": list(self.config.modalities),
            "fusion_strategy": self.config.fusion_strategy,
            "enhanced_engine_available": self._use_enhanced_engine
        }

    # =========================================================================
    # 增强融合引擎公共方法
    # =========================================================================

    def switch_fusion_strategy(self, strategy: str) -> bool:
        """
        切换融合策略

        Args:
            strategy: 策略名称 (early, late, attention, hybrid, bayesian)

        Returns:
            是否切换成功
        """
        valid_strategies = ["early", "late", "attention", "hybrid", "bayesian"]
        if strategy not in valid_strategies:
            logger.warning(f"无效的融合策略: {strategy}")
            return False

        self.config.fusion_strategy = strategy

        if self._use_enhanced_engine and self.strategy_manager is not None:
            try:
                self.strategy_manager.switch_strategy(strategy)
                logger.info(f"融合策略已切换为: {strategy}")
            except Exception as e:
                logger.warning(f"增强引擎策略切换失败: {e}")

        return True

    def auto_select_strategy(self, device_type: str, available_sensors: List[str]) -> str:
        """
        自动选择最佳融合策略

        Args:
            device_type: 设备类型 (transformer, switch, busbar等)
            available_sensors: 可用传感器列表

        Returns:
            推荐的策略名称
        """
        if self._use_enhanced_engine and self.strategy_manager is not None:
            try:
                strategy = self.strategy_manager.auto_select_strategy(
                    device_type=device_type,
                    available_sensors=available_sensors
                )
                return strategy
            except Exception as e:
                logger.warning(f"自动策略选择失败: {e}")

        # 默认策略选择逻辑
        if len(available_sensors) >= 4:
            return "hybrid"
        elif len(available_sensors) >= 2:
            return "attention"
        else:
            return "late"

    def get_bayesian_inference(self, evidence: Dict[str, bool]) -> Optional[Dict]:
        """
        执行贝叶斯故障推断

        Args:
            evidence: 证据字典，例如 {'thermal_anomaly': True, 'gas_H2_elevated': True}

        Returns:
            后验概率和故障链推断结果
        """
        if not self._use_enhanced_engine:
            logger.warning("贝叶斯推断需要增强融合引擎")
            return None

        if self.enhanced_engine is not None:
            try:
                return self.enhanced_engine.bayesian_inference(evidence)
            except Exception as e:
                logger.error(f"贝叶斯推断失败: {e}")
                return None

        return None

    def get_best_strategy(self) -> Optional[str]:
        """获取历史最佳策略"""
        if self._use_enhanced_engine and self.strategy_manager is not None:
            try:
                return self.strategy_manager.get_best_strategy()
            except Exception as e:
                logger.debug(f"获取增强融合历史最佳策略失败，回退配置策略: {e}")
        return self.config.fusion_strategy

    def is_enhanced_engine_active(self) -> bool:
        """检查增强融合引擎是否激活"""
        return self._use_enhanced_engine and self.enhanced_engine is not None


Plugin = MultimodalFusionPlugin
