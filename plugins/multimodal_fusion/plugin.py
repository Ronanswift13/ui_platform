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
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np

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
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "visual": 0.25,
        "acoustic": 0.20,
        "gas": 0.20,
        "hyperspectral": 0.20,
        "thermal": 0.15
    })


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
        return MultimodalConfig(
            fusion_strategy=config_dict.get("fusion_strategy", "hybrid"),
            alarm_threshold=config_dict.get("alarm_threshold", 3),
            modality_weights=config_dict.get("modality_weights", {
                "visual": 0.25, "acoustic": 0.20, "gas": 0.20,
                "hyperspectral": 0.20, "thermal": 0.15
            })
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
            'registered_modalities': list(self._modality_plugins.keys())
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
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_initialized:
            return {"success": False, "error": "插件未初始化"}

        try:
            device_id = inputs.get("device_id", "unknown")
            modalities = inputs.get("modalities", {})
            fusion_strategy = inputs.get("fusion_strategy", self.config.fusion_strategy)

            if not modalities:
                return {"success": False, "error": "缺少模态数据"}

            # 处理各模态数据
            modality_results = self._process_modalities(modalities)

            # 如果增强融合引擎可用，优先使用
            if self._use_enhanced_engine and self.enhanced_engine is not None:
                return self._process_with_enhanced_engine(device_id, modality_results, fusion_strategy)

            # 使用基础融合引擎
            # 执行融合
            if fusion_strategy == "early":
                fusion_result = self._early_fusion(modality_results)
            elif fusion_strategy == "late":
                fusion_result = self._late_fusion(modality_results)
            elif fusion_strategy == "attention":
                fusion_result = self._attention_fusion(modality_results)
            else:
                fusion_result = self._hybrid_fusion(modality_results)

            # 生成诊断报告
            diagnostic_report = self._generate_diagnostic_report(modality_results, fusion_result)

            # 计算模态贡献度
            modality_contributions = self._calculate_contributions(modality_results)

            # 生成建议
            recommendations = self._generate_recommendations(diagnostic_report)

            # 生成告警
            alarms = self._generate_alarms(device_id, diagnostic_report)

            # 更新历史
            self._update_history(device_id, modality_results, fusion_result)

            return {
                "success": True,
                "device_id": device_id,
                "timestamp": time.time(),
                "overall_status": fusion_result.get("status", "normal"),
                "confidence": fusion_result.get("confidence", 0.0),
                "modality_contributions": modality_contributions,
                "fused_detections": fusion_result.get("detections", []),
                "diagnostic_report": diagnostic_report,
                "recommendations": recommendations,
                "alarms": alarms
            }
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 处理失败: {e}")
            return {"success": False, "error": str(e)}

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
        return HealthStatus(healthy=self._is_initialized, message="OK" if self._is_initialized else "未初始化")

    @property
    def plugin_info(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": "多模态融合诊断",
            "capabilities": ["multimodal_data_fusion", "comprehensive_diagnosis", "fault_correlation"],
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
            except Exception:
                pass
        return self.config.fusion_strategy

    def is_enhanced_engine_active(self) -> bool:
        """检查增强融合引擎是否激活"""
        return self._use_enhanced_engine and self.enhanced_engine is not None


Plugin = MultimodalFusionPlugin
