#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高光谱检测插件 - 完整修复版
============================

修复内容:
1. 添加 id 属性 (兼容 platform_core.plugin_manager)
2. 添加 set_status 方法
3. 支持多种构造函数签名

作者: AI巡检系统
版本: 1.0.1
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np

from darkbreaker_sdk.interfaces import HealthStatus
from darkbreaker_sdk.schemas import BoundingBox, RecognitionResult
from platform_core.visual_output_protocol import build_visual_meta

logger = logging.getLogger(__name__)


class PluginStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class HyperspectralConfig:
    wavelength_min: float = 400
    wavelength_max: float = 2500
    num_bands: int = 224
    spatial_resolution: tuple = (256, 256)
    confidence_threshold: float = 0.6
    pca_components: int = 30
    defect_types: List[str] = field(default_factory=lambda: [
        "overheating", "corrosion", "insulation_aging", "oil_leakage",
        "surface_damage", "paint_peeling", "moisture_ingress"
    ])


class HyperspectralDetectionPlugin:
    """高光谱检测插件 - 完整修复版"""

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

    PLUGIN_ID = "hyperspectral_detection"
    PLUGIN_NAME = "高光谱检测"
    PLUGIN_VERSION = "1.0.1"

    def __init__(self, manifest=None, plugin_dir=None, config=None):
        self.manifest = manifest
        self.plugin_dir = plugin_dir if plugin_dir else Path(__file__).parent
        
        self._status: PluginStatus = PluginStatus.UNLOADED
        self._last_error: str = ""
        
        if isinstance(config, HyperspectralConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = self._parse_config(config)
        elif manifest and hasattr(manifest, 'config_schema'):
            self.config = self._parse_config(manifest.config_schema or {})
        else:
            self.config = HyperspectralConfig()
        
        self._model_registry = None
        self._is_initialized = False
        
        logger.info(f"[{self.PLUGIN_NAME}] 实例已创建")
    
    def _parse_config(self, config_dict: Dict) -> HyperspectralConfig:
        return HyperspectralConfig(
            wavelength_min=config_dict.get("wavelength_range", [400, 2500])[0],
            wavelength_max=config_dict.get("wavelength_range", [400, 2500])[1],
            num_bands=config_dict.get("num_bands", 224)
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
            'capabilities': ["hyperspectral_analysis", "defect_detection", "material_identification"]
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
            self._is_initialized = False
            self.set_status(PluginStatus.UNLOADED)
            logger.info(f"[{self.PLUGIN_NAME}] 已关闭")
            return True
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 关闭失败: {e}")
            return False
    
    def cleanup(self) -> None:
        self.shutdown()
    
    # =========================================================================
    # 核心处理方法
    # =========================================================================
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_initialized:
            return {"success": False, "error": "插件未初始化"}
        
        try:
            device_id = inputs.get("device_id", "unknown")
            image = inputs.get("image")
            wavelength_range = inputs.get("wavelength_range", [self.config.wavelength_min, self.config.wavelength_max])
            analysis_type = inputs.get("analysis_type", "full")
            
            # 模拟高光谱分析
            if image is None:
                image = np.random.rand(self.config.num_bands, 256, 256).astype(np.float32)
            
            # 光谱分析
            spectrum_analysis = self._analyze_spectrum(image)
            
            # 缺陷检测
            defect_detection = self._detect_defects(image)
            
            # 材料识别
            material_analysis = self._identify_materials(image)
            
            # 综合状态
            overall_status = "alarm" if defect_detection.get("defects_found") else "normal"
            
            return {
                "success": True,
                "device_id": device_id,
                "timestamp": time.time(),
                "overall_status": overall_status,
                "spectrum_analysis": spectrum_analysis,
                "defect_detection": defect_detection,
                "material_analysis": material_analysis,
                "wavelength_range": wavelength_range,
                "num_bands": self.config.num_bands,
                "recommendations": self._generate_recommendations(defect_detection),
                "alarms": []
            }
        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _analyze_spectrum(self, image: np.ndarray) -> Dict[str, Any]:
        if image.ndim == 3:
            mean_spectrum = np.mean(image, axis=(1, 2)) if image.shape[0] < image.shape[2] else np.mean(image, axis=(0, 1))
        else:
            mean_spectrum = np.mean(image, axis=0)
        
        wavelengths = np.linspace(self.config.wavelength_min, self.config.wavelength_max, len(mean_spectrum))
        
        return {
            "wavelengths": wavelengths.tolist(),
            "mean_spectrum": mean_spectrum.tolist() if hasattr(mean_spectrum, 'tolist') else list(mean_spectrum),
            "spectral_range": {
                "min": float(np.min(mean_spectrum)),
                "max": float(np.max(mean_spectrum)),
                "mean": float(np.mean(mean_spectrum))
            }
        }
    
    def _detect_defects(self, image: np.ndarray) -> Dict[str, Any]:
        # 模拟缺陷检测
        return {
            "defects_found": False,
            "defect_count": 0,
            "defects": [],
            "confidence": 0.95
        }
    
    def _identify_materials(self, image: np.ndarray) -> Dict[str, Any]:
        return {
            "primary_material": "copper",
            "confidence": 0.92,
            "secondary_materials": ["insulation", "paint"]
        }
    
    def _generate_recommendations(self, defect_detection: Dict) -> List[str]:
        if defect_detection.get("defects_found"):
            return ["检测到缺陷，建议进一步检查"]
        return ["设备状态正常"]
    
    # =========================================================================
    # BasePlugin 兼容方法
    # =========================================================================
    
    def infer(self, frame, rois, context):
        """BasePlugin.infer — 占位实现，返回 placeholder RecognitionResult。"""
        if frame is None or not self._is_initialized:
            return []

        task_id = getattr(context, "task_id", "")
        site_id = getattr(context, "site_id", "")
        device_id = getattr(context, "device_id", "")

        meta = build_visual_meta(
            plugin_name="hyperspectral_detection",
            task_type="spectral_analysis",
            modality="hyperspectral",
            runtime_mode="simulation",
            algorithm_stage="placeholder",
            model_status="placeholder",
            quality_gate_status="not_applicable",
            evidence_hint="spectral_cube",
        )

        return [RecognitionResult(
            task_id=task_id,
            site_id=site_id,
            device_id=device_id,
            component_id="",
            roi_id="",
            bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            label="normal_placeholder",
            confidence=0.0,
            model_version=self.PLUGIN_VERSION,
            code_version=self.code_hash,
            metadata=meta,
        )]

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
            "description": "高光谱图像分析与缺陷检测",
            "capabilities": ["hyperspectral_analysis", "defect_detection", "material_identification"]
        }


Plugin = HyperspectralDetectionPlugin
