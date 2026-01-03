#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 模型集成模块
将训练好的模型集成到变电站监测平台

训练路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training
"""

import os
import sys
import json
import yaml
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 路径配置
# =============================================================================
BASE_TRAINING_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/training")
CHECKPOINTS_PATH = BASE_TRAINING_PATH / "checkpoints"
EXPORTS_PATH = BASE_TRAINING_PATH / "exports"

# 平台模型目录 (假设的目标路径)
PLATFORM_MODELS_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/models")


# =============================================================================
# 模型信息数据类
# =============================================================================
class ModelInfo:
    """模型信息"""
    
    def __init__(
        self,
        name: str,
        version: str,
        voltage_level: str,
        plugin: str,
        model_path: Path,
        classes: List[str],
        metrics: Dict[str, float] = None,
        export_formats: List[str] = None
    ):
        self.name = name
        self.version = version
        self.voltage_level = voltage_level
        self.plugin = plugin
        self.model_path = model_path
        self.classes = classes
        self.metrics = metrics or {}
        self.export_formats = export_formats or []
        self.created_at = datetime.now().isoformat()
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """计算模型文件校验和"""
        if self.model_path.exists():
            with open(self.model_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        return ""
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "voltage_level": self.voltage_level,
            "plugin": self.plugin,
            "model_path": str(self.model_path),
            "classes": self.classes,
            "num_classes": len(self.classes),
            "metrics": self.metrics,
            "export_formats": self.export_formats,
            "created_at": self.created_at,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelInfo':
        return cls(
            name=data["name"],
            version=data["version"],
            voltage_level=data["voltage_level"],
            plugin=data["plugin"],
            model_path=Path(data["model_path"]),
            classes=data["classes"],
            metrics=data.get("metrics"),
            export_formats=data.get("export_formats")
        )


# =============================================================================
# 模型注册表
# =============================================================================
class ModelRegistry:
    """
    模型注册表
    管理所有训练好的模型
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or (BASE_TRAINING_PATH / "model_registry.json")
        self.models: Dict[str, ModelInfo] = {}
        self._load_registry()
    
    def _load_registry(self):
        """加载注册表"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for key, model_data in data.get("models", {}).items():
                    self.models[key] = ModelInfo.from_dict(model_data)
    
    def _save_registry(self):
        """保存注册表"""
        data = {
            "updated_at": datetime.now().isoformat(),
            "models": {key: model.to_dict() for key, model in self.models.items()}
        }
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def register(self, model_info: ModelInfo) -> str:
        """注册模型"""
        key = f"{model_info.voltage_level}_{model_info.plugin}_{model_info.version}"
        self.models[key] = model_info
        self._save_registry()
        logger.info(f"模型已注册: {key}")
        return key
    
    def get(self, voltage_level: str, plugin: str, version: str = None) -> Optional[ModelInfo]:
        """获取模型"""
        if version:
            key = f"{voltage_level}_{plugin}_{version}"
            return self.models.get(key)
        
        # 返回最新版本
        matching = [
            (k, v) for k, v in self.models.items()
            if v.voltage_level == voltage_level and v.plugin == plugin
        ]
        if matching:
            return sorted(matching, key=lambda x: x[1].created_at, reverse=True)[0][1]
        return None
    
    def list_models(self, voltage_level: str = None, plugin: str = None) -> List[ModelInfo]:
        """列出模型"""
        models = list(self.models.values())
        
        if voltage_level:
            models = [m for m in models if m.voltage_level == voltage_level]
        if plugin:
            models = [m for m in models if m.plugin == plugin]
        
        return models
    
    def remove(self, key: str) -> bool:
        """移除模型"""
        if key in self.models:
            del self.models[key]
            self._save_registry()
            logger.info(f"模型已移除: {key}")
            return True
        return False


# =============================================================================
# 模型部署器
# =============================================================================
class ModelDeployer:
    """
    模型部署器
    将训练好的模型部署到平台
    """
    
    def __init__(self, registry: ModelRegistry = None):
        self.registry = registry or ModelRegistry()
        
    def scan_trained_models(self) -> List[Dict[str, Any]]:
        """
        扫描所有训练好的模型
        """
        found_models = []
        
        # 扫描checkpoints目录
        for plugin_dir in CHECKPOINTS_PATH.iterdir():
            if not plugin_dir.is_dir():
                continue
            
            plugin = plugin_dir.name
            
            for voltage_dir in plugin_dir.iterdir():
                if not voltage_dir.is_dir():
                    continue
                
                voltage_level = voltage_dir.name
                
                # 查找模型文件
                for model_file in voltage_dir.glob("*.pt"):
                    found_models.append({
                        "voltage_level": voltage_level,
                        "plugin": plugin,
                        "model_path": model_file,
                        "model_type": "pytorch"
                    })
        
        # 扫描exports目录
        for plugin_dir in EXPORTS_PATH.iterdir():
            if not plugin_dir.is_dir():
                continue
            
            plugin = plugin_dir.name
            
            for voltage_dir in plugin_dir.iterdir():
                if not voltage_dir.is_dir():
                    continue
                
                voltage_level = voltage_dir.name
                
                # 查找ONNX模型
                for model_file in voltage_dir.glob("*.onnx"):
                    found_models.append({
                        "voltage_level": voltage_level,
                        "plugin": plugin,
                        "model_path": model_file,
                        "model_type": "onnx"
                    })
        
        return found_models
    
    def deploy_model(
        self,
        voltage_level: str,
        plugin: str,
        model_path: Path,
        version: str = None,
        classes: List[str] = None,
        metrics: Dict[str, float] = None
    ) -> bool:
        """
        部署模型到平台
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型信息
        model_info = ModelInfo(
            name=f"{voltage_level}_{plugin}",
            version=version,
            voltage_level=voltage_level,
            plugin=plugin,
            model_path=model_path,
            classes=classes or [],
            metrics=metrics or {}
        )
        
        # 创建目标目录
        target_dir = PLATFORM_MODELS_PATH / plugin / voltage_level
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制模型文件
        target_path = target_dir / f"{voltage_level}_{plugin}_{version}{model_path.suffix}"
        shutil.copy2(model_path, target_path)
        
        # 更新模型信息路径
        model_info.model_path = target_path
        
        # 注册模型
        self.registry.register(model_info)
        
        # 生成模型配置文件
        self._generate_model_config(model_info, target_dir)
        
        logger.info(f"模型已部署: {target_path}")
        return True
    
    def _generate_model_config(self, model_info: ModelInfo, target_dir: Path):
        """生成模型配置文件"""
        config = {
            "model_info": model_info.to_dict(),
            "inference_config": {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "max_detections": 100,
                "input_size": [640, 640],
                "normalize": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "post_processing": {
                "nms_enabled": True,
                "nms_threshold": 0.45,
                "score_threshold": 0.25
            }
        }
        
        config_path = target_dir / f"{model_info.name}_{model_info.version}_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def deploy_all(self) -> Dict[str, Any]:
        """
        部署所有训练好的模型
        """
        results = {"deployed": [], "failed": []}
        
        models = self.scan_trained_models()
        
        for model in models:
            try:
                # 读取类别信息
                classes = self._get_classes(model["voltage_level"], model["plugin"])
                
                # 部署
                success = self.deploy_model(
                    voltage_level=model["voltage_level"],
                    plugin=model["plugin"],
                    model_path=model["model_path"],
                    classes=classes
                )
                
                if success:
                    results["deployed"].append({
                        "voltage_level": model["voltage_level"],
                        "plugin": model["plugin"],
                        "model_path": str(model["model_path"])
                    })
                else:
                    results["failed"].append(model)
                    
            except Exception as e:
                logger.error(f"部署失败 {model}: {e}")
                results["failed"].append(model)
        
        return results
    
    def _get_classes(self, voltage_level: str, plugin: str) -> List[str]:
        """获取类别列表"""
        # 尝试从data.yaml读取
        data_yaml = BASE_TRAINING_PATH / "data" / "processed" / voltage_level / plugin / "data.yaml"
        
        if data_yaml.exists():
            with open(data_yaml, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                names = config.get("names", {})
                if isinstance(names, dict):
                    return list(names.values())
                return names
        
        return []


# =============================================================================
# 模型版本管理器
# =============================================================================
class ModelVersionManager:
    """
    模型版本管理器
    管理模型的不同版本
    """
    
    def __init__(self, registry: ModelRegistry = None):
        self.registry = registry or ModelRegistry()
    
    def get_versions(self, voltage_level: str, plugin: str) -> List[str]:
        """获取模型的所有版本"""
        models = self.registry.list_models(voltage_level, plugin)
        return sorted([m.version for m in models], reverse=True)
    
    def get_latest(self, voltage_level: str, plugin: str) -> Optional[ModelInfo]:
        """获取最新版本的模型"""
        return self.registry.get(voltage_level, plugin)
    
    def rollback(self, voltage_level: str, plugin: str, version: str) -> bool:
        """回滚到指定版本"""
        model = self.registry.get(voltage_level, plugin, version)
        if model is None:
            logger.error(f"未找到版本: {voltage_level}/{plugin}/{version}")
            return False
        
        # 创建符号链接或复制到"当前"版本
        current_path = PLATFORM_MODELS_PATH / plugin / voltage_level / "current"
        
        if current_path.exists():
            current_path.unlink()
        
        # 创建符号链接
        current_path.symlink_to(model.model_path)
        
        logger.info(f"已回滚到版本: {version}")
        return True
    
    def compare_versions(
        self,
        voltage_level: str,
        plugin: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """比较两个版本的性能"""
        model1 = self.registry.get(voltage_level, plugin, version1)
        model2 = self.registry.get(voltage_level, plugin, version2)
        
        if model1 is None or model2 is None:
            return {"error": "模型版本不存在"}
        
        comparison = {
            "version1": {
                "version": version1,
                "metrics": model1.metrics,
                "created_at": model1.created_at
            },
            "version2": {
                "version": version2,
                "metrics": model2.metrics,
                "created_at": model2.created_at
            },
            "differences": {}
        }
        
        # 计算指标差异
        for metric in set(list(model1.metrics.keys()) + list(model2.metrics.keys())):
            v1 = model1.metrics.get(metric, 0)
            v2 = model2.metrics.get(metric, 0)
            comparison["differences"][metric] = {
                "version1": v1,
                "version2": v2,
                "delta": v2 - v1,
                "improvement": (v2 - v1) / v1 * 100 if v1 > 0 else 0
            }
        
        return comparison


# =============================================================================
# 平台集成代码生成器
# =============================================================================
class PlatformIntegrator:
    """
    平台集成器
    生成集成到变电站监测平台的代码
    """
    
    def __init__(self):
        self.registry = ModelRegistry()
    
    def generate_adapter_code(self, voltage_level: str, plugin: str) -> str:
        """
        生成电压等级适配器代码
        """
        model = self.registry.get(voltage_level, plugin)
        if model is None:
            return "# 模型未找到"
        
        code = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成的模型适配器代码
电压等级: {voltage_level}
插件: {plugin}
生成时间: {datetime.now().isoformat()}
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# 模型配置
MODEL_CONFIG = {{
    "name": "{model.name}",
    "version": "{model.version}",
    "voltage_level": "{voltage_level}",
    "plugin": "{plugin}",
    "model_path": "{model.model_path}",
    "classes": {model.classes},
    "num_classes": {len(model.classes)},
    "input_size": (640, 640),
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45
}}


class {plugin.title()}Detector:
    """
    {voltage_level} {plugin} 检测器
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or MODEL_CONFIG["model_path"]
        self.classes = MODEL_CONFIG["classes"]
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"模型加载成功: {{self.model_path}}")
        except Exception as e:
            print(f"模型加载失败: {{e}}")
    
    def detect(
        self,
        image: np.ndarray,
        conf: float = None,
        iou: float = None
    ) -> List[Dict[str, Any]]:
        """
        执行检测
        
        Args:
            image: BGR格式的图像数组
            conf: 置信度阈值
            iou: IoU阈值
        
        Returns:
            检测结果列表
        """
        if self.model is None:
            return []
        
        conf = conf or MODEL_CONFIG["confidence_threshold"]
        iou = iou or MODEL_CONFIG["iou_threshold"]
        
        results = self.model(image, conf=conf, iou=iou, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({{
                    "class_id": int(box.cls[0]),
                    "class_name": self.classes[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0].tolist()],
                    "bbox_normalized": [float(x) for x in box.xywhn[0].tolist()]
                }})
        
        return detections
    
    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self.classes.copy()


# 创建默认检测器实例
detector = {plugin.title()}Detector()


def detect(image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
    """便捷检测函数"""
    return detector.detect(image, **kwargs)


if __name__ == "__main__":
    import cv2
    
    # 测试检测
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = detect(test_image)
    print(f"检测结果: {{len(results)}} 个目标")
'''
        return code
    
    def generate_all_adapters(self, output_dir: Path = None) -> Dict[str, Path]:
        """
        为所有模型生成适配器代码
        """
        output_dir = output_dir or (BASE_TRAINING_PATH / "adapters")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated = {}
        
        for model in self.registry.list_models():
            code = self.generate_adapter_code(model.voltage_level, model.plugin)
            
            filename = f"{model.voltage_level}_{model.plugin}_adapter.py"
            filepath = output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            
            generated[f"{model.voltage_level}_{model.plugin}"] = filepath
            logger.info(f"生成适配器: {filepath}")
        
        return generated


# =============================================================================
# 命令行接口
# =============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="模型集成工具")
    subparsers = parser.add_subparsers(dest="command")
    
    # scan命令
    scan_parser = subparsers.add_parser("scan", help="扫描训练好的模型")
    
    # deploy命令
    deploy_parser = subparsers.add_parser("deploy", help="部署模型")
    deploy_parser.add_argument("--voltage", "-v", type=str, help="电压等级")
    deploy_parser.add_argument("--plugin", "-p", type=str, help="插件")
    deploy_parser.add_argument("--all", action="store_true", help="部署所有模型")
    
    # list命令
    list_parser = subparsers.add_parser("list", help="列出已注册的模型")
    
    # generate命令
    gen_parser = subparsers.add_parser("generate", help="生成适配器代码")
    gen_parser.add_argument("--output", "-o", type=str, help="输出目录")
    
    args = parser.parse_args()
    
    if args.command == "scan":
        deployer = ModelDeployer()
        models = deployer.scan_trained_models()
        print(f"\n找到 {len(models)} 个模型:")
        for m in models:
            print(f"  - {m['voltage_level']}/{m['plugin']}: {m['model_path']}")
    
    elif args.command == "deploy":
        deployer = ModelDeployer()
        if args.all:
            results = deployer.deploy_all()
            print(f"部署完成: {len(results['deployed'])} 成功, {len(results['failed'])} 失败")
        elif args.voltage and args.plugin:
            # 查找模型
            models = deployer.scan_trained_models()
            for m in models:
                if m['voltage_level'] == args.voltage and m['plugin'] == args.plugin:
                    deployer.deploy_model(m['voltage_level'], m['plugin'], m['model_path'])
                    break
        else:
            print("请指定 --voltage 和 --plugin，或使用 --all")
    
    elif args.command == "list":
        registry = ModelRegistry()
        models = registry.list_models()
        print(f"\n已注册的模型 ({len(models)}):")
        for m in models:
            print(f"  - {m.name} v{m.version}")
            print(f"    路径: {m.model_path}")
            print(f"    类别: {len(m.classes)}")
            if m.metrics:
                print(f"    mAP50: {m.metrics.get('mAP50', 'N/A')}")
    
    elif args.command == "generate":
        integrator = PlatformIntegrator()
        output_dir = Path(args.output) if args.output else None
        generated = integrator.generate_all_adapters(output_dir)
        print(f"生成 {len(generated)} 个适配器")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
