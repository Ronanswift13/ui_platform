#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 训练管理API
提供训练任务管理、数据下载、进度查询等API接口

训练路径: /Users/ronan/Desktop/破夜绘明激光监测平台/training
"""

import os
import sys
import json
import yaml
import time
import hashlib
import threading
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 路径配置
# =============================================================================
BASE_TRAINING_PATH = Path("/Users/ronan/Desktop/破夜绘明激光监测平台/training")
DATA_PATH = BASE_TRAINING_PATH / "data"
CHECKPOINTS_PATH = BASE_TRAINING_PATH / "checkpoints"
EXPORTS_PATH = BASE_TRAINING_PATH / "exports"
LOGS_PATH = BASE_TRAINING_PATH / "logs"

# 确保目录存在
for path in [DATA_PATH, CHECKPOINTS_PATH, EXPORTS_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 枚举和数据类
# =============================================================================
class TrainingStatus(str, Enum):
    IDLE = "idle"
    DOWNLOADING = "downloading"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetStatus(str, Enum):
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PLACEHOLDER = "placeholder"
    READY = "ready"


@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
    voltage_level: str
    plugin: str
    status: TrainingStatus = TrainingStatus.IDLE
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 100
    best_map50: float = 0.0
    message: str = ""
    created_at: str = ""
    updated_at: str = ""
    model_path: Optional[str] = None
    log_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    plugin: str
    source_url: str
    description: str
    image_count: int = 0
    status: DatasetStatus = DatasetStatus.NOT_DOWNLOADED
    local_path: Optional[str] = None
    download_progress: float = 0.0
    is_public: bool = True
    requires_manual: bool = False
    manual_instructions: str = ""
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d


# =============================================================================
# 公开数据集配置
# =============================================================================
PUBLIC_DATASETS = {
    "insulator_defect": DatasetInfo(
        name="Insulator-Defect Detection Dataset",
        plugin="busbar",
        source_url="https://github.com/InsulatorData/InsulatorDataSet/archive/refs/heads/master.zip",
        description="绝缘子缺陷检测数据集 - 来源国家电网",
        image_count=848,
        is_public=True
    ),
    "mpid_insulator": DatasetInfo(
        name="MPID - 合并公开绝缘子数据集",
        plugin="busbar",
        source_url="https://github.com/phd-benel/MPID/archive/refs/heads/main.zip",
        description="合并多个公开绝缘子数据集，YOLO格式",
        image_count=6000,
        is_public=True
    ),
    "ufpr_amr": DatasetInfo(
        name="UFPR-AMR Dataset",
        plugin="meter",
        source_url="https://github.com/raysonlaroca/ufpr-amr-dataset/archive/refs/heads/master.zip",
        description="自动电表读数数据集",
        image_count=2000,
        is_public=True
    ),
    "transformer_thermal": DatasetInfo(
        name="Transformer Thermal Images",
        plugin="transformer",
        source_url="",  # Mendeley需要手动下载
        description="变压器热成像数据集 - 1种正常 + 8种故障类型",
        image_count=255,
        is_public=True,
        requires_manual=True,
        manual_instructions="请从 https://data.mendeley.com/datasets/8mg8mkc7k5/3 手动下载数据集"
    ),
    "switch_indicator": DatasetInfo(
        name="开关指示灯数据集",
        plugin="switch",
        source_url="",
        description="断路器/隔离开关/接地开关状态指示灯数据集",
        image_count=0,
        is_public=False,
        requires_manual=True,
        manual_instructions="需要现场采集：断路器分合闸状态、隔离开关位置、接地开关状态、SF6压力表等"
    ),
    "capacitor_structure": DatasetInfo(
        name="电容器结构数据集",
        plugin="capacitor",
        source_url="",
        description="电容器倾斜、倒塌、缺失检测数据集",
        image_count=0,
        is_public=False,
        requires_manual=True,
        manual_instructions="需要现场采集：电容器正常状态、倾斜状态、倒塌状态、单元缺失等"
    ),
}

# 电压等级配置
VOLTAGE_LEVELS = [
    "UHV_1000kV_AC", "UHV_800kV_DC",
    "EHV_500kV", "EHV_330kV", "EHV_750kV",
    "HV_220kV", "HV_110kV",
    "MV_35kV", "MV_66kV",
    "LV_10kV", "LV_6kV", "LV_380V"
]

PLUGINS = ["transformer", "switch", "busbar", "capacitor", "meter"]


# =============================================================================
# 训练管理器
# =============================================================================
class TrainingManager:
    """训练任务管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.tasks: Dict[str, TrainingTask] = {}
        self.active_downloads: Dict[str, float] = {}
        self.training_process: Optional[subprocess.Popen] = None
        self._load_saved_results()
        self._initialized = True
    
    def _load_saved_results(self):
        """加载已保存的训练结果"""
        results_file = BASE_TRAINING_PATH / "training_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for task_id, task_data in data.get("tasks", {}).items():
                        self.tasks[task_id] = TrainingTask(**task_data)
            except Exception as e:
                logger.error(f"加载训练结果失败: {e}")
    
    def _save_results(self):
        """保存训练结果"""
        results_file = BASE_TRAINING_PATH / "training_results.json"
        try:
            data = {
                "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
                "updated_at": datetime.now().isoformat()
            }
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存训练结果失败: {e}")
    
    def generate_task_id(self, voltage_level: str, plugin: str) -> str:
        """生成任务ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{voltage_level}_{plugin}_{timestamp}"
    
    def get_task(self, voltage_level: str, plugin: str) -> Optional[TrainingTask]:
        """获取最新的训练任务"""
        matching_tasks = [
            t for t in self.tasks.values()
            if t.voltage_level == voltage_level and t.plugin == plugin
        ]
        if matching_tasks:
            return sorted(matching_tasks, key=lambda x: x.created_at, reverse=True)[0]
        return None
    
    def get_completed_model(self, voltage_level: str, plugin: str) -> Optional[Dict]:
        """获取已完成的模型信息"""
        task = self.get_task(voltage_level, plugin)
        if task and task.status == TrainingStatus.COMPLETED and task.model_path:
            model_path = Path(task.model_path)
            if model_path.exists():
                return {
                    "task_id": task.task_id,
                    "model_path": str(model_path),
                    "best_map50": task.best_map50,
                    "created_at": task.created_at,
                    "voltage_level": voltage_level,
                    "plugin": plugin
                }
        return None
    
    def create_task(self, voltage_level: str, plugin: str) -> TrainingTask:
        """创建新训练任务"""
        task_id = self.generate_task_id(voltage_level, plugin)
        task = TrainingTask(
            task_id=task_id,
            voltage_level=voltage_level,
            plugin=plugin,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        self.tasks[task_id] = task
        self._save_results()
        return task
    
    def update_task(self, task_id: str, **kwargs):
        """更新任务状态"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = datetime.now().isoformat()
            self._save_results()
    
    def get_dataset_status(self, plugin: str, voltage_level: str) -> Dict:
        """获取数据集状态"""
        data_path = DATA_PATH / "processed" / voltage_level / plugin
        placeholder_path = DATA_PATH / "placeholder" / voltage_level / plugin
        
        # 检查已处理的数据
        train_images = list((data_path / "images" / "train").glob("*")) if (data_path / "images" / "train").exists() else []
        val_images = list((data_path / "images" / "val").glob("*")) if (data_path / "images" / "val").exists() else []
        test_images = list((data_path / "images" / "test").glob("*")) if (data_path / "images" / "test").exists() else []
        
        # 过滤掉.txt文件等非图像文件
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm'}
        train_count = len([f for f in train_images if f.suffix.lower() in image_exts])
        val_count = len([f for f in val_images if f.suffix.lower() in image_exts])
        test_count = len([f for f in test_images if f.suffix.lower() in image_exts])
        
        total_count = train_count + val_count + test_count
        
        # 检查占位符数据
        has_placeholder = placeholder_path.exists() and any(placeholder_path.iterdir())
        
        # 确定状态
        if total_count > 50:
            status = DatasetStatus.READY
        elif total_count > 0:
            status = DatasetStatus.DOWNLOADED
        elif has_placeholder:
            status = DatasetStatus.PLACEHOLDER
        else:
            status = DatasetStatus.NOT_DOWNLOADED
        
        return {
            "status": status.value,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "total_count": total_count,
            "has_placeholder": has_placeholder,
            "data_path": str(data_path),
            "placeholder_path": str(placeholder_path) if has_placeholder else None
        }
    
    def download_dataset(self, dataset_id: str, voltage_level: str, callback=None) -> bool:
        """下载公开数据集"""
        if dataset_id not in PUBLIC_DATASETS:
            logger.error(f"未知数据集: {dataset_id}")
            return False
        
        dataset = PUBLIC_DATASETS[dataset_id]
        
        if dataset.requires_manual:
            logger.warning(f"数据集 {dataset_id} 需要手动下载")
            return False
        
        if not dataset.source_url:
            logger.error(f"数据集 {dataset_id} 没有下载URL")
            return False
        
        try:
            # 创建下载目录
            raw_path = DATA_PATH / "raw" / dataset_id
            raw_path.mkdir(parents=True, exist_ok=True)
            
            # 下载文件
            zip_path = raw_path / "dataset.zip"
            
            self.active_downloads[dataset_id] = 0.0
            
            def download_progress(block_num, block_size, total_size):
                if total_size > 0:
                    progress = min(block_num * block_size / total_size * 100, 100)
                    self.active_downloads[dataset_id] = progress
                    if callback:
                        callback(progress)
            
            urllib.request.urlretrieve(dataset.source_url, str(zip_path), download_progress)
            
            # 解压
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_path)
            
            # 清理zip文件
            zip_path.unlink()
            
            # 组织数据到processed目录
            self._organize_dataset(dataset_id, dataset.plugin, voltage_level)
            
            del self.active_downloads[dataset_id]
            return True
            
        except Exception as e:
            logger.error(f"下载数据集失败: {e}")
            if dataset_id in self.active_downloads:
                del self.active_downloads[dataset_id]
            return False
    
    def _organize_dataset(self, dataset_id: str, plugin: str, voltage_level: str):
        """组织数据集到processed目录"""
        raw_path = DATA_PATH / "raw" / dataset_id
        processed_path = DATA_PATH / "processed" / voltage_level / plugin
        
        # 创建目录结构
        for split in ["train", "val", "test"]:
            (processed_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (processed_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # 查找图像文件
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm'}
        all_images = []
        for ext in image_exts:
            all_images.extend(raw_path.rglob(f"*{ext}"))
            all_images.extend(raw_path.rglob(f"*{ext.upper()}"))
        
        if not all_images:
            logger.warning(f"在 {raw_path} 中未找到图像文件")
            return
        
        # 按7:2:1分割
        import random
        random.shuffle(all_images)
        
        n = len(all_images)
        train_split = int(n * 0.7)
        val_split = int(n * 0.9)
        
        splits = {
            "train": all_images[:train_split],
            "val": all_images[train_split:val_split],
            "test": all_images[val_split:]
        }
        
        for split, images in splits.items():
            for img_path in images:
                # 复制图像
                dest_img = processed_path / "images" / split / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # 查找对应的标注文件
                label_path = img_path.with_suffix('.txt')
                if not label_path.exists():
                    # 尝试在labels目录中查找
                    label_path = img_path.parent.parent / "labels" / img_path.with_suffix('.txt').name
                
                if label_path.exists():
                    dest_label = processed_path / "labels" / split / label_path.name
                    shutil.copy2(label_path, dest_label)
        
        # 生成data.yaml
        self._generate_data_yaml(plugin, voltage_level)
        
        logger.info(f"数据集组织完成: {voltage_level}/{plugin}")
    
    def _generate_data_yaml(self, plugin: str, voltage_level: str):
        """生成data.yaml配置文件"""
        processed_path = DATA_PATH / "processed" / voltage_level / plugin
        
        # 获取检测类别
        classes = self._get_detection_classes(plugin, voltage_level)
        
        yaml_content = {
            "path": str(processed_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: cls for i, cls in enumerate(classes)},
            "nc": len(classes),
            "voltage_level": voltage_level,
            "plugin": plugin
        }
        
        yaml_path = processed_path / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    def _get_detection_classes(self, plugin: str, voltage_level: str) -> List[str]:
        """获取检测类别"""
        # 根据插件和电压等级返回对应的检测类别
        classes_map = {
            "transformer": {
                "default": ["oil_leak", "rust", "surface_damage", "foreign_object",
                           "silica_gel_normal", "silica_gel_abnormal",
                           "oil_level_normal", "oil_level_abnormal",
                           "bushing_crack", "porcelain_contamination"],
                "UHV": ["oil_leak", "rust", "surface_damage", "foreign_object",
                       "silica_gel_normal", "silica_gel_abnormal",
                       "oil_level_normal", "oil_level_abnormal",
                       "bushing_crack", "porcelain_contamination",
                       "partial_discharge", "core_ground_current", "winding_deformation"]
            },
            "switch": {
                "default": ["breaker_open", "breaker_closed",
                           "isolator_open", "isolator_closed",
                           "grounding_open", "grounding_closed",
                           "indicator_red", "indicator_green"]
            },
            "busbar": {
                "default": ["insulator_crack", "insulator_dirty", "insulator_flashover",
                           "fitting_loose", "fitting_rust", "wire_damage",
                           "foreign_object", "bird", "pin_missing"]
            },
            "capacitor": {
                "default": ["capacitor_unit", "capacitor_tilted", "capacitor_fallen",
                           "capacitor_missing", "person", "vehicle", "fuse_blown"]
            },
            "meter": {
                "default": ["sf6_pressure_gauge", "oil_temp_gauge", "oil_level_gauge",
                           "digital_display", "pointer_gauge", "led_indicator"]
            }
        }
        
        plugin_classes = classes_map.get(plugin, {"default": []})
        
        # 检查是否有特定电压等级的类别
        category = voltage_level.split("_")[0]  # UHV, EHV, HV, MV, LV
        if category in plugin_classes:
            return plugin_classes[category]
        
        return plugin_classes.get("default", [])
    
    def start_training(self, task_id: str, epochs: int = 100, batch_size: int = 16):
        """启动训练任务"""
        if task_id not in self.tasks:
            raise ValueError(f"任务不存在: {task_id}")
        
        task = self.tasks[task_id]
        task.status = TrainingStatus.TRAINING
        task.total_epochs = epochs
        task.message = "正在启动训练..."
        self._save_results()
        
        # 在后台线程中执行训练
        thread = threading.Thread(target=self._run_training, args=(task_id, epochs, batch_size))
        thread.daemon = True
        thread.start()
    
    def _run_training(self, task_id: str, epochs: int, batch_size: int):
        """执行训练（后台线程）"""
        task = self.tasks[task_id]
        
        try:
            # 检查数据
            data_yaml = DATA_PATH / "processed" / task.voltage_level / task.plugin / "data.yaml"
            if not data_yaml.exists():
                raise FileNotFoundError(f"数据配置文件不存在: {data_yaml}")
            
            # 选择模型大小
            category = task.voltage_level.split("_")[0]
            model_size = {
                "UHV": "yolov8l",
                "EHV": "yolov8m",
                "HV": "yolov8m",
                "MV": "yolov8s",
                "LV": "yolov8n"
            }.get(category, "yolov8s")
            
            # 创建检查点目录
            checkpoint_dir = CHECKPOINTS_PATH / task.plugin / task.voltage_level
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建日志目录
            log_dir = LOGS_PATH / task.task_id
            log_dir.mkdir(parents=True, exist_ok=True)
            task.log_path = str(log_dir / "training.log")
            
            # 构建训练命令
            cmd = [
                sys.executable, "-c",
                f"""
from ultralytics import YOLO
import json

model = YOLO('{model_size}.pt')
results = model.train(
    data='{data_yaml}',
    epochs={epochs},
    batch={batch_size},
    imgsz=640,
    project='{checkpoint_dir}',
    name='train',
    exist_ok=True,
    verbose=True
)

# 保存结果
metrics = {{
    'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
    'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
    'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
    'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
}}
print(f'TRAINING_METRICS:{{json.dumps(metrics)}}')
"""
            ]
            
            # 启动训练进程
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # 监控训练进度
            best_map50 = 0.0
            with open(task.log_path, 'w', encoding='utf-8') as log_file:
                for line in self.training_process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    
                    # 解析进度
                    if "epoch" in line.lower():
                        try:
                            # 尝试解析epoch信息
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part.isdigit() and i > 0:
                                    current_epoch = int(part)
                                    task.current_epoch = current_epoch
                                    task.progress = current_epoch / epochs * 100
                                    task.message = f"训练中: Epoch {current_epoch}/{epochs}"
                                    self._save_results()
                                    break
                        except:
                            pass
                    
                    # 解析最终指标
                    if "TRAINING_METRICS:" in line:
                        try:
                            metrics_str = line.split("TRAINING_METRICS:")[1].strip()
                            metrics = json.loads(metrics_str)
                            best_map50 = metrics.get('mAP50', 0)
                        except:
                            pass
            
            # 等待进程完成
            return_code = self.training_process.wait()
            
            if return_code == 0:
                # 训练成功
                model_path = checkpoint_dir / "train" / "weights" / "best.pt"
                if model_path.exists():
                    task.model_path = str(model_path)
                    task.best_map50 = best_map50
                    task.status = TrainingStatus.COMPLETED
                    task.progress = 100.0
                    task.message = f"训练完成! mAP50: {best_map50:.4f}"
                else:
                    task.status = TrainingStatus.FAILED
                    task.message = "训练完成但未找到模型文件"
            else:
                task.status = TrainingStatus.FAILED
                task.message = f"训练失败，返回码: {return_code}"
            
        except Exception as e:
            task.status = TrainingStatus.FAILED
            task.message = f"训练出错: {str(e)}"
            logger.error(f"训练失败: {e}", exc_info=True)
        
        finally:
            self.training_process = None
            self._save_results()
    
    def cancel_training(self, task_id: str):
        """取消训练"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = TrainingStatus.CANCELLED
            task.message = "训练已取消"
            
            if self.training_process:
                self.training_process.terminate()
                self.training_process = None
            
            self._save_results()


# =============================================================================
# API路由
# =============================================================================
router = APIRouter(prefix="/api/training", tags=["训练管理"])

# 请求/响应模型
class TrainRequest(BaseModel):
    voltage_level: str
    plugin: str
    epochs: int = 100
    batch_size: int = 16


class DownloadRequest(BaseModel):
    dataset_id: str
    voltage_level: str


# 创建管理器实例
training_manager = TrainingManager()


@router.get("/status/{voltage_level}/{plugin}")
async def get_training_status(voltage_level: str, plugin: str):
    """获取训练状态"""
    task = training_manager.get_task(voltage_level, plugin)
    dataset_status = training_manager.get_dataset_status(plugin, voltage_level)
    completed_model = training_manager.get_completed_model(voltage_level, plugin)
    
    return {
        "success": True,
        "voltage_level": voltage_level,
        "plugin": plugin,
        "task": task.to_dict() if task else None,
        "dataset": dataset_status,
        "completed_model": completed_model
    }


@router.get("/datasets")
async def list_datasets():
    """列出所有可用数据集"""
    return {
        "success": True,
        "datasets": {k: v.to_dict() for k, v in PUBLIC_DATASETS.items()}
    }


@router.get("/datasets/{plugin}")
async def get_plugin_datasets(plugin: str):
    """获取指定插件的数据集"""
    datasets = {k: v.to_dict() for k, v in PUBLIC_DATASETS.items() if v.plugin == plugin}
    return {
        "success": True,
        "plugin": plugin,
        "datasets": datasets
    }


@router.post("/download")
async def download_dataset(request: DownloadRequest, background_tasks: BackgroundTasks):
    """下载数据集"""
    dataset_id = request.dataset_id
    voltage_level = request.voltage_level
    
    if dataset_id not in PUBLIC_DATASETS:
        raise HTTPException(status_code=404, detail=f"数据集不存在: {dataset_id}")
    
    dataset = PUBLIC_DATASETS[dataset_id]
    
    if dataset.requires_manual:
        return {
            "success": False,
            "message": "该数据集需要手动下载",
            "manual_instructions": dataset.manual_instructions
        }
    
    # 在后台下载
    background_tasks.add_task(training_manager.download_dataset, dataset_id, voltage_level)
    
    return {
        "success": True,
        "message": f"开始下载数据集: {dataset.name}",
        "dataset_id": dataset_id
    }


@router.get("/download/progress/{dataset_id}")
async def get_download_progress(dataset_id: str):
    """获取下载进度"""
    progress = training_manager.active_downloads.get(dataset_id, -1)
    return {
        "success": True,
        "dataset_id": dataset_id,
        "progress": progress,
        "downloading": progress >= 0
    }


@router.post("/start")
async def start_training(request: TrainRequest):
    """启动训练"""
    # 检查数据集状态
    dataset_status = training_manager.get_dataset_status(request.plugin, request.voltage_level)
    
    if dataset_status["total_count"] < 10:
        return {
            "success": False,
            "message": "训练数据不足，请先下载数据集或上传训练数据",
            "dataset_status": dataset_status
        }
    
    # 创建训练任务
    task = training_manager.create_task(request.voltage_level, request.plugin)
    
    # 启动训练
    training_manager.start_training(task.task_id, request.epochs, request.batch_size)
    
    return {
        "success": True,
        "message": "训练任务已启动",
        "task_id": task.task_id,
        "task": task.to_dict()
    }


@router.post("/cancel/{task_id}")
async def cancel_training(task_id: str):
    """取消训练"""
    training_manager.cancel_training(task_id)
    return {
        "success": True,
        "message": "训练已取消"
    }


@router.get("/tasks")
async def list_tasks():
    """列出所有训练任务"""
    return {
        "success": True,
        "tasks": [t.to_dict() for t in training_manager.tasks.values()]
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """获取训练任务详情"""
    if task_id not in training_manager.tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    
    task = training_manager.tasks[task_id]
    return {
        "success": True,
        "task": task.to_dict()
    }


@router.get("/voltage_levels")
async def list_voltage_levels():
    """列出所有电压等级"""
    return {
        "success": True,
        "voltage_levels": VOLTAGE_LEVELS
    }


@router.get("/plugins")
async def list_plugins():
    """列出所有插件"""
    return {
        "success": True,
        "plugins": PLUGINS
    }


# =============================================================================
# 集成函数
# =============================================================================
def integrate_training_routes(app):
    """将训练路由集成到主应用"""
    app.include_router(router)


# =============================================================================
# 测试入口
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="训练管理API测试")
    app.include_router(router)
    
    uvicorn.run(app, host="127.0.0.1", port=8081)
