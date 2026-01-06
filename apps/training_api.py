#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 训练管理API（修复版）
修复内容：
1. 添加真实的开源数据集下载URL
2. 实现实际的下载功能
3. 区分可下载和需要手动获取的数据集
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
import urllib.error
import zipfile
import tarfile
import shutil
import ssl
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
PROJECT_ROOT = Path(__file__).parent.parent
BASE_TRAINING_PATH = PROJECT_ROOT / "training"
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
    can_download: bool = True  # 是否可以自动下载
    unavailable_reason: str = ""  # 无法下载的原因
    manual_url: str = ""  # 手动下载链接
    format_type: str = "yolo"  # 数据格式：yolo, coco, voc, custom
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d


# =============================================================================
# 公开数据集配置 - 真实可下载的数据集
# =============================================================================
PUBLIC_DATASETS = {
    # ==================== 绝缘子/母线巡视数据集 ====================
    "cplid_insulator": DatasetInfo(
        name="CPLID - 中国电力线绝缘子数据集",
        plugin="busbar",
        source_url="https://github.com/InsulatorData/InsulatorDataSet/archive/refs/heads/master.zip",
        description="中国国家电网绝缘子数据集，包含正常和缺陷绝缘子图像",
        image_count=848,
        is_public=True,
        can_download=True,
        format_type="custom"
    ),
    "mpid_insulator": DatasetInfo(
        name="MPID - 合并公开绝缘子数据集",
        plugin="busbar",
        source_url="https://github.com/phd-benel/MPID/archive/refs/heads/main.zip",
        description="合并多个公开绝缘子数据集，已转换为YOLO格式，包含6000+张图像",
        image_count=6000,
        is_public=True,
        can_download=True,
        format_type="yolo"
    ),
    "insulator_defect_detection": DatasetInfo(
        name="Insulator-Defect Detection Dataset",
        plugin="busbar",
        source_url="",  # DatasetNinja需要注册下载
        description="绝缘子缺陷检测数据集(2.43GB, 1600张)，包含insulator, damaged, flashover三类",
        image_count=1600,
        is_public=True,
        can_download=False,
        unavailable_reason="该数据集需要在DatasetNinja网站注册后下载",
        manual_url="https://datasetninja.com/insulator-defect-detection"
    ),
    
    # ==================== 电表读数数据集 ====================
    "ufpr_amr": DatasetInfo(
        name="UFPR-AMR Dataset",
        plugin="meter",
        source_url="https://github.com/raysonlaroca/ufpr-amr-dataset/archive/refs/heads/master.zip",
        description="巴西UFPR大学自动电表读数数据集，2000张电表图像",
        image_count=2000,
        is_public=True,
        can_download=True,
        format_type="custom"
    ),
    "copel_amr": DatasetInfo(
        name="Copel-AMR Dataset",
        plugin="meter",
        source_url="https://github.com/raysonlaroca/copel-amr-dataset/archive/refs/heads/master.zip",
        description="Copel电力公司电表读数数据集，12500张图像，更大规模",
        image_count=12500,
        is_public=True,
        can_download=True,
        format_type="custom"
    ),
    
    # ==================== 变压器/热成像数据集 ====================
    "transformer_thermal": DatasetInfo(
        name="Transformer Thermal Images",
        plugin="transformer",
        source_url="",  # Mendeley需要登录下载
        description="变压器热成像数据集(255张)，包含1种正常状态和8种故障类型",
        image_count=255,
        is_public=True,
        can_download=False,
        unavailable_reason="该数据集托管在Mendeley Data，需要登录后手动下载",
        manual_url="https://data.mendeley.com/datasets/8mg8mkc7k5/3"
    ),
    "flir_thermal": DatasetInfo(
        name="FLIR Thermal Dataset",
        plugin="transformer",
        source_url="",
        description="FLIR热成像数据集，可用于设备温度异常检测",
        image_count=10000,
        is_public=True,
        can_download=False,
        unavailable_reason="需要在FLIR官网注册并申请下载",
        manual_url="https://www.flir.com/oem/adas/adas-dataset-form/"
    ),
    
    # ==================== 开关状态数据集（需要现场采集） ====================
    "switch_indicator": DatasetInfo(
        name="开关状态指示灯数据集",
        plugin="switch",
        source_url="",
        description="断路器/隔离开关/接地开关状态指示灯数据集",
        image_count=0,
        is_public=False,
        can_download=False,
        unavailable_reason="电力行业专用数据，无公开数据集。需要现场采集：断路器分合闸状态、隔离开关位置、接地开关状态、SF6压力表等图像",
        manual_url=""
    ),
    "sf6_gauge": DatasetInfo(
        name="SF6压力表/密度继电器数据集",
        plugin="switch",
        source_url="",
        description="SF6气体绝缘设备压力表和密度继电器读数数据集",
        image_count=0,
        is_public=False,
        can_download=False,
        unavailable_reason="电力行业专用数据，需要在变电站现场采集SF6压力表、密度继电器等设备的图像",
        manual_url=""
    ),
    
    # ==================== 电容器数据集（需要现场采集） ====================
    "capacitor_structure": DatasetInfo(
        name="电容器结构完整性数据集",
        plugin="capacitor",
        source_url="",
        description="电容器倾斜、倒塌、缺失检测数据集",
        image_count=0,
        is_public=False,
        can_download=False,
        unavailable_reason="电容器组结构监测数据无公开数据集。需要现场采集：电容器正常状态、倾斜状态、倒塌状态、单元缺失等图像",
        manual_url=""
    ),
    "intrusion_detection": DatasetInfo(
        name="变电站入侵检测数据集",
        plugin="capacitor",
        source_url="https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
        description="可用COCO数据集训练人员/车辆入侵检测（示例数据）",
        image_count=128,
        is_public=True,
        can_download=True,
        format_type="yolo"
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
# 数据集下载器
# =============================================================================
class DatasetDownloader:
    """真实的数据集下载器"""
    
    def __init__(self):
        self.active_downloads: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        # 创建SSL上下文（处理某些HTTPS证书问题）
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
    
    def download(self, dataset_id: str, voltage_level: str, callback=None) -> bool:
        """
        下载数据集
        
        Args:
            dataset_id: 数据集ID
            voltage_level: 电压等级
            callback: 进度回调函数
            
        Returns:
            是否成功
        """
        if dataset_id not in PUBLIC_DATASETS:
            logger.error(f"未知数据集: {dataset_id}")
            return False
        
        dataset = PUBLIC_DATASETS[dataset_id]
        
        if not dataset.can_download:
            logger.warning(f"数据集 {dataset_id} 无法自动下载: {dataset.unavailable_reason}")
            return False
        
        if not dataset.source_url:
            logger.error(f"数据集 {dataset_id} 没有下载URL")
            return False
        
        # 初始化下载状态
        with self._lock:
            self.active_downloads[dataset_id] = {
                "progress": 0.0,
                "status": "downloading",
                "message": "开始下载...",
                "start_time": datetime.now().isoformat()
            }
        
        try:
            # 创建下载目录
            raw_path = DATA_PATH / "raw" / dataset_id
            raw_path.mkdir(parents=True, exist_ok=True)
            
            # 确定文件名
            url = dataset.source_url
            if url.endswith('.zip'):
                filename = "dataset.zip"
            elif url.endswith('.tar.gz'):
                filename = "dataset.tar.gz"
            elif url.endswith('.tar'):
                filename = "dataset.tar"
            else:
                filename = "dataset.zip"
            
            download_path = raw_path / filename
            
            logger.info(f"开始下载数据集 {dataset_id} 从 {url}")
            
            # 下载文件
            self._download_file(url, download_path, dataset_id, callback)
            
            # 更新状态
            with self._lock:
                self.active_downloads[dataset_id]["status"] = "extracting"
                self.active_downloads[dataset_id]["message"] = "正在解压..."
            
            # 解压文件
            self._extract_file(download_path, raw_path)
            
            # 清理压缩文件
            if download_path.exists():
                download_path.unlink()
            
            # 更新状态
            with self._lock:
                self.active_downloads[dataset_id]["status"] = "organizing"
                self.active_downloads[dataset_id]["message"] = "正在组织数据..."
            
            # 组织数据到processed目录
            self._organize_dataset(dataset_id, dataset.plugin, voltage_level, dataset.format_type)
            
            # 完成
            with self._lock:
                self.active_downloads[dataset_id]["status"] = "completed"
                self.active_downloads[dataset_id]["progress"] = 100.0
                self.active_downloads[dataset_id]["message"] = "下载完成"
            
            logger.info(f"数据集 {dataset_id} 下载完成")
            return True
            
        except Exception as e:
            logger.error(f"下载数据集 {dataset_id} 失败: {e}", exc_info=True)
            with self._lock:
                self.active_downloads[dataset_id]["status"] = "failed"
                self.active_downloads[dataset_id]["message"] = f"下载失败: {str(e)}"
            return False
    
    def _download_file(self, url: str, save_path: Path, dataset_id: str, callback=None):
        """下载文件并显示进度"""
        
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                progress = min(block_num * block_size / total_size * 100, 100)
            else:
                progress = min(block_num * block_size / (50 * 1024 * 1024) * 100, 99)  # 假设50MB
            
            with self._lock:
                self.active_downloads[dataset_id]["progress"] = progress
                self.active_downloads[dataset_id]["message"] = f"下载中: {progress:.1f}%"
            
            if callback:
                callback(progress)
        
        # 设置请求头（模拟浏览器）
        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
        ]
        urllib.request.install_opener(opener)
        
        # 下载
        urllib.request.urlretrieve(url, str(save_path), progress_hook)
    
    def _extract_file(self, file_path: Path, extract_to: Path):
        """解压文件"""
        if file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif file_path.name.endswith('.tar.gz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        elif file_path.suffix == '.tar':
            with tarfile.open(file_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
    
    def _organize_dataset(self, dataset_id: str, plugin: str, voltage_level: str, format_type: str):
        """组织数据集到processed目录"""
        raw_path = DATA_PATH / "raw" / dataset_id
        processed_path = DATA_PATH / "processed" / voltage_level / plugin
        
        # 创建目录结构
        for split in ["train", "val", "test"]:
            (processed_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (processed_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # 查找图像文件
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.tif', '.tiff'}
        all_images = []
        for ext in image_exts:
            all_images.extend(raw_path.rglob(f"*{ext}"))
            all_images.extend(raw_path.rglob(f"*{ext.upper()}"))
        
        if not all_images:
            logger.warning(f"在 {raw_path} 中未找到图像文件")
            return
        
        logger.info(f"找到 {len(all_images)} 张图像")
        
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
                    possible_paths = [
                        img_path.parent.parent / "labels" / img_path.with_suffix('.txt').name,
                        img_path.parent / "labels" / img_path.with_suffix('.txt').name,
                        raw_path / "labels" / img_path.with_suffix('.txt').name,
                    ]
                    for p in possible_paths:
                        if p.exists():
                            label_path = p
                            break
                
                if label_path.exists():
                    dest_label = processed_path / "labels" / split / label_path.name
                    shutil.copy2(label_path, dest_label)
        
        # 生成data.yaml
        self._generate_data_yaml(plugin, voltage_level, dataset_id)
        
        logger.info(f"数据集组织完成: {voltage_level}/{plugin}")
    
    def _generate_data_yaml(self, plugin: str, voltage_level: str, dataset_id: str):
        """生成data.yaml配置文件"""
        processed_path = DATA_PATH / "processed" / voltage_level / plugin
        
        # 获取检测类别
        classes = get_detection_classes(plugin, voltage_level)
        
        yaml_content = {
            "path": str(processed_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: cls for i, cls in enumerate(classes)},
            "nc": len(classes),
            "voltage_level": voltage_level,
            "plugin": plugin,
            "source_dataset": dataset_id
        }
        
        yaml_path = processed_path / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    def get_progress(self, dataset_id: str) -> Dict:
        """获取下载进度"""
        with self._lock:
            return self.active_downloads.get(dataset_id, {
                "progress": -1,
                "status": "not_started",
                "message": ""
            })


def get_detection_classes(plugin: str, voltage_level: str) -> List[str]:
    """获取检测类别"""
    classes_map = {
        "transformer": [
            "oil_leak", "rust", "surface_damage", "foreign_object",
            "silica_gel_normal", "silica_gel_abnormal",
            "oil_level_normal", "oil_level_abnormal",
            "bushing_crack", "porcelain_contamination",
            "thermal_normal", "thermal_warning", "thermal_danger"
        ],
        "switch": [
            "breaker_open", "breaker_closed", "breaker_intermediate",
            "isolator_open", "isolator_closed",
            "grounding_open", "grounding_closed",
            "indicator_red", "indicator_green",
            "sf6_gauge", "sf6_normal", "sf6_abnormal"
        ],
        "busbar": [
            "insulator", "insulator_crack", "insulator_dirty", "insulator_flashover",
            "fitting_loose", "fitting_rust", "wire_damage",
            "foreign_object", "bird", "pin_missing"
        ],
        "capacitor": [
            "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
            "capacitor_missing", "person", "vehicle", "animal", "fuse_blown"
        ],
        "meter": [
            "sf6_pressure_gauge", "oil_temp_gauge", "oil_level_gauge",
            "digital_display", "pointer_gauge", "led_indicator",
            "seven_segment", "dial_meter"
        ]
    }
    
    return classes_map.get(plugin, [])


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
        self.downloader = DatasetDownloader()
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
                        # 处理枚举类型
                        if isinstance(task_data.get("status"), str):
                            task_data["status"] = TrainingStatus(task_data["status"])
                        self.tasks[task_id] = TrainingTask(**task_data)
            except Exception as e:
                logger.error(f"加载训练结果失败: {e}")
    
    def _save_results(self):
        """保存训练结果"""
        results_file = BASE_TRAINING_PATH / "training_results.json"
        try:
            data = {
                "tasks": {},
                "updated_at": datetime.now().isoformat()
            }
            for k, v in self.tasks.items():
                task_dict = v.to_dict()
                task_dict["status"] = v.status.value if isinstance(v.status, TrainingStatus) else v.status
                data["tasks"][k] = task_dict
            
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
    
    def get_dataset_status(self, plugin: str, voltage_level: str) -> Dict:
        """获取数据集状态"""
        data_path = DATA_PATH / "processed" / voltage_level / plugin
        placeholder_path = DATA_PATH / "placeholder" / voltage_level / plugin
        
        # 检查已处理的数据
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm'}
        
        train_count = 0
        val_count = 0
        test_count = 0
        
        train_dir = data_path / "images" / "train"
        val_dir = data_path / "images" / "val"
        test_dir = data_path / "images" / "test"
        
        if train_dir.exists():
            train_count = len([f for f in train_dir.iterdir() if f.suffix.lower() in image_exts])
        if val_dir.exists():
            val_count = len([f for f in val_dir.iterdir() if f.suffix.lower() in image_exts])
        if test_dir.exists():
            test_count = len([f for f in test_dir.iterdir() if f.suffix.lower() in image_exts])
        
        total_count = train_count + val_count + test_count
        
        # 检查占位符数据
        has_placeholder = placeholder_path.exists() and any(placeholder_path.iterdir()) if placeholder_path.exists() else False
        
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
        """下载数据集"""
        return self.downloader.download(dataset_id, voltage_level, callback)
    
    def get_download_progress(self, dataset_id: str) -> Dict:
        """获取下载进度"""
        return self.downloader.get_progress(dataset_id)
    
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
            
            # 模拟训练进度（实际环境中使用ultralytics）
            for epoch in range(1, epochs + 1):
                time.sleep(0.5)  # 模拟训练时间
                task.current_epoch = epoch
                task.progress = epoch / epochs * 100
                task.best_map50 = 0.5 + 0.3 * (epoch / epochs)  # 模拟指标
                task.message = f"训练中: Epoch {epoch}/{epochs}"
                self._save_results()
                
                if task.status == TrainingStatus.CANCELLED:
                    return
            
            # 训练完成
            task.model_path = str(checkpoint_dir / "best.pt")
            task.status = TrainingStatus.COMPLETED
            task.progress = 100.0
            task.message = f"训练完成! mAP50: {task.best_map50:.4f}"
            
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
    
    if not dataset.can_download:
        return {
            "success": False,
            "message": "该数据集无法自动下载",
            "reason": dataset.unavailable_reason,
            "manual_url": dataset.manual_url
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
    progress_info = training_manager.get_download_progress(dataset_id)
    return {
        "success": True,
        "dataset_id": dataset_id,
        "progress": progress_info.get("progress", -1),
        "status": progress_info.get("status", "not_started"),
        "message": progress_info.get("message", ""),
        "downloading": progress_info.get("status") == "downloading"
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


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="训练管理API测试")
    app.include_router(router)
    
    uvicorn.run(app, host="127.0.0.1", port=8081)
