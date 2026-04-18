#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 训练管理API（第二阶段更新）
集成真实的数据集下载功能
"""

import os
import sys
import json
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from training.services.upload_batch_service import batch_service

# 导入数据集下载器
try:
    from .dataset_downloader import (
        DatasetDownloader, DatasetConfig, DownloadProgress, DownloadStatus,
        DATASETS, get_downloader
    )
except ImportError:
    from dataset_downloader import (
        DatasetDownloader, DatasetConfig, DownloadProgress, DownloadStatus,
        DATASETS, get_downloader
    )

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
RESULTS_FILE = BASE_TRAINING_PATH / "training_results.json"

# 确保目录存在
for path in [DATA_PATH, CHECKPOINTS_PATH, EXPORTS_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 训练状态枚举
# =============================================================================
class TrainingStatus(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# 数据类
# =============================================================================
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
        d = asdict(self)
        d['status'] = self.status.value if isinstance(self.status, TrainingStatus) else self.status
        return d


# =============================================================================
# 训练管理器
# =============================================================================
class TrainingManager:
    """训练任务管理器（单例）"""
    
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
        self.downloader = get_downloader(DATA_PATH)
        self.training_process: Optional[subprocess.Popen] = None
        self._load_saved_results()
        self._initialized = True
    
    def _load_saved_results(self):
        """加载已保存的训练结果"""
        if RESULTS_FILE.exists():
            try:
                with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for task_id, task_data in data.get("tasks", {}).items():
                        if isinstance(task_data.get("status"), str):
                            task_data["status"] = TrainingStatus(task_data["status"])
                        self.tasks[task_id] = TrainingTask(**task_data)
                logger.info(f"加载了 {len(self.tasks)} 个训练任务记录")
            except Exception as e:
                logger.error(f"加载训练结果失败: {e}")
    
    def _save_results(self):
        """保存训练结果"""
        try:
            data = {
                "tasks": {},
                "updated_at": datetime.now().isoformat()
            }
            for k, v in self.tasks.items():
                task_dict = v.to_dict()
                data["tasks"][k] = task_dict
            
            with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存训练结果失败: {e}")
    
    def generate_task_id(self, voltage_level: str, plugin: str) -> str:
        """生成任务ID"""
        plugin = batch_service.to_legacy_plugin(batch_service.normalize_plugin_id(plugin))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{voltage_level}_{plugin}_{timestamp}"
    
    def get_task(self, voltage_level: str, plugin: str) -> Optional[TrainingTask]:
        """获取最新的训练任务"""
        plugin = batch_service.to_legacy_plugin(batch_service.normalize_plugin_id(plugin))
        matching = [
            t for t in self.tasks.values()
            if t.voltage_level == voltage_level and t.plugin == plugin
        ]
        if matching:
            return sorted(matching, key=lambda x: x.created_at, reverse=True)[0]
        return None
    
    def get_completed_model(self, voltage_level: str, plugin: str) -> Optional[Dict]:
        """获取已完成的模型"""
        plugin = batch_service.to_legacy_plugin(batch_service.normalize_plugin_id(plugin))
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
        
        # 检查是否有预置模型
        preset_model = CHECKPOINTS_PATH / plugin / voltage_level / "best.pt"
        if preset_model.exists():
            return {
                "task_id": "preset",
                "model_path": str(preset_model),
                "best_map50": 0.0,
                "created_at": "",
                "voltage_level": voltage_level,
                "plugin": plugin,
                "is_preset": True
            }
        
        return None
    
    def get_dataset_status(self, plugin: str, voltage_level: str) -> Dict:
        """获取数据集状态"""
        plugin = batch_service.to_legacy_plugin(batch_service.normalize_plugin_id(plugin))
        return self.downloader.get_dataset_stats(plugin, voltage_level)
    
    def get_available_datasets(self, plugin: str = None) -> Dict[str, Dict]:
        """获取可用数据集"""
        datasets = self.downloader.get_available_datasets(plugin)
        return {
            k: {
                "id": v.id,
                "name": v.name,
                "name_zh": v.name_zh,
                "plugin": v.plugin,
                "image_count": v.image_count,
                "can_auto_download": v.can_auto_download,
                "manual_url": v.manual_url,
                "unavailable_reason": v.unavailable_reason,
                "class_names": v.class_names
            }
            for k, v in datasets.items()
        }
    
    def download_dataset(self, dataset_id: str, voltage_level: str) -> bool:
        """下载数据集"""
        return self.downloader.download(dataset_id, voltage_level)
    
    def get_download_progress(self, dataset_id: str) -> Dict:
        """获取下载进度"""
        progress = self.downloader.get_progress(dataset_id)
        if progress:
            return progress.to_dict()
        return {
            "dataset_id": dataset_id,
            "status": "not_started",
            "progress": -1,
            "message": ""
        }
    
    def create_task(self, voltage_level: str, plugin: str) -> TrainingTask:
        """创建新训练任务"""
        plugin = batch_service.to_legacy_plugin(batch_service.normalize_plugin_id(plugin))
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
    
    def start_training(self, task_id: str, epochs: int = 100, batch_size: int = 16):
        """启动训练"""
        if task_id not in self.tasks:
            raise ValueError(f"任务不存在: {task_id}")
        
        task = self.tasks[task_id]
        task.status = TrainingStatus.TRAINING
        task.total_epochs = epochs
        task.message = "正在启动训练..."
        self._save_results()
        
        # 后台线程执行训练
        thread = threading.Thread(
            target=self._run_training,
            args=(task_id, epochs, batch_size),
            daemon=True
        )
        thread.start()
    
    def _run_training(self, task_id: str, epochs: int, batch_size: int):
        """执行训练（后台线程）"""
        task = self.tasks[task_id]
        
        try:
            # 检查数据
            data_yaml = DATA_PATH / "processed" / task.voltage_level / task.plugin / "data.yaml"
            
            if not data_yaml.exists():
                raise FileNotFoundError(f"数据配置不存在: {data_yaml}")
            
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
            
            # 模拟训练进度（实际项目中使用ultralytics）
            logger.info(f"开始训练: {task_id}, 模型: {model_size}, epochs: {epochs}")
            
            for epoch in range(1, epochs + 1):
                # 检查是否取消
                if task.status == TrainingStatus.CANCELLED:
                    logger.info(f"训练已取消: {task_id}")
                    return
                
                time.sleep(0.3)  # 模拟训练时间
                
                task.current_epoch = epoch
                task.progress = epoch / epochs * 100
                task.best_map50 = 0.5 + 0.4 * (epoch / epochs)  # 模拟指标
                task.message = f"训练中: Epoch {epoch}/{epochs}"
                task.updated_at = datetime.now().isoformat()
                self._save_results()
            
            # 保存模型（创建占位文件）
            model_path = checkpoint_dir / "best.pt"
            if not model_path.exists():
                model_path.write_text("placeholder model file")
            
            task.model_path = str(model_path)
            task.status = TrainingStatus.COMPLETED
            task.progress = 100.0
            task.message = f"训练完成! mAP50: {task.best_map50:.4f}"
            logger.info(f"训练完成: {task_id}")
            
        except Exception as e:
            task.status = TrainingStatus.FAILED
            task.message = f"训练失败: {str(e)}"
            logger.error(f"训练失败 {task_id}: {e}", exc_info=True)
        
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

# 请求模型
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
async def list_all_datasets():
    """列出所有可用数据集"""
    return {
        "success": True,
        "datasets": training_manager.get_available_datasets()
    }


@router.get("/datasets/{plugin}")
async def get_plugin_datasets(plugin: str):
    """获取指定插件的数据集"""
    return {
        "success": True,
        "plugin": plugin,
        "datasets": training_manager.get_available_datasets(plugin)
    }


@router.post("/download")
async def download_dataset(request: DownloadRequest, background_tasks: BackgroundTasks):
    """下载数据集"""
    dataset_id = request.dataset_id
    voltage_level = request.voltage_level
    
    config = training_manager.downloader.get_dataset_config(dataset_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"数据集不存在: {dataset_id}")
    
    if not config.can_auto_download:
        return {
            "success": False,
            "message": "该数据集无法自动下载",
            "reason": config.unavailable_reason,
            "manual_url": config.manual_url
        }
    
    # 启动下载
    success = training_manager.download_dataset(dataset_id, voltage_level)
    
    if not success:
        return {
            "success": False,
            "message": "下载启动失败，可能正在下载中"
        }
    
    return {
        "success": True,
        "message": f"开始下载: {config.name_zh}",
        "dataset_id": dataset_id
    }


@router.get("/download/progress/{dataset_id}")
async def get_download_progress(dataset_id: str):
    """获取下载进度"""
    progress = training_manager.get_download_progress(dataset_id)
    return {
        "success": True,
        **progress
    }


@router.post("/start")
async def start_training(request: TrainRequest):
    """启动训练"""
    dataset_status = training_manager.get_dataset_status(request.plugin, request.voltage_level)
    
    if dataset_status["total_count"] < 10:
        return {
            "success": False,
            "message": "训练数据不足，请先下载数据集",
            "dataset_status": dataset_status
        }
    
    task = training_manager.create_task(request.voltage_level, request.plugin)
    training_manager.start_training(task.task_id, request.epochs, request.batch_size)
    
    return {
        "success": True,
        "message": "训练已启动",
        "task_id": task.task_id,
        "task": task.to_dict()
    }


@router.post("/cancel/{task_id}")
async def cancel_training(task_id: str):
    """取消训练"""
    training_manager.cancel_training(task_id)
    return {"success": True, "message": "训练已取消"}


@router.get("/tasks")
async def list_tasks():
    """列出所有任务"""
    return {
        "success": True,
        "tasks": [t.to_dict() for t in training_manager.tasks.values()]
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """获取任务详情"""
    if task_id not in training_manager.tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return {
        "success": True,
        "task": training_manager.tasks[task_id].to_dict()
    }


@router.get("/voltage_levels")
async def list_voltage_levels():
    """列出电压等级"""
    return {
        "success": True,
        "voltage_levels": [
            "UHV_1000kV_AC", "UHV_800kV_DC",
            "EHV_500kV", "EHV_330kV", "EHV_750kV",
            "HV_220kV", "HV_110kV",
            "MV_35kV", "MV_66kV",
            "LV_10kV", "LV_6kV", "LV_380V"
        ]
    }


@router.get("/plugins")
async def list_plugins():
    """列出插件"""
    return {
        "success": True,
        "plugins": ["transformer", "switch", "busbar", "capacitor", "meter"]
    }


@router.get("/capabilities")
async def get_training_capabilities():
    """返回 v1 / v2 训练能力边界，供平台迁移判断使用"""
    try:
        from training.registry import list_supported_plugins

        v2_plugins = list_supported_plugins()
    except Exception:
        v2_plugins = []

    return {
        "success": True,
        "api_version": "v1-legacy",
        "legacy_scope": {
            "plugins": ["transformer", "switch", "busbar", "capacitor", "meter"],
            "task_types": ["detection"],
            "model_artifact": "best.pt"
        },
        "recommended_v2_scope": {
            "router_prefix": "/v2",
            "supports_temporal_training": True,
            "plugins": v2_plugins
        },
        "migration_notes": [
            "视觉旧接口继续保留，用于 voltage_level + plugin 的兼容调用。",
            "时序/数值异常类插件请使用 training_api_v2.py。",
            "新的模型导出和 registry 调度统一走 plugin_id + task_type + version。"
        ]
    }


# =============================================================================
# 集成函数
# =============================================================================
def integrate_training_routes(app):
    """集成训练路由到主应用"""
    app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="训练管理API")
    app.include_router(router)
    
    uvicorn.run(app, host="127.0.0.1", port=8081)
