"""
MLOps流水线模块
输变电激光星芒破夜绘明监测平台

基于UPDATE.md中期迭代计划:
- 训练流水线自动化
- 模型版本管理与注册
- Canary部署与回滚
- 主动学习策略
- 模型性能监控

功能:
- 训练任务调度
- 模型注册表
- A/B测试
- 数据版本控制
- 自动标注流程

版本: 1.0.0
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import shutil
import time
import threading
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举定义
# =============================================================================

class ModelStatus(Enum):
    """模型状态"""
    DRAFT = "draft"                 # 草稿
    TRAINING = "training"           # 训练中
    VALIDATING = "validating"       # 验证中
    STAGING = "staging"             # 预发布
    PRODUCTION = "production"       # 生产
    ARCHIVED = "archived"           # 已归档
    FAILED = "failed"               # 失败


class TrainingStatus(Enum):
    """训练任务状态"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentStrategy(Enum):
    """部署策略"""
    DIRECT = "direct"               # 直接部署
    CANARY = "canary"               # 金丝雀部署
    BLUE_GREEN = "blue_green"       # 蓝绿部署
    SHADOW = "shadow"               # 影子部署


class SamplingStrategy(Enum):
    """主动学习采样策略"""
    RANDOM = "random"
    UNCERTAINTY = "uncertainty"     # 不确定性采样
    DIVERSITY = "diversity"         # 多样性采样
    ENSEMBLE = "ensemble"           # 集成一致性
    MARGIN = "margin"               # 边界采样


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class ModelVersion:
    """模型版本"""
    model_id: str
    version: str
    name: str
    description: str = ""
    status: ModelStatus = ModelStatus.DRAFT

    # 文件信息
    model_path: str = ""
    config_path: str = ""
    checkpoint_path: str = ""
    file_hash: str = ""
    file_size: int = 0

    # 训练信息
    training_id: Optional[str] = None
    training_config: Dict[str, Any] = field(default_factory=dict)
    dataset_version: str = ""

    # 性能指标
    metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    # 时间戳
    created_at: str = ""
    updated_at: str = ""
    trained_at: Optional[str] = None
    deployed_at: Optional[str] = None

    # 元数据
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "training_id": self.training_id,
            "training_config": self.training_config,
            "dataset_version": self.dataset_version,
            "metrics": self.metrics,
            "validation_metrics": self.validation_metrics,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "trained_at": self.trained_at,
            "deployed_at": self.deployed_at,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """从字典创建"""
        return cls(
            model_id=data.get("model_id", ""),
            version=data.get("version", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            status=ModelStatus(data.get("status", "draft")),
            model_path=data.get("model_path", ""),
            config_path=data.get("config_path", ""),
            checkpoint_path=data.get("checkpoint_path", ""),
            file_hash=data.get("file_hash", ""),
            file_size=data.get("file_size", 0),
            training_id=data.get("training_id"),
            training_config=data.get("training_config", {}),
            dataset_version=data.get("dataset_version", ""),
            metrics=data.get("metrics", {}),
            validation_metrics=data.get("validation_metrics", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            trained_at=data.get("trained_at"),
            deployed_at=data.get("deployed_at"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TrainingJob:
    """训练任务"""
    job_id: str
    model_id: str
    status: TrainingStatus = TrainingStatus.PENDING

    # 配置
    config: Dict[str, Any] = field(default_factory=dict)
    dataset_path: str = ""
    dataset_version: str = ""
    output_path: str = ""

    # 进度
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0

    # 指标
    train_loss: float = 0.0
    val_loss: float = 0.0
    best_metric: float = 0.0
    metrics_history: List[Dict[str, float]] = field(default_factory=list)

    # 资源
    gpu_ids: List[int] = field(default_factory=list)
    memory_usage: float = 0.0

    # 时间戳
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # 错误信息
    error_message: str = ""

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetVersion:
    """数据集版本"""
    dataset_id: str
    version: str
    name: str
    description: str = ""

    # 文件信息
    data_path: str = ""
    annotation_path: str = ""
    split_info: Dict[str, int] = field(default_factory=dict)  # train/val/test数量

    # 统计
    total_samples: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    annotation_quality: float = 0.0

    # 来源
    parent_version: Optional[str] = None
    created_from: str = ""  # upload/augmentation/active_learning

    # 时间戳
    created_at: str = ""

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentConfig:
    """部署配置"""
    strategy: DeploymentStrategy = DeploymentStrategy.CANARY
    canary_percentage: float = 10.0     # 金丝雀流量比例
    rollout_duration: float = 3600.0    # 渐进式发布时长(秒)
    rollback_threshold: float = 0.1     # 回滚阈值(性能下降比例)
    validation_samples: int = 1000      # 验证样本数
    auto_rollback: bool = True          # 自动回滚


@dataclass
class ActiveLearningConfig:
    """主动学习配置"""
    strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY
    sample_size: int = 100              # 每批采样数量
    uncertainty_threshold: float = 0.7  # 不确定性阈值
    diversity_weight: float = 0.3       # 多样性权重
    min_confidence: float = 0.3         # 最小置信度
    max_confidence: float = 0.7         # 最大置信度


# =============================================================================
# 模型注册表
# =============================================================================

class ModelRegistry:
    """
    模型注册表

    管理所有模型版本的生命周期
    """

    def __init__(self, storage_path: str = "mlops/models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._models: Dict[str, Dict[str, ModelVersion]] = {}  # model_id -> version -> ModelVersion
        self._production_versions: Dict[str, str] = {}  # model_id -> production_version

        self._lock = threading.Lock()
        self._load_registry()

    def _load_registry(self):
        """加载注册表"""
        registry_file = self.storage_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    data = json.load(f)

                for model_id, versions in data.get("models", {}).items():
                    self._models[model_id] = {}
                    for version, version_data in versions.items():
                        self._models[model_id][version] = ModelVersion.from_dict(version_data)

                self._production_versions = data.get("production_versions", {})
                logger.info(f"[ModelRegistry] 加载了 {len(self._models)} 个模型")

            except Exception as e:
                logger.warning(f"[ModelRegistry] 加载注册表失败: {e}")

    def _save_registry(self):
        """保存注册表"""
        registry_file = self.storage_path / "registry.json"

        data = {
            "models": {},
            "production_versions": self._production_versions,
        }

        for model_id, versions in self._models.items():
            data["models"][model_id] = {
                v: mv.to_dict() for v, mv in versions.items()
            }

        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def register(
        self,
        model_id: str,
        version: str,
        name: str,
        model_path: str,
        config: Optional[Dict] = None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> ModelVersion:
        """
        注册新模型版本

        Args:
            model_id: 模型ID
            version: 版本号
            name: 模型名称
            model_path: 模型文件路径
            config: 训练配置
            metrics: 性能指标
            **kwargs: 其他参数

        Returns:
            注册的模型版本
        """
        with self._lock:
            # 计算文件哈希
            file_hash = self._compute_file_hash(model_path)
            file_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0

            # 复制模型文件到存储目录
            model_dir = self.storage_path / model_id / version
            model_dir.mkdir(parents=True, exist_ok=True)

            stored_path = model_dir / Path(model_path).name
            if os.path.exists(model_path):
                shutil.copy2(model_path, stored_path)

            # 创建版本记录
            now = datetime.now().isoformat()
            model_version = ModelVersion(
                model_id=model_id,
                version=version,
                name=name,
                model_path=str(stored_path),
                file_hash=file_hash,
                file_size=file_size,
                training_config=config or {},
                metrics=metrics or {},
                created_at=now,
                updated_at=now,
                **kwargs
            )

            # 保存
            if model_id not in self._models:
                self._models[model_id] = {}
            self._models[model_id][version] = model_version

            self._save_registry()
            logger.info(f"[ModelRegistry] 注册模型: {model_id}:{version}")

            return model_version

    def get_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """获取指定版本"""
        return self._models.get(model_id, {}).get(version)

    def get_production_version(self, model_id: str) -> Optional[ModelVersion]:
        """获取生产版本"""
        prod_version = self._production_versions.get(model_id)
        if prod_version:
            return self.get_version(model_id, prod_version)
        return None

    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """列出所有版本"""
        versions = self._models.get(model_id, {})
        return sorted(versions.values(), key=lambda v: v.created_at, reverse=True)

    def list_models(self) -> List[str]:
        """列出所有模型ID"""
        return list(self._models.keys())

    def promote_to_staging(self, model_id: str, version: str) -> bool:
        """提升到预发布"""
        with self._lock:
            mv = self.get_version(model_id, version)
            if mv is None:
                return False

            mv.status = ModelStatus.STAGING
            mv.updated_at = datetime.now().isoformat()
            self._save_registry()
            return True

    def promote_to_production(self, model_id: str, version: str) -> bool:
        """提升到生产"""
        with self._lock:
            mv = self.get_version(model_id, version)
            if mv is None:
                return False

            # 将当前生产版本归档
            current_prod = self._production_versions.get(model_id)
            if current_prod and current_prod != version:
                old_mv = self.get_version(model_id, current_prod)
                if old_mv:
                    old_mv.status = ModelStatus.ARCHIVED
                    old_mv.updated_at = datetime.now().isoformat()

            # 设置新的生产版本
            mv.status = ModelStatus.PRODUCTION
            mv.deployed_at = datetime.now().isoformat()
            mv.updated_at = mv.deployed_at
            self._production_versions[model_id] = version

            self._save_registry()
            logger.info(f"[ModelRegistry] 部署模型到生产: {model_id}:{version}")
            return True

    def rollback(self, model_id: str, target_version: str) -> bool:
        """回滚到指定版本"""
        return self.promote_to_production(model_id, target_version)

    def archive(self, model_id: str, version: str) -> bool:
        """归档版本"""
        with self._lock:
            mv = self.get_version(model_id, version)
            if mv is None:
                return False

            if self._production_versions.get(model_id) == version:
                return False  # 不能归档生产版本

            mv.status = ModelStatus.ARCHIVED
            mv.updated_at = datetime.now().isoformat()
            self._save_registry()
            return True

    def _compute_file_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        if not os.path.exists(file_path):
            return ""

        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()


# =============================================================================
# 训练调度器
# =============================================================================

class TrainingScheduler:
    """
    训练任务调度器

    管理训练任务的创建、排队和执行
    """

    def __init__(
        self,
        max_concurrent_jobs: int = 2,
        storage_path: str = "mlops/training"
    ):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._job_queue: deque = deque()
        self._running_jobs: Dict[str, TrainingJob] = {}
        self._completed_jobs: Dict[str, TrainingJob] = {}

        self._lock = threading.Lock()
        self._stop_flag = False

        # 训练回调
        self._training_callback: Optional[Callable] = None

    def set_training_callback(self, callback: Callable[[TrainingJob], None]):
        """设置训练回调"""
        self._training_callback = callback

    def submit(
        self,
        model_id: str,
        config: Dict[str, Any],
        dataset_path: str,
        dataset_version: str = "",
        priority: int = 0,
        **kwargs
    ) -> TrainingJob:
        """
        提交训练任务

        Args:
            model_id: 模型ID
            config: 训练配置
            dataset_path: 数据集路径
            dataset_version: 数据集版本
            priority: 优先级
            **kwargs: 其他参数

        Returns:
            训练任务
        """
        job_id = str(uuid.uuid4())[:8]
        output_path = str(self.storage_path / model_id / job_id)
        os.makedirs(output_path, exist_ok=True)

        job = TrainingJob(
            job_id=job_id,
            model_id=model_id,
            status=TrainingStatus.PENDING,
            config=config,
            dataset_path=dataset_path,
            dataset_version=dataset_version,
            output_path=output_path,
            created_at=datetime.now().isoformat(),
            metadata={"priority": priority, **kwargs},
        )

        with self._lock:
            self._job_queue.append(job)
            job.status = TrainingStatus.QUEUED

        logger.info(f"[TrainingScheduler] 提交训练任务: {job_id}")
        return job

    def cancel(self, job_id: str) -> bool:
        """取消任务"""
        with self._lock:
            # 检查队列
            for job in self._job_queue:
                if job.job_id == job_id:
                    job.status = TrainingStatus.CANCELLED
                    self._job_queue.remove(job)
                    return True

            # 检查运行中
            if job_id in self._running_jobs:
                job = self._running_jobs[job_id]
                job.status = TrainingStatus.CANCELLED
                # 实际停止需要在训练循环中检查
                return True

        return False

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """获取任务"""
        with self._lock:
            # 检查运行中
            if job_id in self._running_jobs:
                return self._running_jobs[job_id]

            # 检查已完成
            if job_id in self._completed_jobs:
                return self._completed_jobs[job_id]

            # 检查队列
            for job in self._job_queue:
                if job.job_id == job_id:
                    return job

        return None

    def list_jobs(
        self,
        model_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None
    ) -> List[TrainingJob]:
        """列出任务"""
        with self._lock:
            jobs = list(self._job_queue) + list(self._running_jobs.values())

            if model_id:
                jobs = [j for j in jobs if j.model_id == model_id]
            if status:
                jobs = [j for j in jobs if j.status == status]

            return jobs

    def process_queue(self):
        """处理队列 (在后台线程调用)"""
        while not self._stop_flag:
            with self._lock:
                # 检查是否有空闲槽位
                if len(self._running_jobs) >= self.max_concurrent_jobs:
                    time.sleep(1)
                    continue

                # 获取下一个任务
                if not self._job_queue:
                    time.sleep(1)
                    continue

                job = self._job_queue.popleft()

            # 执行任务
            self._execute_job(job)

    def _execute_job(self, job: TrainingJob):
        """执行训练任务"""
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now().isoformat()

        with self._lock:
            self._running_jobs[job.job_id] = job

        try:
            # 调用训练回调
            if self._training_callback:
                self._training_callback(job)
            else:
                # 模拟训练
                self._simulate_training(job)

            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now().isoformat()

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now().isoformat()
            logger.error(f"[TrainingScheduler] 训练任务失败: {job.job_id}, {e}")

        finally:
            with self._lock:
                del self._running_jobs[job.job_id]
                self._completed_jobs[job.job_id] = job

    def _simulate_training(self, job: TrainingJob):
        """模拟训练过程"""
        job.total_epochs = job.config.get("epochs", 10)
        job.total_steps = job.total_epochs * 100

        for epoch in range(job.total_epochs):
            if job.status == TrainingStatus.CANCELLED:
                break

            job.current_epoch = epoch + 1

            for step in range(100):
                job.current_step = epoch * 100 + step + 1
                job.progress = job.current_step / job.total_steps

                # 模拟损失下降
                job.train_loss = 1.0 / (epoch + 1) + 0.1 * (1 - step / 100)
                job.val_loss = job.train_loss * 1.1

                time.sleep(0.01)  # 模拟计算

            job.metrics_history.append({
                "epoch": epoch + 1,
                "train_loss": job.train_loss,
                "val_loss": job.val_loss,
            })

        job.best_metric = min(m["val_loss"] for m in job.metrics_history) if job.metrics_history else 0


# =============================================================================
# 主动学习引擎
# =============================================================================

class ActiveLearningEngine:
    """
    主动学习引擎

    选择最有价值的样本进行标注
    """

    def __init__(self, config: ActiveLearningConfig):
        self.config = config

        # 待标注队列
        self._pending_samples: deque = deque(maxlen=10000)

        # 已选择的样本
        self._selected_samples: List[Dict] = []

    def add_sample(
        self,
        sample_id: str,
        features: Optional[np.ndarray],
        predictions: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        """
        添加样本到候选池

        Args:
            sample_id: 样本ID
            features: 特征向量
            predictions: 预测概率
            metadata: 元数据
        """
        self._pending_samples.append({
            "sample_id": sample_id,
            "features": features,
            "predictions": predictions,
            "metadata": metadata or {},
        })

    def select_samples(self, n_samples: Optional[int] = None) -> List[Dict]:
        """
        选择样本进行标注

        Args:
            n_samples: 选择数量

        Returns:
            选中的样本列表
        """
        if n_samples is None:
            n_samples = self.config.sample_size

        if len(self._pending_samples) == 0:
            return []

        samples = list(self._pending_samples)

        # 计算采样分数
        for sample in samples:
            sample["score"] = self._compute_score(sample)

        # 根据策略选择
        if self.config.strategy == SamplingStrategy.RANDOM:
            import random
            selected = random.sample(samples, min(n_samples, len(samples)))

        elif self.config.strategy == SamplingStrategy.UNCERTAINTY:
            # 按不确定性排序
            samples.sort(key=lambda x: x["score"], reverse=True)
            selected = samples[:n_samples]

        elif self.config.strategy == SamplingStrategy.DIVERSITY:
            selected = self._select_diverse(samples, n_samples)

        elif self.config.strategy == SamplingStrategy.MARGIN:
            # 按边界排序
            samples.sort(key=lambda x: x["score"], reverse=True)
            selected = samples[:n_samples]

        else:
            selected = samples[:n_samples]

        self._selected_samples.extend(selected)

        # 从待选池移除
        selected_ids = {s["sample_id"] for s in selected}
        self._pending_samples = deque(
            [s for s in self._pending_samples if s["sample_id"] not in selected_ids],
            maxlen=10000
        )

        return selected

    def _compute_score(self, sample: Dict) -> float:
        """计算样本分数"""
        predictions = sample.get("predictions", {})
        if not predictions:
            return 0.0

        probs = list(predictions.values())

        if self.config.strategy == SamplingStrategy.UNCERTAINTY:
            # 熵
            import math
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            return entropy

        elif self.config.strategy == SamplingStrategy.MARGIN:
            # 边界: 最高两个概率的差值
            sorted_probs = sorted(probs, reverse=True)
            if len(sorted_probs) >= 2:
                return 1.0 - (sorted_probs[0] - sorted_probs[1])
            return 0.0

        elif self.config.strategy == SamplingStrategy.ENSEMBLE:
            # 需要多个模型的预测
            return max(probs) if probs else 0.0

        return 0.5

    def _select_diverse(self, samples: List[Dict], n_samples: int) -> List[Dict]:
        """多样性选择"""
        if not samples:
            return []

        # 简单的贪婪多样性选择
        selected = [samples[0]]
        remaining = samples[1:]

        while len(selected) < n_samples and remaining:
            # 找到与已选择样本最不相似的
            best_sample = None
            best_min_sim = -float('inf')

            for sample in remaining:
                if sample.get("features") is None:
                    continue

                min_sim = float('inf')
                for sel in selected:
                    if sel.get("features") is not None:
                        sim = self._compute_similarity(
                            sample["features"],
                            sel["features"]
                        )
                        min_sim = min(min_sim, sim)

                if min_sim > best_min_sim:
                    best_min_sim = min_sim
                    best_sample = sample

            if best_sample:
                selected.append(best_sample)
                remaining.remove(best_sample)
            else:
                break

        return selected

    def _compute_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """计算特征相似度"""
        if f1 is None or f2 is None:
            return 0.0
        return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-10))

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "pending_samples": len(self._pending_samples),
            "selected_samples": len(self._selected_samples),
            "strategy": self.config.strategy.value,
        }


# =============================================================================
# 部署管理器
# =============================================================================

class DeploymentManager:
    """
    部署管理器

    管理模型的部署、金丝雀发布和回滚
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

        # 当前部署状态
        self._deployments: Dict[str, Dict] = {}  # model_id -> deployment_info

    def deploy(
        self,
        model_id: str,
        version: str,
        config: Optional[DeploymentConfig] = None
    ) -> Dict[str, Any]:
        """
        部署模型

        Args:
            model_id: 模型ID
            version: 版本号
            config: 部署配置

        Returns:
            部署信息
        """
        if config is None:
            config = DeploymentConfig()

        model_version = self.registry.get_version(model_id, version)
        if model_version is None:
            raise ValueError(f"模型版本不存在: {model_id}:{version}")

        deployment_id = str(uuid.uuid4())[:8]
        deployment_info = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "version": version,
            "strategy": config.strategy.value,
            "status": "deploying",
            "traffic_percentage": 0.0,
            "started_at": datetime.now().isoformat(),
            "config": {
                "canary_percentage": config.canary_percentage,
                "rollout_duration": config.rollout_duration,
                "rollback_threshold": config.rollback_threshold,
            },
            "metrics": {},
        }

        if config.strategy == DeploymentStrategy.DIRECT:
            # 直接部署
            self.registry.promote_to_production(model_id, version)
            deployment_info["traffic_percentage"] = 100.0
            deployment_info["status"] = "completed"

        elif config.strategy == DeploymentStrategy.CANARY:
            # 金丝雀部署
            deployment_info["traffic_percentage"] = config.canary_percentage
            deployment_info["status"] = "canary"
            # 实际流量切换需要在负载均衡器层面实现

        elif config.strategy == DeploymentStrategy.BLUE_GREEN:
            # 蓝绿部署
            deployment_info["status"] = "validating"
            # 需要准备两套环境

        elif config.strategy == DeploymentStrategy.SHADOW:
            # 影子部署
            deployment_info["traffic_percentage"] = 0.0
            deployment_info["status"] = "shadow"
            # 影子模式只记录不响应

        self._deployments[model_id] = deployment_info
        logger.info(f"[DeploymentManager] 部署模型: {model_id}:{version}, 策略: {config.strategy.value}")

        return deployment_info

    def update_traffic(self, model_id: str, percentage: float) -> bool:
        """更新流量比例"""
        if model_id not in self._deployments:
            return False

        self._deployments[model_id]["traffic_percentage"] = percentage
        self._deployments[model_id]["updated_at"] = datetime.now().isoformat()

        if percentage >= 100.0:
            # 完成部署
            version = self._deployments[model_id]["version"]
            self.registry.promote_to_production(model_id, version)
            self._deployments[model_id]["status"] = "completed"

        return True

    def rollback(self, model_id: str, reason: str = "") -> bool:
        """回滚部署"""
        if model_id not in self._deployments:
            return False

        # 获取上一个生产版本
        versions = self.registry.list_versions(model_id)
        previous_prod = None
        for v in versions:
            if v.status == ModelStatus.ARCHIVED and v.deployed_at:
                previous_prod = v
                break

        if previous_prod:
            self.registry.promote_to_production(model_id, previous_prod.version)
            self._deployments[model_id]["status"] = "rolled_back"
            self._deployments[model_id]["rollback_reason"] = reason
            logger.warning(f"[DeploymentManager] 回滚模型: {model_id} -> {previous_prod.version}")
            return True

        return False

    def get_deployment_status(self, model_id: str) -> Optional[Dict]:
        """获取部署状态"""
        return self._deployments.get(model_id)

    def record_metrics(self, model_id: str, metrics: Dict[str, float]):
        """记录部署指标"""
        if model_id not in self._deployments:
            return

        deployment = self._deployments[model_id]
        if "metrics_history" not in deployment:
            deployment["metrics_history"] = []

        deployment["metrics_history"].append({
            "timestamp": datetime.now().isoformat(),
            **metrics
        })

        # 检查是否需要自动回滚
        config = deployment.get("config", {})
        if config.get("auto_rollback") and deployment["status"] == "canary":
            baseline = deployment.get("baseline_metrics", {})
            if baseline:
                for key, value in metrics.items():
                    if key in baseline:
                        degradation = (baseline[key] - value) / baseline[key]
                        if degradation > config.get("rollback_threshold", 0.1):
                            self.rollback(model_id, f"Performance degradation: {key}")
                            break


# =============================================================================
# MLOps管道
# =============================================================================

class MLOpsPipeline:
    """
    MLOps管道

    整合模型注册、训练调度、主动学习和部署管理
    """

    def __init__(
        self,
        storage_path: str = "mlops",
        max_concurrent_jobs: int = 2
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 组件
        self.registry = ModelRegistry(str(self.storage_path / "models"))
        self.scheduler = TrainingScheduler(
            max_concurrent_jobs=max_concurrent_jobs,
            storage_path=str(self.storage_path / "training")
        )
        self.active_learning = ActiveLearningEngine(ActiveLearningConfig())
        self.deployment = DeploymentManager(self.registry)

        # 设置训练完成回调
        self.scheduler.set_training_callback(self._on_training_complete)

    def _on_training_complete(self, job: TrainingJob):
        """训练完成回调"""
        if job.status == TrainingStatus.COMPLETED:
            # 自动注册模型
            model_path = Path(job.output_path) / "best_model.onnx"
            if model_path.exists():
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.registry.register(
                    model_id=job.model_id,
                    version=version,
                    name=f"{job.model_id}_{version}",
                    model_path=str(model_path),
                    config=job.config,
                    metrics={"val_loss": job.best_metric},
                    training_id=job.job_id,
                )

    def train_model(
        self,
        model_id: str,
        config: Dict[str, Any],
        dataset_path: str,
        **kwargs
    ) -> TrainingJob:
        """训练模型"""
        return self.scheduler.submit(
            model_id=model_id,
            config=config,
            dataset_path=dataset_path,
            **kwargs
        )

    def deploy_model(
        self,
        model_id: str,
        version: str,
        strategy: DeploymentStrategy = DeploymentStrategy.CANARY
    ) -> Dict[str, Any]:
        """部署模型"""
        config = DeploymentConfig(strategy=strategy)
        return self.deployment.deploy(model_id, version, config)

    def get_model_info(self, model_id: str, version: Optional[str] = None) -> Optional[Dict]:
        """获取模型信息"""
        if version:
            mv = self.registry.get_version(model_id, version)
        else:
            mv = self.registry.get_production_version(model_id)

        if mv:
            return mv.to_dict()
        return None

    def get_status(self) -> Dict[str, Any]:
        """获取MLOps状态"""
        return {
            "models": len(self.registry.list_models()),
            "training_jobs": len(self.scheduler.list_jobs()),
            "active_learning": self.active_learning.get_statistics(),
        }


# =============================================================================
# 全局实例
# =============================================================================

_mlops_pipeline: Optional[MLOpsPipeline] = None


def get_mlops_pipeline() -> MLOpsPipeline:
    """获取MLOps管道实例"""
    global _mlops_pipeline
    if _mlops_pipeline is None:
        _mlops_pipeline = MLOpsPipeline()
    return _mlops_pipeline
