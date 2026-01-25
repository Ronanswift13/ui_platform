#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据导入和训练管道
======================

基于迭代方案实现:
- 多格式数据导入 (图像/点云/时序)
- 自动标注增强
- 训练样本管理
- 数据验证和清洗

版本: 2.0.0
作者: 变电站AI巡检系统
"""

from __future__ import annotations
import logging
import os
import json
import hashlib
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举和数据结构
# =============================================================================

class DataType(Enum):
    """数据类型"""
    IMAGE = "image"
    POINT_CLOUD = "point_cloud"
    TIME_SERIES = "time_series"
    VIDEO = "video"
    AUDIO = "audio"
    THERMAL = "thermal"
    MIXED = "mixed"


class AnnotationType(Enum):
    """标注类型"""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    KEYPOINT = "keypoint"
    REGRESSION = "regression"
    SEQUENCE = "sequence"


class DataQuality(Enum):
    """数据质量"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


class ImportStatus(Enum):
    """导入状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DataConfig:
    """数据配置"""
    # 存储路径
    data_root: str = "data"
    raw_dir: str = "raw"
    processed_dir: str = "processed"
    annotations_dir: str = "annotations"
    
    # 支持的格式
    image_formats: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"])
    video_formats: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
    pointcloud_formats: List[str] = field(default_factory=lambda: [".pcd", ".ply", ".bin", ".npy"])
    timeseries_formats: List[str] = field(default_factory=lambda: [".csv", ".json", ".parquet"])
    
    # 图像预处理
    image_size: Tuple[int, int] = (640, 640)
    normalize: bool = True
    augment: bool = True
    
    # 验证参数
    min_image_size: Tuple[int, int] = (32, 32)
    max_image_size: Tuple[int, int] = (8192, 8192)
    min_points: int = 100
    
    # 批处理
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class DataSample:
    """数据样本"""
    sample_id: str
    data_type: DataType
    file_path: str
    annotation_path: Optional[str] = None
    
    # 元数据
    timestamp: float = 0.0
    source: str = ""
    device_id: str = ""
    
    # 标注
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    # 质量
    quality: DataQuality = DataQuality.GOOD
    quality_score: float = 1.0
    
    # 状态
    status: ImportStatus = ImportStatus.PENDING
    
    # 哈希
    file_hash: str = ""
    
    def __post_init__(self):
        if not self.sample_id:
            self.sample_id = self._generate_id()
    
    def _generate_id(self) -> str:
        content = f"{self.file_path}_{self.timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class ImportResult:
    """导入结果"""
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    samples: List[DataSample] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0


@dataclass
class DatasetStats:
    """数据集统计"""
    total_samples: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_quality: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    total_size_mb: float = 0.0
    annotation_coverage: float = 0.0


# =============================================================================
# 数据验证器
# =============================================================================

class DataValidator:
    """数据验证器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def validate_image(self, file_path: str) -> Tuple[bool, DataQuality, str]:
        """验证图像"""
        try:
            import cv2
            
            img = cv2.imread(file_path)
            if img is None:
                return False, DataQuality.INVALID, "无法读取图像"
            
            h, w = img.shape[:2]
            
            # 检查尺寸
            min_h, min_w = self.config.min_image_size
            max_h, max_w = self.config.max_image_size
            
            if h < min_h or w < min_w:
                return False, DataQuality.INVALID, f"图像太小: {w}x{h}"
            
            if h > max_h or w > max_w:
                return True, DataQuality.ACCEPTABLE, f"图像较大: {w}x{h}"
            
            # 检查质量
            blur = cv2.Laplacian(img, cv2.CV_64F).var()
            
            if blur < 50:
                return True, DataQuality.POOR, "图像模糊"
            elif blur < 100:
                return True, DataQuality.ACCEPTABLE, "图像清晰度一般"
            elif blur < 500:
                return True, DataQuality.GOOD, ""
            else:
                return True, DataQuality.EXCELLENT, ""
                
        except ImportError:
            # 无cv2，简单验证
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return True, DataQuality.GOOD, ""
            return False, DataQuality.INVALID, "文件不存在或为空"
        except Exception as e:
            return False, DataQuality.INVALID, str(e)
    
    def validate_pointcloud(self, file_path: str) -> Tuple[bool, DataQuality, str]:
        """验证点云"""
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == ".npy":
                points = np.load(file_path)
            elif ext == ".bin":
                points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            elif ext == ".pcd":
                # 简单PCD解析
                points = self._read_pcd(file_path)
            else:
                return False, DataQuality.INVALID, f"不支持的格式: {ext}"
            
            num_points = len(points)
            
            if num_points < self.config.min_points:
                return False, DataQuality.INVALID, f"点数太少: {num_points}"
            
            if num_points < 1000:
                return True, DataQuality.POOR, f"点数较少: {num_points}"
            elif num_points < 10000:
                return True, DataQuality.ACCEPTABLE, ""
            elif num_points < 100000:
                return True, DataQuality.GOOD, ""
            else:
                return True, DataQuality.EXCELLENT, ""
                
        except Exception as e:
            return False, DataQuality.INVALID, str(e)
    
    def _read_pcd(self, file_path: str) -> np.ndarray:
        """简单PCD读取"""
        points = []
        data_started = False
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('DATA'):
                    data_started = True
                    continue
                if data_started:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        points.append([float(p) for p in parts[:3]])
        
        return np.array(points)
    
    def validate_timeseries(self, file_path: str) -> Tuple[bool, DataQuality, str]:
        """验证时序数据"""
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == ".csv":
                import csv
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                num_rows = len(rows) - 1  # 减去表头
            elif ext == ".json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
                num_rows = len(data) if isinstance(data, list) else 1
            else:
                return False, DataQuality.INVALID, f"不支持的格式: {ext}"
            
            if num_rows < 10:
                return False, DataQuality.INVALID, f"数据点太少: {num_rows}"
            elif num_rows < 100:
                return True, DataQuality.POOR, ""
            elif num_rows < 1000:
                return True, DataQuality.ACCEPTABLE, ""
            else:
                return True, DataQuality.GOOD, ""
                
        except Exception as e:
            return False, DataQuality.INVALID, str(e)
    
    def validate(self, file_path: str, data_type: DataType) -> Tuple[bool, DataQuality, str]:
        """通用验证"""
        if data_type == DataType.IMAGE:
            return self.validate_image(file_path)
        elif data_type == DataType.POINT_CLOUD:
            return self.validate_pointcloud(file_path)
        elif data_type == DataType.TIME_SERIES:
            return self.validate_timeseries(file_path)
        elif data_type == DataType.THERMAL:
            return self.validate_image(file_path)
        else:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return True, DataQuality.GOOD, ""
            return False, DataQuality.INVALID, "文件无效"


# =============================================================================
# 自动标注器
# =============================================================================

class AutoAnnotator:
    """自动标注器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._annotators: Dict[str, Callable] = {}
    
    def register_annotator(self, name: str, func: Callable):
        """注册标注器"""
        self._annotators[name] = func
    
    def annotate(self, sample: DataSample, annotator_name: str = None) -> Dict[str, Any]:
        """自动标注"""
        annotations = {}
        
        if annotator_name and annotator_name in self._annotators:
            annotations = self._annotators[annotator_name](sample)
        else:
            # 默认标注
            if sample.data_type == DataType.IMAGE:
                annotations = self._annotate_image(sample)
            elif sample.data_type == DataType.POINT_CLOUD:
                annotations = self._annotate_pointcloud(sample)
            elif sample.data_type == DataType.TIME_SERIES:
                annotations = self._annotate_timeseries(sample)
        
        return annotations
    
    def _annotate_image(self, sample: DataSample) -> Dict[str, Any]:
        """图像自动标注"""
        try:
            import cv2
            
            img = cv2.imread(sample.file_path)
            if img is None:
                return {}
            
            h, w = img.shape[:2]
            
            # 基本信息
            annotations = {
                "width": w,
                "height": h,
                "channels": img.shape[2] if len(img.shape) > 2 else 1,
            }
            
            # 简单分析
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            annotations["brightness"] = float(np.mean(gray))
            annotations["contrast"] = float(np.std(gray))
            
            return annotations
            
        except Exception as e:
            logger.error(f"图像标注失败: {e}")
            return {}
    
    def _annotate_pointcloud(self, sample: DataSample) -> Dict[str, Any]:
        """点云自动标注"""
        try:
            ext = Path(sample.file_path).suffix.lower()
            
            if ext == ".npy":
                points = np.load(sample.file_path)
            elif ext == ".bin":
                points = np.fromfile(sample.file_path, dtype=np.float32).reshape(-1, 4)
            else:
                return {}
            
            return {
                "num_points": len(points),
                "bbox_min": points[:, :3].min(axis=0).tolist(),
                "bbox_max": points[:, :3].max(axis=0).tolist(),
                "centroid": points[:, :3].mean(axis=0).tolist(),
            }
            
        except Exception as e:
            logger.error(f"点云标注失败: {e}")
            return {}
    
    def _annotate_timeseries(self, sample: DataSample) -> Dict[str, Any]:
        """时序数据自动标注"""
        try:
            ext = Path(sample.file_path).suffix.lower()
            
            if ext == ".csv":
                import csv
                with open(sample.file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                
                if not rows:
                    return {}
                
                return {
                    "num_rows": len(rows),
                    "columns": list(rows[0].keys()),
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"时序标注失败: {e}")
            return {}


# =============================================================================
# 数据处理器
# =============================================================================

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def process_image(self, src_path: str, dst_path: str) -> bool:
        """处理图像"""
        try:
            import cv2
            
            img = cv2.imread(src_path)
            if img is None:
                return False
            
            # 调整大小
            target_size = self.config.image_size
            img = cv2.resize(img, target_size)
            
            # 归一化 (如果需要)
            if self.config.normalize:
                img = img.astype(np.float32) / 255.0
            
            # 保存
            if self.config.normalize:
                np.save(dst_path.replace('.jpg', '.npy').replace('.png', '.npy'), img)
            else:
                cv2.imwrite(dst_path, img)
            
            return True
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            return False
    
    def process_pointcloud(self, src_path: str, dst_path: str) -> bool:
        """处理点云"""
        try:
            ext = Path(src_path).suffix.lower()
            
            if ext == ".npy":
                points = np.load(src_path)
            elif ext == ".bin":
                points = np.fromfile(src_path, dtype=np.float32).reshape(-1, 4)
            else:
                return False
            
            # 下采样
            if len(points) > 100000:
                indices = np.random.choice(len(points), 100000, replace=False)
                points = points[indices]
            
            # 归一化坐标
            center = points[:, :3].mean(axis=0)
            points[:, :3] -= center
            
            scale = np.abs(points[:, :3]).max()
            if scale > 0:
                points[:, :3] /= scale
            
            np.save(dst_path, points)
            return True
            
        except Exception as e:
            logger.error(f"点云处理失败: {e}")
            return False
    
    def process(self, sample: DataSample, output_dir: str) -> Optional[str]:
        """处理样本"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            src_path = sample.file_path
            filename = Path(src_path).name
            dst_path = os.path.join(output_dir, filename)
            
            if sample.data_type == DataType.IMAGE:
                success = self.process_image(src_path, dst_path)
            elif sample.data_type == DataType.POINT_CLOUD:
                dst_path = dst_path.rsplit('.', 1)[0] + '.npy'
                success = self.process_pointcloud(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
                success = True
            
            return dst_path if success else None
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            return None


# =============================================================================
# 数据导入管道
# =============================================================================

class DataImportPipeline:
    """数据导入管道"""
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        
        # 组件
        self.validator = DataValidator(self.config)
        self.annotator = AutoAnnotator(self.config)
        self.processor = DataProcessor(self.config)
        
        # 存储
        self._samples: Dict[str, DataSample] = {}
        
        # 确保目录存在
        self._ensure_directories()
        
        # 统计
        self._stats = DatasetStats()
        
        logger.info("数据导入管道初始化完成")
    
    def _ensure_directories(self):
        """确保目录存在"""
        base = Path(self.config.data_root)
        (base / self.config.raw_dir).mkdir(parents=True, exist_ok=True)
        (base / self.config.processed_dir).mkdir(parents=True, exist_ok=True)
        (base / self.config.annotations_dir).mkdir(parents=True, exist_ok=True)
    
    def _detect_data_type(self, file_path: str) -> DataType:
        """检测数据类型"""
        ext = Path(file_path).suffix.lower()
        
        if ext in self.config.image_formats:
            return DataType.IMAGE
        elif ext in self.config.video_formats:
            return DataType.VIDEO
        elif ext in self.config.pointcloud_formats:
            return DataType.POINT_CLOUD
        elif ext in self.config.timeseries_formats:
            return DataType.TIME_SERIES
        else:
            return DataType.MIXED
    
    def _compute_file_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def import_file(self, 
                    file_path: str,
                    source: str = "",
                    device_id: str = "",
                    auto_annotate: bool = True,
                    auto_process: bool = True) -> Optional[DataSample]:
        """
        导入单个文件
        
        Args:
            file_path: 文件路径
            source: 数据来源
            device_id: 设备ID
            auto_annotate: 自动标注
            auto_process: 自动处理
            
        Returns:
            数据样本
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return None
        
        # 检测类型
        data_type = self._detect_data_type(file_path)
        
        # 计算哈希
        file_hash = self._compute_file_hash(file_path)
        
        # 检查重复
        for sample in self._samples.values():
            if sample.file_hash == file_hash:
                logger.warning(f"文件已存在: {file_path}")
                return sample
        
        # 创建样本
        sample = DataSample(
            sample_id="",
            data_type=data_type,
            file_path=file_path,
            timestamp=time.time(),
            source=source,
            device_id=device_id,
            file_hash=file_hash,
            status=ImportStatus.PROCESSING
        )
        
        # 验证
        valid, quality, message = self.validator.validate(file_path, data_type)
        sample.quality = quality
        
        if not valid:
            sample.status = ImportStatus.FAILED
            logger.error(f"验证失败: {file_path} - {message}")
            return None
        
        # 自动标注
        if auto_annotate:
            sample.annotations = self.annotator.annotate(sample)
        
        # 处理
        if auto_process:
            processed_dir = os.path.join(self.config.data_root, self.config.processed_dir)
            processed_path = self.processor.process(sample, processed_dir)
            
            if processed_path:
                sample.file_path = processed_path
        
        # 保存标注
        if sample.annotations:
            ann_path = os.path.join(
                self.config.data_root,
                self.config.annotations_dir,
                f"{sample.sample_id}.json"
            )
            with open(ann_path, 'w') as f:
                json.dump(sample.annotations, f, indent=2)
            sample.annotation_path = ann_path
        
        sample.status = ImportStatus.COMPLETED
        self._samples[sample.sample_id] = sample
        
        # 更新统计
        self._update_stats()
        
        logger.info(f"导入成功: {file_path} -> {sample.sample_id}")
        return sample
    
    def import_directory(self,
                         directory: str,
                         recursive: bool = True,
                         source: str = "",
                         **kwargs) -> ImportResult:
        """
        导入目录
        
        Args:
            directory: 目录路径
            recursive: 递归导入
            source: 数据来源
            
        Returns:
            导入结果
        """
        start_time = time.time()
        result = ImportResult()
        
        # 收集文件
        files = []
        if recursive:
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
        else:
            files = [os.path.join(directory, f) for f in os.listdir(directory)
                    if os.path.isfile(os.path.join(directory, f))]
        
        result.total = len(files)
        
        # 导入每个文件
        for file_path in files:
            try:
                sample = self.import_file(file_path, source=source, **kwargs)
                
                if sample:
                    if sample.status == ImportStatus.COMPLETED:
                        result.successful += 1
                        result.samples.append(sample)
                    else:
                        result.failed += 1
                else:
                    result.failed += 1
                    
            except Exception as e:
                result.failed += 1
                result.errors.append(f"{file_path}: {str(e)}")
        
        result.duration = time.time() - start_time
        
        logger.info(f"批量导入完成: {result.successful}/{result.total} 成功, "
                   f"耗时 {result.duration:.1f}s")
        
        return result
    
    def _update_stats(self):
        """更新统计"""
        self._stats.total_samples = len(self._samples)
        
        # 按类型统计
        self._stats.by_type = {}
        for sample in self._samples.values():
            t = sample.data_type.value
            self._stats.by_type[t] = self._stats.by_type.get(t, 0) + 1
        
        # 按质量统计
        self._stats.by_quality = {}
        for sample in self._samples.values():
            q = sample.quality.value
            self._stats.by_quality[q] = self._stats.by_quality.get(q, 0) + 1
        
        # 按来源统计
        self._stats.by_source = {}
        for sample in self._samples.values():
            s = sample.source or "unknown"
            self._stats.by_source[s] = self._stats.by_source.get(s, 0) + 1
        
        # 标注覆盖率
        annotated = sum(1 for s in self._samples.values() if s.annotations)
        self._stats.annotation_coverage = annotated / len(self._samples) if self._samples else 0
    
    def get_sample(self, sample_id: str) -> Optional[DataSample]:
        """获取样本"""
        return self._samples.get(sample_id)
    
    def list_samples(self, 
                     data_type: DataType = None,
                     quality: DataQuality = None,
                     source: str = None) -> List[DataSample]:
        """列出样本"""
        samples = list(self._samples.values())
        
        if data_type:
            samples = [s for s in samples if s.data_type == data_type]
        
        if quality:
            samples = [s for s in samples if s.quality == quality]
        
        if source:
            samples = [s for s in samples if s.source == source]
        
        return samples
    
    def export_dataset(self, 
                       output_dir: str,
                       split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)) -> Dict[str, int]:
        """
        导出数据集
        
        Args:
            output_dir: 输出目录
            split_ratio: 训练/验证/测试比例
            
        Returns:
            各集合样本数
        """
        os.makedirs(output_dir, exist_ok=True)
        
        samples = list(self._samples.values())
        np.random.shuffle(samples)
        
        n = len(samples)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])
        
        splits = {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:]
        }
        
        counts = {}
        
        for split_name, split_samples in splits.items():
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            manifest = []
            
            for sample in split_samples:
                # 复制文件
                src = sample.file_path
                dst = os.path.join(split_dir, os.path.basename(src))
                
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                
                manifest.append({
                    "sample_id": sample.sample_id,
                    "file_path": os.path.basename(dst),
                    "data_type": sample.data_type.value,
                    "annotations": sample.annotations
                })
            
            # 保存清单
            manifest_path = os.path.join(split_dir, "manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            counts[split_name] = len(split_samples)
        
        logger.info(f"数据集导出完成: train={counts['train']}, val={counts['val']}, test={counts['test']}")
        
        return counts
    
    def get_statistics(self) -> DatasetStats:
        """获取统计信息"""
        self._update_stats()
        return self._stats


# =============================================================================
# 工厂函数
# =============================================================================

def create_import_pipeline(data_root: str = "data", **kwargs) -> DataImportPipeline:
    """创建导入管道"""
    config = DataConfig(data_root=data_root, **kwargs)
    return DataImportPipeline(config)


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建管道
    pipeline = create_import_pipeline(data_root="/tmp/test_data")
    
    # 创建测试数据
    test_dir = "/tmp/test_import"
    os.makedirs(test_dir, exist_ok=True)
    
    # 生成测试图像
    for i in range(5):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        try:
            import cv2
            cv2.imwrite(os.path.join(test_dir, f"test_{i}.jpg"), img)
        except ImportError:
            # 无cv2，创建占位文件
            with open(os.path.join(test_dir, f"test_{i}.dat"), 'wb') as f:
                f.write(img.tobytes())
    
    # 生成测试时序数据
    import csv
    with open(os.path.join(test_dir, "timeseries.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "value"])
        for i in range(100):
            writer.writerow([i, np.random.random()])
    
    # 导入目录
    result = pipeline.import_directory(test_dir, source="test")
    
    print(f"\n导入结果:")
    print(f"  总数: {result.total}")
    print(f"  成功: {result.successful}")
    print(f"  失败: {result.failed}")
    print(f"  耗时: {result.duration:.2f}s")
    
    # 统计
    stats = pipeline.get_statistics()
    print(f"\n数据集统计:")
    print(f"  总样本: {stats.total_samples}")
    print(f"  按类型: {stats.by_type}")
    print(f"  按质量: {stats.by_quality}")
    print(f"  标注覆盖: {stats.annotation_coverage:.1%}")
    
    # 导出
    counts = pipeline.export_dataset("/tmp/exported_dataset")
    print(f"\n导出结果: {counts}")
