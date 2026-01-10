# -*- coding: utf-8 -*-
"""
数据采集管理器
==============

实现方案中的数据采集功能：
1. 多模态数据收集计划
2. 数据与设备工况、故障日志对齐
3. 标签管理
4. 数据预处理

作者: AI巡检系统
版本: 1.0.0
"""

from __future__ import annotations
import os
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 数据类型定义
# =============================================================================

class ModalityType(str, Enum):
    """模态类型"""
    ACOUSTIC = "acoustic"           # 声学数据
    ULTRASONIC = "ultrasonic"       # 超声波数据
    GAS = "gas"                     # 气体数据
    HYPERSPECTRAL = "hyperspectral" # 高光谱数据
    THERMAL = "thermal"             # 热成像数据
    VISUAL = "visual"               # 可见光图像
    LIDAR = "lidar"                 # 激光雷达点云
    VIBRATION = "vibration"         # 振动数据
    ELECTRICAL = "electrical"       # 电气参数


class DataQuality(str, Enum):
    """数据质量等级"""
    HIGH = "high"           # 高质量
    MEDIUM = "medium"       # 中等质量
    LOW = "low"             # 低质量
    INVALID = "invalid"     # 无效


class LabelStatus(str, Enum):
    """标注状态"""
    UNLABELED = "unlabeled"     # 未标注
    AUTO_LABELED = "auto"       # 自动标注
    MANUAL_LABELED = "manual"   # 人工标注
    VERIFIED = "verified"       # 已验证
    REJECTED = "rejected"       # 已拒绝


@dataclass
class DataSample:
    """数据样本"""
    sample_id: str
    modality: ModalityType
    device_id: str
    timestamp: float
    data_path: str
    data_hash: str
    quality: DataQuality = DataQuality.MEDIUM
    label_status: LabelStatus = LabelStatus.UNLABELED
    labels: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    aligned_samples: List[str] = field(default_factory=list)  # 对齐的其他模态样本ID


@dataclass
class CollectionTask:
    """采集任务"""
    task_id: str
    name: str
    modalities: List[ModalityType]
    device_ids: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    interval_seconds: int = 3600
    status: str = "pending"
    collected_count: int = 0
    target_count: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FaultRecord:
    """故障记录"""
    record_id: str
    device_id: str
    fault_type: str
    fault_level: str  # critical, alarm, warning, attention
    start_time: datetime
    end_time: Optional[datetime] = None
    description: str = ""
    verified: bool = False
    related_samples: List[str] = field(default_factory=list)


# =============================================================================
# 数据采集管理器
# =============================================================================

class DataCollectionManager:
    """
    数据采集管理器
    
    负责：
    - 管理多模态数据采集任务
    - 数据质量评估
    - 数据对齐
    - 标签管理
    - 与故障日志对接
    """
    
    def __init__(self, data_dir: str = "./collected_data", config: Optional[Dict] = None):
        """
        Args:
            data_dir: 数据存储目录
            config: 配置参数
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        
        # 创建目录结构
        self._init_directories()
        
        # 数据索引
        self.samples: Dict[str, DataSample] = {}
        self.tasks: Dict[str, CollectionTask] = {}
        self.fault_records: Dict[str, FaultRecord] = {}
        
        # 采集处理器
        self.collectors: Dict[ModalityType, Callable] = {}
        
        # 预处理器
        self.preprocessors: Dict[ModalityType, Callable] = {}
        
        # 质量评估器
        self.quality_assessors: Dict[ModalityType, Callable] = {}
        
        # 锁
        self._lock = threading.RLock()
        
        # 加载已有数据索引
        self._load_index()
        
        logger.info(f"DataCollectionManager 初始化完成, 数据目录: {self.data_dir}")
    
    def _init_directories(self):
        """初始化目录结构"""
        dirs = [
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "labels",
            self.data_dir / "index",
            self.data_dir / "exports"
        ]
        
        for modality in ModalityType:
            dirs.append(self.data_dir / "raw" / modality.value)
            dirs.append(self.data_dir / "processed" / modality.value)
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _load_index(self):
        """加载数据索引"""
        index_file = self.data_dir / "index" / "samples.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for sample_data in data.get('samples', []):
                        sample = DataSample(
                            sample_id=sample_data['sample_id'],
                            modality=ModalityType(sample_data['modality']),
                            device_id=sample_data['device_id'],
                            timestamp=sample_data['timestamp'],
                            data_path=sample_data['data_path'],
                            data_hash=sample_data['data_hash'],
                            quality=DataQuality(sample_data.get('quality', 'medium')),
                            label_status=LabelStatus(sample_data.get('label_status', 'unlabeled')),
                            labels=sample_data.get('labels', {}),
                            metadata=sample_data.get('metadata', {}),
                            aligned_samples=sample_data.get('aligned_samples', [])
                        )
                        self.samples[sample.sample_id] = sample
                logger.info(f"加载了 {len(self.samples)} 个样本索引")
            except Exception as e:
                logger.error(f"加载索引失败: {e}")
    
    def _save_index(self):
        """保存数据索引"""
        index_file = self.data_dir / "index" / "samples.json"
        try:
            with self._lock:
                samples_data = []
                for sample in self.samples.values():
                    samples_data.append({
                        'sample_id': sample.sample_id,
                        'modality': sample.modality.value,
                        'device_id': sample.device_id,
                        'timestamp': sample.timestamp,
                        'data_path': sample.data_path,
                        'data_hash': sample.data_hash,
                        'quality': sample.quality.value,
                        'label_status': sample.label_status.value,
                        'labels': sample.labels,
                        'metadata': sample.metadata,
                        'aligned_samples': sample.aligned_samples
                    })
                
                with open(index_file, 'w', encoding='utf-8') as f:
                    json.dump({'samples': samples_data, 'updated': time.time()}, f, indent=2)
                    
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    # =========================================================================
    # 采集器注册
    # =========================================================================
    
    def register_collector(self, modality: ModalityType, collector: Callable):
        """
        注册模态采集器
        
        Args:
            modality: 模态类型
            collector: 采集函数 (device_id, config) -> (data, metadata)
        """
        self.collectors[modality] = collector
        logger.info(f"注册采集器: {modality.value}")
    
    def register_preprocessor(self, modality: ModalityType, preprocessor: Callable):
        """
        注册预处理器
        
        Args:
            modality: 模态类型
            preprocessor: 预处理函数 (raw_data, config) -> processed_data
        """
        self.preprocessors[modality] = preprocessor
        logger.info(f"注册预处理器: {modality.value}")
    
    def register_quality_assessor(self, modality: ModalityType, assessor: Callable):
        """
        注册质量评估器
        
        Args:
            modality: 模态类型
            assessor: 评估函数 (data) -> DataQuality
        """
        self.quality_assessors[modality] = assessor
        logger.info(f"注册质量评估器: {modality.value}")
    
    # =========================================================================
    # 采集任务管理
    # =========================================================================
    
    def create_collection_task(self, 
                              name: str,
                              modalities: List[ModalityType],
                              device_ids: List[str],
                              duration_hours: int = 168,  # 默认一周
                              interval_seconds: int = 3600,
                              config: Optional[Dict] = None) -> CollectionTask:
        """
        创建采集任务
        
        Args:
            name: 任务名称
            modalities: 要采集的模态
            device_ids: 设备ID列表
            duration_hours: 采集持续时间(小时)
            interval_seconds: 采集间隔(秒)
            config: 额外配置
            
        Returns:
            task: 采集任务
        """
        task_id = f"task_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        target_count = len(modalities) * len(device_ids) * (duration_hours * 3600 // interval_seconds)
        
        task = CollectionTask(
            task_id=task_id,
            name=name,
            modalities=modalities,
            device_ids=device_ids,
            start_time=start_time,
            end_time=end_time,
            interval_seconds=interval_seconds,
            target_count=target_count,
            config=config or {}
        )
        
        self.tasks[task_id] = task
        logger.info(f"创建采集任务: {name} ({task_id}), 目标: {target_count} 样本")
        
        return task
    
    def collect_sample(self, 
                      modality: ModalityType,
                      device_id: str,
                      raw_data: Any = None,
                      metadata: Optional[Dict] = None,
                      auto_preprocess: bool = True) -> Optional[DataSample]:
        """
        收集单个样本
        
        Args:
            modality: 模态类型
            device_id: 设备ID
            raw_data: 原始数据 (如果为None则使用注册的采集器)
            metadata: 元数据
            auto_preprocess: 是否自动预处理
            
        Returns:
            sample: 数据样本
        """
        timestamp = time.time()
        sample_id = f"{modality.value}_{device_id}_{int(timestamp)}"
        
        try:
            # 获取原始数据
            if raw_data is None:
                if modality in self.collectors:
                    raw_data, collected_metadata = self.collectors[modality](device_id, {})
                    metadata = {**(metadata or {}), **collected_metadata}
                else:
                    logger.warning(f"未注册采集器: {modality.value}")
                    return None
            
            # 保存原始数据
            raw_path = self.data_dir / "raw" / modality.value / f"{sample_id}.npy"
            if isinstance(raw_data, np.ndarray):
                np.save(raw_path, raw_data)
            elif isinstance(raw_data, (dict, list)):
                raw_path = raw_path.with_suffix('.json')
                with open(raw_path, 'w', encoding='utf-8') as f:
                    json.dump(raw_data, f)
            else:
                raw_path = raw_path.with_suffix('.bin')
                with open(raw_path, 'wb') as f:
                    f.write(raw_data if isinstance(raw_data, bytes) else str(raw_data).encode())
            
            # 计算数据哈希
            with open(raw_path, 'rb') as f:
                data_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            
            # 预处理
            if auto_preprocess and modality in self.preprocessors:
                try:
                    processed_data = self.preprocessors[modality](raw_data, {})
                    processed_path = self.data_dir / "processed" / modality.value / f"{sample_id}.npy"
                    np.save(processed_path, processed_data)
                except Exception as e:
                    logger.warning(f"预处理失败: {e}")
            
            # 质量评估
            quality = DataQuality.MEDIUM
            if modality in self.quality_assessors:
                try:
                    quality = self.quality_assessors[modality](raw_data)
                except Exception as e:
                    logger.warning(f"质量评估失败: {e}")
            
            # 创建样本记录
            sample = DataSample(
                sample_id=sample_id,
                modality=modality,
                device_id=device_id,
                timestamp=timestamp,
                data_path=str(raw_path),
                data_hash=data_hash,
                quality=quality,
                metadata=metadata or {}
            )
            
            with self._lock:
                self.samples[sample_id] = sample
            
            # 定期保存索引
            if len(self.samples) % 100 == 0:
                self._save_index()
            
            return sample
            
        except Exception as e:
            logger.error(f"采集样本失败: {e}")
            return None
    
    # =========================================================================
    # 数据对齐
    # =========================================================================
    
    def align_samples(self, 
                     sample_ids: List[str] = None,
                     time_window_seconds: float = 10.0,
                     device_filter: Optional[str] = None) -> List[List[str]]:
        """
        对齐多模态样本
        
        Args:
            sample_ids: 要对齐的样本ID (None表示所有)
            time_window_seconds: 时间窗口
            device_filter: 设备过滤
            
        Returns:
            aligned_groups: 对齐的样本组列表
        """
        # 收集样本
        if sample_ids:
            samples = [self.samples[sid] for sid in sample_ids if sid in self.samples]
        else:
            samples = list(self.samples.values())
        
        if device_filter:
            samples = [s for s in samples if s.device_id == device_filter]
        
        # 按设备和时间分组
        device_samples: Dict[str, List[DataSample]] = defaultdict(list)
        for sample in samples:
            device_samples[sample.device_id].append(sample)
        
        aligned_groups = []
        
        for device_id, device_samples_list in device_samples.items():
            # 按时间排序
            device_samples_list.sort(key=lambda x: x.timestamp)
            
            # 滑动窗口对齐
            i = 0
            while i < len(device_samples_list):
                anchor = device_samples_list[i]
                group = [anchor.sample_id]
                
                j = i + 1
                while j < len(device_samples_list):
                    candidate = device_samples_list[j]
                    if candidate.timestamp - anchor.timestamp <= time_window_seconds:
                        # 检查是否是不同模态
                        if candidate.modality != anchor.modality:
                            group.append(candidate.sample_id)
                    else:
                        break
                    j += 1
                
                if len(group) > 1:
                    aligned_groups.append(group)
                    # 更新样本的对齐信息
                    for sid in group:
                        self.samples[sid].aligned_samples = [s for s in group if s != sid]
                
                i += 1
        
        logger.info(f"对齐完成, 共 {len(aligned_groups)} 组")
        return aligned_groups
    
    # =========================================================================
    # 故障日志对接
    # =========================================================================
    
    def import_fault_records(self, records: List[Dict]) -> int:
        """
        导入故障记录
        
        Args:
            records: 故障记录列表
            
        Returns:
            imported_count: 导入数量
        """
        imported = 0
        
        for record_data in records:
            try:
                record_id = record_data.get('record_id', f"fault_{int(time.time())}_{imported}")
                
                start_time = record_data.get('start_time')
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time)
                elif isinstance(start_time, (int, float)):
                    start_time = datetime.fromtimestamp(start_time)
                
                end_time = record_data.get('end_time')
                if end_time:
                    if isinstance(end_time, str):
                        end_time = datetime.fromisoformat(end_time)
                    elif isinstance(end_time, (int, float)):
                        end_time = datetime.fromtimestamp(end_time)
                
                record = FaultRecord(
                    record_id=record_id,
                    device_id=record_data['device_id'],
                    fault_type=record_data['fault_type'],
                    fault_level=record_data.get('fault_level', 'warning'),
                    start_time=start_time,
                    end_time=end_time,
                    description=record_data.get('description', ''),
                    verified=record_data.get('verified', False)
                )
                
                self.fault_records[record_id] = record
                imported += 1
                
            except Exception as e:
                logger.warning(f"导入故障记录失败: {e}")
        
        logger.info(f"导入 {imported} 条故障记录")
        return imported
    
    def link_samples_to_faults(self, time_buffer_seconds: float = 300) -> int:
        """
        将样本与故障记录关联
        
        Args:
            time_buffer_seconds: 时间缓冲(秒)
            
        Returns:
            linked_count: 关联数量
        """
        linked = 0
        
        for sample in self.samples.values():
            sample_time = datetime.fromtimestamp(sample.timestamp)
            
            for record in self.fault_records.values():
                if sample.device_id != record.device_id:
                    continue
                
                # 检查时间是否匹配
                start_with_buffer = record.start_time - timedelta(seconds=time_buffer_seconds)
                end_time = record.end_time or datetime.now()
                end_with_buffer = end_time + timedelta(seconds=time_buffer_seconds)
                
                if start_with_buffer <= sample_time <= end_with_buffer:
                    # 关联样本和故障
                    if sample.sample_id not in record.related_samples:
                        record.related_samples.append(sample.sample_id)
                    
                    # 自动标注
                    if sample.label_status == LabelStatus.UNLABELED:
                        sample.labels['fault_type'] = record.fault_type
                        sample.labels['fault_level'] = record.fault_level
                        sample.label_status = LabelStatus.AUTO_LABELED
                    
                    linked += 1
        
        self._save_index()
        logger.info(f"关联 {linked} 个样本到故障记录")
        return linked
    
    # =========================================================================
    # 标签管理
    # =========================================================================
    
    def add_label(self, sample_id: str, labels: Dict[str, Any], 
                  status: LabelStatus = LabelStatus.MANUAL_LABELED) -> bool:
        """
        添加标签
        
        Args:
            sample_id: 样本ID
            labels: 标签字典
            status: 标注状态
            
        Returns:
            success: 是否成功
        """
        if sample_id not in self.samples:
            return False
        
        sample = self.samples[sample_id]
        sample.labels.update(labels)
        sample.label_status = status
        
        # 保存标签文件
        label_path = self.data_dir / "labels" / f"{sample_id}.json"
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump({
                'sample_id': sample_id,
                'labels': sample.labels,
                'status': status.value,
                'updated': time.time()
            }, f, indent=2)
        
        return True
    
    def batch_label(self, sample_ids: List[str], labels: Dict[str, Any],
                   status: LabelStatus = LabelStatus.MANUAL_LABELED) -> int:
        """批量标注"""
        success_count = 0
        for sid in sample_ids:
            if self.add_label(sid, labels, status):
                success_count += 1
        self._save_index()
        return success_count
    
    def verify_labels(self, sample_ids: List[str]) -> int:
        """验证标签"""
        verified = 0
        for sid in sample_ids:
            if sid in self.samples:
                self.samples[sid].label_status = LabelStatus.VERIFIED
                verified += 1
        self._save_index()
        return verified
    
    # =========================================================================
    # 数据导出
    # =========================================================================
    
    def export_dataset(self, 
                      output_name: str,
                      modalities: Optional[List[ModalityType]] = None,
                      label_filter: Optional[Dict[str, Any]] = None,
                      quality_filter: Optional[List[DataQuality]] = None,
                      split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                      include_aligned_only: bool = False) -> Dict[str, Any]:
        """
        导出训练数据集
        
        Args:
            output_name: 输出名称
            modalities: 模态过滤
            label_filter: 标签过滤
            quality_filter: 质量过滤
            split_ratios: (train, val, test) 比例
            include_aligned_only: 只包含对齐的样本
            
        Returns:
            export_info: 导出信息
        """
        # 过滤样本
        filtered_samples = []
        
        for sample in self.samples.values():
            # 模态过滤
            if modalities and sample.modality not in modalities:
                continue
            
            # 质量过滤
            if quality_filter and sample.quality not in quality_filter:
                continue
            
            # 对齐过滤
            if include_aligned_only and not sample.aligned_samples:
                continue
            
            # 标签过滤
            if label_filter:
                match = True
                for key, value in label_filter.items():
                    if key not in sample.labels or sample.labels[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            # 只包含已标注的
            if sample.label_status in [LabelStatus.UNLABELED, LabelStatus.REJECTED]:
                continue
            
            filtered_samples.append(sample)
        
        # 随机打乱
        np.random.shuffle(filtered_samples)
        
        # 划分数据集
        n = len(filtered_samples)
        train_end = int(n * split_ratios[0])
        val_end = train_end + int(n * split_ratios[1])
        
        splits = {
            'train': filtered_samples[:train_end],
            'val': filtered_samples[train_end:val_end],
            'test': filtered_samples[val_end:]
        }
        
        # 创建导出目录
        export_dir = self.data_dir / "exports" / output_name
        export_dir.mkdir(parents=True, exist_ok=True)
        
        export_info = {
            'name': output_name,
            'created': time.time(),
            'total_samples': n,
            'splits': {}
        }
        
        for split_name, samples in splits.items():
            split_dir = export_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            manifest = []
            for sample in samples:
                # 复制数据文件
                src_path = Path(sample.data_path)
                if src_path.exists():
                    dst_path = split_dir / src_path.name
                    import shutil
                    shutil.copy2(src_path, dst_path)
                
                manifest.append({
                    'sample_id': sample.sample_id,
                    'modality': sample.modality.value,
                    'device_id': sample.device_id,
                    'file': src_path.name,
                    'labels': sample.labels
                })
            
            # 保存manifest
            with open(split_dir / 'manifest.json', 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            
            export_info['splits'][split_name] = len(samples)
        
        # 保存导出信息
        with open(export_dir / 'export_info.json', 'w', encoding='utf-8') as f:
            json.dump(export_info, f, indent=2)
        
        logger.info(f"导出数据集: {output_name}, 总计 {n} 样本")
        return export_info
    
    # =========================================================================
    # 统计信息
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {
            'total_samples': len(self.samples),
            'by_modality': defaultdict(int),
            'by_device': defaultdict(int),
            'by_quality': defaultdict(int),
            'by_label_status': defaultdict(int),
            'fault_records': len(self.fault_records),
            'aligned_groups': 0
        }
        
        aligned_samples = set()
        for sample in self.samples.values():
            stats['by_modality'][sample.modality.value] += 1
            stats['by_device'][sample.device_id] += 1
            stats['by_quality'][sample.quality.value] += 1
            stats['by_label_status'][sample.label_status.value] += 1
            
            if sample.aligned_samples:
                aligned_samples.add(sample.sample_id)
        
        stats['aligned_samples'] = len(aligned_samples)
        
        # 转换为普通dict
        stats['by_modality'] = dict(stats['by_modality'])
        stats['by_device'] = dict(stats['by_device'])
        stats['by_quality'] = dict(stats['by_quality'])
        stats['by_label_status'] = dict(stats['by_label_status'])
        
        return stats


# =============================================================================
# 预定义的采集器和预处理器
# =============================================================================

class AcousticCollector:
    """声学数据采集器"""
    
    def __init__(self, sample_rate: int = 16000, duration: float = 2.0):
        self.sample_rate = sample_rate
        self.duration = duration
    
    def __call__(self, device_id: str, config: Dict) -> Tuple[np.ndarray, Dict]:
        """模拟采集声学数据"""
        n_samples = int(self.sample_rate * self.duration)
        # 模拟数据
        data = np.random.randn(n_samples).astype(np.float32) * 0.1
        
        metadata = {
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'channels': 1,
            'device_id': device_id,
            'collected_at': time.time()
        }
        
        return data, metadata


class GasCollector:
    """气体数据采集器"""
    
    def __call__(self, device_id: str, config: Dict) -> Tuple[Dict, Dict]:
        """模拟采集气体数据"""
        data = {
            'SF6': np.random.uniform(800, 1200),
            'H2': np.random.uniform(30, 100),
            'CO': np.random.uniform(50, 200),
            'C2H2': np.random.uniform(0, 10),
            'CH4': np.random.uniform(10, 50),
            'temperature': np.random.uniform(20, 40),
            'humidity': np.random.uniform(30, 70),
            'pressure': np.random.uniform(0.9, 1.1)
        }
        
        metadata = {
            'device_id': device_id,
            'collected_at': time.time()
        }
        
        return data, metadata


class AcousticPreprocessor:
    """声学数据预处理器"""
    
    def __init__(self, n_mels: int = 128, hop_length: int = 512):
        self.n_mels = n_mels
        self.hop_length = hop_length
    
    def __call__(self, data: np.ndarray, config: Dict) -> np.ndarray:
        """提取Mel频谱特征"""
        # 简化的Mel频谱计算
        n_fft = 2048
        
        # STFT
        frames = []
        for i in range(0, len(data) - n_fft, self.hop_length):
            frame = data[i:i+n_fft]
            windowed = frame * np.hanning(n_fft)
            spectrum = np.abs(np.fft.rfft(windowed))
            frames.append(spectrum)
        
        if not frames:
            return np.zeros((self.n_mels, 1))
        
        spectrogram = np.array(frames).T
        
        # 简化的Mel滤波
        mel_features = np.zeros((self.n_mels, spectrogram.shape[1]))
        bin_width = spectrogram.shape[0] // self.n_mels
        for i in range(self.n_mels):
            mel_features[i] = np.mean(spectrogram[i*bin_width:(i+1)*bin_width], axis=0)
        
        # 转为dB
        mel_features = 20 * np.log10(np.maximum(mel_features, 1e-10))
        
        return mel_features.astype(np.float32)


def assess_acoustic_quality(data: np.ndarray) -> DataQuality:
    """评估声学数据质量"""
    if data is None or len(data) == 0:
        return DataQuality.INVALID
    
    # 检查信噪比
    rms = np.sqrt(np.mean(data**2))
    if rms < 0.001:
        return DataQuality.LOW
    
    # 检查是否有异常值
    max_val = np.max(np.abs(data))
    if max_val > 1.0:
        return DataQuality.LOW
    
    # 检查是否有信号
    if rms > 0.01:
        return DataQuality.HIGH
    
    return DataQuality.MEDIUM


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'ModalityType',
    'DataQuality',
    'LabelStatus',
    'DataSample',
    'CollectionTask',
    'FaultRecord',
    'DataCollectionManager',
    'AcousticCollector',
    'GasCollector',
    'AcousticPreprocessor',
    'assess_acoustic_quality'
]
