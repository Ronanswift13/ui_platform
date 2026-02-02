#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平台核心 - 数据管理器 (Data Manager V2.0)
=============================================

实现修复方案中的四大核心模块:
  模块一: 结构化数据仓库架构 (Structured Data Warehouse)
  模块二: 健壮的上传与摄入引擎 (Ingestion Engine) — 后端部分
  模块三: 数据清洗与校验守门员 (The Gatekeeper)
  模块四: 动态训练数据聚合器 (Dynamic Data Aggregator)

目录层级规范 (三级索引):
  training/data/raw/{Voltage_Level}/{Plugin_Type}/{Timestamp}_{Uploader}_{Name}/
  例: training/data/raw/HV_220kV/transformer/20260131_StudentA_ThermalData_v1/

每个 Dataset_Instance 文件夹自动生成 meta.json:
  {
    "uploader": "StudentA",
    "upload_time": "2026-01-31T14:30:00",
    "image_count": 120,
    "label_count": 115,
    "format": "YOLO",
    "status": "verified",
    "voltage_level": "HV_220kV",
    "plugin_type": "transformer",
    "validation_report": { ... }
  }

作者: 破夜绘明团队
日期: 2026-02-01
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import shutil
import tempfile
import threading
import time
import zipfile
import tarfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# 常量与枚举
# =============================================================================

# 根路径 - 相对于项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TRAINING_BASE = PROJECT_ROOT / "training"
DATA_ROOT = TRAINING_BASE / "data"
RAW_DATA_PATH = DATA_ROOT / "raw"
PROCESSED_DATA_PATH = DATA_ROOT / "processed"
TEMP_UPLOAD_PATH = DATA_ROOT / "temp_upload"      # 服务器端扫描导入暂存区
CHUNK_TEMP_PATH = DATA_ROOT / "chunk_temp"         # 分片上传临时目录

# 元数据索引文件
METADATA_INDEX_PATH = DATA_ROOT / "metadata_index.json"

# 支持的图像扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.ppm', '.webp'}

# 支持的视频扩展名
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg'}

# 支持的音频扩展名
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus'}

# 支持的标注扩展名
LABEL_EXTENSIONS = {'.txt', '.xml', '.json'}

# 支持的压缩包扩展名
ARCHIVE_EXTENSIONS = {'.zip', '.tar', '.tar.gz', '.tgz', '.rar'}

# 垃圾文件/目录 (模块三: 清洗)
GARBAGE_PATTERNS = {
    '__MACOSX', '.DS_Store', 'Thumbs.db', '._.', 'desktop.ini',
    '.ipynb_checkpoints', '__pycache__', '.git',
}

# 分片大小 (5MB)
CHUNK_SIZE = 5 * 1024 * 1024


class VoltageLevel(str, Enum):
    """电压等级枚举"""
    UHV_1000kV_AC = "UHV_1000kV_AC"
    UHV_800kV_DC = "UHV_800kV_DC"
    EHV_500kV = "EHV_500kV"
    EHV_330kV = "EHV_330kV"
    HV_220kV = "HV_220kV"
    HV_110kV = "HV_110kV"
    MV_35kV = "MV_35kV"
    LV_10kV = "LV_10kV"

    @classmethod
    def values(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls.values()


class PluginType(str, Enum):
    """插件类型枚举"""
    transformer = "transformer"
    switch = "switch"
    busbar = "busbar"
    capacitor = "capacitor"
    meter = "meter"
    thermal = "thermal"
    acoustic = "acoustic"
    gas = "gas"
    hyperspectral = "hyperspectral"
    slam = "slam"
    fusion = "fusion"

    @classmethod
    def values(cls) -> List[str]:
        return [e.value for e in cls]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls.values()


class DatasetStatus(str, Enum):
    """数据集状态"""
    PENDING = "pending"           # 等待处理 (上传中/解压中)
    VALIDATING = "validating"     # 校验中
    VERIFIED = "verified"         # 校验通过
    WARNING = "warning"           # 有警告但可用
    ERROR = "error"               # 错误，不可用
    ARCHIVED = "archived"         # 已归档


class AnnotationFormat(str, Enum):
    """标注格式"""
    YOLO = "YOLO"
    COCO = "COCO"
    VOC = "VOC"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# 模块一: 结构化数据仓库架构
# =============================================================================

class DataWarehouse:
    """
    结构化数据仓库
    
    职责:
    - 管理三级目录索引: Voltage -> Plugin -> Dataset_Instance
    - 自动创建目录结构
    - 管理元数据 (meta.json)
    - 维护全局元数据索引
    """

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else DATA_ROOT
        self.raw_path = self.data_root / "raw"
        self.processed_path = self.data_root / "processed"
        self.temp_upload_path = self.data_root / "temp_upload"
        self.chunk_temp_path = self.data_root / "chunk_temp"
        self.metadata_index_path = self.data_root / "metadata_index.json"

        # 全局元数据索引 (内存缓存)
        self._metadata_index: Dict[str, Dict] = {}
        self._lock = threading.Lock()

        # 初始化目录
        self._init_directories()
        # 加载索引
        self._load_metadata_index()

    def _init_directories(self):
        """初始化基础目录结构"""
        for path in [self.raw_path, self.processed_path,
                     self.temp_upload_path, self.chunk_temp_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _load_metadata_index(self):
        """从磁盘加载全局元数据索引"""
        if self.metadata_index_path.exists():
            try:
                with open(self.metadata_index_path, 'r', encoding='utf-8') as f:
                    self._metadata_index = json.load(f)
                logger.info(f"[DataWarehouse] 加载元数据索引: {len(self._metadata_index)} 条记录")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[DataWarehouse] 加载索引失败, 重建: {e}")
                self._metadata_index = {}
                self._rebuild_metadata_index()
        else:
            self._rebuild_metadata_index()

    def _rebuild_metadata_index(self):
        """扫描磁盘重建全局元数据索引"""
        logger.info("[DataWarehouse] 重建元数据索引...")
        self._metadata_index = {}

        if not self.raw_path.exists():
            return

        for voltage_dir in self.raw_path.iterdir():
            if not voltage_dir.is_dir():
                continue
            for plugin_dir in voltage_dir.iterdir():
                if not plugin_dir.is_dir():
                    continue
                for dataset_dir in plugin_dir.iterdir():
                    if not dataset_dir.is_dir():
                        continue
                    meta_file = dataset_dir / "meta.json"
                    if meta_file.exists():
                        try:
                            with open(meta_file, 'r', encoding='utf-8') as f:
                                meta = json.load(f)
                            dataset_id = dataset_dir.name
                            self._metadata_index[dataset_id] = meta
                        except Exception:
                            pass

        self._save_metadata_index()
        logger.info(f"[DataWarehouse] 索引重建完成: {len(self._metadata_index)} 条")

    def _save_metadata_index(self):
        """持久化全局元数据索引"""
        try:
            with open(self.metadata_index_path, 'w', encoding='utf-8') as f:
                json.dump(self._metadata_index, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"[DataWarehouse] 保存索引失败: {e}")

    def create_dataset_instance(
        self,
        voltage_level: str,
        plugin_type: str,
        uploader: str = "anonymous",
        dataset_name: str = "upload"
    ) -> Tuple[Path, str]:
        """
        创建新的数据集实例目录 (三级索引)
        
        Args:
            voltage_level: 电压等级 (枚举值)
            plugin_type: 插件类型 (枚举值)
            uploader: 上传者ID
            dataset_name: 数据集名称
            
        Returns:
            (dataset_path, dataset_id) 数据集路径与唯一ID
        """
        # 验证参数
        if not VoltageLevel.is_valid(voltage_level):
            raise ValueError(
                f"无效的电压等级: {voltage_level}。"
                f"可选值: {VoltageLevel.values()}"
            )

        if not PluginType.is_valid(plugin_type):
            raise ValueError(
                f"无效的插件类型: {plugin_type}。"
                f"可选值: {PluginType.values()}"
            )

        # 清理上传者名和数据集名中的特殊字符
        safe_uploader = re.sub(r'[^\w\-]', '_', uploader)[:30]
        safe_name = re.sub(r'[^\w\-]', '_', dataset_name)[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        dataset_id = f"{timestamp}_{safe_uploader}_{safe_name}"
        dataset_path = self.raw_path / voltage_level / plugin_type / dataset_id

        # 创建目录
        dataset_path.mkdir(parents=True, exist_ok=True)
        (dataset_path / "images").mkdir(exist_ok=True)
        (dataset_path / "labels").mkdir(exist_ok=True)
        (dataset_path / "videos").mkdir(exist_ok=True)
        (dataset_path / "audios").mkdir(exist_ok=True)

        # 生成初始 meta.json
        meta = {
            "dataset_id": dataset_id,
            "uploader": uploader,
            "upload_time": datetime.now().isoformat(),
            "voltage_level": voltage_level,
            "plugin_type": plugin_type,
            "dataset_name": dataset_name,
            "image_count": 0,
            "label_count": 0,
            "format": AnnotationFormat.UNKNOWN.value,
            "status": DatasetStatus.PENDING.value,
            "validation_report": None,
            "data_path": str(dataset_path),
        }

        meta_file = dataset_path / "meta.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 更新全局索引
        with self._lock:
            self._metadata_index[dataset_id] = meta
            self._save_metadata_index()

        logger.info(f"[DataWarehouse] 创建数据集实例: {dataset_path}")
        return dataset_path, dataset_id

    def update_metadata(self, dataset_id: str, updates: Dict[str, Any]):
        """更新数据集元数据"""
        with self._lock:
            if dataset_id not in self._metadata_index:
                logger.warning(f"[DataWarehouse] 未找到数据集: {dataset_id}")
                return

            meta = self._metadata_index[dataset_id]
            meta.update(updates)
            meta["updated_at"] = datetime.now().isoformat()

            # 写入 meta.json
            data_path = Path(meta["data_path"])
            meta_file = data_path / "meta.json"
            if meta_file.parent.exists():
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

            self._save_metadata_index()

    def get_metadata(self, dataset_id: str) -> Optional[Dict]:
        """获取数据集元数据"""
        return self._metadata_index.get(dataset_id)

    def list_datasets(
        self,
        voltage_level: Optional[str] = None,
        plugin_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict]:
        """
        列出数据集，支持过滤
        
        Args:
            voltage_level: 过滤电压等级
            plugin_type: 过滤插件类型
            status: 过滤状态
            
        Returns:
            符合条件的数据集元数据列表
        """
        results = []
        for ds_id, meta in self._metadata_index.items():
            if voltage_level and meta.get("voltage_level") != voltage_level:
                continue
            if plugin_type and meta.get("plugin_type") != plugin_type:
                continue
            if status and meta.get("status") != status:
                continue
            results.append(meta)

        # 按上传时间倒序
        results.sort(key=lambda x: x.get("upload_time", ""), reverse=True)
        return results

    def delete_dataset(self, dataset_id: str) -> bool:
        """删除数据集实例"""
        with self._lock:
            meta = self._metadata_index.get(dataset_id)
            if not meta:
                return False

            data_path = Path(meta["data_path"])
            if data_path.exists():
                shutil.rmtree(data_path, ignore_errors=True)
                logger.info(f"[DataWarehouse] 删除数据集目录: {data_path}")

            del self._metadata_index[dataset_id]
            self._save_metadata_index()
        return True

    def get_dataset_path(
        self, voltage_level: str, plugin_type: str
    ) -> Path:
        """获取指定 电压+插件 的原始数据根路径"""
        return self.raw_path / voltage_level / plugin_type

    def get_all_valid_dataset_dirs(
        self, voltage_level: str, plugin_type: str
    ) -> List[Path]:
        """
        模块四需要: 获取指定 电压+插件 下所有校验通过的数据集子目录
        
        Returns:
            所有 status 为 verified / warning 的子目录列表
        """
        base = self.raw_path / voltage_level / plugin_type
        valid_dirs = []

        if not base.exists():
            return valid_dirs

        for dataset_dir in sorted(base.iterdir()):
            if not dataset_dir.is_dir():
                continue
            ds_id = dataset_dir.name
            meta = self._metadata_index.get(ds_id, {})
            status = meta.get("status", "")
            if status in (DatasetStatus.VERIFIED.value, DatasetStatus.WARNING.value):
                valid_dirs.append(dataset_dir)

        return valid_dirs


# =============================================================================
# 模块三: 数据清洗与校验守门员 (The Gatekeeper)
# =============================================================================

class DataGatekeeper:
    """
    数据校验守门员
    
    职责:
    - 垃圾文件清洗 (__MACOSX, .DS_Store, Thumbs.db 等)
    - 数量对齐检查 (图片-标签匹配率)
    - 格式探针 (自动识别 YOLO / COCO / VOC)
    - 生成结构化校验报告
    """

    MATCH_THRESHOLD = 0.9    # 匹配率阈值
    FORMAT_PROBE_COUNT = 5   # 格式探测采样数

    @staticmethod
    def clean_garbage(dataset_path: Path) -> int:
        """
        清洗垃圾文件和目录
        
        Returns:
            删除的文件/目录数量
        """
        removed_count = 0

        for root, dirs, files in os.walk(dataset_path, topdown=False):
            root_path = Path(root)

            # 删除垃圾文件
            for fname in files:
                if any(pat in fname for pat in GARBAGE_PATTERNS):
                    fpath = root_path / fname
                    try:
                        fpath.unlink()
                        removed_count += 1
                    except OSError:
                        pass

            # 删除垃圾目录
            for dname in dirs:
                if any(pat in dname for pat in GARBAGE_PATTERNS):
                    dpath = root_path / dname
                    try:
                        shutil.rmtree(dpath)
                        removed_count += 1
                    except OSError:
                        pass

        if removed_count > 0:
            logger.info(f"[Gatekeeper] 清理垃圾文件: {removed_count} 项 in {dataset_path.name}")
        return removed_count

    @staticmethod
    def find_images(dataset_path: Path) -> List[Path]:
        """递归查找所有图像文件"""
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(dataset_path.rglob(f"*{ext}"))
            images.extend(dataset_path.rglob(f"*{ext.upper()}"))
        return sorted(set(images))

    @staticmethod
    def find_labels(dataset_path: Path) -> List[Path]:
        """递归查找所有标注文件 (.txt, .xml, .json)"""
        labels = []
        for ext in LABEL_EXTENSIONS:
            labels.extend(dataset_path.rglob(f"*{ext}"))
        # 过滤掉 meta.json, classes.txt 等非标注文件
        skip_names = {'meta.json', 'classes.txt', 'dataset_config.json',
                      'data.yaml', 'annotations.json', '_annotations.json'}
        labels = [l for l in labels if l.name not in skip_names]
        return sorted(set(labels))

    @classmethod
    def check_alignment(
        cls, dataset_path: Path
    ) -> Dict[str, Any]:
        """
        数量对齐检查: 比较图片集合和标签集合的文件名(不含扩展名)交集
        
        Returns:
            {
                "image_count": int,
                "label_count": int,
                "matched_count": int,
                "match_ratio": float,
                "missing_labels": List[str],  # 有图无标注的文件名
                "orphan_labels": List[str],   # 有标注无图的文件名
                "aligned": bool
            }
        """
        images = cls.find_images(dataset_path)
        labels = cls.find_labels(dataset_path)

        image_stems = {img.stem for img in images}
        label_stems = {lbl.stem for lbl in labels}

        matched = image_stems & label_stems
        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)

        image_count = len(image_stems)
        match_ratio = len(matched) / image_count if image_count > 0 else 0.0

        return {
            "image_count": image_count,
            "label_count": len(label_stems),
            "matched_count": len(matched),
            "match_ratio": round(match_ratio, 4),
            "missing_labels": missing_labels[:50],     # 最多返回50条
            "orphan_labels": orphan_labels[:50],
            "aligned": match_ratio >= cls.MATCH_THRESHOLD,
        }

    @classmethod
    def probe_format(cls, dataset_path: Path) -> AnnotationFormat:
        """
        格式探针: 随机抽取标注文件，自动识别格式
        
        策略:
        - 抽取最多 5 个 .txt 文件，尝试按 YOLO 格式解析
        - 如果发现 .xml 文件，检查是否为 VOC 格式
        - 如果发现 annotations.json / _annotations.json，检查 COCO
        
        Returns:
            AnnotationFormat 枚举值
        """
        # 优先检查 COCO
        coco_files = list(dataset_path.rglob("annotations.json")) + \
                     list(dataset_path.rglob("_annotations.json"))
        for cf in coco_files:
            try:
                with open(cf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if "images" in data and "annotations" in data:
                    logger.info(f"[Gatekeeper] 格式探测: COCO ({cf.name})")
                    return AnnotationFormat.COCO
            except (json.JSONDecodeError, IOError):
                pass

        # 检查 VOC (.xml)
        xml_files = list(dataset_path.rglob("*.xml"))
        for xf in xml_files[:cls.FORMAT_PROBE_COUNT]:
            try:
                with open(xf, 'r', encoding='utf-8') as f:
                    content = f.read(500)
                if "<annotation>" in content or "<object>" in content:
                    logger.info(f"[Gatekeeper] 格式探测: VOC ({xf.name})")
                    return AnnotationFormat.VOC
            except IOError:
                pass

        # 检查 YOLO (.txt)
        txt_files = list(dataset_path.rglob("*.txt"))
        # 过滤掉特殊文件
        txt_files = [t for t in txt_files
                     if t.name not in {'classes.txt', 'meta.json', 'dataset_config.json'}]

        sample = txt_files[:cls.FORMAT_PROBE_COUNT]
        yolo_hits = 0

        for tf in sample:
            try:
                with open(tf, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if not lines:
                    continue
                # YOLO 格式: class_id x_center y_center width height
                for line in lines[:5]:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        coords = [float(p) for p in parts[1:]]
                        # YOLO 坐标在 [0, 1] 范围内
                        if all(0 <= c <= 1 for c in coords) and class_id >= 0:
                            yolo_hits += 1
                            break
            except (ValueError, IOError):
                pass

        if yolo_hits > 0 and yolo_hits >= len(sample) * 0.5:
            logger.info("[Gatekeeper] 格式探测: YOLO")
            return AnnotationFormat.YOLO

        logger.info("[Gatekeeper] 格式探测: UNKNOWN")
        return AnnotationFormat.UNKNOWN

    @classmethod
    def validate_yolo_dataset(cls, dataset_path: Path) -> Dict[str, Any]:
        """
        完整校验流程 (修复方案中的核心函数)
        
        步骤:
        1. 清洗垃圾文件
        2. 数量对齐检查
        3. 格式探针
        4. 生成结构化报告
        
        Args:
            dataset_path: 数据集目录路径
            
        Returns:
            {
                "valid": bool,
                "image_count": int,
                "label_count": int,
                "matched_count": int,
                "match_ratio": float,
                "missing_labels": int,
                "format": str,
                "garbage_removed": int,
                "warnings": List[str],
                "errors": List[str]
            }
        """
        warnings = []
        errors = []

        # Step 1: 清洗
        garbage_count = cls.clean_garbage(dataset_path)

        # Step 2: 对齐检查
        alignment = cls.check_alignment(dataset_path)

        # Step 3: 格式探针
        fmt = cls.probe_format(dataset_path)

        # 生成诊断信息
        if alignment["image_count"] == 0:
            errors.append("未找到任何图像文件")

        if not alignment["aligned"]:
            if alignment["match_ratio"] < 0.5:
                errors.append(
                    f"图片-标签匹配率极低: {alignment['match_ratio']:.1%}，"
                    f"缺少 {len(alignment['missing_labels'])} 个标注文件"
                )
            else:
                warnings.append(
                    f"图片-标签匹配率: {alignment['match_ratio']:.1%}，"
                    f"缺少 {len(alignment['missing_labels'])} 个标注文件"
                )

        if fmt == AnnotationFormat.UNKNOWN and alignment["label_count"] > 0:
            warnings.append("无法自动识别标注格式，可能需要手动转换")
        elif fmt == AnnotationFormat.VOC:
            warnings.append("检测到 VOC(XML) 格式，建议运行格式转换脚本转为 YOLO")
        elif fmt == AnnotationFormat.COCO:
            warnings.append("检测到 COCO(JSON) 格式，建议运行格式转换脚本转为 YOLO")

        # 判断整体有效性
        is_valid = len(errors) == 0

        # 确定状态
        if not is_valid:
            status = DatasetStatus.ERROR.value
        elif len(warnings) > 0:
            status = DatasetStatus.WARNING.value
        else:
            status = DatasetStatus.VERIFIED.value

        report = {
            "valid": is_valid,
            "status": status,
            "image_count": alignment["image_count"],
            "label_count": alignment["label_count"],
            "matched_count": alignment["matched_count"],
            "match_ratio": alignment["match_ratio"],
            "missing_labels": len(alignment["missing_labels"]),
            "orphan_labels": len(alignment["orphan_labels"]),
            "format": fmt.value,
            "garbage_removed": garbage_count,
            "warnings": warnings,
            "errors": errors,
            "validated_at": datetime.now().isoformat(),
        }

        logger.info(
            f"[Gatekeeper] 校验完成: valid={is_valid}, "
            f"images={alignment['image_count']}, "
            f"labels={alignment['label_count']}, "
            f"format={fmt.value}"
        )
        return report


# =============================================================================
# 模块二: 健壮的上传与摄入引擎 (Ingestion Engine) — 后端核心逻辑
# =============================================================================

class IngestionEngine:
    """
    数据摄入引擎
    
    策略 A: 分片断点续传 (Web端 < 2GB)
      - 前端将文件切为 5MB Chunk, 计算 MD5, 并发上传
      - 后端接收 Chunk -> 临时存储 -> 合并 -> 校验完整性
    
    策略 B: 服务器端扫描导入 (> 2GB 超大数据集)
      - 学生通过 U盘/FTP 拷贝到 /temp_upload 目录
      - 后端 scan_local_import() 扫描、解压、移入结构化仓库
    
    异步解压队列:
      - 使用 BackgroundTasks 或线程, 解压放在后台执行
    """

    def __init__(self, warehouse: DataWarehouse):
        self.warehouse = warehouse
        # 分片上传状态跟踪
        self._chunk_sessions: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        # 异步任务状态
        self._task_status: Dict[str, Dict] = {}

    # ---- 策略 A: 直接上传 (中小文件) ----

    def receive_files(
        self,
        files: List[Any],          # FastAPI UploadFile 列表
        voltage_level: str,
        plugin_type: str,
        uploader: str = "anonymous",
        dataset_name: str = "upload"
    ) -> Tuple[str, Path]:
        """
        接收直接上传的文件 (非分片)
        
        Args:
            files: 上传的文件列表 (UploadFile)
            voltage_level: 电压等级
            plugin_type: 插件类型
            uploader: 上传者
            dataset_name: 数据集名称
            
        Returns:
            (dataset_id, dataset_path)
        """
        dataset_path, dataset_id = self.warehouse.create_dataset_instance(
            voltage_level=voltage_level,
            plugin_type=plugin_type,
            uploader=uploader,
            dataset_name=dataset_name,
        )

        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        videos_dir = dataset_path / "videos"
        audios_dir = dataset_path / "audios"

        saved_count = 0
        archive_files = []

        for upload_file in files:
            filename = upload_file.filename or f"file_{saved_count}"
            ext = Path(filename).suffix.lower()

            if ext in ARCHIVE_EXTENSIONS:
                # 压缩包 -> 存到临时位置, 稍后解压
                archive_path = dataset_path / filename
                with open(archive_path, 'wb') as f:
                    content = upload_file.file.read()
                    f.write(content)
                archive_files.append(archive_path)
                saved_count += 1

            elif ext in IMAGE_EXTENSIONS:
                dest = images_dir / filename
                with open(dest, 'wb') as f:
                    content = upload_file.file.read()
                    f.write(content)
                saved_count += 1

            elif ext in VIDEO_EXTENSIONS:
                dest = videos_dir / filename
                with open(dest, 'wb') as f:
                    content = upload_file.file.read()
                    f.write(content)
                saved_count += 1

            elif ext in AUDIO_EXTENSIONS:
                dest = audios_dir / filename
                with open(dest, 'wb') as f:
                    content = upload_file.file.read()
                    f.write(content)
                saved_count += 1

            elif ext in LABEL_EXTENSIONS:
                dest = labels_dir / filename
                with open(dest, 'wb') as f:
                    content = upload_file.file.read()
                    f.write(content)
                saved_count += 1

            else:
                # 其他文件直接存到数据集根目录
                dest = dataset_path / filename
                with open(dest, 'wb') as f:
                    content = upload_file.file.read()
                    f.write(content)
                saved_count += 1

        logger.info(f"[Ingestion] 接收 {saved_count} 个文件 -> {dataset_id}")

        # 如果有压缩包, 标记需要后台解压
        if archive_files:
            self._task_status[dataset_id] = {
                "status": "extracting",
                "message": f"正在解压 {len(archive_files)} 个压缩包...",
                "archives": [str(a) for a in archive_files],
            }

        return dataset_id, dataset_path

    def extract_archives_in_dataset(self, dataset_id: str, dataset_path: Path):
        """
        后台解压数据集中的压缩包 (供 BackgroundTasks 调用)
        """
        try:
            self._task_status[dataset_id] = {
                "status": "extracting",
                "message": "正在解压...",
            }

            archive_exts = {'.zip', '.tar', '.tar.gz', '.tgz'}
            archives = [f for f in dataset_path.iterdir()
                       if f.suffix.lower() in archive_exts or f.name.endswith('.tar.gz')]

            for archive in archives:
                logger.info(f"[Ingestion] 解压: {archive.name}")
                try:
                    if archive.suffix == '.zip':
                        with zipfile.ZipFile(archive, 'r') as zf:
                            zf.extractall(dataset_path)
                    elif archive.name.endswith('.tar.gz') or archive.suffix == '.tgz':
                        with tarfile.open(archive, 'r:gz') as tf:
                            tf.extractall(dataset_path)
                    elif archive.suffix == '.tar':
                        with tarfile.open(archive, 'r') as tf:
                            tf.extractall(dataset_path)
                    # 删除压缩包
                    archive.unlink()
                except zipfile.BadZipFile:
                    logger.error(f"[Ingestion] 损坏的ZIP文件: {archive.name}")
                    self._task_status[dataset_id] = {
                        "status": "error",
                        "message": f"损坏的压缩文件: {archive.name}",
                    }
                    self.warehouse.update_metadata(dataset_id, {
                        "status": DatasetStatus.ERROR.value,
                    })
                    return
                except tarfile.TarError as e:
                    logger.error(f"[Ingestion] TAR解压失败: {e}")
                    self._task_status[dataset_id] = {
                        "status": "error",
                        "message": f"TAR解压失败: {archive.name}",
                    }
                    self.warehouse.update_metadata(dataset_id, {
                        "status": DatasetStatus.ERROR.value,
                    })
                    return

            # 解压完成后, 整理文件结构
            self._reorganize_extracted_files(dataset_path)

            # 解压完成 -> 触发校验
            self._task_status[dataset_id] = {
                "status": "validating",
                "message": "解压完成, 正在校验...",
            }

            self._run_validation(dataset_id, dataset_path)

        except Exception as e:
            logger.error(f"[Ingestion] 解压异常: {e}", exc_info=True)
            self._task_status[dataset_id] = {
                "status": "error",
                "message": f"处理异常: {str(e)}",
            }
            self.warehouse.update_metadata(dataset_id, {
                "status": DatasetStatus.ERROR.value,
            })

    def _reorganize_extracted_files(self, dataset_path: Path):
        """
        解压后整理: 将散落的图片/标注文件移入 images/ 和 labels/ 子目录
        """
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        for root, dirs, files in os.walk(dataset_path):
            root_path = Path(root)
            # 跳过 images/ labels/ 自身
            if root_path == images_dir or root_path == labels_dir:
                continue
            # 跳过 meta.json 所在层
            if root_path == dataset_path:
                for fname in files:
                    fpath = root_path / fname
                    ext = fpath.suffix.lower()
                    if ext in IMAGE_EXTENSIONS:
                        shutil.move(str(fpath), str(images_dir / fname))
                    elif ext in LABEL_EXTENSIONS and fname != "meta.json":
                        shutil.move(str(fpath), str(labels_dir / fname))
                continue

            # 子目录中的文件
            for fname in files:
                fpath = root_path / fname
                ext = fpath.suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    dest = images_dir / fname
                    if not dest.exists():
                        shutil.move(str(fpath), str(dest))
                elif ext in LABEL_EXTENSIONS and fname not in {'meta.json', 'classes.txt'}:
                    dest = labels_dir / fname
                    if not dest.exists():
                        shutil.move(str(fpath), str(dest))

    def _run_validation(self, dataset_id: str, dataset_path: Path):
        """运行校验并更新元数据"""
        report = DataGatekeeper.validate_yolo_dataset(dataset_path)

        self.warehouse.update_metadata(dataset_id, {
            "image_count": report["image_count"],
            "label_count": report["label_count"],
            "format": report["format"],
            "status": report["status"],
            "validation_report": report,
        })

        self._task_status[dataset_id] = {
            "status": "completed",
            "message": f"处理完成: {report['image_count']}张图像, "
                       f"匹配率{report['match_ratio']:.0%}",
            "validation_report": report,
        }

    # ---- 策略 A-2: 分片断点续传 ----

    def init_chunk_upload(
        self,
        file_name: str,
        file_size: int,
        total_chunks: int,
        file_md5: str,
        voltage_level: str,
        plugin_type: str,
        uploader: str = "anonymous",
    ) -> str:
        """
        初始化分片上传会话
        
        Returns:
            upload_session_id
        """
        session_id = hashlib.md5(
            f"{file_name}_{file_size}_{file_md5}_{time.time()}".encode()
        ).hexdigest()[:16]

        session_dir = self.warehouse.chunk_temp_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            self._chunk_sessions[session_id] = {
                "file_name": file_name,
                "file_size": file_size,
                "total_chunks": total_chunks,
                "file_md5": file_md5,
                "received_chunks": set(),
                "session_dir": str(session_dir),
                "voltage_level": voltage_level,
                "plugin_type": plugin_type,
                "uploader": uploader,
                "created_at": datetime.now().isoformat(),
            }

        logger.info(f"[Ingestion] 分片上传会话: {session_id}, "
                    f"{total_chunks} chunks, {file_size} bytes")
        return session_id

    def receive_chunk(
        self,
        session_id: str,
        chunk_index: int,
        chunk_data: bytes,
        chunk_md5: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        接收单个分片
        
        Returns:
            {"received": True, "completed": bool, "progress": float}
        """
        with self._lock:
            session = self._chunk_sessions.get(session_id)
            if not session:
                raise ValueError(f"无效的上传会话: {session_id}")

        session_dir = Path(session["session_dir"])
        chunk_path = session_dir / f"chunk_{chunk_index:06d}"

        # 写入分片
        with open(chunk_path, 'wb') as f:
            f.write(chunk_data)

        # 可选: MD5 校验
        if chunk_md5:
            actual_md5 = hashlib.md5(chunk_data).hexdigest()
            if actual_md5 != chunk_md5:
                chunk_path.unlink()
                raise ValueError(
                    f"分片 {chunk_index} MD5 不匹配: "
                    f"期望 {chunk_md5}, 实际 {actual_md5}"
                )

        with self._lock:
            session["received_chunks"].add(chunk_index)
            received = len(session["received_chunks"])
            total = session["total_chunks"]

        progress = received / total if total > 0 else 0
        completed = received >= total

        return {
            "received": True,
            "chunk_index": chunk_index,
            "completed": completed,
            "progress": round(progress, 4),
            "received_count": received,
            "total_chunks": total,
        }

    def merge_chunks(self, session_id: str) -> Tuple[str, Path]:
        """
        合并所有分片 -> 创建数据集实例 -> 解压 (如果是压缩包)
        
        Returns:
            (dataset_id, dataset_path)
        """
        with self._lock:
            session = self._chunk_sessions.get(session_id)
            if not session:
                raise ValueError(f"无效的上传会话: {session_id}")

        session_dir = Path(session["session_dir"])
        file_name = session["file_name"]
        total_chunks = session["total_chunks"]

        # 合并
        merged_path = session_dir / file_name
        with open(merged_path, 'wb') as outf:
            for i in range(total_chunks):
                chunk_path = session_dir / f"chunk_{i:06d}"
                if not chunk_path.exists():
                    raise FileNotFoundError(f"缺少分片: {i}")
                with open(chunk_path, 'rb') as cf:
                    outf.write(cf.read())

        # 校验文件完整性 (可选MD5)
        if session.get("file_md5"):
            md5_hash = hashlib.md5()
            with open(merged_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    md5_hash.update(chunk)
            if md5_hash.hexdigest() != session["file_md5"]:
                logger.error(f"[Ingestion] 文件MD5不匹配: {session_id}")
                # 不抛异常, 仅记录警告

        # 创建数据集实例
        dataset_path, dataset_id = self.warehouse.create_dataset_instance(
            voltage_level=session["voltage_level"],
            plugin_type=session["plugin_type"],
            uploader=session["uploader"],
            dataset_name=Path(file_name).stem,
        )

        # 移动合并后的文件到数据集目录
        dest = dataset_path / file_name
        shutil.move(str(merged_path), str(dest))

        # 清理分片临时目录
        shutil.rmtree(session_dir, ignore_errors=True)

        with self._lock:
            del self._chunk_sessions[session_id]

        logger.info(f"[Ingestion] 分片合并完成: {dataset_id}")
        return dataset_id, dataset_path

    # ---- 策略 B: 服务器端扫描导入 ----

    def scan_local_import(self, voltage_level: str, plugin_type: str,
                          uploader: str = "server_import") -> List[Dict]:
        """
        扫描 temp_upload/ 目录下的压缩包, 导入到结构化仓库
        
        Returns:
            导入结果列表
        """
        results = []
        scan_dir = self.warehouse.temp_upload_path

        if not scan_dir.exists():
            return results

        archives = []
        for ext in ARCHIVE_EXTENSIONS:
            archives.extend(scan_dir.glob(f"*{ext}"))

        logger.info(f"[Ingestion] 扫描 {scan_dir}: 发现 {len(archives)} 个压缩包")

        for archive_path in archives:
            try:
                dataset_name = archive_path.stem
                dataset_path, dataset_id = self.warehouse.create_dataset_instance(
                    voltage_level=voltage_level,
                    plugin_type=plugin_type,
                    uploader=uploader,
                    dataset_name=dataset_name,
                )

                # 移动压缩包到数据集目录
                dest = dataset_path / archive_path.name
                shutil.move(str(archive_path), str(dest))

                results.append({
                    "dataset_id": dataset_id,
                    "file_name": archive_path.name,
                    "status": "queued",
                    "message": "已加入解压队列",
                })

            except Exception as e:
                results.append({
                    "dataset_id": None,
                    "file_name": archive_path.name,
                    "status": "error",
                    "message": str(e),
                })

        return results

    def get_task_status(self, dataset_id: str) -> Dict:
        """获取异步任务状态"""
        return self._task_status.get(dataset_id, {
            "status": "unknown",
            "message": "无任务信息",
        })


# =============================================================================
# 模块四: 动态训练数据聚合器 (Dynamic Data Aggregator)
# =============================================================================

class DataAggregator:
    """
    动态训练数据聚合器
    
    核心能力:
    - 递归加载指定 电压+插件 下所有有效数据集
    - 虚拟合并: 不物理移动文件, 而是生成临时 dataset_config.yaml
    - 聚合后的 YAML 可直接喂给 YOLO 训练引擎
    
    优势:
    - 学生A昨天传了1000张, 学生B今天传了500张
    - 无需手动合并, 训练时自动把这1500张一起用
    """

    def __init__(self, warehouse: DataWarehouse):
        self.warehouse = warehouse

    def aggregate_datasets(
        self,
        voltage_level: str,
        plugin_type: str,
        output_yaml_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        聚合指定 电压+插件 下的所有有效数据集, 生成 YOLO 训练 YAML
        
        Args:
            voltage_level: 电压等级
            plugin_type: 插件类型
            output_yaml_path: 输出 YAML 路径 (默认自动生成)
            
        Returns:
            {
                "yaml_path": str,
                "dataset_count": int,
                "total_images": int,
                "total_labels": int,
                "datasets": [{ "id": ..., "images": ..., "labels": ... }],
                "classes": {...}
            }
        """
        # 获取所有有效数据集
        valid_dirs = self.warehouse.get_all_valid_dataset_dirs(
            voltage_level, plugin_type
        )

        if not valid_dirs:
            logger.warning(
                f"[Aggregator] 未找到有效数据集: {voltage_level}/{plugin_type}"
            )
            return {
                "yaml_path": None,
                "dataset_count": 0,
                "total_images": 0,
                "total_labels": 0,
                "datasets": [],
                "classes": {},
            }

        # 收集所有数据集信息
        all_train_image_dirs = []
        all_train_label_dirs = []
        total_images = 0
        total_labels = 0
        dataset_details = []

        for ds_dir in valid_dirs:
            ds_id = ds_dir.name
            meta = self.warehouse.get_metadata(ds_id) or {}

            images_dir = ds_dir / "images"
            labels_dir = ds_dir / "labels"

            img_count = len(DataGatekeeper.find_images(ds_dir))
            lbl_count = len(DataGatekeeper.find_labels(ds_dir))

            # 图像目录可能在 images/ 子目录或直接在根目录
            if images_dir.exists() and any(images_dir.iterdir()):
                all_train_image_dirs.append(str(images_dir))
            elif img_count > 0:
                all_train_image_dirs.append(str(ds_dir))

            if labels_dir.exists() and any(labels_dir.iterdir()):
                all_train_label_dirs.append(str(labels_dir))
            elif lbl_count > 0:
                all_train_label_dirs.append(str(ds_dir))

            total_images += img_count
            total_labels += lbl_count

            dataset_details.append({
                "id": ds_id,
                "path": str(ds_dir),
                "images": img_count,
                "labels": lbl_count,
                "format": meta.get("format", "UNKNOWN"),
                "uploader": meta.get("uploader", "unknown"),
                "upload_time": meta.get("upload_time", ""),
            })

        # 获取检测类别
        classes = self._get_detection_classes(plugin_type, voltage_level)

        # 生成 YAML 配置
        if output_yaml_path is None:
            output_dir = self.warehouse.processed_path / voltage_level / plugin_type
            output_dir.mkdir(parents=True, exist_ok=True)
            output_yaml_path = output_dir / "aggregated_data.yaml"

        yaml_config = {
            "# Auto-generated by DataAggregator": None,
            "# Do not edit manually": None,
            "path": str(
                self.warehouse.raw_path / voltage_level / plugin_type
            ),
            "train": all_train_image_dirs,
            "val": all_train_image_dirs,     # 训练时可进一步拆分
            "nc": len(classes),
            "names": classes,
            "# Aggregation metadata": None,
            "voltage_level": voltage_level,
            "plugin_type": plugin_type,
            "aggregated_datasets": len(valid_dirs),
            "total_images": total_images,
            "generated_at": datetime.now().isoformat(),
        }

        # 清理 None 注释键
        clean_config = {k: v for k, v in yaml_config.items() if v is not None}

        if YAML_AVAILABLE:
            with open(output_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(clean_config, f, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)
        else:
            with open(output_yaml_path, 'w', encoding='utf-8') as f:
                json.dump(clean_config, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[Aggregator] 聚合完成: {len(valid_dirs)} 个数据集, "
            f"{total_images} 张图像 -> {output_yaml_path}"
        )

        return {
            "yaml_path": str(output_yaml_path),
            "dataset_count": len(valid_dirs),
            "total_images": total_images,
            "total_labels": total_labels,
            "datasets": dataset_details,
            "classes": classes,
        }

    def generate_split_yaml(
        self,
        voltage_level: str,
        plugin_type: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        output_yaml_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        生成带 train/val/test 拆分的 YAML (需要物理拷贝)
        
        这个方法会:
        1. 收集所有有效数据集的图片和标签
        2. 按比例随机拆分到 processed/{voltage}/{plugin}/images/{train,val,test}
        3. 生成标准 YOLO data.yaml
        """
        valid_dirs = self.warehouse.get_all_valid_dataset_dirs(
            voltage_level, plugin_type
        )

        if not valid_dirs:
            return {"error": "No valid datasets found"}

        # 收集所有 图片-标签 对
        all_pairs = []  # [(image_path, label_path_or_None), ...]

        for ds_dir in valid_dirs:
            images = DataGatekeeper.find_images(ds_dir)
            labels = DataGatekeeper.find_labels(ds_dir)
            label_map = {l.stem: l for l in labels}

            for img in images:
                lbl = label_map.get(img.stem)
                all_pairs.append((img, lbl))

        random.shuffle(all_pairs)
        n = len(all_pairs)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {
            "train": all_pairs[:train_end],
            "val": all_pairs[train_end:val_end],
            "test": all_pairs[val_end:],
        }

        # 目标目录
        out_base = self.warehouse.processed_path / voltage_level / plugin_type
        for split_name in ["train", "val", "test"]:
            (out_base / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (out_base / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        # 拷贝文件
        stats = {}
        for split_name, pairs in splits.items():
            img_dest = out_base / "images" / split_name
            lbl_dest = out_base / "labels" / split_name

            for img_path, lbl_path in pairs:
                shutil.copy2(img_path, img_dest / img_path.name)
                if lbl_path and lbl_path.exists():
                    shutil.copy2(lbl_path, lbl_dest / lbl_path.name)

            stats[split_name] = len(pairs)

        # 生成 data.yaml
        classes = self._get_detection_classes(plugin_type, voltage_level)
        yaml_config = {
            "path": str(out_base),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(classes),
            "names": classes,
            "voltage_level": voltage_level,
            "plugin_type": plugin_type,
            "generated_at": datetime.now().isoformat(),
            "split_stats": stats,
        }

        if output_yaml_path is None:
            output_yaml_path = out_base / "data.yaml"

        if YAML_AVAILABLE:
            with open(output_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_config, f, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)
        else:
            with open(output_yaml_path, 'w', encoding='utf-8') as f:
                json.dump(yaml_config, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[Aggregator] 拆分完成: train={stats['train']}, "
            f"val={stats['val']}, test={stats['test']}"
        )

        return {
            "yaml_path": str(output_yaml_path),
            "total_pairs": n,
            "splits": stats,
            "classes": classes,
            "output_base": str(out_base),
        }

    @staticmethod
    def _get_detection_classes(
        plugin_type: str, voltage_level: str
    ) -> Dict[int, str]:
        """获取检测类别 (复用 training_api 中的定义)"""
        classes_map = {
            "transformer": [
                "oil_leak", "rust", "surface_damage", "foreign_object",
                "silica_gel_normal", "silica_gel_abnormal",
                "oil_level_normal", "oil_level_abnormal",
                "bushing_crack", "porcelain_contamination",
                "thermal_normal", "thermal_warning", "thermal_danger",
            ],
            "switch": [
                "breaker_open", "breaker_closed", "breaker_intermediate",
                "isolator_open", "isolator_closed",
                "grounding_open", "grounding_closed",
                "indicator_red", "indicator_green",
            ],
            "busbar": [
                "insulator_crack", "insulator_dirty", "insulator_flashover",
                "fitting_loose", "fitting_rust", "wire_damage",
                "foreign_object", "bird", "pin_missing",
            ],
            "capacitor": [
                "capacitor_unit", "capacitor_tilted", "capacitor_fallen",
                "capacitor_missing", "person", "vehicle", "fuse_blown",
            ],
            "meter": [
                "sf6_pressure_gauge", "oil_temp_gauge", "oil_level_gauge",
                "digital_display", "pointer_gauge", "led_indicator",
            ],
            "thermal": [
                "thermal_normal", "thermal_warning", "thermal_danger",
                "hotspot", "cold_spot",
            ],
        }

        class_list = classes_map.get(plugin_type, ["class_0", "class_1"])
        return {i: name for i, name in enumerate(class_list)}


# =============================================================================
# 统一门面 (Facade)
# =============================================================================

class DataManager:
    """
    数据管理器 - 统一门面
    
    整合四大模块, 提供简洁的 API 供 FastAPI 路由调用
    """

    _instance: Optional['DataManager'] = None

    def __init__(self, data_root: Optional[Path] = None):
        self.warehouse = DataWarehouse(data_root)
        self.ingestion = IngestionEngine(self.warehouse)
        self.gatekeeper = DataGatekeeper
        self.aggregator = DataAggregator(self.warehouse)

    @classmethod
    def get_instance(cls, data_root: Optional[Path] = None) -> 'DataManager':
        """单例获取"""
        if cls._instance is None:
            cls._instance = cls(data_root)
        return cls._instance

    # ---- 便捷方法 ----

    def upload_files(
        self,
        files: List[Any],
        voltage_level: str,
        plugin_type: str,
        uploader: str = "anonymous",
        dataset_name: str = "upload",
    ) -> Dict[str, Any]:
        """
        上传文件 -> 创建数据集 -> 返回 dataset_id
        (解压和校验在后台异步完成)
        """
        dataset_id, dataset_path = self.ingestion.receive_files(
            files=files,
            voltage_level=voltage_level,
            plugin_type=plugin_type,
            uploader=uploader,
            dataset_name=dataset_name,
        )
        return {
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "status": "received",
        }

    def process_dataset_background(self, dataset_id: str, dataset_path: str):
        """
        后台处理: 解压 + 校验 (供 BackgroundTasks 调用)
        """
        path = Path(dataset_path)
        self.ingestion.extract_archives_in_dataset(dataset_id, path)

        # 如果没有压缩包, 直接校验
        task_status = self.ingestion.get_task_status(dataset_id)
        if task_status.get("status") == "unknown":
            self.ingestion._run_validation(dataset_id, path)

    def validate_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """手动触发校验"""
        meta = self.warehouse.get_metadata(dataset_id)
        if not meta:
            return {"valid": False, "errors": ["数据集不存在"]}

        dataset_path = Path(meta["data_path"])
        if not dataset_path.exists():
            return {"valid": False, "errors": ["数据集目录不存在"]}

        report = self.gatekeeper.validate_yolo_dataset(dataset_path)
        self.warehouse.update_metadata(dataset_id, {
            "image_count": report["image_count"],
            "label_count": report["label_count"],
            "format": report["format"],
            "status": report["status"],
            "validation_report": report,
        })
        return report

    def list_datasets(self, **filters) -> List[Dict]:
        """列出数据集"""
        return self.warehouse.list_datasets(**filters)

    def delete_dataset(self, dataset_id: str) -> bool:
        """删除数据集"""
        return self.warehouse.delete_dataset(dataset_id)

    def get_task_status(self, dataset_id: str) -> Dict:
        """获取异步任务状态"""
        return self.ingestion.get_task_status(dataset_id)

    def scan_and_import(
        self,
        voltage_level: str,
        plugin_type: str,
        uploader: str = "server_import"
    ) -> List[Dict]:
        """扫描本地目录导入"""
        return self.ingestion.scan_local_import(
            voltage_level, plugin_type, uploader
        )

    def aggregate_for_training(
        self,
        voltage_level: str,
        plugin_type: str,
    ) -> Dict[str, Any]:
        """聚合数据集用于训练"""
        return self.aggregator.aggregate_datasets(
            voltage_level, plugin_type
        )

    def prepare_training_split(
        self,
        voltage_level: str,
        plugin_type: str,
        train_ratio: float = 0.8,
    ) -> Dict[str, Any]:
        """准备训练拆分"""
        return self.aggregator.generate_split_yaml(
            voltage_level, plugin_type,
            train_ratio=train_ratio,
            val_ratio=(1 - train_ratio) * 0.75,
            test_ratio=(1 - train_ratio) * 0.25,
        )

    @property
    def voltage_levels(self) -> List[str]:
        return VoltageLevel.values()

    @property
    def plugin_types(self) -> List[str]:
        return PluginType.values()
