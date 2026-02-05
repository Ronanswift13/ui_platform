#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据上传API - 完整修复版本 V3.0
输变电激光星芒破夜绘明监测平台

解决的问题:
1. 后端上传端点完整实现
2. 电压等级命名标准化 (220kv -> HV_220kV)
3. 插件ID标准化映射
4. 上传数据与训练管道完整集成
5. YOLO格式数据组织

作者: Claude AI Assistant
日期: 2026-02-02
"""

import os
import json
import shutil
import zipfile
import tarfile
import logging
import hashlib
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# FastAPI相关导入
try:
    from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("警告: FastAPI未安装，API功能不可用")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 路径配置
# =============================================================================

# 项目根目录 - 自动检测
PROJECT_ROOT = Path(__file__).parent.parent
if not (PROJECT_ROOT / "training").exists():
    PROJECT_ROOT = Path.cwd()

# 训练数据目录
TRAINING_BASE = PROJECT_ROOT / "training"
DATA_RAW = TRAINING_BASE / "data" / "raw"
DATA_PROCESSED = TRAINING_BASE / "data" / "processed"
DATA_AUGMENTED = TRAINING_BASE / "data" / "augmented"

# 上传记录文件
UPLOAD_RECORDS_FILE = TRAINING_BASE / "data" / "upload_records.json"

# 确保目录存在
for dir_path in [DATA_RAW, DATA_PROCESSED, DATA_AUGMENTED]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 电压等级标准化映射 (关键修复 #4)
# =============================================================================

VOLTAGE_NORMALIZE = {
    # 前端简化格式 -> 训练系统内部格式
    "220kv": "HV_220kV",
    "110kv": "HV_110kV",
    "35kv": "MV_35kV",
    "10kv": "LV_10kV",
    "500kv": "EHV_500kV",
    "330kv": "EHV_330kV",
    "1000kv": "UHV_1000kV_AC",
    "800kv": "UHV_800kV_DC",
    # 已标准化格式 (保持不变)
    "HV_220kV": "HV_220kV",
    "HV_110kV": "HV_110kV",
    "MV_35kV": "MV_35kV",
    "LV_10kV": "LV_10kV",
    "EHV_500kV": "EHV_500kV",
    "EHV_330kV": "EHV_330kV",
    "UHV_1000kV_AC": "UHV_1000kV_AC",
    "UHV_800kV_DC": "UHV_800kV_DC",
}

# 逆向映射 (用于前端显示)
VOLTAGE_DISPLAY = {v: k for k, v in VOLTAGE_NORMALIZE.items() if not k.startswith("HV_") and not k.startswith("MV_")}

# =============================================================================
# 插件ID标准化映射
# =============================================================================

PLUGIN_NORMALIZE = {
    # 前端插件名 -> 标准ID
    "transformer_inspection": "transformer",
    "transformer_check": "transformer",
    "switch_inspection": "switch",
    "switch_check": "switch",
    "busbar_inspection": "busbar",
    "busbar_check": "busbar",
    "capacitor_inspection": "capacitor",
    "capacitor_check": "capacitor",
    "meter_reading": "meter",
    "meter_inspection": "meter",
    "bird_monitoring": "bird",
    "bird_detection": "bird",
    "acoustic_monitoring": "acoustic",
    "acoustic_detection": "acoustic",
    "gas_detection": "gas",
    "gas_monitoring": "gas",
    "hyperspectral": "hyperspectral",
    "hyperspectral_detection": "hyperspectral",
    "slam_mapping": "slam",
    "slam": "slam",
    "multimodal_fusion": "fusion",
    "fusion": "fusion",
    "indoor_fence": "indoor_fence",
    # 已标准化的ID (保持不变)
    "transformer": "transformer",
    "switch": "switch",
    "busbar": "busbar",
    "capacitor": "capacitor",
    "meter": "meter",
    "bird": "bird",
    "acoustic": "acoustic",
    "gas": "gas",
}

# =============================================================================
# 插件检测类别定义 (与训练系统保持一致)
# =============================================================================

PLUGIN_CLASSES = {
    "transformer": [
        "oil_leak", "oil_level_low", "oil_level_high", "rust", "corrosion",
        "crack", "damage", "foreign_object", "bird_nest", "overheating",
        "insulator_damage", "bushing_crack", "normal"
    ],
    "switch": [
        "breaker_open", "breaker_closed", "breaker_fault", "isolator_open",
        "isolator_closed", "isolator_fault", "contact_wear", "arc_damage",
        "mechanism_fault", "overheating", "rust", "normal"
    ],
    "busbar": [
        "pin_missing", "pin_loose", "crack", "corrosion", "foreign_object",
        "bird_nest", "overheating", "insulator_damage", "normal"
    ],
    "capacitor": [
        "tilt", "collapse", "missing", "oil_leak", "bulge", "crack", "normal"
    ],
    "meter": [
        "digit_0", "digit_1", "digit_2", "digit_3", "digit_4",
        "digit_5", "digit_6", "digit_7", "digit_8", "digit_9"
    ],
    "bird": [
        "sparrow", "crow", "pigeon", "eagle", "unknown_bird"
    ],
    "acoustic": [
        "normal", "partial_discharge", "corona", "mechanical_fault", "abnormal"
    ],
    "gas": [
        "sf6_normal", "sf6_leak", "oil_gas_normal", "oil_gas_abnormal", "alarm"
    ],
    "hyperspectral": [
        "normal", "contamination", "aging", "damage", "abnormal"
    ],
    "slam": [
        "obstacle", "path", "equipment", "marker", "unknown"
    ],
    "fusion": [
        "defect_confirmed", "defect_suspected", "normal", "needs_review",
        "thermal_anomaly", "visual_anomaly"
    ],
    "indoor_fence": [
        "person", "intrusion", "equipment", "normal"
    ],
}

# =============================================================================
# 支持的文件格式 (修复 #3)
# =============================================================================

SUPPORTED_FORMATS = {
    "images": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".webp"],
    "videos": [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".mpeg", ".mpg"],
    "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"],
    "labels": [".txt", ".xml", ".json", ".csv", ".yaml", ".yml"],
    "archives": [".zip", ".tar", ".gz", ".tgz", ".rar", ".7z"],
}

ALL_SUPPORTED_EXTENSIONS = []
for exts in SUPPORTED_FORMATS.values():
    ALL_SUPPORTED_EXTENSIONS.extend(exts)

# =============================================================================
# 数据模型定义
# =============================================================================

class UploadStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class UploadRecord:
    """上传记录"""
    id: str
    created_at: str
    voltage_level: str
    voltage_level_raw: str  # 原始前端值
    plugins: List[str]
    plugins_raw: List[str]  # 原始前端值
    file_count: int
    total_size: int
    status: str
    data_path: str
    message: str = ""
    image_count: int = 0
    video_count: int = 0
    audio_count: int = 0
    label_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


if FASTAPI_AVAILABLE:
    class ValidateRequest(BaseModel):
        """验证请求模型"""
        voltage_level: str
        plugins: List[str]
        files: List[Dict[str, Any]]

    class ImportLocalRequest(BaseModel):
        """本地导入请求"""
        source_path: str
        voltage_level: str
        plugins: List[str]

# =============================================================================
# 上传记录管理器
# =============================================================================

class UploadRecordManager:
    """上传记录管理器"""
    
    def __init__(self, records_file: Path = UPLOAD_RECORDS_FILE):
        self.records_file = records_file
        self.records_file.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, UploadRecord] = {}
        self._load()
    
    def _load(self):
        """加载记录"""
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for record_data in data.get("records", []):
                        record = UploadRecord(**record_data)
                        self._records[record.id] = record
                logger.info(f"[UploadRecordManager] 加载 {len(self._records)} 条上传记录")
            except Exception as e:
                logger.error(f"[UploadRecordManager] 加载记录失败: {e}")
                self._records = {}
    
    def _save(self):
        """保存记录"""
        try:
            data = {
                "updated_at": datetime.now().isoformat(),
                "records": [r.to_dict() for r in self._records.values()]
            }
            with open(self.records_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[UploadRecordManager] 保存记录失败: {e}")
    
    def create(self, record: UploadRecord) -> UploadRecord:
        """创建记录"""
        self._records[record.id] = record
        self._save()
        return record
    
    def update(self, record_id: str, **kwargs) -> Optional[UploadRecord]:
        """更新记录"""
        if record_id in self._records:
            record = self._records[record_id]
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            self._save()
            return record
        return None
    
    def get(self, record_id: str) -> Optional[UploadRecord]:
        """获取记录"""
        return self._records.get(record_id)
    
    def list_all(self, limit: int = 100) -> List[UploadRecord]:
        """列出所有记录"""
        records = sorted(
            self._records.values(),
            key=lambda r: r.created_at,
            reverse=True
        )
        return records[:limit]
    
    def delete(self, record_id: str) -> bool:
        """删除记录"""
        if record_id in self._records:
            del self._records[record_id]
            self._save()
            return True
        return False


# 全局记录管理器
record_manager = UploadRecordManager()

# =============================================================================
# 数据处理工具函数
# =============================================================================

def normalize_voltage_level(voltage_level: str) -> str:
    """标准化电压等级命名"""
    normalized = VOLTAGE_NORMALIZE.get(voltage_level.lower(), voltage_level)
    if normalized != voltage_level:
        logger.info(f"[DataUpload] 电压等级标准化: {voltage_level} -> {normalized}")
    return normalized


def normalize_plugin_id(plugin_id: str) -> str:
    """标准化插件ID"""
    normalized = PLUGIN_NORMALIZE.get(plugin_id.lower(), plugin_id.lower())
    if normalized != plugin_id:
        logger.info(f"[DataUpload] 插件ID标准化: {plugin_id} -> {normalized}")
    return normalized


def normalize_plugin_list(plugins: List[str]) -> List[str]:
    """标准化插件列表"""
    return [normalize_plugin_id(p) for p in plugins]


def get_file_category(filename: str) -> str:
    """获取文件类别"""
    ext = Path(filename).suffix.lower()
    for category, extensions in SUPPORTED_FORMATS.items():
        if ext in extensions:
            return category
    return "unknown"


def is_supported_file(filename: str) -> bool:
    """检查文件是否支持"""
    ext = Path(filename).suffix.lower()
    return ext in ALL_SUPPORTED_EXTENSIONS


def generate_upload_id() -> str:
    """生成上传ID"""
    return f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def calculate_file_hash(file_path: Path) -> str:
    """计算文件哈希"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def extract_archive(archive_path: Path, extract_to: Path) -> List[Path]:
    """解压压缩包"""
    extracted_files = []
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_to)
                extracted_files = [extract_to / name for name in zf.namelist()]
        
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            mode = 'r:gz' if archive_path.suffix in ['.gz', '.tgz'] else 'r'
            with tarfile.open(archive_path, mode) as tf:
                tf.extractall(extract_to)
                extracted_files = [extract_to / m.name for m in tf.getmembers()]
        
        logger.info(f"[DataUpload] 解压 {archive_path.name} -> {len(extracted_files)} 个文件")
    
    except Exception as e:
        logger.error(f"[DataUpload] 解压失败: {e}")
    
    return extracted_files


def organize_yolo_structure(
    source_dir: Path,
    voltage_level: str,
    plugin: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2
) -> Tuple[int, int, int]:
    """
    组织数据为YOLO训练结构
    
    返回: (train_count, val_count, test_count)
    """
    import random
    
    # 目标目录
    dest_base = DATA_PROCESSED / voltage_level / plugin
    
    # 创建YOLO目录结构
    for split in ["train", "val", "test"]:
        (dest_base / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_base / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # 收集所有图像文件
    image_files = []
    for ext in SUPPORTED_FORMATS["images"]:
        image_files.extend(source_dir.rglob(f"*{ext}"))
        image_files.extend(source_dir.rglob(f"*{ext.upper()}"))
    
    # 去重
    image_files = list(set(image_files))
    random.shuffle(image_files)
    
    if not image_files:
        logger.warning(f"[DataUpload] 未找到图像文件: {source_dir}")
        return (0, 0, 0)
    
    # 计算分割点
    n = len(image_files)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }
    
    counts = {"train": 0, "val": 0, "test": 0}
    
    for split_name, files in splits.items():
        for img_path in files:
            # 复制图像
            dest_img = dest_base / "images" / split_name / img_path.name
            shutil.copy2(img_path, dest_img)
            counts[split_name] += 1
            
            # 查找对应的标注文件
            label_path = img_path.with_suffix('.txt')
            if not label_path.exists():
                # 尝试在labels子目录中查找
                parent_labels = img_path.parent.parent / "labels"
                if parent_labels.exists():
                    label_path = parent_labels / img_path.with_suffix('.txt').name
            
            if label_path.exists():
                dest_label = dest_base / "labels" / split_name / label_path.name
                shutil.copy2(label_path, dest_label)
    
    # 生成data.yaml
    generate_data_yaml(voltage_level, plugin)
    
    logger.info(f"[DataUpload] YOLO结构组织完成: {voltage_level}/{plugin} "
                f"(train={counts['train']}, val={counts['val']}, test={counts['test']})")
    
    return (counts["train"], counts["val"], counts["test"])


def generate_data_yaml(voltage_level: str, plugin: str):
    """生成YOLO训练配置文件"""
    data_path = DATA_PROCESSED / voltage_level / plugin
    
    # 获取类别列表
    classes = PLUGIN_CLASSES.get(plugin, ["defect", "normal"])
    
    yaml_content = f"""# YOLO训练数据配置
# 自动生成于: {datetime.now().isoformat()}
# 电压等级: {voltage_level}
# 插件类型: {plugin}

path: {data_path}
train: images/train
val: images/val
test: images/test

nc: {len(classes)}
names: {classes}
"""
    
    yaml_path = data_path / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    logger.info(f"[DataUpload] 生成 data.yaml: {yaml_path}")
    return yaml_path


# =============================================================================
# 后台数据处理任务
# =============================================================================

def process_upload_task(
    record_id: str,
    raw_path: Path,
    voltage_level: str,
    plugins: List[str]
):
    """后台处理上传的数据"""
    try:
        record_manager.update(record_id, status=UploadStatus.PROCESSING.value)
        
        # 解压压缩包
        for archive in raw_path.glob("*"):
            if archive.suffix in SUPPORTED_FORMATS["archives"]:
                extract_archive(archive, raw_path)
        
        # 为每个插件组织数据
        total_train = total_val = total_test = 0
        for plugin in plugins:
            train, val, test = organize_yolo_structure(raw_path, voltage_level, plugin)
            total_train += train
            total_val += val
            total_test += test
        
        # 更新记录
        record_manager.update(
            record_id,
            status=UploadStatus.COMPLETED.value,
            message=f"处理完成: train={total_train}, val={total_val}, test={total_test}"
        )
        
        logger.info(f"[DataUpload] 后台处理完成: {record_id}")
        
    except Exception as e:
        logger.error(f"[DataUpload] 后台处理失败: {e}")
        record_manager.update(
            record_id,
            status=UploadStatus.FAILED.value,
            message=str(e)
        )


# =============================================================================
# API路由定义
# =============================================================================

if FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/api/training/data", tags=["数据上传"])

    @router.get("/voltage-levels")
    async def get_voltage_levels():
        """获取支持的电压等级列表"""
        levels = [
            {"id": "UHV_1000kV_AC", "name": "特高压1000kV交流", "display": "1000kV"},
            {"id": "UHV_800kV_DC", "name": "特高压800kV直流", "display": "800kV"},
            {"id": "EHV_500kV", "name": "超高压500kV", "display": "500kV"},
            {"id": "EHV_330kV", "name": "超高压330kV", "display": "330kV"},
            {"id": "HV_220kV", "name": "高压220kV", "display": "220kV"},
            {"id": "HV_110kV", "name": "高压110kV", "display": "110kV"},
            {"id": "MV_35kV", "name": "中压35kV", "display": "35kV"},
            {"id": "LV_10kV", "name": "低压10kV", "display": "10kV"},
        ]
        return {"success": True, "levels": levels}

    @router.get("/plugin-types")
    async def get_plugin_types():
        """获取支持的插件类型列表"""
        plugins = [
            {"id": "transformer", "name": "变压器巡检", "classes": len(PLUGIN_CLASSES.get("transformer", []))},
            {"id": "switch", "name": "开关设备巡检", "classes": len(PLUGIN_CLASSES.get("switch", []))},
            {"id": "busbar", "name": "母线巡检", "classes": len(PLUGIN_CLASSES.get("busbar", []))},
            {"id": "capacitor", "name": "电容器巡检", "classes": len(PLUGIN_CLASSES.get("capacitor", []))},
            {"id": "meter", "name": "仪表读数", "classes": len(PLUGIN_CLASSES.get("meter", []))},
            {"id": "bird", "name": "鸟害监测", "classes": len(PLUGIN_CLASSES.get("bird", []))},
            {"id": "acoustic", "name": "声学监测", "classes": len(PLUGIN_CLASSES.get("acoustic", []))},
            {"id": "gas", "name": "气体检测", "classes": len(PLUGIN_CLASSES.get("gas", []))},
            {"id": "hyperspectral", "name": "高光谱检测", "classes": len(PLUGIN_CLASSES.get("hyperspectral", []))},
            {"id": "slam", "name": "SLAM建图", "classes": len(PLUGIN_CLASSES.get("slam", []))},
            {"id": "fusion", "name": "多模态融合", "classes": len(PLUGIN_CLASSES.get("fusion", []))},
        ]
        return {"success": True, "plugins": plugins}

    @router.get("/supported-formats")
    async def get_supported_formats():
        """获取支持的文件格式"""
        return {
            "success": True,
            "formats": SUPPORTED_FORMATS,
            "total_extensions": len(ALL_SUPPORTED_EXTENSIONS)
        }

    @router.post("/validate")
    async def validate_upload(request: ValidateRequest):
        """验证上传数据"""
        errors = []
        warnings = []
        
        # 验证电压等级
        voltage_normalized = normalize_voltage_level(request.voltage_level)
        if voltage_normalized not in [v for v in VOLTAGE_NORMALIZE.values()]:
            warnings.append(f"非标准电压等级: {request.voltage_level} (将使用: {voltage_normalized})")
        
        # 验证插件
        plugins_normalized = normalize_plugin_list(request.plugins)
        for plugin in plugins_normalized:
            if plugin not in PLUGIN_CLASSES:
                errors.append(f"不支持的插件类型: {plugin}")
        
        # 验证文件
        image_count = video_count = label_count = unsupported_count = 0
        total_size = 0
        
        for file_info in request.files:
            name = file_info.get("name", "")
            size = file_info.get("size", 0)
            total_size += size
            
            category = get_file_category(name)
            if category == "images":
                image_count += 1
            elif category == "videos":
                video_count += 1
            elif category == "labels":
                label_count += 1
            elif category == "unknown":
                unsupported_count += 1
                warnings.append(f"不支持的文件格式: {name}")
        
        # 检查图像-标注匹配
        if image_count > 0 and label_count == 0:
            warnings.append("未检测到标注文件，将需要手动标注或使用自动标注")
        
        valid = len(errors) == 0
        
        return {
            "valid": valid,
            "message": "验证通过" if valid else "验证失败，请检查错误信息",
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "voltage_level": voltage_normalized,
                "plugins": plugins_normalized,
                "image_count": image_count,
                "video_count": video_count,
                "label_count": label_count,
                "unsupported_count": unsupported_count,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
        }

    @router.post("/upload")
    async def upload_training_data(
        background_tasks: BackgroundTasks,
        voltage_level: str = Form(...),
        plugins: str = Form(...),  # JSON字符串
        files: List[UploadFile] = File(...)
    ):
        """
        上传训练数据
        
        这是主要的数据上传端点，接收文件并保存到训练目录
        """
        try:
            # 解析插件列表
            try:
                plugin_list = json.loads(plugins)
            except json.JSONDecodeError:
                plugin_list = [p.strip() for p in plugins.split(',')]
            
            # 🔧 关键修复: 标准化电压等级和插件ID
            voltage_level_raw = voltage_level
            voltage_level = normalize_voltage_level(voltage_level)
            
            plugins_raw = plugin_list.copy()
            plugin_list = normalize_plugin_list(plugin_list)
            
            logger.info(f"[DataUpload] 接收上传请求: voltage={voltage_level}, plugins={plugin_list}, files={len(files)}")
            
            # 生成上传ID和目录
            upload_id = generate_upload_id()
            raw_path = DATA_RAW / voltage_level / upload_id
            raw_path.mkdir(parents=True, exist_ok=True)
            
            # 统计信息
            total_size = 0
            image_count = video_count = audio_count = label_count = 0
            
            # 保存文件
            for file in files:
                file_path = raw_path / file.filename
                
                # 读取并保存文件
                content = await file.read()
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                total_size += len(content)
                
                # 统计文件类型
                category = get_file_category(file.filename)
                if category == "images":
                    image_count += 1
                elif category == "videos":
                    video_count += 1
                elif category == "audio":
                    audio_count += 1
                elif category == "labels":
                    label_count += 1
                
                logger.debug(f"[DataUpload] 保存文件: {file.filename} ({len(content)} bytes)")
            
            # 创建上传记录
            record = UploadRecord(
                id=upload_id,
                created_at=datetime.now().isoformat(),
                voltage_level=voltage_level,
                voltage_level_raw=voltage_level_raw,
                plugins=plugin_list,
                plugins_raw=plugins_raw,
                file_count=len(files),
                total_size=total_size,
                status=UploadStatus.PENDING.value,
                data_path=str(raw_path),
                image_count=image_count,
                video_count=video_count,
                audio_count=audio_count,
                label_count=label_count
            )
            record_manager.create(record)
            
            # 启动后台处理任务
            background_tasks.add_task(
                process_upload_task,
                upload_id,
                raw_path,
                voltage_level,
                plugin_list
            )
            
            logger.info(f"[DataUpload] 上传成功: {upload_id}, {len(files)} 个文件, {total_size/(1024*1024):.2f} MB")
            
            return {
                "success": True,
                "message": "数据上传成功，正在后台处理",
                "upload_id": upload_id,
                "voltage_level": voltage_level,
                "plugins": plugin_list,
                "file_count": len(files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "data_path": str(raw_path),
                "summary": {
                    "images": image_count,
                    "videos": video_count,
                    "audio": audio_count,
                    "labels": label_count
                }
            }
            
        except Exception as e:
            logger.error(f"[DataUpload] 上传失败: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/list")
    async def list_upload_records(limit: int = 100):
        """获取上传记录列表"""
        records = record_manager.list_all(limit)
        return {
            "success": True,
            "count": len(records),
            "records": [r.to_dict() for r in records]
        }

    @router.get("/record/{upload_id}")
    async def get_upload_record(upload_id: str):
        """获取单个上传记录"""
        record = record_manager.get(upload_id)
        if not record:
            raise HTTPException(status_code=404, detail="上传记录不存在")
        return {
            "success": True,
            "record": record.to_dict()
        }

    @router.delete("/record/{upload_id}")
    async def delete_upload_record(upload_id: str, delete_files: bool = False):
        """删除上传记录"""
        record = record_manager.get(upload_id)
        if not record:
            raise HTTPException(status_code=404, detail="上传记录不存在")
        
        # 删除文件
        if delete_files and record.data_path:
            data_path = Path(record.data_path)
            if data_path.exists():
                shutil.rmtree(data_path)
                logger.info(f"[DataUpload] 删除数据目录: {data_path}")
        
        # 删除记录
        record_manager.delete(upload_id)
        
        return {
            "success": True,
            "message": "上传记录已删除",
            "files_deleted": delete_files
        }

    @router.post("/import/local")
    async def import_local_data(
        request: ImportLocalRequest,
        background_tasks: BackgroundTasks
    ):
        """从服务器本地目录导入数据"""
        source_path = Path(request.source_path)
        
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"源目录不存在: {source_path}")
        
        # 标准化
        voltage_level = normalize_voltage_level(request.voltage_level)
        plugins = normalize_plugin_list(request.plugins)
        
        # 生成上传ID
        upload_id = generate_upload_id()
        
        # 复制到raw目录
        raw_path = DATA_RAW / voltage_level / upload_id
        shutil.copytree(source_path, raw_path)
        
        # 统计文件
        file_count = sum(1 for _ in raw_path.rglob("*") if _.is_file())
        
        # 创建记录
        record = UploadRecord(
            id=upload_id,
            created_at=datetime.now().isoformat(),
            voltage_level=voltage_level,
            voltage_level_raw=request.voltage_level,
            plugins=plugins,
            plugins_raw=request.plugins,
            file_count=file_count,
            total_size=sum(f.stat().st_size for f in raw_path.rglob("*") if f.is_file()),
            status=UploadStatus.PENDING.value,
            data_path=str(raw_path)
        )
        record_manager.create(record)
        
        # 后台处理
        background_tasks.add_task(process_upload_task, upload_id, raw_path, voltage_level, plugins)
        
        return {
            "success": True,
            "message": "本地数据导入成功，正在后台处理",
            "upload_id": upload_id,
            "file_count": file_count
        }

    @router.get("/stats")
    async def get_upload_stats():
        """获取上传统计信息"""
        records = record_manager.list_all(1000)
        
        total_files = sum(r.file_count for r in records)
        total_size = sum(r.total_size for r in records)
        
        # 按状态统计
        status_counts = {}
        for r in records:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
        
        # 按电压等级统计
        voltage_counts = {}
        for r in records:
            voltage_counts[r.voltage_level] = voltage_counts.get(r.voltage_level, 0) + 1
        
        return {
            "success": True,
            "stats": {
                "total_uploads": len(records),
                "total_files": total_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "by_status": status_counts,
                "by_voltage_level": voltage_counts
            }
        }


# =============================================================================
# 集成函数 - 将路由注册到FastAPI应用
# =============================================================================

def integrate_data_upload_routes(app):
    """
    将数据上传API路由集成到FastAPI应用
    
    使用方法:
        from data_upload_api_complete import integrate_data_upload_routes
        integrate_data_upload_routes(app)
    """
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI不可用，无法集成数据上传路由")
        return False
    
    try:
        app.include_router(router)
        logger.info("✓ 数据上传API路由已集成 (V3.0)")
        logger.info(f"  - POST /api/training/data/upload")
        logger.info(f"  - POST /api/training/data/validate")
        logger.info(f"  - GET  /api/training/data/list")
        logger.info(f"  - GET  /api/training/data/voltage-levels")
        logger.info(f"  - GET  /api/training/data/plugin-types")
        logger.info(f"  - DELETE /api/training/data/record/{{upload_id}}")
        return True
    except Exception as e:
        logger.error(f"集成数据上传路由失败: {e}")
        return False


# =============================================================================
# 独立运行测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("数据上传API - 功能测试")
    print("=" * 60)
    
    # 测试电压等级标准化
    print("\n1. 电压等级标准化测试:")
    test_voltages = ["220kv", "110kv", "500kv", "HV_220kV"]
    for v in test_voltages:
        normalized = normalize_voltage_level(v)
        print(f"   {v} -> {normalized}")
    
    # 测试插件标准化
    print("\n2. 插件ID标准化测试:")
    test_plugins = ["transformer_inspection", "switch", "meter_reading"]
    for p in test_plugins:
        normalized = normalize_plugin_id(p)
        print(f"   {p} -> {normalized}")
    
    # 测试文件分类
    print("\n3. 文件分类测试:")
    test_files = ["image.jpg", "video.mp4", "label.txt", "data.zip", "unknown.xyz"]
    for f in test_files:
        category = get_file_category(f)
        print(f"   {f} -> {category}")
    
    # 测试上传记录管理
    print("\n4. 上传记录管理测试:")
    test_record = UploadRecord(
        id=generate_upload_id(),
        created_at=datetime.now().isoformat(),
        voltage_level="HV_220kV",
        voltage_level_raw="220kv",
        plugins=["transformer"],
        plugins_raw=["transformer_inspection"],
        file_count=10,
        total_size=1024000,
        status=UploadStatus.COMPLETED.value,
        data_path="/test/path"
    )
    record_manager.create(test_record)
    print(f"   创建记录: {test_record.id}")
    
    records = record_manager.list_all()
    print(f"   总记录数: {len(records)}")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试完成")
    print("=" * 60)
    
    # 启动API服务器 (可选)
    if FASTAPI_AVAILABLE and os.environ.get("RUN_SERVER"):
        import uvicorn
        from fastapi import FastAPI
        
        app = FastAPI(title="数据上传API测试服务器")
        integrate_data_upload_routes(app)
        
        uvicorn.run(app, host="0.0.0.0", port=8081)
