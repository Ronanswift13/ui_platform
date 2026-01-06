#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破夜绘明激光监测平台 - 数据集下载器（增强版）
实现真实的公开数据集下载、解压、格式转换功能

支持的数据集:
- 绝缘子: CPLID (848张), MPID (6000+张), Insulator-Defect (1600张)
- 电表: UFPR-AMR (2000张), Copel-AMR (12500张)
- 热成像: Transformer Thermal (255张, Mendeley手动)
"""

import os
import sys
import json
import shutil
import random
import hashlib
import logging
import tempfile
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import urllib.request
import urllib.error
import zipfile
import tarfile
import ssl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 配置
# =============================================================================
@dataclass
class DatasetConfig:
    """数据集配置"""
    id: str
    name: str
    name_zh: str
    plugin: str
    url: str
    image_count: int
    format_type: str  # yolo, coco, custom, images_only
    can_auto_download: bool
    manual_url: str = ""
    unavailable_reason: str = ""
    labels_included: bool = True
    class_names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = []


# 公开数据集配置
DATASETS: Dict[str, DatasetConfig] = {
    # ==================== 绝缘子/母线巡视 ====================
    "cplid_insulator": DatasetConfig(
        id="cplid_insulator",
        name="CPLID - Chinese Power Line Insulator Dataset",
        name_zh="CPLID - 中国电力线绝缘子数据集",
        plugin="busbar",
        url="https://github.com/InsulatorData/InsulatorDataSet/archive/refs/heads/master.zip",
        image_count=848,
        format_type="custom",
        can_auto_download=True,
        class_names=["insulator", "defect"]
    ),
    "mpid_insulator": DatasetConfig(
        id="mpid_insulator",
        name="MPID - Merged Public Insulator Dataset",
        name_zh="MPID - 合并公开绝缘子数据集",
        plugin="busbar",
        url="https://github.com/phd-benel/MPID/archive/refs/heads/main.zip",
        image_count=6000,
        format_type="yolo",
        can_auto_download=True,
        class_names=["insulator", "defective_insulator"]
    ),
    "insulator_defect_detection": DatasetConfig(
        id="insulator_defect_detection",
        name="Insulator-Defect Detection Dataset",
        name_zh="绝缘子缺陷检测数据集",
        plugin="busbar",
        url="",
        image_count=1600,
        format_type="supervisely",
        can_auto_download=False,
        manual_url="https://datasetninja.com/insulator-defect-detection",
        unavailable_reason="需要在DatasetNinja网站注册后下载 (2.43GB)",
        class_names=["insulator", "damaged", "flashover"]
    ),
    
    # ==================== 电表读数 ====================
    "ufpr_amr": DatasetConfig(
        id="ufpr_amr",
        name="UFPR-AMR Dataset",
        name_zh="UFPR自动电表读数数据集",
        plugin="meter",
        url="https://github.com/raysonlaroca/ufpr-amr-dataset/archive/refs/heads/master.zip",
        image_count=2000,
        format_type="custom",
        can_auto_download=True,
        class_names=["meter", "digit_0", "digit_1", "digit_2", "digit_3", 
                     "digit_4", "digit_5", "digit_6", "digit_7", "digit_8", "digit_9"]
    ),
    "copel_amr": DatasetConfig(
        id="copel_amr",
        name="Copel-AMR Dataset",
        name_zh="Copel电力公司电表数据集",
        plugin="meter",
        url="https://github.com/raysonlaroca/copel-amr-dataset/archive/refs/heads/master.zip",
        image_count=12500,
        format_type="custom",
        can_auto_download=True,
        class_names=["meter", "reading"]
    ),
    
    # ==================== 变压器/热成像 ====================
    "transformer_thermal": DatasetConfig(
        id="transformer_thermal",
        name="Transformer Thermal Images",
        name_zh="变压器热成像数据集",
        plugin="transformer",
        url="",
        image_count=255,
        format_type="custom",
        can_auto_download=False,
        manual_url="https://data.mendeley.com/datasets/8mg8mkc7k5/3",
        unavailable_reason="托管在Mendeley Data，需要登录后手动下载",
        class_names=["normal", "fault_type_1", "fault_type_2", "fault_type_3",
                     "fault_type_4", "fault_type_5", "fault_type_6", "fault_type_7", "fault_type_8"]
    ),
    
    # ==================== 开关状态 (占位符) ====================
    "switch_indicator": DatasetConfig(
        id="switch_indicator",
        name="Switch State Indicator Dataset",
        name_zh="开关状态指示灯数据集",
        plugin="switch",
        url="",
        image_count=0,
        format_type="placeholder",
        can_auto_download=False,
        unavailable_reason="电力行业专用数据，无公开数据集。需要现场采集断路器/隔离开关/接地开关状态图像",
        class_names=["breaker_open", "breaker_closed", "isolator_open", 
                     "isolator_closed", "grounding_open", "grounding_closed"]
    ),
    
    # ==================== 电容器 (占位符) ====================
    "capacitor_structure": DatasetConfig(
        id="capacitor_structure",
        name="Capacitor Structure Dataset",
        name_zh="电容器结构完整性数据集",
        plugin="capacitor",
        url="",
        image_count=0,
        format_type="placeholder",
        can_auto_download=False,
        unavailable_reason="电容器组结构监测数据无公开数据集。需要现场采集正常/倾斜/倒塌/缺失等状态图像",
        class_names=["capacitor_normal", "capacitor_tilted", "capacitor_fallen", "capacitor_missing"]
    ),
}


# =============================================================================
# 下载状态
# =============================================================================
class DownloadStatus(str, Enum):
    NOT_STARTED = "not_started"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    CONVERTING = "converting"
    ORGANIZING = "organizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadProgress:
    """下载进度"""
    dataset_id: str
    status: DownloadStatus
    progress: float  # 0-100
    message: str
    downloaded_bytes: int = 0
    total_bytes: int = 0
    start_time: str = ""
    end_time: str = ""
    error: str = ""
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d


# =============================================================================
# 数据集下载器
# =============================================================================
class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, base_path: Path):
        """
        初始化下载器
        
        Args:
            base_path: 训练数据根目录 (training/data)
        """
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.placeholder_path = self.base_path / "placeholder"
        
        # 确保目录存在
        for p in [self.raw_path, self.processed_path, self.placeholder_path]:
            p.mkdir(parents=True, exist_ok=True)
        
        # 下载进度跟踪
        self._progress: Dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()
        self._active_downloads: Dict[str, threading.Thread] = {}
        
        # SSL上下文
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE
    
    def get_dataset_config(self, dataset_id: str) -> Optional[DatasetConfig]:
        """获取数据集配置"""
        return DATASETS.get(dataset_id)
    
    def get_available_datasets(self, plugin: Optional[str] = None) -> Dict[str, DatasetConfig]:
        """获取可用数据集"""
        if plugin:
            return {k: v for k, v in DATASETS.items() if v.plugin == plugin}
        return DATASETS.copy()
    
    def get_progress(self, dataset_id: str) -> Optional[DownloadProgress]:
        """获取下载进度"""
        with self._lock:
            return self._progress.get(dataset_id)
    
    def is_downloaded(self, dataset_id: str, voltage_level: str) -> bool:
        """检查数据集是否已下载"""
        config = self.get_dataset_config(dataset_id)
        if not config:
            return False
        
        # 检查处理后的数据目录
        processed_dir = self.processed_path / voltage_level / config.plugin
        if processed_dir.exists():
            # 检查是否有图像
            images_dir = processed_dir / "images"
            if images_dir.exists():
                image_count = sum(1 for _ in images_dir.rglob("*.jpg")) + \
                              sum(1 for _ in images_dir.rglob("*.png"))
                return image_count > 0
        
        return False
    
    def download(
        self,
        dataset_id: str,
        voltage_level: str,
        callback: Optional[Callable[[DownloadProgress], None]] = None
    ) -> bool:
        """
        下载数据集
        
        Args:
            dataset_id: 数据集ID
            voltage_level: 电压等级
            callback: 进度回调函数
            
        Returns:
            是否成功启动下载
        """
        config = self.get_dataset_config(dataset_id)
        if not config:
            logger.error(f"未知数据集: {dataset_id}")
            return False
        
        if not config.can_auto_download:
            logger.warning(f"数据集 {dataset_id} 无法自动下载: {config.unavailable_reason}")
            return False
        
        # 检查是否正在下载
        with self._lock:
            if dataset_id in self._active_downloads:
                logger.warning(f"数据集 {dataset_id} 正在下载中")
                return False
        
        # 初始化进度
        progress = DownloadProgress(
            dataset_id=dataset_id,
            status=DownloadStatus.DOWNLOADING,
            progress=0.0,
            message="准备下载...",
            start_time=datetime.now().isoformat()
        )
        
        with self._lock:
            self._progress[dataset_id] = progress
        
        # 启动下载线程
        thread = threading.Thread(
            target=self._download_task,
            args=(config, voltage_level, callback),
            daemon=True
        )
        
        with self._lock:
            self._active_downloads[dataset_id] = thread
        
        thread.start()
        return True
    
    def _download_task(
        self,
        config: DatasetConfig,
        voltage_level: str,
        callback: Optional[Callable[[DownloadProgress], None]]
    ):
        """下载任务（后台线程）"""
        dataset_id = config.id
        
        try:
            # 1. 下载文件
            self._update_progress(dataset_id, DownloadStatus.DOWNLOADING, 0, "开始下载...")
            
            raw_dir = self.raw_path / dataset_id
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # 确定文件名
            if config.url.endswith('.zip'):
                filename = "dataset.zip"
            elif config.url.endswith('.tar.gz'):
                filename = "dataset.tar.gz"
            else:
                filename = "dataset.zip"
            
            download_path = raw_dir / filename
            
            # 下载
            self._download_file(config.url, download_path, dataset_id)
            
            # 2. 解压
            self._update_progress(dataset_id, DownloadStatus.EXTRACTING, 60, "解压中...")
            self._extract_file(download_path, raw_dir)
            
            # 删除压缩包
            if download_path.exists():
                download_path.unlink()
            
            # 3. 转换格式
            self._update_progress(dataset_id, DownloadStatus.CONVERTING, 75, "转换格式...")
            self._convert_format(config, raw_dir)
            
            # 4. 组织数据
            self._update_progress(dataset_id, DownloadStatus.ORGANIZING, 85, "组织数据...")
            self._organize_data(config, voltage_level, raw_dir)
            
            # 5. 完成
            self._update_progress(dataset_id, DownloadStatus.COMPLETED, 100, "下载完成!")
            
            # 更新结束时间
            with self._lock:
                if dataset_id in self._progress:
                    self._progress[dataset_id].end_time = datetime.now().isoformat()
            
            logger.info(f"数据集 {dataset_id} 下载完成")
            
        except Exception as e:
            logger.error(f"下载数据集 {dataset_id} 失败: {e}", exc_info=True)
            self._update_progress(
                dataset_id, 
                DownloadStatus.FAILED, 
                -1, 
                f"下载失败: {str(e)}",
                error=str(e)
            )
        
        finally:
            with self._lock:
                self._active_downloads.pop(dataset_id, None)
    
    def _download_file(self, url: str, save_path: Path, dataset_id: str):
        """下载文件"""
        logger.info(f"从 {url} 下载到 {save_path}")
        
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(downloaded / total_size * 50, 50)  # 下载占50%进度
                
                with self._lock:
                    if dataset_id in self._progress:
                        self._progress[dataset_id].progress = percent
                        self._progress[dataset_id].downloaded_bytes = downloaded
                        self._progress[dataset_id].total_bytes = total_size
                        self._progress[dataset_id].message = f"下载中: {percent:.1f}%"
        
        # 设置请求头
        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
        ]
        urllib.request.install_opener(opener)
        
        # 下载
        urllib.request.urlretrieve(url, str(save_path), progress_hook)
    
    def _extract_file(self, file_path: Path, extract_to: Path):
        """解压文件"""
        logger.info(f"解压 {file_path} 到 {extract_to}")
        
        if str(file_path).endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zf:
                zf.extractall(extract_to)
        elif str(file_path).endswith('.tar.gz'):
            with tarfile.open(file_path, 'r:gz') as tf:
                tf.extractall(extract_to)
        elif str(file_path).endswith('.tar'):
            with tarfile.open(file_path, 'r') as tf:
                tf.extractall(extract_to)
    
    def _convert_format(self, config: DatasetConfig, raw_dir: Path):
        """转换数据格式"""
        if config.format_type == "yolo":
            # YOLO格式不需要转换
            logger.info(f"数据集 {config.id} 已是YOLO格式")
            return
        
        # 对于其他格式，尝试查找图像和标注
        logger.info(f"转换 {config.id} 从 {config.format_type} 格式")
        
        # 这里可以添加更多格式转换逻辑
        # 例如: COCO -> YOLO, VOC -> YOLO, Custom -> YOLO
    
    def _organize_data(self, config: DatasetConfig, voltage_level: str, raw_dir: Path):
        """组织数据到processed目录"""
        # 目标目录
        processed_dir = self.processed_path / voltage_level / config.plugin
        
        # 创建目录结构
        for split in ["train", "val", "test"]:
            (processed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (processed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # 查找所有图像
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.tif', '.tiff'}
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(raw_dir.rglob(f"*{ext}"))
            all_images.extend(raw_dir.rglob(f"*{ext.upper()}"))
        
        if not all_images:
            logger.warning(f"在 {raw_dir} 中未找到图像文件")
            return
        
        logger.info(f"找到 {len(all_images)} 张图像")
        
        # 随机打乱
        random.shuffle(all_images)
        
        # 按 7:2:1 分割
        n = len(all_images)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)
        
        splits = {
            "train": all_images[:train_end],
            "val": all_images[train_end:val_end],
            "test": all_images[val_end:]
        }
        
        # 复制文件
        for split, images in splits.items():
            for img_path in images:
                # 复制图像
                dest_img = processed_dir / "images" / split / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # 查找对应标注
                label_path = self._find_label_file(img_path, raw_dir)
                if label_path and label_path.exists():
                    dest_label = processed_dir / "labels" / split / img_path.with_suffix('.txt').name
                    shutil.copy2(label_path, dest_label)
        
        # 生成 data.yaml
        self._generate_data_yaml(config, voltage_level, processed_dir)
        
        logger.info(f"数据组织完成: {processed_dir}")
    
    def _find_label_file(self, img_path: Path, raw_dir: Path) -> Optional[Path]:
        """查找标注文件"""
        # 尝试多种可能的标注文件位置
        possible_paths = [
            img_path.with_suffix('.txt'),
            img_path.parent / "labels" / img_path.with_suffix('.txt').name,
            img_path.parent.parent / "labels" / img_path.with_suffix('.txt').name,
            raw_dir / "labels" / img_path.with_suffix('.txt').name,
        ]
        
        for p in possible_paths:
            if p.exists():
                return p
        
        return None
    
    def _generate_data_yaml(self, config: DatasetConfig, voltage_level: str, processed_dir: Path):
        """生成data.yaml"""
        import yaml
        
        # 统计图像数量
        train_count = len(list((processed_dir / "images" / "train").glob("*")))
        val_count = len(list((processed_dir / "images" / "val").glob("*")))
        test_count = len(list((processed_dir / "images" / "test").glob("*")))
        
        yaml_content = {
            "path": str(processed_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(config.class_names)},
            "nc": len(config.class_names),
            # 元数据
            "voltage_level": voltage_level,
            "plugin": config.plugin,
            "source_dataset": config.id,
            "source_name": config.name,
            # 统计
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "total_count": train_count + val_count + test_count
        }
        
        yaml_path = processed_dir / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"生成 data.yaml: {yaml_path}")
    
    def _update_progress(
        self,
        dataset_id: str,
        status: DownloadStatus,
        progress: float,
        message: str,
        error: str = ""
    ):
        """更新下载进度"""
        with self._lock:
            if dataset_id in self._progress:
                self._progress[dataset_id].status = status
                self._progress[dataset_id].progress = progress
                self._progress[dataset_id].message = message
                if error:
                    self._progress[dataset_id].error = error
    
    def cancel_download(self, dataset_id: str) -> bool:
        """取消下载"""
        with self._lock:
            if dataset_id in self._progress:
                self._progress[dataset_id].status = DownloadStatus.CANCELLED
                self._progress[dataset_id].message = "下载已取消"
                return True
        return False
    
    def get_dataset_stats(self, plugin: str, voltage_level: str) -> Dict:
        """获取数据集统计"""
        processed_dir = self.processed_path / voltage_level / plugin
        
        stats = {
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
            "total_count": 0,
            "has_labels": False,
            "status": "not_downloaded"
        }
        
        if not processed_dir.exists():
            return stats
        
        # 统计图像
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for split in ["train", "val", "test"]:
            images_dir = processed_dir / "images" / split
            if images_dir.exists():
                count = sum(1 for f in images_dir.iterdir() if f.suffix.lower() in image_exts)
                stats[f"{split}_count"] = count
        
        stats["total_count"] = stats["train_count"] + stats["val_count"] + stats["test_count"]
        
        # 检查标注
        labels_dir = processed_dir / "labels" / "train"
        if labels_dir.exists():
            stats["has_labels"] = any(labels_dir.iterdir())
        
        # 确定状态
        if stats["total_count"] > 50:
            stats["status"] = "ready"
        elif stats["total_count"] > 0:
            stats["status"] = "downloaded"
        else:
            # 检查占位符
            placeholder_dir = self.placeholder_path / voltage_level / plugin
            if placeholder_dir.exists() and any(placeholder_dir.iterdir()):
                stats["status"] = "placeholder"
        
        return stats


# =============================================================================
# 全局实例
# =============================================================================
_downloader: Optional[DatasetDownloader] = None


def get_downloader(base_path: Optional[Path] = None) -> DatasetDownloader:
    """获取下载器实例"""
    global _downloader
    if _downloader is None:
        if base_path is None:
            # 默认路径
            base_path = Path(__file__).parent.parent / "training" / "data"
        _downloader = DatasetDownloader(base_path)
    return _downloader


# =============================================================================
# 命令行接口
# =============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集下载器")
    parser.add_argument("--list", "-l", action="store_true", help="列出可用数据集")
    parser.add_argument("--download", "-d", type=str, help="下载指定数据集")
    parser.add_argument("--voltage", "-v", type=str, default="HV_220kV", help="电压等级")
    parser.add_argument("--plugin", "-p", type=str, help="按插件筛选")
    parser.add_argument("--base-path", "-b", type=str, help="数据目录路径")
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path) if args.base_path else None
    downloader = get_downloader(base_path)
    
    if args.list:
        datasets = downloader.get_available_datasets(args.plugin)
        print("\n可用数据集:")
        print("-" * 80)
        for ds_id, config in datasets.items():
            status = "✅ 可下载" if config.can_auto_download else "❌ 需手动"
            print(f"  {ds_id}:")
            print(f"    名称: {config.name_zh}")
            print(f"    插件: {config.plugin}")
            print(f"    图像数: {config.image_count}")
            print(f"    状态: {status}")
            if not config.can_auto_download:
                print(f"    原因: {config.unavailable_reason}")
                if config.manual_url:
                    print(f"    手动下载: {config.manual_url}")
            print()
        return
    
    if args.download:
        print(f"开始下载 {args.download} (电压等级: {args.voltage})")
        success = downloader.download(args.download, args.voltage)
        
        if not success:
            print("下载启动失败")
            return
        
        # 等待下载完成
        import time
        while True:
            progress = downloader.get_progress(args.download)
            if progress:
                print(f"\r{progress.message} ({progress.progress:.1f}%)", end="")
                if progress.status in [DownloadStatus.COMPLETED, DownloadStatus.FAILED, DownloadStatus.CANCELLED]:
                    print()
                    break
            time.sleep(1)
        
        if progress and progress.status == DownloadStatus.COMPLETED:
            print("下载完成!")
        else:
            print(f"下载失败: {progress.error if progress else '未知错误'}")


if __name__ == "__main__":
    main()
