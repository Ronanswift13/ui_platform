"""
批量数据导入器
================

支持批量导入历史数据:
- CSV文件导入
- JSON文件导入
- Excel文件导入
- 数据库导入
- API批量拉取

作者: 破夜绘明团队
版本: 1.0.0
"""

from __future__ import annotations
import logging
import os
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class ImportFormat(Enum):
    """导入格式"""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    EXCEL = "excel"
    PARQUET = "parquet"
    DATABASE = "database"
    API = "api"


class ImportStatus(Enum):
    """导入状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ImportConfig:
    """导入配置"""
    format: ImportFormat = ImportFormat.CSV
    source_path: str = ""

    # CSV配置
    delimiter: str = ","
    encoding: str = "utf-8"
    has_header: bool = True
    skip_rows: int = 0

    # 列映射
    column_mapping: Dict[str, str] = field(default_factory=dict)  # {源列: 目标字段}
    timestamp_column: str = "timestamp"
    timestamp_format: str = ""  # 留空则自动检测

    # 数据库配置
    db_connection_string: str = ""
    db_table: str = ""
    db_query: str = ""

    # API配置
    api_url: str = ""
    api_method: str = "GET"
    api_headers: Dict[str, str] = field(default_factory=dict)
    api_params: Dict[str, Any] = field(default_factory=dict)
    api_pagination: bool = False
    api_page_size: int = 1000

    # 处理配置
    batch_size: int = 1000
    validate_data: bool = True
    skip_invalid: bool = True

    # 过滤配置
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    sensor_ids: List[str] = field(default_factory=list)


@dataclass
class ImportResult:
    """导入结果"""
    status: ImportStatus = ImportStatus.PENDING
    total_records: int = 0
    imported_records: int = 0
    skipped_records: int = 0
    failed_records: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    source: str = ""

    @property
    def duration_seconds(self) -> float:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def success_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.imported_records / self.total_records

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "total_records": self.total_records,
            "imported_records": self.imported_records,
            "skipped_records": self.skipped_records,
            "failed_records": self.failed_records,
            "error_count": len(self.errors),
            "errors": self.errors[:10],  # 只返回前10个错误
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "success_rate": self.success_rate,
            "source": self.source,
        }


class BatchImporter:
    """
    批量数据导入器

    支持多种数据源和格式的批量导入
    """

    def __init__(self, config: ImportConfig = None):
        """
        初始化导入器

        Args:
            config: 导入配置
        """
        self.config = config or ImportConfig()
        self._result = ImportResult()
        self._cancelled = False
        self._progress_callback: Optional[Callable[[int, int], None]] = None
        self._data_callback: Optional[Callable[[Dict], None]] = None

    def set_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        """设置进度回调"""
        self._progress_callback = callback

    def set_data_callback(self, callback: Callable[[Dict], None]) -> None:
        """设置数据回调"""
        self._data_callback = callback

    def import_data(self, config: ImportConfig = None) -> ImportResult:
        """
        执行数据导入

        Args:
            config: 导入配置 (可选，覆盖初始配置)

        Returns:
            导入结果
        """
        if config:
            self.config = config

        self._result = ImportResult(
            status=ImportStatus.RUNNING,
            start_time=time.time(),
            source=self.config.source_path or self.config.api_url
        )
        self._cancelled = False

        try:
            # 根据格式选择导入方法
            if self.config.format == ImportFormat.CSV:
                self._import_csv()
            elif self.config.format == ImportFormat.JSON:
                self._import_json()
            elif self.config.format == ImportFormat.JSONL:
                self._import_jsonl()
            elif self.config.format == ImportFormat.EXCEL:
                self._import_excel()
            elif self.config.format == ImportFormat.PARQUET:
                self._import_parquet()
            elif self.config.format == ImportFormat.DATABASE:
                self._import_database()
            elif self.config.format == ImportFormat.API:
                self._import_api()
            else:
                raise ValueError(f"不支持的格式: {self.config.format}")

            self._result.status = ImportStatus.COMPLETED

        except Exception as e:
            self._result.status = ImportStatus.FAILED
            self._result.errors.append({
                "type": "import_error",
                "message": str(e),
                "timestamp": time.time()
            })
            logger.error(f"导入失败: {e}")

        finally:
            self._result.end_time = time.time()

        return self._result

    def cancel(self) -> None:
        """取消导入"""
        self._cancelled = True
        self._result.status = ImportStatus.CANCELLED

    def _import_csv(self) -> None:
        """导入CSV文件"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas未安装: pip install pandas")

        path = Path(self.config.source_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        # 读取CSV
        df = pd.read_csv(
            path,
            delimiter=self.config.delimiter,
            encoding=self.config.encoding,
            skiprows=self.config.skip_rows if not self.config.has_header else self.config.skip_rows,
            header=0 if self.config.has_header else None
        )

        self._result.total_records = len(df)

        # 列映射
        if self.config.column_mapping:
            df = df.rename(columns=self.config.column_mapping)

        # 时间戳处理
        if self.config.timestamp_column in df.columns:
            if self.config.timestamp_format:
                df['_timestamp'] = pd.to_datetime(
                    df[self.config.timestamp_column],
                    format=self.config.timestamp_format
                )
            else:
                df['_timestamp'] = pd.to_datetime(df[self.config.timestamp_column])

        # 时间过滤
        if self.config.start_time and '_timestamp' in df.columns:
            df = df[df['_timestamp'] >= self.config.start_time]
        if self.config.end_time and '_timestamp' in df.columns:
            df = df[df['_timestamp'] <= self.config.end_time]

        # 批量处理
        for i in range(0, len(df), self.config.batch_size):
            if self._cancelled:
                break

            batch = df.iloc[i:i + self.config.batch_size]
            self._process_batch(batch.to_dict('records'))

            if self._progress_callback:
                self._progress_callback(i + len(batch), self._result.total_records)

    def _import_json(self) -> None:
        """导入JSON文件"""
        path = Path(self.config.source_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        with open(path, 'r', encoding=self.config.encoding) as f:
            data = json.load(f)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and 'data' in data:
            records = data['data']
        else:
            records = [data]

        self._result.total_records = len(records)

        # 批量处理
        for i in range(0, len(records), self.config.batch_size):
            if self._cancelled:
                break

            batch = records[i:i + self.config.batch_size]
            self._process_batch(batch)

            if self._progress_callback:
                self._progress_callback(i + len(batch), self._result.total_records)

    def _import_jsonl(self) -> None:
        """导入JSONL文件 (每行一个JSON对象)"""
        path = Path(self.config.source_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        # 计算总行数
        with open(path, 'r', encoding=self.config.encoding) as f:
            self._result.total_records = sum(1 for _ in f)

        # 批量读取处理
        batch = []
        count = 0

        with open(path, 'r', encoding=self.config.encoding) as f:
            for line in f:
                if self._cancelled:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    batch.append(record)
                except json.JSONDecodeError as e:
                    self._result.errors.append({
                        "type": "parse_error",
                        "line": count + 1,
                        "message": str(e)
                    })
                    self._result.failed_records += 1

                count += 1

                if len(batch) >= self.config.batch_size:
                    self._process_batch(batch)
                    batch = []

                    if self._progress_callback:
                        self._progress_callback(count, self._result.total_records)

        # 处理剩余
        if batch:
            self._process_batch(batch)

    def _import_excel(self) -> None:
        """导入Excel文件"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas未安装: pip install pandas openpyxl")

        path = Path(self.config.source_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        # 读取Excel
        df = pd.read_excel(
            path,
            skiprows=self.config.skip_rows,
            header=0 if self.config.has_header else None
        )

        self._result.total_records = len(df)

        # 列映射
        if self.config.column_mapping:
            df = df.rename(columns=self.config.column_mapping)

        # 批量处理
        for i in range(0, len(df), self.config.batch_size):
            if self._cancelled:
                break

            batch = df.iloc[i:i + self.config.batch_size]
            self._process_batch(batch.to_dict('records'))

            if self._progress_callback:
                self._progress_callback(i + len(batch), self._result.total_records)

    def _import_parquet(self) -> None:
        """导入Parquet文件"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas和pyarrow未安装: pip install pandas pyarrow")

        path = Path(self.config.source_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        df = pd.read_parquet(path)
        self._result.total_records = len(df)

        # 列映射
        if self.config.column_mapping:
            df = df.rename(columns=self.config.column_mapping)

        # 批量处理
        for i in range(0, len(df), self.config.batch_size):
            if self._cancelled:
                break

            batch = df.iloc[i:i + self.config.batch_size]
            self._process_batch(batch.to_dict('records'))

            if self._progress_callback:
                self._progress_callback(i + len(batch), self._result.total_records)

    def _import_database(self) -> None:
        """从数据库导入"""
        try:
            import pandas as pd
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError("pandas和sqlalchemy未安装: pip install pandas sqlalchemy")

        if not self.config.db_connection_string:
            raise ValueError("未配置数据库连接字符串")

        engine = create_engine(self.config.db_connection_string)

        # 构建查询
        if self.config.db_query:
            query = self.config.db_query
        elif self.config.db_table:
            query = f"SELECT * FROM {self.config.db_table}"
        else:
            raise ValueError("未配置表名或查询语句")

        # 分批读取
        for chunk in pd.read_sql(query, engine, chunksize=self.config.batch_size):
            if self._cancelled:
                break

            self._result.total_records += len(chunk)

            if self.config.column_mapping:
                chunk = chunk.rename(columns=self.config.column_mapping)

            self._process_batch(chunk.to_dict('records'))

            if self._progress_callback:
                self._progress_callback(
                    self._result.imported_records + self._result.skipped_records + self._result.failed_records,
                    self._result.total_records
                )

    def _import_api(self) -> None:
        """从API导入"""
        try:
            import requests
        except ImportError:
            raise ImportError("requests未安装: pip install requests")

        if not self.config.api_url:
            raise ValueError("未配置API URL")

        session = requests.Session()
        session.headers.update(self.config.api_headers)

        params = dict(self.config.api_params)
        page = 1

        while not self._cancelled:
            # 添加分页参数
            if self.config.api_pagination:
                params['page'] = page
                params['page_size'] = self.config.api_page_size

            # 发送请求
            if self.config.api_method.upper() == "GET":
                response = session.get(self.config.api_url, params=params)
            else:
                response = session.post(self.config.api_url, json=params)

            response.raise_for_status()
            data = response.json()

            # 提取数据
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get('data', data.get('results', data.get('items', [])))
            else:
                records = []

            if not records:
                break

            self._result.total_records += len(records)
            self._process_batch(records)

            if self._progress_callback:
                self._progress_callback(
                    self._result.imported_records,
                    self._result.total_records
                )

            # 检查是否还有更多数据
            if not self.config.api_pagination or len(records) < self.config.api_page_size:
                break

            page += 1

    def _process_batch(self, records: List[Dict]) -> None:
        """处理一批数据"""
        for record in records:
            if self._cancelled:
                break

            try:
                # 应用列映射
                if self.config.column_mapping:
                    mapped_record = {}
                    for src, dst in self.config.column_mapping.items():
                        if src in record:
                            mapped_record[dst] = record[src]
                    # 保留未映射的字段
                    for key, value in record.items():
                        if key not in self.config.column_mapping and key not in mapped_record:
                            mapped_record[key] = value
                    record = mapped_record

                # 时间过滤
                if self.config.timestamp_column in record:
                    ts = record[self.config.timestamp_column]
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    elif isinstance(ts, (int, float)):
                        ts = datetime.fromtimestamp(ts)

                    if self.config.start_time and ts < self.config.start_time:
                        self._result.skipped_records += 1
                        continue
                    if self.config.end_time and ts > self.config.end_time:
                        self._result.skipped_records += 1
                        continue

                # 传感器过滤
                if self.config.sensor_ids:
                    sensor_id = record.get('sensor_id', record.get('device_id', ''))
                    if sensor_id not in self.config.sensor_ids:
                        self._result.skipped_records += 1
                        continue

                # 数据回调
                if self._data_callback:
                    self._data_callback(record)

                self._result.imported_records += 1

            except Exception as e:
                self._result.failed_records += 1
                if len(self._result.errors) < 100:  # 限制错误数量
                    self._result.errors.append({
                        "type": "process_error",
                        "record": str(record)[:200],
                        "message": str(e)
                    })


# =============================================================================
# 数据导出器
# =============================================================================
class DataExporter:
    """数据导出器"""

    @staticmethod
    def export_csv(data: List[Dict], path: str, columns: List[str] = None) -> bool:
        """导出为CSV"""
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            if columns:
                df = df[columns]
            df.to_csv(path, index=False, encoding='utf-8')
            return True
        except Exception as e:
            logger.error(f"CSV导出失败: {e}")
            return False

    @staticmethod
    def export_json(data: List[Dict], path: str, indent: int = 2) -> bool:
        """导出为JSON"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
            return True
        except Exception as e:
            logger.error(f"JSON导出失败: {e}")
            return False

    @staticmethod
    def export_parquet(data: List[Dict], path: str) -> bool:
        """导出为Parquet"""
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_parquet(path, index=False)
            return True
        except Exception as e:
            logger.error(f"Parquet导出失败: {e}")
            return False
