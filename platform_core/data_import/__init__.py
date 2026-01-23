"""
数据导入系统
==============

提供统一的数据导入接口，支持:
- 实时传感器数据采集
- 历史数据批量导入
- 数据验证和预处理
- 多协议设备连接

作者: 破夜绘明团队
版本: 1.0.0
"""

from .sensor_interface import (
    SensorInterface,
    SensorConfig,
    SensorData,
    SensorType,
    ConnectionStatus,
)
from .data_validator import (
    DataValidator,
    ValidationResult,
    ValidationRule,
)
from .batch_importer import (
    BatchImporter,
    ImportResult,
    ImportConfig,
)
from .protocol_adapters import (
    ModbusAdapter,
    OPCUAAdapter,
    MQTTAdapter,
    HTTPAdapter,
)

__all__ = [
    # 传感器接口
    'SensorInterface',
    'SensorConfig',
    'SensorData',
    'SensorType',
    'ConnectionStatus',
    # 数据验证
    'DataValidator',
    'ValidationResult',
    'ValidationRule',
    # 批量导入
    'BatchImporter',
    'ImportResult',
    'ImportConfig',
    # 协议适配器
    'ModbusAdapter',
    'OPCUAAdapter',
    'MQTTAdapter',
    'HTTPAdapter',
]
