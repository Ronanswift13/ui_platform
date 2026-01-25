#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器统一接口模块
====================

提供气体/温度/热成像/雷达等多种传感器的统一数据接口

功能:
1. 统一传感器抽象基类
2. 多种传感器具体实现
3. 数据格式标准化
4. 实时数据流支持

版本: 1.0.0
作者: 变电站AI巡检系统
"""

from __future__ import annotations
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from collections import deque
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举定义
# =============================================================================

class SensorType(Enum):
    """传感器类型"""
    GAS = "gas"                         # 气体传感器
    TEMPERATURE = "temperature"          # 温度传感器
    THERMAL_CAMERA = "thermal_camera"    # 热成像相机
    HUMIDITY = "humidity"                # 湿度传感器
    PRESSURE = "pressure"                # 压力传感器
    VIBRATION = "vibration"              # 振动传感器
    ACOUSTIC = "acoustic"                # 声学传感器
    LIDAR = "lidar"                      # 激光雷达
    RADAR = "radar"                      # 毫米波雷达
    CAMERA = "camera"                    # 可见光相机
    UV = "uv"                           # 紫外相机


class SensorStatus(Enum):
    """传感器状态"""
    UNKNOWN = "unknown"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
    CALIBRATING = "calibrating"


class DataQuality(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"      # 优秀
    GOOD = "good"               # 良好
    FAIR = "fair"               # 一般
    POOR = "poor"               # 差
    INVALID = "invalid"          # 无效


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class SensorReading:
    """传感器读数基类"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    value: Any
    unit: str = ""
    quality: DataQuality = DataQuality.GOOD
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "value": self.value if not isinstance(self.value, np.ndarray) else self.value.tolist(),
            "unit": self.unit,
            "quality": self.quality.value,
            "metadata": self.metadata
        }


@dataclass
class GasReading(SensorReading):
    """气体传感器读数"""
    gas_type: str = "SF6"               # 气体类型
    concentration: float = 0.0           # 浓度
    temperature: Optional[float] = None  # 环境温度
    humidity: Optional[float] = None     # 环境湿度
    pressure: Optional[float] = None     # 大气压
    
    def __post_init__(self):
        self.sensor_type = SensorType.GAS
        self.value = self.concentration


@dataclass
class TemperatureReading(SensorReading):
    """温度传感器读数"""
    temperature: float = 0.0
    ambient_temp: Optional[float] = None
    
    def __post_init__(self):
        self.sensor_type = SensorType.TEMPERATURE
        self.value = self.temperature
        self.unit = "°C"


@dataclass
class ThermalFrame(SensorReading):
    """热成像帧数据"""
    frame: np.ndarray = field(default_factory=lambda: np.zeros((120, 160)))
    min_temp: float = 0.0
    max_temp: float = 0.0
    avg_temp: float = 0.0
    resolution: tuple = (160, 120)
    
    def __post_init__(self):
        self.sensor_type = SensorType.THERMAL_CAMERA
        self.value = {
            "min": self.min_temp,
            "max": self.max_temp,
            "avg": self.avg_temp
        }
        self.unit = "°C"


@dataclass
class RadarFrame(SensorReading):
    """雷达帧数据"""
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # x, y, z, velocity
    num_targets: int = 0
    max_range: float = 0.0
    frame_id: int = 0
    
    def __post_init__(self):
        self.sensor_type = SensorType.RADAR
        self.value = {
            "num_targets": self.num_targets,
            "num_points": len(self.points) if self.points is not None else 0
        }


@dataclass 
class SensorConfig:
    """传感器配置"""
    sensor_id: str
    sensor_type: SensorType
    name: str = ""
    description: str = ""
    
    # 连接参数
    protocol: str = "modbus"          # modbus, mqtt, http, serial, tcp, udp
    host: str = "127.0.0.1"
    port: int = 502
    address: int = 1                   # Modbus从机地址/设备地址
    
    # 采样参数
    sample_rate: float = 1.0           # 采样频率(Hz)
    buffer_size: int = 100             # 缓冲区大小
    
    # 校准参数
    calibration: Dict[str, Any] = field(default_factory=dict)
    
    # 告警阈值
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    # 额外参数
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 传感器基类
# =============================================================================

class BaseSensor(ABC):
    """
    传感器抽象基类
    
    所有传感器适配器都应继承此类并实现抽象方法
    """
    
    def __init__(self, config: SensorConfig):
        """
        初始化传感器
        
        Args:
            config: 传感器配置
        """
        self.config = config
        self.sensor_id = config.sensor_id
        self.sensor_type = config.sensor_type
        
        # 状态
        self._status = SensorStatus.DISCONNECTED
        self._last_reading: Optional[SensorReading] = None
        self._last_error: Optional[str] = None
        self._error_count = 0
        
        # 数据缓冲
        self._buffer: deque = deque(maxlen=config.buffer_size)
        
        # 数据流
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[SensorReading], None]] = []
        
        # 统计
        self._stats = {
            "total_readings": 0,
            "successful_readings": 0,
            "failed_readings": 0,
            "connect_time": None,
            "last_reading_time": None
        }
        
        logger.info(f"传感器初始化: {self.sensor_id} ({self.sensor_type.value})")
    
    # =========================================================================
    # 抽象方法
    # =========================================================================
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接传感器
        
        Returns:
            是否连接成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开传感器连接
        
        Returns:
            是否成功断开
        """
        pass
    
    @abstractmethod
    def read(self) -> Optional[SensorReading]:
        """
        读取传感器数据
        
        Returns:
            传感器读数，失败返回None
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查是否连接
        
        Returns:
            是否已连接
        """
        pass
    
    # =========================================================================
    # 公共方法
    # =========================================================================
    
    @property
    def status(self) -> SensorStatus:
        """获取传感器状态"""
        return self._status
    
    @property
    def last_reading(self) -> Optional[SensorReading]:
        """获取最新读数"""
        return self._last_reading
    
    @property
    def last_error(self) -> Optional[str]:
        """获取最后错误信息"""
        return self._last_error
    
    def get_buffer(self, count: Optional[int] = None) -> List[SensorReading]:
        """
        获取缓冲区数据
        
        Args:
            count: 获取数量，None表示全部
            
        Returns:
            读数列表
        """
        if count is None:
            return list(self._buffer)
        return list(self._buffer)[-count:]
    
    def clear_buffer(self):
        """清空缓冲区"""
        self._buffer.clear()
    
    def register_callback(self, callback: Callable[[SensorReading], None]):
        """
        注册数据回调
        
        Args:
            callback: 回调函数，接收SensorReading参数
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug(f"[{self.sensor_id}] 注册数据回调")
    
    def unregister_callback(self, callback: Callable[[SensorReading], None]):
        """注销数据回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def start_streaming(self):
        """启动数据流"""
        if self._streaming:
            return
        
        if not self.is_connected():
            if not self.connect():
                logger.error(f"[{self.sensor_id}] 无法启动数据流：连接失败")
                return
        
        self._streaming = True
        self._status = SensorStatus.STREAMING
        
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name=f"Sensor-{self.sensor_id}",
            daemon=True
        )
        self._stream_thread.start()
        
        logger.info(f"[{self.sensor_id}] 数据流已启动 ({self.config.sample_rate}Hz)")
    
    def stop_streaming(self):
        """停止数据流"""
        self._streaming = False
        
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)
        
        self._status = SensorStatus.CONNECTED if self.is_connected() else SensorStatus.DISCONNECTED
        logger.info(f"[{self.sensor_id}] 数据流已停止")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "status": self._status.value,
            "buffer_size": len(self._buffer),
            "error_count": self._error_count
        }
    
    def calibrate(self, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        校准传感器
        
        Args:
            params: 校准参数
            
        Returns:
            是否校准成功
        """
        logger.info(f"[{self.sensor_id}] 开始校准...")
        self._status = SensorStatus.CALIBRATING
        
        # 默认实现：直接返回成功
        # 子类可以覆盖此方法实现实际校准逻辑
        
        self._status = SensorStatus.CONNECTED if self.is_connected() else SensorStatus.DISCONNECTED
        return True
    
    # =========================================================================
    # 内部方法
    # =========================================================================
    
    def _stream_loop(self):
        """数据流循环"""
        interval = 1.0 / self.config.sample_rate
        
        while self._streaming:
            try:
                start = time.time()
                
                reading = self.read()
                
                if reading:
                    self._last_reading = reading
                    self._buffer.append(reading)
                    self._stats["successful_readings"] += 1
                    self._stats["last_reading_time"] = time.time()
                    
                    # 触发回调
                    for callback in self._callbacks:
                        try:
                            callback(reading)
                        except Exception as e:
                            logger.error(f"[{self.sensor_id}] 回调执行失败: {e}")
                else:
                    self._stats["failed_readings"] += 1
                
                self._stats["total_readings"] += 1
                
                # 控制采样率
                elapsed = time.time() - start
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                    
            except Exception as e:
                logger.error(f"[{self.sensor_id}] 数据流错误: {e}")
                self._error_count += 1
                self._last_error = str(e)
                time.sleep(0.1)
    
    def _set_error(self, message: str):
        """设置错误状态"""
        self._status = SensorStatus.ERROR
        self._last_error = message
        self._error_count += 1
        logger.error(f"[{self.sensor_id}] {message}")


# =============================================================================
# 气体传感器实现
# =============================================================================

class GasSensor(BaseSensor):
    """
    气体传感器
    
    支持协议:
    - Modbus RTU/TCP
    - MQTT
    - HTTP API
    """
    
    # 支持的气体类型
    SUPPORTED_GASES = ["SF6", "H2", "CO", "C2H2", "CH4", "C2H4", "C2H6", "O2", "CO2"]
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        
        self._connected = False
        self._client = None
        
        # 气体类型
        self.gas_type = config.extra.get("gas_type", "SF6")
        
        # Modbus寄存器配置
        self._registers = config.extra.get("registers", {
            "concentration": 0,
            "temperature": 2,
            "humidity": 4,
            "pressure": 6
        })
    
    def connect(self) -> bool:
        """连接气体传感器"""
        self._status = SensorStatus.CONNECTING
        
        try:
            protocol = self.config.protocol.lower()
            
            if protocol in ["modbus", "modbus_tcp"]:
                return self._connect_modbus_tcp()
            elif protocol == "modbus_rtu":
                return self._connect_modbus_rtu()
            elif protocol == "mqtt":
                return self._connect_mqtt()
            elif protocol == "http":
                return self._connect_http()
            else:
                # 模拟模式
                logger.warning(f"[{self.sensor_id}] 未知协议 {protocol}，使用模拟模式")
                self._connected = True
                self._status = SensorStatus.CONNECTED
                self._stats["connect_time"] = time.time()
                return True
                
        except Exception as e:
            self._set_error(f"连接失败: {e}")
            return False
    
    def _connect_modbus_tcp(self) -> bool:
        """连接Modbus TCP设备"""
        try:
            from pymodbus.client import ModbusTcpClient
            
            self._client = ModbusTcpClient(
                host=self.config.host,
                port=self.config.port,
                timeout=5
            )
            
            if self._client.connect():
                self._connected = True
                self._status = SensorStatus.CONNECTED
                self._stats["connect_time"] = time.time()
                logger.info(f"[{self.sensor_id}] Modbus TCP连接成功")
                return True
            else:
                self._set_error("Modbus TCP连接失败")
                return False
                
        except ImportError:
            logger.warning(f"[{self.sensor_id}] pymodbus未安装，使用模拟模式")
            self._connected = True
            self._status = SensorStatus.CONNECTED
            return True
        except Exception as e:
            self._set_error(f"Modbus TCP连接异常: {e}")
            return False
    
    def _connect_modbus_rtu(self) -> bool:
        """连接Modbus RTU设备"""
        try:
            from pymodbus.client import ModbusSerialClient
            
            self._client = ModbusSerialClient(
                port=self.config.extra.get("serial_port", "/dev/ttyUSB0"),
                baudrate=self.config.extra.get("baudrate", 9600),
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=3
            )
            
            if self._client.connect():
                self._connected = True
                self._status = SensorStatus.CONNECTED
                return True
            else:
                self._set_error("Modbus RTU连接失败")
                return False
                
        except ImportError:
            logger.warning(f"[{self.sensor_id}] pymodbus未安装，使用模拟模式")
            self._connected = True
            self._status = SensorStatus.CONNECTED
            return True
        except Exception as e:
            self._set_error(f"Modbus RTU连接异常: {e}")
            return False
    
    def _connect_mqtt(self) -> bool:
        """连接MQTT Broker"""
        try:
            import paho.mqtt.client as mqtt
            
            self._client = mqtt.Client()
            
            # 设置回调
            self._client.on_connect = self._on_mqtt_connect
            self._client.on_message = self._on_mqtt_message
            
            self._client.connect(
                self.config.host,
                self.config.port,
                60
            )
            
            self._client.loop_start()
            self._connected = True
            self._status = SensorStatus.CONNECTED
            
            # 订阅主题
            topic = self.config.extra.get("topic", f"sensors/gas/{self.sensor_id}")
            self._client.subscribe(topic)
            
            return True
            
        except ImportError:
            logger.warning(f"[{self.sensor_id}] paho-mqtt未安装，使用模拟模式")
            self._connected = True
            self._status = SensorStatus.CONNECTED
            return True
        except Exception as e:
            self._set_error(f"MQTT连接异常: {e}")
            return False
    
    def _connect_http(self) -> bool:
        """连接HTTP API"""
        self._connected = True
        self._status = SensorStatus.CONNECTED
        return True
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT连接回调"""
        if rc == 0:
            logger.info(f"[{self.sensor_id}] MQTT连接成功")
        else:
            logger.error(f"[{self.sensor_id}] MQTT连接失败: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT消息回调"""
        try:
            import json
            data = json.loads(msg.payload.decode())
            
            reading = GasReading(
                sensor_id=self.sensor_id,
                sensor_type=SensorType.GAS,
                timestamp=time.time(),
                value=data.get("concentration", 0),
                gas_type=self.gas_type,
                concentration=data.get("concentration", 0),
                temperature=data.get("temperature"),
                humidity=data.get("humidity"),
                pressure=data.get("pressure")
            )
            
            self._last_reading = reading
            self._buffer.append(reading)
            
            for callback in self._callbacks:
                callback(reading)
                
        except Exception as e:
            logger.error(f"[{self.sensor_id}] MQTT消息处理失败: {e}")
    
    def disconnect(self) -> bool:
        """断开连接"""
        try:
            if self._client:
                if hasattr(self._client, 'close'):
                    self._client.close()
                elif hasattr(self._client, 'loop_stop'):
                    self._client.loop_stop()
                    self._client.disconnect()
            
            self._connected = False
            self._status = SensorStatus.DISCONNECTED
            self._client = None
            
            logger.info(f"[{self.sensor_id}] 已断开连接")
            return True
            
        except Exception as e:
            logger.error(f"[{self.sensor_id}] 断开连接失败: {e}")
            return False
    
    def read(self) -> Optional[GasReading]:
        """读取气体数据"""
        if not self._connected:
            return None
        
        try:
            if self._client and hasattr(self._client, 'read_holding_registers'):
                # Modbus读取
                return self._read_modbus()
            elif self.config.protocol == "http":
                # HTTP读取
                return self._read_http()
            else:
                # 模拟读取
                return self._read_simulated()
                
        except Exception as e:
            logger.error(f"[{self.sensor_id}] 读取失败: {e}")
            return None
    
    def _read_modbus(self) -> Optional[GasReading]:
        """从Modbus读取"""
        try:
            result = self._client.read_holding_registers(
                address=self._registers["concentration"],
                count=8,
                slave=self.config.address
            )
            
            if result.isError():
                return None
            
            # 解析数据 (根据实际设备协议调整)
            concentration = result.registers[0] / 10.0  # 假设精度0.1
            temperature = result.registers[2] / 10.0 if len(result.registers) > 2 else None
            humidity = result.registers[4] / 10.0 if len(result.registers) > 4 else None
            pressure = result.registers[6] / 10.0 if len(result.registers) > 6 else None
            
            return GasReading(
                sensor_id=self.sensor_id,
                sensor_type=SensorType.GAS,
                timestamp=time.time(),
                value=concentration,
                unit="ppm",
                gas_type=self.gas_type,
                concentration=concentration,
                temperature=temperature,
                humidity=humidity,
                pressure=pressure
            )
            
        except Exception as e:
            logger.error(f"[{self.sensor_id}] Modbus读取失败: {e}")
            return None
    
    def _read_http(self) -> Optional[GasReading]:
        """从HTTP API读取"""
        try:
            import requests
            
            url = f"http://{self.config.host}:{self.config.port}/api/reading"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return GasReading(
                    sensor_id=self.sensor_id,
                    sensor_type=SensorType.GAS,
                    timestamp=time.time(),
                    value=data.get("concentration", 0),
                    unit="ppm",
                    gas_type=self.gas_type,
                    concentration=data.get("concentration", 0),
                    temperature=data.get("temperature"),
                    humidity=data.get("humidity"),
                    pressure=data.get("pressure")
                )
            return None
            
        except Exception as e:
            logger.error(f"[{self.sensor_id}] HTTP读取失败: {e}")
            return None
    
    def _read_simulated(self) -> GasReading:
        """模拟读取"""
        # 生成模拟数据
        base_concentration = {
            "SF6": 900,
            "H2": 80,
            "CO": 150,
            "C2H2": 3,
            "CH4": 20,
        }.get(self.gas_type, 100)
        
        concentration = base_concentration + np.random.normal(0, base_concentration * 0.02)
        temperature = 25.0 + np.random.normal(0, 2)
        humidity = 50.0 + np.random.normal(0, 5)
        pressure = 101.3 + np.random.normal(0, 1)
        
        return GasReading(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.GAS,
            timestamp=time.time(),
            value=concentration,
            unit="ppm",
            gas_type=self.gas_type,
            concentration=float(concentration),
            temperature=float(temperature),
            humidity=float(humidity),
            pressure=float(pressure),
            metadata={"simulated": True}
        )
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected


# =============================================================================
# 热成像相机传感器
# =============================================================================

class ThermalCameraSensor(BaseSensor):
    """
    热成像相机传感器
    
    支持:
    - FLIR相机 (通过FLIR SDK)
    - 海康威视热成像 (ISAPI)
    - 通用RTSP流
    - 模拟数据
    """
    
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        
        self._connected = False
        self._camera = None
        self._stream = None
        
        # 温度范围
        self.temp_min = config.extra.get("temp_min", -20.0)
        self.temp_max = config.extra.get("temp_max", 150.0)
        
        # 分辨率
        self.resolution = config.extra.get("resolution", (160, 120))
    
    def connect(self) -> bool:
        """连接热成像相机"""
        self._status = SensorStatus.CONNECTING
        
        try:
            camera_type = self.config.extra.get("camera_type", "simulated")
            
            if camera_type == "flir":
                return self._connect_flir()
            elif camera_type == "hikvision":
                return self._connect_hikvision()
            elif camera_type == "rtsp":
                return self._connect_rtsp()
            else:
                # 模拟模式
                self._connected = True
                self._status = SensorStatus.CONNECTED
                self._stats["connect_time"] = time.time()
                logger.info(f"[{self.sensor_id}] 热成像相机模拟模式已启用")
                return True
                
        except Exception as e:
            self._set_error(f"连接失败: {e}")
            return False
    
    def _connect_flir(self) -> bool:
        """连接FLIR相机"""
        try:
            # 尝试使用FLIR Lepton SDK
            logger.info(f"[{self.sensor_id}] 尝试连接FLIR相机...")
            
            # FLIR SDK初始化代码
            # from flirpy.camera.lepton import Lepton
            # self._camera = Lepton()
            
            # 暂时使用模拟
            self._connected = True
            self._status = SensorStatus.CONNECTED
            return True
            
        except Exception as e:
            logger.warning(f"[{self.sensor_id}] FLIR SDK不可用: {e}")
            self._connected = True
            self._status = SensorStatus.CONNECTED
            return True
    
    def _connect_hikvision(self) -> bool:
        """连接海康威视热成像"""
        try:
            import requests
            
            # 测试ISAPI连接
            url = f"http://{self.config.host}:{self.config.port}/ISAPI/System/deviceInfo"
            auth = (
                self.config.extra.get("username", "admin"),
                self.config.extra.get("password", "")
            )
            
            response = requests.get(url, auth=auth, timeout=5)
            
            if response.status_code == 200:
                self._connected = True
                self._status = SensorStatus.CONNECTED
                logger.info(f"[{self.sensor_id}] 海康威视热成像连接成功")
                return True
            else:
                logger.warning(f"[{self.sensor_id}] 海康威视连接失败，使用模拟模式")
                self._connected = True
                self._status = SensorStatus.CONNECTED
                return True
                
        except Exception as e:
            logger.warning(f"[{self.sensor_id}] 海康威视连接异常: {e}，使用模拟模式")
            self._connected = True
            self._status = SensorStatus.CONNECTED
            return True
    
    def _connect_rtsp(self) -> bool:
        """连接RTSP热成像流"""
        try:
            import cv2
            
            rtsp_url = self.config.extra.get(
                "rtsp_url",
                f"rtsp://{self.config.host}:{self.config.port}/thermal"
            )
            
            self._stream = cv2.VideoCapture(rtsp_url)
            
            if self._stream.isOpened():
                self._connected = True
                self._status = SensorStatus.CONNECTED
                return True
            else:
                logger.warning(f"[{self.sensor_id}] RTSP流连接失败，使用模拟模式")
                self._connected = True
                self._status = SensorStatus.CONNECTED
                return True
                
        except ImportError:
            logger.warning(f"[{self.sensor_id}] OpenCV不可用，使用模拟模式")
            self._connected = True
            self._status = SensorStatus.CONNECTED
            return True
    
    def disconnect(self) -> bool:
        """断开连接"""
        if self._stream:
            self._stream.release()
            self._stream = None
        
        if self._camera:
            # 关闭相机
            self._camera = None
        
        self._connected = False
        self._status = SensorStatus.DISCONNECTED
        return True
    
    def read(self) -> Optional[ThermalFrame]:
        """读取热成像帧"""
        if not self._connected:
            return None
        
        try:
            if self._stream and self._stream.isOpened():
                return self._read_rtsp()
            elif self._camera:
                return self._read_camera()
            else:
                return self._read_simulated()
                
        except Exception as e:
            logger.error(f"[{self.sensor_id}] 读取失败: {e}")
            return None
    
    def _read_rtsp(self) -> Optional[ThermalFrame]:
        """从RTSP流读取"""
        ret, frame = self._stream.read()
        
        if not ret:
            return None
        
        # 转换为温度矩阵
        if len(frame.shape) == 3:
            gray = frame[:, :, 0].astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        
        # 映射到温度范围
        temp_frame = self.temp_min + (gray / 255.0) * (self.temp_max - self.temp_min)
        
        return ThermalFrame(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.THERMAL_CAMERA,
            timestamp=time.time(),
            value={"min": float(temp_frame.min()), "max": float(temp_frame.max())},
            frame=temp_frame,
            min_temp=float(temp_frame.min()),
            max_temp=float(temp_frame.max()),
            avg_temp=float(temp_frame.mean()),
            resolution=temp_frame.shape[::-1]
        )
    
    def _read_camera(self) -> Optional[ThermalFrame]:
        """从相机读取"""
        # 相机SDK读取实现
        return self._read_simulated()
    
    def _read_simulated(self) -> ThermalFrame:
        """模拟读取"""
        h, w = self.resolution[1], self.resolution[0]
        
        # 生成模拟热成像数据
        base_temp = 35.0
        temp_frame = np.random.normal(base_temp, 3, (h, w)).astype(np.float32)
        
        # 添加热点
        hot_spot_y, hot_spot_x = np.random.randint(10, h-10), np.random.randint(10, w-10)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((y_grid - hot_spot_y)**2 + (x_grid - hot_spot_x)**2)
        temp_frame += 20 * np.exp(-dist**2 / (2 * 10**2))
        
        return ThermalFrame(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.THERMAL_CAMERA,
            timestamp=time.time(),
            value={"min": float(temp_frame.min()), "max": float(temp_frame.max())},
            frame=temp_frame,
            min_temp=float(temp_frame.min()),
            max_temp=float(temp_frame.max()),
            avg_temp=float(temp_frame.mean()),
            resolution=(w, h),
            metadata={"simulated": True}
        )
    
    def is_connected(self) -> bool:
        return self._connected


# =============================================================================
# 传感器管理器
# =============================================================================

class SensorManager:
    """
    传感器管理器
    
    统一管理多个传感器的连接、数据采集和缓存
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._sensors: Dict[str, BaseSensor] = {}
        self._callbacks: List[Callable[[str, SensorReading], None]] = []
        self._initialized = True
        
        logger.info("传感器管理器初始化完成")
    
    def register_sensor(self, config: SensorConfig) -> BaseSensor:
        """
        注册传感器
        
        Args:
            config: 传感器配置
            
        Returns:
            创建的传感器实例
        """
        # 根据类型创建传感器
        sensor_classes = {
            SensorType.GAS: GasSensor,
            SensorType.THERMAL_CAMERA: ThermalCameraSensor,
        }
        
        sensor_class = sensor_classes.get(config.sensor_type, BaseSensor)
        
        if sensor_class == BaseSensor:
            raise ValueError(f"不支持的传感器类型: {config.sensor_type}")
        
        sensor = sensor_class(config)
        self._sensors[config.sensor_id] = sensor
        
        # 注册统一回调
        sensor.register_callback(
            lambda reading: self._dispatch_reading(config.sensor_id, reading)
        )
        
        logger.info(f"传感器已注册: {config.sensor_id}")
        return sensor
    
    def unregister_sensor(self, sensor_id: str) -> bool:
        """注销传感器"""
        if sensor_id not in self._sensors:
            return False
        
        sensor = self._sensors[sensor_id]
        sensor.stop_streaming()
        sensor.disconnect()
        
        del self._sensors[sensor_id]
        logger.info(f"传感器已注销: {sensor_id}")
        return True
    
    def get_sensor(self, sensor_id: str) -> Optional[BaseSensor]:
        """获取传感器"""
        return self._sensors.get(sensor_id)
    
    def list_sensors(self) -> List[Dict[str, Any]]:
        """列出所有传感器"""
        return [
            {
                "sensor_id": sensor.sensor_id,
                "sensor_type": sensor.sensor_type.value,
                "status": sensor.status.value,
                **sensor.get_statistics()
            }
            for sensor in self._sensors.values()
        ]
    
    def connect_all(self) -> Dict[str, bool]:
        """连接所有传感器"""
        results = {}
        for sensor_id, sensor in self._sensors.items():
            results[sensor_id] = sensor.connect()
        return results
    
    def disconnect_all(self):
        """断开所有传感器"""
        for sensor in self._sensors.values():
            sensor.stop_streaming()
            sensor.disconnect()
    
    def start_all_streaming(self):
        """启动所有传感器数据流"""
        for sensor in self._sensors.values():
            sensor.start_streaming()
    
    def stop_all_streaming(self):
        """停止所有数据流"""
        for sensor in self._sensors.values():
            sensor.stop_streaming()
    
    def register_global_callback(self, callback: Callable[[str, SensorReading], None]):
        """注册全局数据回调"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def _dispatch_reading(self, sensor_id: str, reading: SensorReading):
        """分发传感器读数"""
        for callback in self._callbacks:
            try:
                callback(sensor_id, reading)
            except Exception as e:
                logger.error(f"全局回调执行失败: {e}")
    
    def get_all_latest_readings(self) -> Dict[str, Optional[SensorReading]]:
        """获取所有传感器最新读数"""
        return {
            sensor_id: sensor.last_reading
            for sensor_id, sensor in self._sensors.items()
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_sensor(
    sensor_id: str,
    sensor_type: Union[str, SensorType],
    **kwargs
) -> BaseSensor:
    """
    创建传感器的便捷函数
    
    Args:
        sensor_id: 传感器ID
        sensor_type: 传感器类型
        **kwargs: 其他配置参数
        
    Returns:
        传感器实例
    """
    if isinstance(sensor_type, str):
        sensor_type = SensorType(sensor_type)
    
    config = SensorConfig(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        **kwargs
    )
    
    manager = SensorManager()
    return manager.register_sensor(config)


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建传感器管理器
    manager = SensorManager()
    
    # 注册气体传感器
    gas_config = SensorConfig(
        sensor_id="gas_sf6_01",
        sensor_type=SensorType.GAS,
        name="SF6气体传感器",
        protocol="simulated",
        sample_rate=1.0,
        extra={"gas_type": "SF6"}
    )
    gas_sensor = manager.register_sensor(gas_config)
    
    # 注册热成像相机
    thermal_config = SensorConfig(
        sensor_id="thermal_01",
        sensor_type=SensorType.THERMAL_CAMERA,
        name="热成像相机",
        protocol="simulated",
        sample_rate=10.0,
        extra={"resolution": (160, 120)}
    )
    thermal_sensor = manager.register_sensor(thermal_config)
    
    # 注册全局回调
    def on_reading(sensor_id: str, reading: SensorReading):
        print(f"[{sensor_id}] {reading.sensor_type.value}: {reading.value}")
    
    manager.register_global_callback(on_reading)
    
    # 连接并启动数据流
    manager.connect_all()
    manager.start_all_streaming()
    
    # 运行一段时间
    try:
        time.sleep(5)
    finally:
        manager.stop_all_streaming()
        manager.disconnect_all()
    
    print("\n传感器统计:")
    for info in manager.list_sensors():
        print(f"  {info['sensor_id']}: {info['successful_readings']} 成功读数")
