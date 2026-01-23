"""
传感器统一接口
================

提供统一的传感器数据采集接口，支持:
- 气体传感器 (SF6, H2, CO, O2等)
- 温湿度传感器
- 红外热成像相机
- 2D激光雷达
- 可见光相机
- 声学传感器

作者: 破夜绘明团队
版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """传感器类型"""
    GAS_SF6 = "gas_sf6"              # SF6气体传感器
    GAS_H2 = "gas_h2"                # 氢气传感器
    GAS_CO = "gas_co"                # 一氧化碳传感器
    GAS_O2 = "gas_o2"                # 氧气传感器
    GAS_MULTI = "gas_multi"          # 多气体传感器
    TEMPERATURE = "temperature"       # 温度传感器
    HUMIDITY = "humidity"            # 湿度传感器
    PRESSURE = "pressure"            # 压力传感器
    THERMAL_CAMERA = "thermal_camera"     # 红外热成像
    VISIBLE_CAMERA = "visible_camera"     # 可见光相机
    LIDAR_2D = "lidar_2d"            # 2D激光雷达
    LIDAR_3D = "lidar_3d"            # 3D激光雷达
    ACOUSTIC = "acoustic"            # 声学传感器
    VIBRATION = "vibration"          # 振动传感器
    SMOKE = "smoke"                  # 烟雾传感器
    FLAME = "flame"                  # 火焰传感器


class ConnectionStatus(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    SIMULATED = "simulated"


@dataclass
class SensorConfig:
    """传感器配置"""
    sensor_id: str                   # 传感器唯一ID
    sensor_type: SensorType          # 传感器类型
    name: str = ""                   # 显示名称

    # 连接配置
    protocol: str = "modbus"         # modbus, opcua, mqtt, http, serial
    host: str = "127.0.0.1"
    port: int = 502
    slave_id: int = 1                # Modbus从站地址

    # 串口配置 (如果使用串口)
    serial_port: str = ""
    baudrate: int = 9600

    # 采集配置
    poll_interval_ms: int = 1000     # 轮询间隔(毫秒)
    timeout_ms: int = 5000           # 超时时间(毫秒)
    retry_count: int = 3             # 重试次数

    # 数据配置
    register_address: int = 0        # 寄存器起始地址
    register_count: int = 1          # 寄存器数量
    data_type: str = "float32"       # 数据类型: int16, int32, float32, float64
    scale_factor: float = 1.0        # 缩放因子
    offset: float = 0.0              # 偏移量
    unit: str = ""                   # 单位

    # 校验配置
    min_value: float = -float('inf')
    max_value: float = float('inf')

    # 模拟配置
    simulate_if_unavailable: bool = True
    simulate_base_value: float = 0.0
    simulate_noise_std: float = 0.0

    # 额外配置
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorData:
    """传感器数据"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float                  # Unix时间戳
    value: Union[float, np.ndarray, Dict[str, float]]  # 数值/数组/字典
    unit: str = ""
    quality: float = 1.0              # 数据质量 0-1
    is_simulated: bool = False
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
            "quality": self.quality,
            "is_simulated": self.is_simulated,
            "metadata": self.metadata,
        }


class SensorInterface(ABC):
    """
    传感器接口基类

    所有传感器适配器都应继承此类并实现抽象方法
    """

    def __init__(self, config: SensorConfig):
        self.config = config
        self._status = ConnectionStatus.DISCONNECTED
        self._last_error: str = ""
        self._last_read_time: float = 0
        self._read_count: int = 0
        self._error_count: int = 0

        # 数据缓冲
        self._data_buffer: queue.Queue = queue.Queue(maxlen=1000)

        # 回调
        self._data_callbacks: List[Callable[[SensorData], None]] = []
        self._error_callbacks: List[Callable[[str], None]] = []

        # 采集线程
        self._polling_thread: Optional[threading.Thread] = None
        self._polling_active = False

        logger.info(f"传感器接口初始化: {config.sensor_id} ({config.sensor_type.value})")

    @property
    def sensor_id(self) -> str:
        return self.config.sensor_id

    @property
    def sensor_type(self) -> SensorType:
        return self.config.sensor_type

    @property
    def status(self) -> ConnectionStatus:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._status in (ConnectionStatus.CONNECTED, ConnectionStatus.SIMULATED)

    @property
    def last_error(self) -> str:
        return self._last_error

    # ==========================================================================
    # 抽象方法 - 子类必须实现
    # ==========================================================================

    @abstractmethod
    def _connect_impl(self) -> bool:
        """实际连接实现"""
        pass

    @abstractmethod
    def _disconnect_impl(self) -> None:
        """实际断开实现"""
        pass

    @abstractmethod
    def _read_impl(self) -> Optional[SensorData]:
        """实际读取实现"""
        pass

    # ==========================================================================
    # 公共方法
    # ==========================================================================

    def connect(self) -> bool:
        """
        连接传感器

        Returns:
            是否成功连接
        """
        self._status = ConnectionStatus.CONNECTING

        try:
            success = self._connect_impl()

            if success:
                self._status = ConnectionStatus.CONNECTED
                self._error_count = 0
                logger.info(f"传感器连接成功: {self.sensor_id}")
                return True
            else:
                raise Exception("连接失败")

        except Exception as e:
            self._last_error = str(e)
            self._error_count += 1

            # 尝试模拟模式
            if self.config.simulate_if_unavailable:
                self._status = ConnectionStatus.SIMULATED
                logger.warning(f"传感器 {self.sensor_id} 连接失败，使用模拟模式: {e}")
                return True
            else:
                self._status = ConnectionStatus.ERROR
                logger.error(f"传感器连接失败: {self.sensor_id} - {e}")
                return False

    def disconnect(self) -> None:
        """断开连接"""
        self.stop_polling()

        try:
            self._disconnect_impl()
        except Exception as e:
            logger.warning(f"断开连接时出错: {e}")

        self._status = ConnectionStatus.DISCONNECTED
        logger.info(f"传感器已断开: {self.sensor_id}")

    def read(self) -> Optional[SensorData]:
        """
        读取传感器数据

        Returns:
            传感器数据，失败返回None
        """
        if not self.is_connected:
            logger.warning(f"传感器未连接: {self.sensor_id}")
            return None

        try:
            # 模拟模式
            if self._status == ConnectionStatus.SIMULATED:
                data = self._simulate_read()
            else:
                data = self._read_impl()

            if data is not None:
                # 验证数据
                data = self._validate_data(data)

                # 更新统计
                self._last_read_time = time.time()
                self._read_count += 1

                # 触发回调
                self._notify_data(data)

                # 加入缓冲
                try:
                    self._data_buffer.put_nowait(data)
                except queue.Full:
                    self._data_buffer.get()
                    self._data_buffer.put_nowait(data)

            return data

        except Exception as e:
            self._last_error = str(e)
            self._error_count += 1
            logger.error(f"读取传感器失败: {self.sensor_id} - {e}")
            self._notify_error(str(e))
            return None

    def start_polling(self) -> None:
        """启动轮询采集"""
        if self._polling_active:
            return

        self._polling_active = True
        self._polling_thread = threading.Thread(
            target=self._polling_loop,
            daemon=True,
            name=f"sensor_poll_{self.sensor_id}"
        )
        self._polling_thread.start()
        logger.info(f"开始轮询采集: {self.sensor_id}")

    def stop_polling(self) -> None:
        """停止轮询采集"""
        self._polling_active = False
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=2.0)
        self._polling_thread = None
        logger.info(f"停止轮询采集: {self.sensor_id}")

    def add_data_callback(self, callback: Callable[[SensorData], None]) -> None:
        """添加数据回调"""
        self._data_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str], None]) -> None:
        """添加错误回调"""
        self._error_callbacks.append(callback)

    def get_recent_data(self, count: int = 100) -> List[SensorData]:
        """获取最近的数据"""
        data_list = []
        try:
            while len(data_list) < count and not self._data_buffer.empty():
                data_list.append(self._data_buffer.get_nowait())
        except queue.Empty:
            pass
        return data_list

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "status": self._status.value,
            "read_count": self._read_count,
            "error_count": self._error_count,
            "last_read_time": self._last_read_time,
            "last_error": self._last_error,
            "buffer_size": self._data_buffer.qsize(),
        }

    # ==========================================================================
    # 内部方法
    # ==========================================================================

    def _polling_loop(self) -> None:
        """轮询循环"""
        interval_sec = self.config.poll_interval_ms / 1000.0

        while self._polling_active:
            try:
                self.read()
            except Exception as e:
                logger.error(f"轮询异常: {self.sensor_id} - {e}")

            time.sleep(interval_sec)

    def _simulate_read(self) -> SensorData:
        """模拟读取"""
        base = self.config.simulate_base_value
        noise = np.random.normal(0, self.config.simulate_noise_std)
        value = base + noise

        return SensorData(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=time.time(),
            value=float(value),
            unit=self.config.unit,
            quality=0.8,  # 模拟数据质量稍低
            is_simulated=True,
            metadata={"simulated": True}
        )

    def _validate_data(self, data: SensorData) -> SensorData:
        """验证数据"""
        if isinstance(data.value, (int, float)):
            # 范围检查
            if data.value < self.config.min_value or data.value > self.config.max_value:
                data.quality *= 0.5
                data.metadata["out_of_range"] = True

            # 应用缩放和偏移
            data.value = data.value * self.config.scale_factor + self.config.offset

        return data

    def _notify_data(self, data: SensorData) -> None:
        """通知数据回调"""
        for callback in self._data_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"数据回调异常: {e}")

    def _notify_error(self, error: str) -> None:
        """通知错误回调"""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"错误回调异常: {e}")


# =============================================================================
# 气体传感器接口
# =============================================================================
class GasSensorInterface(SensorInterface):
    """
    气体传感器接口

    支持SF6、H2、CO、O2等气体传感器
    常见协议: Modbus RTU/TCP
    """

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self._client = None

    def _connect_impl(self) -> bool:
        """连接气体传感器"""
        protocol = self.config.protocol.lower()

        if protocol == "modbus":
            return self._connect_modbus()
        elif protocol == "serial":
            return self._connect_serial()
        else:
            raise ValueError(f"不支持的协议: {protocol}")

    def _connect_modbus(self) -> bool:
        """Modbus连接"""
        try:
            from pymodbus.client import ModbusTcpClient

            self._client = ModbusTcpClient(
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.timeout_ms / 1000
            )
            return self._client.connect()

        except ImportError:
            logger.warning("pymodbus未安装，无法使用Modbus连接")
            return False
        except Exception as e:
            logger.error(f"Modbus连接失败: {e}")
            return False

    def _connect_serial(self) -> bool:
        """串口连接"""
        try:
            import serial

            self._client = serial.Serial(
                port=self.config.serial_port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout_ms / 1000
            )
            return self._client.is_open

        except ImportError:
            logger.warning("pyserial未安装，无法使用串口连接")
            return False
        except Exception as e:
            logger.error(f"串口连接失败: {e}")
            return False

    def _disconnect_impl(self) -> None:
        """断开连接"""
        if self._client:
            try:
                self._client.close()
            except:
                pass
            self._client = None

    def _read_impl(self) -> Optional[SensorData]:
        """读取气体浓度"""
        if self._client is None:
            return None

        protocol = self.config.protocol.lower()

        if protocol == "modbus":
            return self._read_modbus()
        elif protocol == "serial":
            return self._read_serial()
        else:
            return None

    def _read_modbus(self) -> Optional[SensorData]:
        """Modbus读取"""
        try:
            result = self._client.read_holding_registers(
                address=self.config.register_address,
                count=self.config.register_count,
                slave=self.config.slave_id
            )

            if result.isError():
                raise Exception(f"Modbus读取错误: {result}")

            # 解析数据
            value = self._parse_registers(result.registers)

            return SensorData(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                value=value,
                unit=self.config.unit,
                quality=1.0,
                is_simulated=False
            )

        except Exception as e:
            raise Exception(f"Modbus读取失败: {e}")

    def _read_serial(self) -> Optional[SensorData]:
        """串口读取"""
        try:
            # 发送读取命令 (根据具体协议调整)
            command = self.config.extra.get("read_command", b"\x01\x03\x00\x00\x00\x02")
            self._client.write(command)

            # 读取响应
            response = self._client.read(self.config.extra.get("response_length", 9))

            if len(response) < 5:
                raise Exception("响应数据不完整")

            # 解析响应 (根据具体协议调整)
            value = self._parse_serial_response(response)

            return SensorData(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                value=value,
                unit=self.config.unit,
                quality=1.0,
                is_simulated=False
            )

        except Exception as e:
            raise Exception(f"串口读取失败: {e}")

    def _parse_registers(self, registers: List[int]) -> float:
        """解析Modbus寄存器"""
        data_type = self.config.data_type.lower()

        if data_type == "int16":
            return float(registers[0])
        elif data_type == "int32":
            return float((registers[0] << 16) | registers[1])
        elif data_type == "float32":
            import struct
            raw = struct.pack('>HH', registers[0], registers[1])
            return struct.unpack('>f', raw)[0]
        else:
            return float(registers[0])

    def _parse_serial_response(self, response: bytes) -> float:
        """解析串口响应"""
        # 默认解析: 第4-5字节为数据 (大端)
        if len(response) >= 5:
            value = (response[3] << 8) | response[4]
            return float(value)
        return 0.0


# =============================================================================
# 温度传感器接口
# =============================================================================
class TemperatureSensorInterface(SensorInterface):
    """温度传感器接口"""

    def __init__(self, config: SensorConfig):
        config.unit = config.unit or "°C"
        config.min_value = config.min_value if config.min_value > -float('inf') else -40.0
        config.max_value = config.max_value if config.max_value < float('inf') else 125.0
        super().__init__(config)
        self._client = None

    def _connect_impl(self) -> bool:
        try:
            from pymodbus.client import ModbusTcpClient
            self._client = ModbusTcpClient(
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.timeout_ms / 1000
            )
            return self._client.connect()
        except ImportError:
            return False
        except Exception:
            return False

    def _disconnect_impl(self) -> None:
        if self._client:
            try:
                self._client.close()
            except:
                pass
            self._client = None

    def _read_impl(self) -> Optional[SensorData]:
        if self._client is None:
            return None

        try:
            result = self._client.read_holding_registers(
                address=self.config.register_address,
                count=2,
                slave=self.config.slave_id
            )

            if result.isError():
                raise Exception(f"读取错误: {result}")

            # 温度通常是有符号数，需要处理
            raw_value = result.registers[0]
            if raw_value > 32767:
                raw_value -= 65536

            temperature = raw_value / 10.0  # 常见的0.1度分辨率

            return SensorData(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=time.time(),
                value=temperature,
                unit=self.config.unit,
                quality=1.0,
                is_simulated=False
            )
        except Exception as e:
            raise Exception(f"温度读取失败: {e}")


# =============================================================================
# 热成像相机接口
# =============================================================================
class ThermalCameraInterface(SensorInterface):
    """
    热成像相机接口

    支持:
    - FLIR相机 (通过SDK)
    - 海康威视红外相机 (通过SDK)
    - 通用RTSP流 (通过OpenCV)
    """

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self._capture = None
        self._sdk = None
        self.frame_width = config.extra.get("width", 640)
        self.frame_height = config.extra.get("height", 480)

    def _connect_impl(self) -> bool:
        """连接热成像相机"""
        camera_type = self.config.extra.get("camera_type", "rtsp")

        if camera_type == "rtsp":
            return self._connect_rtsp()
        elif camera_type == "flir":
            return self._connect_flir()
        elif camera_type == "hikvision":
            return self._connect_hikvision()
        else:
            return self._connect_rtsp()

    def _connect_rtsp(self) -> bool:
        """RTSP连接"""
        try:
            import cv2

            rtsp_url = self.config.extra.get("rtsp_url",
                f"rtsp://{self.config.host}:{self.config.port}/stream")

            self._capture = cv2.VideoCapture(rtsp_url)

            if not self._capture.isOpened():
                raise Exception("无法打开RTSP流")

            return True

        except ImportError:
            logger.warning("OpenCV未安装")
            return False
        except Exception as e:
            logger.error(f"RTSP连接失败: {e}")
            return False

    def _connect_flir(self) -> bool:
        """FLIR SDK连接"""
        try:
            # 尝试导入FLIR SDK
            import PySpin

            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()

            if cam_list.GetSize() == 0:
                raise Exception("未找到FLIR相机")

            self._sdk = cam_list.GetByIndex(0)
            self._sdk.Init()
            self._sdk.BeginAcquisition()

            return True

        except ImportError:
            logger.warning("PySpin SDK未安装")
            return False
        except Exception as e:
            logger.error(f"FLIR连接失败: {e}")
            return False

    def _connect_hikvision(self) -> bool:
        """海康威视SDK连接"""
        try:
            # 使用海康威视SDK
            # 这里需要安装海康威视的Python SDK
            from HCNetSDK import HCNetSDK

            sdk = HCNetSDK()
            sdk.NET_DVR_Init()

            device_info = sdk.NET_DVR_Login_V40(
                self.config.host,
                self.config.port,
                self.config.extra.get("username", "admin"),
                self.config.extra.get("password", "")
            )

            if device_info < 0:
                raise Exception("海康威视登录失败")

            self._sdk = {"sdk": sdk, "handle": device_info}
            return True

        except ImportError:
            logger.warning("海康威视SDK未安装")
            return False
        except Exception as e:
            logger.error(f"海康威视连接失败: {e}")
            return False

    def _disconnect_impl(self) -> None:
        """断开连接"""
        if self._capture:
            self._capture.release()
            self._capture = None

        if self._sdk:
            try:
                if isinstance(self._sdk, dict):
                    self._sdk["sdk"].NET_DVR_Logout(self._sdk["handle"])
                else:
                    self._sdk.EndAcquisition()
                    self._sdk.DeInit()
            except:
                pass
            self._sdk = None

    def _read_impl(self) -> Optional[SensorData]:
        """读取热成像数据"""
        camera_type = self.config.extra.get("camera_type", "rtsp")

        if camera_type == "rtsp":
            return self._read_rtsp()
        elif camera_type == "flir":
            return self._read_flir()
        else:
            return self._read_rtsp()

    def _read_rtsp(self) -> Optional[SensorData]:
        """RTSP读取"""
        if self._capture is None:
            return None

        ret, frame = self._capture.read()
        if not ret:
            raise Exception("读取帧失败")

        # 如果是红外相机，帧可能是单通道温度数据
        # 需要根据相机型号进行转换

        return SensorData(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=time.time(),
            value=frame,  # numpy数组
            unit="thermal_frame",
            quality=1.0,
            is_simulated=False,
            metadata={
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1
            }
        )

    def _read_flir(self) -> Optional[SensorData]:
        """FLIR读取"""
        if self._sdk is None:
            return None

        import PySpin

        image = self._sdk.GetNextImage()

        if image.IsIncomplete():
            raise Exception("图像不完整")

        # 获取温度数据
        frame = image.GetNDArray()

        # FLIR相机通常输出16位温度数据
        # 需要根据相机型号进行温度转换
        temperature_frame = self._convert_to_temperature(frame)

        image.Release()

        return SensorData(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=time.time(),
            value=temperature_frame,
            unit="thermal_frame",
            quality=1.0,
            is_simulated=False,
            metadata={
                "width": frame.shape[1],
                "height": frame.shape[0],
                "min_temp": float(np.min(temperature_frame)),
                "max_temp": float(np.max(temperature_frame)),
                "mean_temp": float(np.mean(temperature_frame))
            }
        )

    def _convert_to_temperature(self, raw_frame: np.ndarray) -> np.ndarray:
        """将原始数据转换为温度 (摄氏度)"""
        # FLIR相机的典型转换公式
        # 实际参数需要根据相机型号和标定获取

        # 常见的线性转换
        # T = raw * scale + offset
        scale = self.config.extra.get("temp_scale", 0.04)
        offset = self.config.extra.get("temp_offset", -273.15)

        temperature = raw_frame.astype(np.float32) * scale + offset

        return temperature

    def _simulate_read(self) -> SensorData:
        """模拟热成像数据"""
        # 生成模拟温度场
        frame = np.random.normal(
            self.config.simulate_base_value,
            self.config.simulate_noise_std,
            (self.frame_height, self.frame_width)
        ).astype(np.float32)

        # 添加一些热点
        num_hotspots = np.random.randint(0, 3)
        for _ in range(num_hotspots):
            cx = np.random.randint(50, self.frame_width - 50)
            cy = np.random.randint(50, self.frame_height - 50)
            radius = np.random.randint(10, 30)
            intensity = np.random.uniform(10, 30)

            y, x = np.ogrid[:self.frame_height, :self.frame_width]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            frame[mask] += intensity

        return SensorData(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=time.time(),
            value=frame,
            unit="thermal_frame",
            quality=0.8,
            is_simulated=True,
            metadata={
                "width": self.frame_width,
                "height": self.frame_height,
                "min_temp": float(np.min(frame)),
                "max_temp": float(np.max(frame)),
                "mean_temp": float(np.mean(frame)),
                "simulated": True
            }
        )


# =============================================================================
# 传感器工厂
# =============================================================================
class SensorFactory:
    """传感器工厂"""

    _registry: Dict[SensorType, type] = {
        SensorType.GAS_SF6: GasSensorInterface,
        SensorType.GAS_H2: GasSensorInterface,
        SensorType.GAS_CO: GasSensorInterface,
        SensorType.GAS_O2: GasSensorInterface,
        SensorType.GAS_MULTI: GasSensorInterface,
        SensorType.TEMPERATURE: TemperatureSensorInterface,
        SensorType.THERMAL_CAMERA: ThermalCameraInterface,
    }

    @classmethod
    def create(cls, config: SensorConfig) -> SensorInterface:
        """创建传感器实例"""
        interface_class = cls._registry.get(config.sensor_type)

        if interface_class is None:
            raise ValueError(f"不支持的传感器类型: {config.sensor_type}")

        return interface_class(config)

    @classmethod
    def register(cls, sensor_type: SensorType, interface_class: type) -> None:
        """注册传感器类型"""
        cls._registry[sensor_type] = interface_class
