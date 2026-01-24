#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激光雷达适配器 V2.1
==========================================

封装2D激光雷达测距:
- 扫描数据获取 (真实硬件/模拟)
- 点云聚类与目标检测
- 距离测量与跟踪

支持设备:
- RPLIDAR系列 (A1/A2/A3/S1) - 串口
- SICK TiM系列 - TCP/UDP
- Hokuyo UST系列 - TCP
- 自定义UDP协议

作者: G组 | 版本: 2.1.0
更新: 2026-01-24 - 升级为工程级实现
"""

from __future__ import annotations
import logging
import time
import random
import math
import socket
import struct
import threading
import serial
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from .base_adapter import BaseAdapter, AdapterConfig, AdapterStatus

logger = logging.getLogger(__name__)


class LidarProtocol(Enum):
    """激光雷达协议类型"""
    RPLIDAR = "rplidar"          # 思岚RPLIDAR系列
    SICK_TIM = "sick_tim"        # SICK TiM系列
    HOKUYO = "hokuyo"            # Hokuyo UST系列
    CUSTOM_UDP = "custom_udp"    # 自定义UDP协议
    SIMULATION = "simulation"    # 模拟模式


@dataclass
class LidarConfig(AdapterConfig):
    """雷达配置"""
    # 连接参数
    device_ip: str = "192.168.1.100"
    device_port: int = 2368
    serial_port: str = "/dev/ttyUSB0"  # 串口设备
    serial_baudrate: int = 256000      # 串口波特率
    protocol: str = "rplidar"          # rplidar/sick_tim/hokuyo/custom_udp/simulation

    # 扫描参数
    scan_rate_hz: float = 10.0      # 扫描频率
    angle_min_deg: float = -45.0    # 最小角度 (视场限制)
    angle_max_deg: float = 45.0     # 最大角度 (视场限制)
    angle_resolution_deg: float = 0.5  # 角度分辨率

    # 测距参数
    range_min_m: float = 0.1        # 最小测量距离
    range_max_m: float = 10.0       # 最大测量距离

    # 安装参数
    height_m: float = 2.5           # 安装高度
    tilt_deg: float = 20.0          # 向下倾斜角度
    angle_offset_deg: float = 0.0   # 角度偏移

    # 聚类参数
    cluster_distance_threshold: float = 0.3  # 聚类距离阈值 (米)
    min_cluster_points: int = 3     # 最小聚类点数

    # 跟踪参数
    enable_tracking: bool = True     # 启用目标跟踪
    tracking_max_age: int = 5        # 最大丢失帧数
    tracking_iou_threshold: float = 0.3  # 跟踪匹配阈值

    # 数据质量
    filter_invalid_points: bool = True   # 过滤无效点
    median_filter_window: int = 3        # 中值滤波窗口


# ==========================================
# 协议驱动基类和实现
# ==========================================

class LidarDriver(ABC):
    """激光雷达驱动基类"""

    def __init__(self, config: LidarConfig):
        self.config = config
        self._connected = False
        self._running = False
        self._last_scan: Optional['LidarScan'] = None
        self._scan_callback: Optional[Callable] = None

    @abstractmethod
    def connect(self) -> bool:
        """连接设备"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开设备"""
        pass

    @abstractmethod
    def start_scan(self) -> bool:
        """开始扫描"""
        pass

    @abstractmethod
    def stop_scan(self) -> None:
        """停止扫描"""
        pass

    @abstractmethod
    def get_scan(self) -> Optional['LidarScan']:
        """获取扫描数据"""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    def set_scan_callback(self, callback: Callable) -> None:
        """设置扫描回调"""
        self._scan_callback = callback


class RPLidarDriver(LidarDriver):
    """
    RPLIDAR驱动 - 思岚科技激光雷达

    支持型号: A1, A2, A3, S1
    通信方式: 串口 (USB转TTL)
    协议: RPLIDAR Binary Protocol
    """

    # RPLIDAR命令
    CMD_STOP = 0x25
    CMD_RESET = 0x40
    CMD_SCAN = 0x20
    CMD_EXPRESS_SCAN = 0x82
    CMD_GET_INFO = 0x50
    CMD_GET_HEALTH = 0x52

    # 响应类型
    RESP_MEASUREMENT = 0x81
    RESP_DEVICE_INFO = 0x04
    RESP_DEVICE_HEALTH = 0x06

    def __init__(self, config: LidarConfig):
        super().__init__(config)
        self._serial: Optional[serial.Serial] = None
        self._read_thread: Optional[threading.Thread] = None
        self._scan_buffer: deque = deque(maxlen=10)
        self._current_scan_points: List[Tuple[float, float]] = []

    def connect(self) -> bool:
        """连接RPLIDAR"""
        try:
            self._serial = serial.Serial(
                port=self.config.serial_port,
                baudrate=self.config.serial_baudrate,
                timeout=1.0,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )

            # 等待设备就绪
            time.sleep(0.5)

            # 发送停止命令确保干净状态
            self._send_command(self.CMD_STOP)
            time.sleep(0.1)
            self._serial.flushInput()

            # 获取设备信息
            device_info = self._get_device_info()
            if device_info:
                logger.info(f"RPLIDAR连接成功: 型号={device_info.get('model', 'unknown')}, "
                           f"固件={device_info.get('firmware', 'unknown')}")
                self._connected = True
                return True
            else:
                logger.warning("RPLIDAR: 无法获取设备信息")
                self._connected = True  # 仍然尝试使用
                return True

        except serial.SerialException as e:
            logger.error(f"RPLIDAR连接失败: {e}")
            return False
        except Exception as e:
            logger.error(f"RPLIDAR连接异常: {e}")
            return False

    def disconnect(self) -> None:
        """断开RPLIDAR"""
        self.stop_scan()

        if self._serial and self._serial.is_open:
            try:
                self._send_command(self.CMD_STOP)
                time.sleep(0.1)
                self._serial.close()
            except Exception as e:
                logger.warning(f"RPLIDAR断开异常: {e}")

        self._serial = None
        self._connected = False
        logger.info("RPLIDAR已断开")

    def start_scan(self) -> bool:
        """开始扫描"""
        if not self._connected or not self._serial:
            return False

        try:
            # 发送扫描命令
            self._send_command(self.CMD_SCAN)

            # 启动读取线程
            self._running = True
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()

            logger.info("RPLIDAR开始扫描")
            return True

        except Exception as e:
            logger.error(f"RPLIDAR启动扫描失败: {e}")
            return False

    def stop_scan(self) -> None:
        """停止扫描"""
        self._running = False

        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)

        if self._serial and self._serial.is_open:
            try:
                self._send_command(self.CMD_STOP)
            except Exception:
                pass

    def get_scan(self) -> Optional['LidarScan']:
        """获取扫描数据"""
        if self._scan_buffer:
            return self._scan_buffer.popleft()
        return self._last_scan

    def _send_command(self, cmd: int, payload: bytes = b'') -> None:
        """发送命令"""
        if not self._serial:
            return

        # 构建命令包: 0xA5 + cmd + [payload_size] + [payload] + [checksum]
        packet = bytes([0xA5, cmd])
        if payload:
            packet += bytes([len(payload)]) + payload

        self._serial.write(packet)

    def _get_device_info(self) -> Optional[Dict]:
        """获取设备信息"""
        try:
            self._send_command(self.CMD_GET_INFO)
            time.sleep(0.1)

            # 读取响应头
            header = self._serial.read(7)
            if len(header) < 7:
                return None

            if header[0:2] != b'\xA5\x5A':
                return None

            # 读取设备信息
            info_data = self._serial.read(20)
            if len(info_data) < 20:
                return None

            return {
                "model": info_data[0],
                "firmware_minor": info_data[1],
                "firmware_major": info_data[2],
                "firmware": f"{info_data[2]}.{info_data[1]}",
                "hardware": info_data[3],
                "serial": info_data[4:20].hex()
            }

        except Exception as e:
            logger.warning(f"获取RPLIDAR信息失败: {e}")
            return None

    def _read_loop(self) -> None:
        """扫描数据读取循环"""
        scan_start_angle = None
        points = []

        while self._running and self._serial and self._serial.is_open:
            try:
                # 读取测量数据 (5字节)
                data = self._serial.read(5)
                if len(data) < 5:
                    continue

                # 解析测量点
                # Byte 0: quality (6bit) + start_flag (1bit) + ~start_flag (1bit)
                # Byte 1-2: angle_q6 (15bit) + check_bit (1bit)
                # Byte 3-4: distance_q2 (16bit)

                quality = data[0] >> 2
                start_flag = bool(data[0] & 0x01)

                angle_raw = ((data[2] << 8) | data[1]) >> 1
                angle = angle_raw / 64.0  # Q6格式

                distance_raw = (data[4] << 8) | data[3]
                distance = distance_raw / 4.0 / 1000.0  # Q2格式,转换为米

                # 检查新扫描开始
                if start_flag:
                    if points:
                        # 完成一帧扫描
                        scan = self._create_scan(points)
                        self._scan_buffer.append(scan)
                        self._last_scan = scan

                        if self._scan_callback:
                            self._scan_callback(scan)

                    points = []

                # 添加有效点
                if quality > 0 and distance > 0:
                    points.append((angle, distance))

            except Exception as e:
                if self._running:
                    logger.warning(f"RPLIDAR读取异常: {e}")
                time.sleep(0.01)

    def _create_scan(self, points: List[Tuple[float, float]]) -> 'LidarScan':
        """创建扫描数据对象"""
        if not points:
            return LidarScan(
                timestamp=time.time(),
                angles=np.array([]),
                distances=np.array([])
            )

        # 排序并转换
        points.sort(key=lambda p: p[0])
        angles = np.array([p[0] for p in points])
        distances = np.array([p[1] for p in points])

        # 应用视场限制
        mask = (angles >= self.config.angle_min_deg) & (angles <= self.config.angle_max_deg)
        angles = angles[mask]
        distances = distances[mask]

        # 应用距离限制
        distances = np.clip(distances, self.config.range_min_m, self.config.range_max_m)

        return LidarScan(
            timestamp=time.time(),
            angles=angles,
            distances=distances
        )


class SICKTiMDriver(LidarDriver):
    """
    SICK TiM系列驱动

    支持型号: TiM551, TiM561, TiM571
    通信方式: TCP (SOPAS协议)
    """

    def __init__(self, config: LidarConfig):
        super().__init__(config)
        self._socket: Optional[socket.socket] = None
        self._read_thread: Optional[threading.Thread] = None
        self._scan_buffer: deque = deque(maxlen=10)

    def connect(self) -> bool:
        """连接SICK TiM"""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.config.connection_timeout)
            self._socket.connect((self.config.device_ip, self.config.device_port))

            # 发送设备名称查询
            self._send_sopas("sRN DeviceIdent")
            response = self._recv_sopas()

            if response:
                logger.info(f"SICK TiM连接成功: {response}")
                self._connected = True
                return True
            else:
                logger.warning("SICK TiM: 无响应")
                self._connected = True
                return True

        except socket.error as e:
            logger.error(f"SICK TiM连接失败: {e}")
            return False

    def disconnect(self) -> None:
        """断开SICK TiM"""
        self.stop_scan()

        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass

        self._socket = None
        self._connected = False

    def start_scan(self) -> bool:
        """开始扫描"""
        if not self._connected or not self._socket:
            return False

        try:
            # 启动扫描数据输出
            self._send_sopas("sEN LMDscandata 1")
            response = self._recv_sopas()

            if "sEA" in response:
                self._running = True
                self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
                self._read_thread.start()
                logger.info("SICK TiM开始扫描")
                return True

            return False

        except Exception as e:
            logger.error(f"SICK TiM启动扫描失败: {e}")
            return False

    def stop_scan(self) -> None:
        """停止扫描"""
        self._running = False

        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)

        if self._socket:
            try:
                self._send_sopas("sEN LMDscandata 0")
            except Exception:
                pass

    def get_scan(self) -> Optional['LidarScan']:
        """获取扫描数据"""
        if self._scan_buffer:
            return self._scan_buffer.popleft()
        return self._last_scan

    def _send_sopas(self, cmd: str) -> None:
        """发送SOPAS命令"""
        if not self._socket:
            return
        packet = f"\x02{cmd}\x03".encode('latin-1')
        self._socket.send(packet)

    def _recv_sopas(self, timeout: float = 1.0) -> str:
        """接收SOPAS响应"""
        if not self._socket:
            return ""

        try:
            self._socket.settimeout(timeout)
            data = self._socket.recv(4096)
            # 去除STX/ETX
            return data.decode('latin-1').strip('\x02\x03')
        except socket.timeout:
            return ""

    def _read_loop(self) -> None:
        """扫描数据读取循环"""
        buffer = b""

        while self._running and self._socket:
            try:
                self._socket.settimeout(0.1)
                data = self._socket.recv(4096)
                buffer += data

                # 查找完整的扫描数据包
                while b'\x02' in buffer and b'\x03' in buffer:
                    start = buffer.find(b'\x02')
                    end = buffer.find(b'\x03', start)

                    if end > start:
                        packet = buffer[start+1:end].decode('latin-1')
                        buffer = buffer[end+1:]

                        if packet.startswith("sSN LMDscandata"):
                            scan = self._parse_scan_data(packet)
                            if scan:
                                self._scan_buffer.append(scan)
                                self._last_scan = scan
                    else:
                        break

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.warning(f"SICK TiM读取异常: {e}")

    def _parse_scan_data(self, packet: str) -> Optional['LidarScan']:
        """解析扫描数据包"""
        try:
            parts = packet.split()
            if len(parts) < 26:
                return None

            # 解析关键参数
            # parts[16]: 起始角度 (1/10000度)
            # parts[17]: 角度分辨率 (1/10000度)
            # parts[19]: 测量点数
            start_angle = int(parts[16], 16) / 10000.0 - 90  # 转换为-90~90范围
            angle_step = int(parts[17], 16) / 10000.0
            num_points = int(parts[19], 16)

            # 解析距离数据 (mm)
            distances = []
            for i in range(num_points):
                if 20 + i < len(parts):
                    dist_mm = int(parts[20 + i], 16)
                    distances.append(dist_mm / 1000.0)  # 转换为米

            if not distances:
                return None

            angles = np.array([start_angle + i * angle_step for i in range(len(distances))])
            distances = np.array(distances)

            # 应用视场限制
            mask = (angles >= self.config.angle_min_deg) & (angles <= self.config.angle_max_deg)

            return LidarScan(
                timestamp=time.time(),
                angles=angles[mask],
                distances=np.clip(distances[mask], self.config.range_min_m, self.config.range_max_m)
            )

        except Exception as e:
            logger.warning(f"SICK TiM解析失败: {e}")
            return None


class CustomUDPDriver(LidarDriver):
    """
    自定义UDP协议驱动

    支持自定义的UDP扫描数据格式
    数据包格式: [header(4)] [timestamp(8)] [num_points(2)] [points...]
    每个点: [angle_deg(4)] [distance_m(4)] [intensity(2)]
    """

    HEADER_MAGIC = b'LIDR'

    def __init__(self, config: LidarConfig):
        super().__init__(config)
        self._socket: Optional[socket.socket] = None
        self._read_thread: Optional[threading.Thread] = None
        self._scan_buffer: deque = deque(maxlen=10)

    def connect(self) -> bool:
        """连接UDP雷达"""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(('0.0.0.0', self.config.device_port))
            self._socket.settimeout(1.0)

            logger.info(f"UDP雷达监听端口: {self.config.device_port}")
            self._connected = True
            return True

        except socket.error as e:
            logger.error(f"UDP雷达绑定失败: {e}")
            return False

    def disconnect(self) -> None:
        """断开UDP雷达"""
        self.stop_scan()

        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass

        self._socket = None
        self._connected = False

    def start_scan(self) -> bool:
        """开始接收扫描"""
        if not self._connected:
            return False

        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        return True

    def stop_scan(self) -> None:
        """停止接收"""
        self._running = False
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)

    def get_scan(self) -> Optional['LidarScan']:
        """获取扫描数据"""
        if self._scan_buffer:
            return self._scan_buffer.popleft()
        return self._last_scan

    def _read_loop(self) -> None:
        """UDP数据读取循环"""
        while self._running and self._socket:
            try:
                data, addr = self._socket.recvfrom(65535)

                if len(data) < 14:
                    continue

                # 检查header
                if data[:4] != self.HEADER_MAGIC:
                    continue

                # 解析
                timestamp = struct.unpack('<d', data[4:12])[0]
                num_points = struct.unpack('<H', data[12:14])[0]

                points_data = data[14:]
                point_size = 10  # 4 + 4 + 2

                if len(points_data) < num_points * point_size:
                    continue

                angles = []
                distances = []
                intensities = []

                for i in range(num_points):
                    offset = i * point_size
                    angle = struct.unpack('<f', points_data[offset:offset+4])[0]
                    distance = struct.unpack('<f', points_data[offset+4:offset+8])[0]
                    intensity = struct.unpack('<H', points_data[offset+8:offset+10])[0]

                    angles.append(angle)
                    distances.append(distance)
                    intensities.append(intensity)

                scan = LidarScan(
                    timestamp=timestamp,
                    angles=np.array(angles),
                    distances=np.clip(np.array(distances),
                                     self.config.range_min_m,
                                     self.config.range_max_m),
                    intensities=np.array(intensities)
                )

                self._scan_buffer.append(scan)
                self._last_scan = scan

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.warning(f"UDP读取异常: {e}")


class SimulationDriver(LidarDriver):
    """模拟驱动 - 用于测试和演示"""

    def __init__(self, config: LidarConfig):
        super().__init__(config)
        self._background: Optional[np.ndarray] = None
        self._scan_thread: Optional[threading.Thread] = None
        self._scan_buffer: deque = deque(maxlen=10)

    def connect(self) -> bool:
        self._init_background()
        self._connected = True
        logger.info("模拟雷达已连接")
        return True

    def disconnect(self) -> None:
        self.stop_scan()
        self._connected = False

    def start_scan(self) -> bool:
        if not self._connected:
            return False

        self._running = True
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        return True

    def stop_scan(self) -> None:
        self._running = False
        if self._scan_thread and self._scan_thread.is_alive():
            self._scan_thread.join(timeout=2.0)

    def get_scan(self) -> Optional['LidarScan']:
        if self._scan_buffer:
            return self._scan_buffer.popleft()
        return self._last_scan

    def _init_background(self):
        num_points = int((self.config.angle_max_deg - self.config.angle_min_deg)
                        / self.config.angle_resolution_deg)
        self._background = np.full(num_points, self.config.range_max_m - 1.0)

    def _scan_loop(self) -> None:
        scan_interval = 1.0 / self.config.scan_rate_hz

        while self._running:
            scan = self._generate_scan()
            self._scan_buffer.append(scan)
            self._last_scan = scan

            if self._scan_callback:
                self._scan_callback(scan)

            time.sleep(scan_interval)

    def _generate_scan(self) -> 'LidarScan':
        angles = np.arange(
            self.config.angle_min_deg,
            self.config.angle_max_deg,
            self.config.angle_resolution_deg
        )

        distances = self._background.copy() if self._background is not None else np.full(len(angles), 8.0)
        distances = distances[:len(angles)]
        distances += np.random.normal(0, 0.02, len(distances))

        # 模拟1-3个移动目标
        num_targets = random.randint(1, 3)
        for _ in range(num_targets):
            center_angle = random.uniform(-30, 30)
            center_distance = random.uniform(1.5, 4.0)
            width_deg = random.uniform(3, 8)

            for i, angle in enumerate(angles):
                if abs(angle - center_angle) < width_deg / 2:
                    target_dist = center_distance + random.gauss(0, 0.05)
                    distances[i] = min(distances[i], max(0.2, target_dist))

        distances = np.clip(distances, self.config.range_min_m, self.config.range_max_m)

        return LidarScan(
            timestamp=time.time(),
            angles=angles,
            distances=distances
        )


def create_lidar_driver(config: LidarConfig) -> LidarDriver:
    """创建雷达驱动工厂函数"""
    protocol = config.protocol.lower()

    driver_map = {
        "rplidar": RPLidarDriver,
        "sick_tim": SICKTiMDriver,
        "hokuyo": SICKTiMDriver,  # Hokuyo使用类似协议
        "custom_udp": CustomUDPDriver,
        "simulation": SimulationDriver
    }

    driver_class = driver_map.get(protocol, SimulationDriver)
    return driver_class(config)


@dataclass
class LidarScan:
    """雷达扫描数据"""
    timestamp: float
    angles: np.ndarray              # 角度数组 (度)
    distances: np.ndarray           # 距离数组 (米)
    intensities: Optional[np.ndarray] = None  # 强度数组 (可选)
    
    @property
    def num_points(self) -> int:
        return len(self.distances)
    
    def to_cartesian(self) -> Tuple[np.ndarray, np.ndarray]:
        """转换为笛卡尔坐标"""
        angles_rad = np.radians(self.angles)
        x = self.distances * np.sin(angles_rad)
        y = self.distances * np.cos(angles_rad)
        return x, y


@dataclass
class LidarCluster:
    """雷达聚类结果"""
    cluster_id: str
    center_distance: float          # 聚类中心距离 (米)
    center_angle: float             # 聚类中心角度 (度)
    min_distance: float             # 最小距离
    max_distance: float             # 最大距离
    point_count: int                # 点数
    confidence: float               # 置信度
    timestamp: float
    
    # 原始点
    points_angles: Optional[np.ndarray] = None
    points_distances: Optional[np.ndarray] = None


class LidarTracker:
    """
    雷达目标跟踪器

    使用简单的IoU匹配进行多目标跟踪
    """

    def __init__(self, max_age: int = 5, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self._tracks: Dict[str, Dict] = {}
        self._next_id = 0

    def update(self, clusters: List[LidarCluster]) -> List[LidarCluster]:
        """更新跟踪状态"""
        if not clusters:
            # 增加所有轨迹的age
            self._age_tracks()
            return []

        # 计算匹配矩阵
        track_ids = list(self._tracks.keys())
        matched_clusters = []
        unmatched_clusters = list(range(len(clusters)))
        matched_tracks = set()

        for i, cluster in enumerate(clusters):
            best_match = None
            best_score = self.iou_threshold

            for track_id in track_ids:
                if track_id in matched_tracks:
                    continue

                track = self._tracks[track_id]
                score = self._compute_similarity(cluster, track)

                if score > best_score:
                    best_score = score
                    best_match = track_id

            if best_match:
                # 更新已有轨迹
                self._tracks[best_match]['cluster'] = cluster
                self._tracks[best_match]['age'] = 0
                cluster.cluster_id = best_match
                matched_tracks.add(best_match)
                unmatched_clusters.remove(i)

        # 创建新轨迹
        for i in unmatched_clusters:
            track_id = f"T{self._next_id:05d}"
            self._next_id += 1

            self._tracks[track_id] = {
                'cluster': clusters[i],
                'age': 0,
                'history': []
            }
            clusters[i].cluster_id = track_id

        # 老化未匹配轨迹
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self._tracks[track_id]['age'] += 1

        # 移除过期轨迹
        expired = [tid for tid, t in self._tracks.items() if t['age'] > self.max_age]
        for tid in expired:
            del self._tracks[tid]

        return clusters

    def _compute_similarity(self, cluster: LidarCluster, track: Dict) -> float:
        """计算聚类与轨迹的相似度"""
        prev_cluster = track['cluster']

        # 基于角度和距离的相似度
        angle_diff = abs(cluster.center_angle - prev_cluster.center_angle)
        dist_diff = abs(cluster.center_distance - prev_cluster.center_distance)

        # 归一化
        angle_score = max(0, 1 - angle_diff / 30.0)
        dist_score = max(0, 1 - dist_diff / 2.0)

        return (angle_score + dist_score) / 2

    def _age_tracks(self) -> None:
        """增加所有轨迹age"""
        expired = []
        for track_id in self._tracks:
            self._tracks[track_id]['age'] += 1
            if self._tracks[track_id]['age'] > self.max_age:
                expired.append(track_id)

        for tid in expired:
            del self._tracks[tid]


class LidarAdapter(BaseAdapter):
    """
    激光雷达适配器 V2.1

    提供统一的测距接口:
    - get_scan() -> LidarScan
    - get_clusters() -> List[LidarCluster]

    支持多种雷达协议:
    - RPLIDAR (串口)
    - SICK TiM (TCP)
    - 自定义UDP
    - 模拟模式

    内部封装雷达通信、点云处理和目标跟踪
    """

    def __init__(self, config: LidarConfig):
        super().__init__(config)
        self.config: LidarConfig = config

        # 雷达驱动
        self._driver: Optional[LidarDriver] = None

        # 扫描缓冲
        self._scan_buffer: List[LidarScan] = []
        self._buffer_size = 10

        # 背景模型 (用于前景提取)
        self._background: Optional[np.ndarray] = None

        # 聚类ID计数器
        self._cluster_id_counter = 0

        # 目标跟踪器
        self._tracker: Optional[LidarTracker] = None
        if config.enable_tracking:
            self._tracker = LidarTracker(
                max_age=config.tracking_max_age,
                iou_threshold=config.tracking_iou_threshold
            )

        # 滤波器
        self._median_buffer: deque = deque(maxlen=config.median_filter_window)

        protocol_name = config.protocol
        logger.info(f"雷达适配器初始化: 协议={protocol_name}, "
                   f"IP={config.device_ip}, 端口={config.device_port}")

    def connect(self) -> bool:
        """连接雷达"""
        self._stats['connect_attempts'] += 1
        self._status = AdapterStatus.CONNECTING

        try:
            # 创建对应协议的驱动
            self._driver = create_lidar_driver(self.config)

            # 尝试连接
            if self._driver.connect():
                # 启动扫描
                if self._driver.start_scan():
                    self._connected = True
                    self._status = AdapterStatus.CONNECTED

                    # 初始化背景模型
                    self._init_background()

                    if isinstance(self._driver, SimulationDriver):
                        self._simulated = True
                        logger.info(f"雷达适配器: 模拟模式已启用")
                    else:
                        logger.info(f"雷达适配器: 已连接真实设备 ({self.config.protocol})")

                    return True

            # 连接失败,尝试模拟模式
            if self.config.simulate_if_unavailable:
                logger.warning(f"雷达 {self.config.protocol} 连接失败,切换到模拟模式")
                self._driver = SimulationDriver(self.config)
                self._driver.connect()
                self._driver.start_scan()
                self._set_simulated()
                self._init_background()
                return True
            else:
                self._set_error("雷达连接失败,且未启用模拟模式")
                return False

        except Exception as e:
            logger.error(f"雷达连接异常: {e}")
            if self.config.simulate_if_unavailable:
                self._driver = SimulationDriver(self.config)
                self._driver.connect()
                self._driver.start_scan()
                self._set_simulated()
                self._init_background()
                return True
            else:
                self._set_error(f"雷达连接异常: {e}")
                return False

    def disconnect(self):
        """断开雷达"""
        if self._driver:
            self._driver.stop_scan()
            self._driver.disconnect()
            self._driver = None

        self._connected = False
        self._simulated = False
        self._status = AdapterStatus.DISCONNECTED
        self._scan_buffer.clear()
        logger.info("雷达适配器已断开")

    def healthcheck(self) -> bool:
        """健康检查"""
        if self._driver:
            return self._driver.is_connected
        return False

    def _init_background(self):
        """初始化背景模型"""
        num_points = int((self.config.angle_max_deg - self.config.angle_min_deg)
                        / self.config.angle_resolution_deg)
        self._background = np.full(num_points, self.config.range_max_m - 1.0)

    def get_scan(self) -> Optional[LidarScan]:
        """
        获取一帧扫描数据

        Returns:
            LidarScan: 扫描数据 (None表示失败)
        """
        if not self.is_connected or not self._driver:
            logger.warning("雷达未连接")
            return None

        try:
            scan = self._driver.get_scan()

            if scan is None:
                return None

            # 应用滤波
            if self.config.filter_invalid_points:
                scan = self._apply_filter(scan)

            self._stats['successful_reads'] += 1
            self._stats['last_read_time'] = time.time()

            return scan

        except Exception as e:
            self._stats['failed_reads'] += 1
            logger.error(f"扫描读取失败: {e}")
            return None

    def _apply_filter(self, scan: LidarScan) -> LidarScan:
        """应用滤波处理"""
        # 添加到中值滤波缓冲
        self._median_buffer.append(scan.distances.copy())

        if len(self._median_buffer) >= self.config.median_filter_window:
            # 计算中值
            stacked = np.stack(list(self._median_buffer), axis=0)
            filtered = np.median(stacked, axis=0)
            scan.distances = filtered

        return scan
    
    def get_clusters(self, scan: Optional[LidarScan] = None) -> List[LidarCluster]:
        """
        获取聚类结果

        Args:
            scan: 扫描数据 (可选,如果为None则获取新扫描)

        Returns:
            List[LidarCluster]: 聚类结果列表
        """
        if scan is None:
            scan = self.get_scan()

        if scan is None:
            return []

        # 前景提取
        fg_mask = self._extract_foreground(scan)

        if not np.any(fg_mask):
            return []

        # 聚类
        clusters = self._cluster_points(scan, fg_mask)

        # 目标跟踪
        if self._tracker and clusters:
            clusters = self._tracker.update(clusters)

        return clusters

    def get_tracked_targets(self) -> Dict[str, Dict]:
        """获取当前跟踪的所有目标"""
        if self._tracker:
            return {tid: {'cluster': t['cluster'], 'age': t['age']}
                    for tid, t in self._tracker._tracks.items()}
        return {}
    
    def _extract_foreground(self, scan: LidarScan) -> np.ndarray:
        """提取前景点"""
        if self._background is None:
            # 没有背景模型,使用固定阈值
            threshold = self.config.range_max_m - 2.0
            return scan.distances < threshold
        
        # 与背景比较
        diff = self._background[:len(scan.distances)] - scan.distances
        
        # 前景判定: 距离明显小于背景
        fg_mask = diff > 0.5
        
        return fg_mask
    
    def _cluster_points(self, scan: LidarScan, fg_mask: np.ndarray) -> List[LidarCluster]:
        """对前景点聚类"""
        clusters = []
        
        fg_indices = np.where(fg_mask)[0]
        if len(fg_indices) == 0:
            return clusters
        
        # 简单的连续性聚类
        current_cluster_indices = []
        
        for i, idx in enumerate(fg_indices):
            if not current_cluster_indices:
                current_cluster_indices.append(idx)
            else:
                # 检查连续性 (角度和距离)
                prev_idx = current_cluster_indices[-1]
                angle_diff = abs(scan.angles[idx] - scan.angles[prev_idx])
                dist_diff = abs(scan.distances[idx] - scan.distances[prev_idx])
                
                if angle_diff < 3 * self.config.angle_resolution_deg and dist_diff < self.config.cluster_distance_threshold:
                    current_cluster_indices.append(idx)
                else:
                    # 保存当前聚类,开始新聚类
                    if len(current_cluster_indices) >= self.config.min_cluster_points:
                        cluster = self._create_cluster(scan, current_cluster_indices)
                        clusters.append(cluster)
                    current_cluster_indices = [idx]
        
        # 处理最后一个聚类
        if len(current_cluster_indices) >= self.config.min_cluster_points:
            cluster = self._create_cluster(scan, current_cluster_indices)
            clusters.append(cluster)
        
        return clusters
    
    def _create_cluster(self, scan: LidarScan, indices: List[int]) -> LidarCluster:
        """创建聚类对象"""
        self._cluster_id_counter += 1
        
        cluster_angles = scan.angles[indices]
        cluster_distances = scan.distances[indices]
        
        return LidarCluster(
            cluster_id=f"LC{self._cluster_id_counter:05d}",
            center_distance=float(np.mean(cluster_distances)),
            center_angle=float(np.mean(cluster_angles)),
            min_distance=float(np.min(cluster_distances)),
            max_distance=float(np.max(cluster_distances)),
            point_count=len(indices),
            confidence=min(1.0, len(indices) / 10.0),
            timestamp=scan.timestamp,
            points_angles=cluster_angles,
            points_distances=cluster_distances,
        )
    
    def update_background(self, scan: LidarScan):
        """更新背景模型"""
        if self._background is None:
            self._background = scan.distances.copy()
        else:
            # 指数移动平均
            alpha = 0.1
            min_len = min(len(self._background), len(scan.distances))
            self._background[:min_len] = (
                alpha * scan.distances[:min_len] + 
                (1 - alpha) * self._background[:min_len]
            )


__all__ = [
    'LidarProtocol',
    'LidarConfig',
    'LidarScan',
    'LidarCluster',
    'LidarDriver',
    'RPLidarDriver',
    'SICKTiMDriver',
    'CustomUDPDriver',
    'SimulationDriver',
    'create_lidar_driver',
    'LidarTracker',
    'LidarAdapter'
]
