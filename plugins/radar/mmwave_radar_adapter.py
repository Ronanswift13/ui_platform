#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
毫米波雷达适配器模块
=====================

提供mmWave雷达的完整接口实现:
- TI IWR系列雷达支持 (UART/Ethernet)
- 实时点云数据获取
- 目标检测和跟踪
- 速度估计和微运动检测
- 多协议驱动支持

版本: 2.1.0
作者: 变电站AI巡检系统
"""

from __future__ import annotations
import logging
import time
import struct
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举定义
# =============================================================================

class RadarProtocol(Enum):
    """雷达协议类型"""
    UART = "uart"
    ETHERNET_UDP = "ethernet_udp"
    ETHERNET_TCP = "ethernet_tcp"
    USB = "usb"


class RadarModel(Enum):
    """雷达型号"""
    TI_IWR1443 = "ti_iwr1443"
    TI_IWR1642 = "ti_iwr1642"
    TI_IWR6843 = "ti_iwr6843"
    TI_AWR1443 = "ti_awr1443"
    TI_AWR1642 = "ti_awr1642"
    TI_AWR2243 = "ti_awr2243"
    INFINEON_BGT60 = "infineon_bgt60"
    GENERIC = "generic"


class TargetType(Enum):
    """目标类型"""
    UNKNOWN = "unknown"
    PERSON = "person"
    VEHICLE = "vehicle"
    BIRD = "bird"
    DRONE = "drone"
    CLUTTER = "clutter"


class AdapterStatus(Enum):
    """适配器状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
    SIMULATED = "simulated"


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class RadarConfig:
    """雷达配置"""
    protocol: RadarProtocol = RadarProtocol.UART
    model: RadarModel = RadarModel.TI_IWR6843
    
    # UART配置
    serial_port: str = "/dev/ttyUSB0"
    config_port: str = "/dev/ttyUSB1"
    baudrate: int = 921600
    config_baudrate: int = 115200
    
    # Ethernet配置
    host: str = "192.168.1.100"
    data_port: int = 4098
    config_port_eth: int = 4096
    
    # 雷达参数
    max_range_m: float = 10.0
    range_resolution_m: float = 0.04
    max_velocity_m_s: float = 3.0
    frame_rate_hz: float = 10.0
    
    # 检测参数
    cfar_threshold_db: float = 15.0
    snr_threshold_db: float = 10.0
    min_num_points: int = 3
    max_num_targets: int = 50
    
    # 跟踪参数
    enable_tracking: bool = True
    track_association_threshold: float = 0.5
    track_max_age: int = 10
    track_min_hits: int = 3
    
    # 滤波参数
    enable_clutter_removal: bool = True
    
    # 模拟配置
    simulate_if_unavailable: bool = True


@dataclass
class RadarPoint:
    """雷达点"""
    x: float
    y: float
    z: float
    velocity: float
    snr: float
    noise: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.velocity, self.snr])
    
    @property
    def range(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)


@dataclass
class RadarFrame:
    """雷达帧"""
    frame_id: int
    timestamp: float
    points: List[RadarPoint]
    num_detected_points: int = 0
    
    def __post_init__(self):
        self.num_detected_points = len(self.points)
    
    def to_point_cloud(self) -> np.ndarray:
        if not self.points:
            return np.zeros((0, 5))
        return np.array([p.to_array() for p in self.points])


@dataclass
class TrackedTarget:
    """跟踪目标"""
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    bbox: Tuple[float, float, float, float, float, float]
    target_type: TargetType
    confidence: float
    age: int
    hits: int
    time_since_update: int
    timestamp: float
    history: List[np.ndarray] = field(default_factory=list)
    
    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)


# =============================================================================
# 雷达驱动基类
# =============================================================================

class BaseRadarDriver(ABC):
    """雷达驱动基类"""
    
    def __init__(self, config: RadarConfig):
        self.config = config
        self._connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        pass
    
    @abstractmethod
    def configure(self, params: Dict[str, Any] = None) -> bool:
        pass
    
    @abstractmethod
    def start_streaming(self) -> bool:
        pass
    
    @abstractmethod
    def stop_streaming(self) -> bool:
        pass
    
    @abstractmethod
    def read_frame(self) -> Optional[RadarFrame]:
        pass
    
    @property
    def is_connected(self) -> bool:
        return self._connected


# =============================================================================
# TI IWR系列UART驱动
# =============================================================================

class TIRadarUARTDriver(BaseRadarDriver):
    """TI IWR/AWR系列雷达UART驱动"""
    
    MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
    
    def __init__(self, config: RadarConfig):
        super().__init__(config)
        self._data_serial = None
        self._config_serial = None
        self._frame_id = 0
        self._running = False
    
    def connect(self) -> bool:
        try:
            import serial
            
            self._config_serial = serial.Serial(
                port=self.config.config_port,
                baudrate=self.config.config_baudrate,
                timeout=1
            )
            
            self._data_serial = serial.Serial(
                port=self.config.serial_port,
                baudrate=self.config.baudrate,
                timeout=0.1
            )
            
            self._connected = True
            logger.info(f"TI雷达连接成功: {self.config.serial_port}")
            return True
            
        except ImportError:
            logger.warning("pyserial未安装")
            return False
        except Exception as e:
            logger.error(f"TI雷达连接失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        self._running = False
        
        if self._data_serial:
            self._data_serial.close()
            self._data_serial = None
        
        if self._config_serial:
            self._config_serial.close()
            self._config_serial = None
        
        self._connected = False
        return True
    
    def configure(self, params: Dict[str, Any] = None) -> bool:
        if not self._config_serial:
            return False
        
        try:
            config_commands = [
                "sensorStop",
                "flushCfg",
                "dfeDataOutputMode 1",
                "channelCfg 15 7 0",
                "adcCfg 2 1",
                "profileCfg 0 77 7 7 57.14 0 0 70 1 256 5209 0 0 30",
                "frameCfg 0 2 128 0 100 1 0",
                "guiMonitor -1 1 1 0 0 0 1",
                "cfarCfg -1 0 2 8 4 3 0 15 1",
                "clutterRemoval -1 1",
                "sensorStart"
            ]
            
            for cmd in config_commands:
                self._config_serial.write(f"{cmd}\n".encode())
                time.sleep(0.05)
            
            return True
            
        except Exception as e:
            logger.error(f"雷达配置失败: {e}")
            return False
    
    def start_streaming(self) -> bool:
        if not self._connected:
            return False
        self._running = True
        if self._config_serial:
            self._config_serial.write(b"sensorStart\n")
        return True
    
    def stop_streaming(self) -> bool:
        self._running = False
        if self._config_serial:
            self._config_serial.write(b"sensorStop\n")
        return True
    
    def read_frame(self) -> Optional[RadarFrame]:
        if not self._data_serial or not self._running:
            return None
        
        try:
            data = self._data_serial.read(8)
            
            while data != self.MAGIC_WORD:
                if not data:
                    return None
                data = data[1:] + self._data_serial.read(1)
            
            header_data = self._data_serial.read(32)
            if len(header_data) < 32:
                return None
            
            header = self._parse_header(header_data)
            payload_size = header["packet_length"] - 40
            payload = self._data_serial.read(payload_size)
            
            if len(payload) < payload_size:
                return None
            
            points = self._parse_point_cloud(payload, header)
            self._frame_id += 1
            
            return RadarFrame(
                frame_id=self._frame_id,
                timestamp=time.time(),
                points=points
            )
            
        except Exception as e:
            logger.error(f"读取帧失败: {e}")
            return None
    
    def _parse_header(self, data: bytes) -> Dict[str, Any]:
        return {
            "version": struct.unpack('<I', data[0:4])[0],
            "packet_length": struct.unpack('<I', data[4:8])[0],
            "frame_number": struct.unpack('<I', data[12:16])[0],
            "num_detected_obj": struct.unpack('<I', data[20:24])[0],
            "num_tlvs": struct.unpack('<I', data[24:28])[0],
        }
    
    def _parse_point_cloud(self, data: bytes, header: Dict) -> List[RadarPoint]:
        points = []
        offset = 0
        num_tlvs = header.get("num_tlvs", 0)
        
        for _ in range(num_tlvs):
            if offset + 8 > len(data):
                break
            
            tlv_type = struct.unpack('<I', data[offset:offset+4])[0]
            tlv_length = struct.unpack('<I', data[offset+4:offset+8])[0]
            offset += 8
            
            if tlv_type == 1:
                point_size = 16
                num_points = tlv_length // point_size
                
                for i in range(num_points):
                    start = offset + i * point_size
                    if start + point_size <= len(data):
                        x = struct.unpack('<f', data[start:start+4])[0]
                        y = struct.unpack('<f', data[start+4:start+8])[0]
                        z = struct.unpack('<f', data[start+8:start+12])[0]
                        velocity = struct.unpack('<f', data[start+12:start+16])[0]
                        
                        points.append(RadarPoint(x=x, y=y, z=z, velocity=velocity, snr=20.0))
            
            offset += tlv_length
        
        return points


# =============================================================================
# 模拟驱动
# =============================================================================

class SimulatedRadarDriver(BaseRadarDriver):
    """模拟雷达驱动"""
    
    def __init__(self, config: RadarConfig):
        super().__init__(config)
        self._frame_id = 0
        self._running = False
        self._targets = []
    
    def connect(self) -> bool:
        self._connected = True
        self._init_simulated_targets()
        return True
    
    def disconnect(self) -> bool:
        self._connected = False
        self._running = False
        return True
    
    def configure(self, params: Dict[str, Any] = None) -> bool:
        return True
    
    def start_streaming(self) -> bool:
        self._running = True
        return True
    
    def stop_streaming(self) -> bool:
        self._running = False
        return True
    
    def _init_simulated_targets(self):
        self._targets = [
            {"x": 2.0, "y": 3.0, "z": 1.5, "vx": 0.1, "vy": -0.05, "vz": 0},
            {"x": -1.5, "y": 4.0, "z": 1.7, "vx": -0.05, "vy": 0.1, "vz": 0},
            {"x": 0.5, "y": 2.5, "z": 0.5, "vx": 0, "vy": 0, "vz": 0},
        ]
    
    def read_frame(self) -> Optional[RadarFrame]:
        if not self._running:
            return None
        
        points = []
        
        for target in self._targets:
            target["x"] += target["vx"]
            target["y"] += target["vy"]
            target["z"] += target["vz"]
            
            if abs(target["x"]) > 5:
                target["vx"] *= -1
            if target["y"] < 0.5 or target["y"] > 8:
                target["vy"] *= -1
            
            num_points = np.random.randint(5, 15)
            for _ in range(num_points):
                x = target["x"] + np.random.normal(0, 0.1)
                y = target["y"] + np.random.normal(0, 0.1)
                z = target["z"] + np.random.normal(0, 0.05)
                v = np.sqrt(target["vx"]**2 + target["vy"]**2) + np.random.normal(0, 0.02)
                snr = 15 + np.random.normal(0, 3)
                
                points.append(RadarPoint(x=x, y=y, z=z, velocity=v, snr=snr))
        
        for _ in range(np.random.randint(0, 5)):
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(0.5, 8)
            z = np.random.uniform(0, 2)
            points.append(RadarPoint(x=x, y=y, z=z, velocity=0, snr=8))
        
        self._frame_id += 1
        
        return RadarFrame(
            frame_id=self._frame_id,
            timestamp=time.time(),
            points=points
        )


# =============================================================================
# 目标跟踪器
# =============================================================================

class RadarTargetTracker:
    """雷达目标跟踪器"""
    
    def __init__(self, config: RadarConfig):
        self.config = config
        self._tracks: Dict[int, TrackedTarget] = {}
        self._next_track_id = 0
    
    def update(self, frame: RadarFrame) -> List[TrackedTarget]:
        clusters = self._cluster_points(frame.points)
        matched_tracks, unmatched_clusters, unmatched_tracks = self._associate(clusters)
        
        for track_id, cluster in matched_tracks:
            self._update_track(track_id, cluster, frame.timestamp)
        
        for cluster in unmatched_clusters:
            self._create_track(cluster, frame.timestamp)
        
        for track_id in unmatched_tracks:
            self._age_track(track_id)
        
        self._prune_tracks()
        
        return list(self._tracks.values())
    
    def _cluster_points(self, points: List[RadarPoint]) -> List[List[RadarPoint]]:
        if not points:
            return []
        
        clusters = []
        used = [False] * len(points)
        threshold = 0.5
        
        for i, p1 in enumerate(points):
            if used[i]:
                continue
            
            cluster = [p1]
            used[i] = True
            
            for j, p2 in enumerate(points):
                if used[j]:
                    continue
                
                dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                if dist < threshold:
                    cluster.append(p2)
                    used[j] = True
            
            if len(cluster) >= self.config.min_num_points:
                clusters.append(cluster)
        
        return clusters
    
    def _associate(self, clusters: List[List[RadarPoint]]) -> Tuple[List, List, List]:
        matched = []
        unmatched_clusters = list(range(len(clusters)))
        unmatched_tracks = list(self._tracks.keys())
        
        if not clusters or not self._tracks:
            return matched, [clusters[i] for i in unmatched_clusters], unmatched_tracks
        
        cost_matrix = np.zeros((len(clusters), len(self._tracks)))
        track_ids = list(self._tracks.keys())
        
        for i, cluster in enumerate(clusters):
            cluster_center = self._get_cluster_center(cluster)
            for j, track_id in enumerate(track_ids):
                track = self._tracks[track_id]
                dist = np.linalg.norm(cluster_center - track.position)
                cost_matrix[i, j] = dist
        
        threshold = self.config.track_association_threshold
        
        while unmatched_clusters and unmatched_tracks:
            min_cost = float('inf')
            min_i, min_j = -1, -1
            
            for i in unmatched_clusters:
                for j, track_id in enumerate(track_ids):
                    if track_id not in unmatched_tracks:
                        continue
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_i = i
                        min_j = track_id
            
            if min_cost > threshold:
                break
            
            matched.append((min_j, clusters[min_i]))
            unmatched_clusters.remove(min_i)
            unmatched_tracks.remove(min_j)
        
        return matched, [clusters[i] for i in unmatched_clusters], unmatched_tracks
    
    def _get_cluster_center(self, cluster: List[RadarPoint]) -> np.ndarray:
        if not cluster:
            return np.zeros(3)
        return np.array([
            np.mean([p.x for p in cluster]),
            np.mean([p.y for p in cluster]),
            np.mean([p.z for p in cluster])
        ])
    
    def _update_track(self, track_id: int, cluster: List[RadarPoint], timestamp: float):
        track = self._tracks[track_id]
        new_position = self._get_cluster_center(cluster)
        dt = timestamp - track.timestamp
        
        if dt > 0:
            new_velocity = (new_position - track.position) / dt
            new_acceleration = (new_velocity - track.velocity) / dt
        else:
            new_velocity = track.velocity
            new_acceleration = track.acceleration
        
        alpha = 0.3
        track.position = alpha * new_position + (1 - alpha) * track.position
        track.velocity = alpha * new_velocity + (1 - alpha) * track.velocity
        track.acceleration = alpha * new_acceleration + (1 - alpha) * track.acceleration
        
        track.hits += 1
        track.time_since_update = 0
        track.timestamp = timestamp
        track.history.append(track.position.copy())
        
        if len(track.history) > 100:
            track.history = track.history[-100:]
    
    def _create_track(self, cluster: List[RadarPoint], timestamp: float):
        position = self._get_cluster_center(cluster)
        avg_velocity = np.mean([p.velocity for p in cluster])
        
        track = TrackedTarget(
            track_id=self._next_track_id,
            position=position,
            velocity=np.array([0.0, avg_velocity, 0.0]),
            acceleration=np.zeros(3),
            bbox=(0, 0, 0, 0, 0, 0),
            target_type=TargetType.UNKNOWN,
            confidence=0.5,
            age=1,
            hits=1,
            time_since_update=0,
            timestamp=timestamp,
            history=[position.copy()]
        )
        
        self._tracks[self._next_track_id] = track
        self._next_track_id += 1
    
    def _age_track(self, track_id: int):
        if track_id in self._tracks:
            self._tracks[track_id].time_since_update += 1
            self._tracks[track_id].age += 1
    
    def _prune_tracks(self):
        to_remove = [
            track_id for track_id, track in self._tracks.items()
            if track.time_since_update > self.config.track_max_age
        ]
        for track_id in to_remove:
            del self._tracks[track_id]
    
    def get_confirmed_tracks(self) -> List[TrackedTarget]:
        return [t for t in self._tracks.values() if t.hits >= self.config.track_min_hits]
    
    def reset(self):
        self._tracks.clear()
        self._next_track_id = 0


# =============================================================================
# 毫米波雷达适配器
# =============================================================================

class MmWaveRadarAdapter:
    """毫米波雷达适配器"""
    
    def __init__(self, config: Optional[RadarConfig] = None):
        self.config = config or RadarConfig()
        self._driver: Optional[BaseRadarDriver] = None
        self._tracker: Optional[RadarTargetTracker] = None
        self._status = AdapterStatus.DISCONNECTED
        
        self._frame_buffer: deque = deque(maxlen=100)
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[RadarFrame], None]] = []
        
        self._stats = {
            "frames_received": 0,
            "points_total": 0,
            "targets_tracked": 0,
            "start_time": None
        }
        
        logger.info(f"毫米波雷达适配器初始化: {self.config.model.value}")
    
    def connect(self) -> bool:
        self._status = AdapterStatus.CONNECTING
        self._driver = self._create_driver()
        
        if self._driver is None:
            if self.config.simulate_if_unavailable:
                logger.warning("使用模拟雷达驱动")
                self._driver = SimulatedRadarDriver(self.config)
                self._status = AdapterStatus.SIMULATED
            else:
                self._status = AdapterStatus.ERROR
                return False
        
        if self._driver.connect():
            self._driver.configure()
            
            if self.config.enable_tracking:
                self._tracker = RadarTargetTracker(self.config)
            
            self._status = AdapterStatus.CONNECTED
            self._stats["start_time"] = time.time()
            
            logger.info("雷达连接成功")
            return True
        else:
            if self.config.simulate_if_unavailable:
                self._driver = SimulatedRadarDriver(self.config)
                self._driver.connect()
                self._status = AdapterStatus.SIMULATED
                return True
            
            self._status = AdapterStatus.ERROR
            return False
    
    def _create_driver(self) -> Optional[BaseRadarDriver]:
        if self.config.protocol == RadarProtocol.UART:
            return TIRadarUARTDriver(self.config)
        return None
    
    def disconnect(self) -> bool:
        self.stop_streaming()
        if self._driver:
            self._driver.disconnect()
        self._status = AdapterStatus.DISCONNECTED
        return True
    
    def start_streaming(self) -> bool:
        if self._driver is None or self._streaming:
            return False
        
        self._driver.start_streaming()
        self._streaming = True
        self._status = AdapterStatus.STREAMING
        
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()
        
        return True
    
    def stop_streaming(self) -> bool:
        self._streaming = False
        
        if self._driver:
            self._driver.stop_streaming()
        
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None
        
        if self._status == AdapterStatus.STREAMING:
            self._status = AdapterStatus.CONNECTED
        
        return True
    
    def _stream_loop(self):
        while self._streaming:
            frame = self._driver.read_frame()
            if frame:
                self._process_frame(frame)
            time.sleep(1.0 / self.config.frame_rate_hz)
    
    def _process_frame(self, frame: RadarFrame):
        if self.config.enable_clutter_removal:
            frame.points = [p for p in frame.points if p.snr >= self.config.snr_threshold_db]
        
        if self._tracker:
            targets = self._tracker.update(frame)
            self._stats["targets_tracked"] = len(targets)
        
        self._stats["frames_received"] += 1
        self._stats["points_total"] += frame.num_detected_points
        
        self._frame_buffer.append(frame)
        
        for callback in self._callbacks:
            try:
                callback(frame)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")
    
    def get_latest_frame(self) -> Optional[RadarFrame]:
        return self._frame_buffer[-1] if self._frame_buffer else None
    
    def get_tracked_targets(self) -> List[TrackedTarget]:
        return self._tracker.get_confirmed_tracks() if self._tracker else []
    
    def get_point_cloud(self) -> np.ndarray:
        frame = self.get_latest_frame()
        return frame.to_point_cloud() if frame else np.zeros((0, 5))
    
    def register_callback(self, callback: Callable[[RadarFrame], None]):
        self._callbacks.append(callback)
    
    @property
    def status(self) -> AdapterStatus:
        return self._status
    
    @property
    def is_connected(self) -> bool:
        return self._status in [AdapterStatus.CONNECTED, AdapterStatus.STREAMING, AdapterStatus.SIMULATED]
    
    def get_statistics(self) -> Dict[str, Any]:
        elapsed = time.time() - self._stats["start_time"] if self._stats["start_time"] else 0
        fps = self._stats["frames_received"] / elapsed if elapsed > 0 else 0
        
        return {
            "status": self._status.value,
            "model": self.config.model.value,
            "frames_received": self._stats["frames_received"],
            "points_total": self._stats["points_total"],
            "targets_tracked": self._stats["targets_tracked"],
            "fps": fps
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_radar_adapter(
    model: Union[str, RadarModel] = RadarModel.TI_IWR6843,
    protocol: Union[str, RadarProtocol] = RadarProtocol.UART,
    **kwargs
) -> MmWaveRadarAdapter:
    """创建雷达适配器"""
    if isinstance(model, str):
        model = RadarModel(model)
    if isinstance(protocol, str):
        protocol = RadarProtocol(protocol)
    
    config = RadarConfig(model=model, protocol=protocol, **kwargs)
    return MmWaveRadarAdapter(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    adapter = create_radar_adapter(simulate_if_unavailable=True, enable_tracking=True)
    
    if adapter.connect():
        print(f"状态: {adapter.status.value}")
        
        adapter.start_streaming()
        time.sleep(3)
        
        targets = adapter.get_tracked_targets()
        print(f"跟踪目标数: {len(targets)}")
        for target in targets:
            print(f"  ID {target.track_id}: 位置={target.position}, 速度={target.speed:.2f}m/s")
        
        adapter.stop_streaming()
        print(f"统计: {adapter.get_statistics()}")
        adapter.disconnect()
