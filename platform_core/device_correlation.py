# -*- coding: utf-8 -*-
"""
一次设备—间隔—二次设备—信号点 关系映射
=======================================

核心功能:
1. 根据 secondary_device_id + signal_id 查找:
   - 所属厂站
   - 所属间隔
   - 所属一次设备
   - 所属二次设备屏柜/装置
2. 根据一次设备ID反查关联二次设备
3. 设备拓扑图谱管理

schema设计支持接入: OCS系统、变电管理平台、OMS系统、电网管理平台、
保信子站、外信子站、故障录波、保护装置、备自投装置、出口压板监测、
端子箱、直流系统监测、电能质量、PT并列装置、行波测距

复用: device_adapter/manager.py 的设备注册模式
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json

logger = logging.getLogger(__name__)


# =============================================================================
# 数据模型
# =============================================================================

@dataclass
class Station:
    """厂站"""
    station_id: str = ""
    station_name: str = ""
    voltage_level: str = ""         # 500kV / 220kV / 110kV / 35kV ...
    region: str = ""                # 区域(如"大理")
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bay:
    """间隔"""
    bay_id: str = ""
    bay_name: str = ""
    station_id: str = ""
    voltage_level: str = ""
    bay_type: str = ""              # line/transformer/bus_coupler/bus_section...
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrimaryDevice:
    """一次设备"""
    device_id: str = ""
    device_name: str = ""
    device_type: str = ""           # transformer/breaker/disconnector/busbar/capacitor/line/pt/ct...
    station_id: str = ""
    bay_id: str = ""
    voltage_level: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecondaryDevice:
    """二次设备"""
    device_id: str = ""
    device_name: str = ""
    device_type: str = ""           # protection/recloser/backup_auto/monitor/fault_recorder/dc_monitor...
    station_id: str = ""
    bay_id: str = ""
    cabinet_id: str = ""            # 屏柜编号
    manufacturer: str = ""
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalPoint:
    """信号点"""
    signal_id: str = ""
    signal_name: str = ""
    signal_group: str = ""          # 对应 SignalGroup 枚举
    secondary_device_id: str = ""
    primary_device_id: str = ""     # 关联的一次设备(可选)
    bay_id: str = ""
    station_id: str = ""
    data_type: str = "bool"         # bool/int/float/string
    source_system: str = ""         # 数据来源系统
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BusSection:
    """母线段"""
    bus_id: str = ""
    bus_name: str = ""
    station_id: str = ""
    voltage_level: str = ""
    bus_type: str = ""              # single/double/one_and_half/ring
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BayBusConnection:
    """间隔-母线连接关系"""
    bay_id: str = ""
    bus_id: str = ""
    breaker_id: str = ""            # 该连接上的断路器
    disconnector_ids: List[str] = field(default_factory=list)


@dataclass
class ProtectionZone:
    """保护区域(一个保护装置保护哪些设备/间隔)"""
    zone_id: str = ""
    secondary_device_id: str = ""   # 保护装置ID
    protection_type: str = ""       # ProtectionType 枚举值
    protected_bay_ids: List[str] = field(default_factory=list)
    protected_primary_ids: List[str] = field(default_factory=list)
    trip_breaker_ids: List[str] = field(default_factory=list)


@dataclass
class CorrelationResult:
    """关系查询结果"""
    station: Optional[Station] = None
    bay: Optional[Bay] = None
    primary_device: Optional[PrimaryDevice] = None
    secondary_device: Optional[SecondaryDevice] = None
    signal_point: Optional[SignalPoint] = None
    related_primary_devices: List[PrimaryDevice] = field(default_factory=list)
    related_secondary_devices: List[SecondaryDevice] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.station:
            result['station'] = asdict(self.station)
        if self.bay:
            result['bay'] = asdict(self.bay)
        if self.primary_device:
            result['primary_device'] = asdict(self.primary_device)
        if self.secondary_device:
            result['secondary_device'] = asdict(self.secondary_device)
        if self.signal_point:
            result['signal_point'] = asdict(self.signal_point)
        result['related_primary_devices'] = [asdict(d) for d in self.related_primary_devices]
        result['related_secondary_devices'] = [asdict(d) for d in self.related_secondary_devices]
        return result


# =============================================================================
# 设备关系映射服务
# =============================================================================

class DeviceCorrelationService:
    """
    设备关系映射服务

    管理厂站-间隔-一次设备-二次设备-信号点的关系图谱。
    支持配置文件加载或运行时注册。
    """

    def __init__(self):
        self._stations: Dict[str, Station] = {}
        self._bays: Dict[str, Bay] = {}
        self._primary_devices: Dict[str, PrimaryDevice] = {}
        self._secondary_devices: Dict[str, SecondaryDevice] = {}
        self._signal_points: Dict[str, SignalPoint] = {}
        self._bus_sections: Dict[str, BusSection] = {}
        self._bay_bus_connections: List[BayBusConnection] = []
        self._protection_zones: Dict[str, ProtectionZone] = {}

        # 索引
        self._bay_to_station: Dict[str, str] = {}
        self._device_to_bay: Dict[str, str] = {}
        self._signal_to_secondary: Dict[str, str] = {}
        self._bay_primary_devices: Dict[str, List[str]] = {}
        self._bay_secondary_devices: Dict[str, List[str]] = {}
        # 母线拓扑索引
        self._bus_to_bays: Dict[str, List[str]] = {}           # bus_id -> [bay_id]
        self._bay_to_buses: Dict[str, List[str]] = {}           # bay_id -> [bus_id]
        self._breaker_to_connection: Dict[str, BayBusConnection] = {}  # breaker_id -> connection
        # 保护区域索引
        self._device_protection_zones: Dict[str, List[str]] = {}  # secondary_device_id -> [zone_id]

        logger.info("DeviceCorrelationService 初始化完成")

    # -------------------------------------------------------------------------
    # 注册
    # -------------------------------------------------------------------------

    def register_station(self, station: Station) -> None:
        self._stations[station.station_id] = station

    def register_bay(self, bay: Bay) -> None:
        self._bays[bay.bay_id] = bay
        self._bay_to_station[bay.bay_id] = bay.station_id

    def register_primary_device(self, device: PrimaryDevice) -> None:
        self._primary_devices[device.device_id] = device
        self._device_to_bay[device.device_id] = device.bay_id
        self._bay_primary_devices.setdefault(device.bay_id, []).append(device.device_id)

    def register_secondary_device(self, device: SecondaryDevice) -> None:
        self._secondary_devices[device.device_id] = device
        self._device_to_bay[device.device_id] = device.bay_id
        self._bay_secondary_devices.setdefault(device.bay_id, []).append(device.device_id)

    def register_signal_point(self, signal: SignalPoint) -> None:
        self._signal_points[signal.signal_id] = signal
        self._signal_to_secondary[signal.signal_id] = signal.secondary_device_id

    def register_bus_section(self, bus: BusSection) -> None:
        self._bus_sections[bus.bus_id] = bus

    def register_bay_bus_connection(self, conn: BayBusConnection) -> None:
        self._bay_bus_connections.append(conn)
        self._bus_to_bays.setdefault(conn.bus_id, [])
        if conn.bay_id not in self._bus_to_bays[conn.bus_id]:
            self._bus_to_bays[conn.bus_id].append(conn.bay_id)
        self._bay_to_buses.setdefault(conn.bay_id, [])
        if conn.bus_id not in self._bay_to_buses[conn.bay_id]:
            self._bay_to_buses[conn.bay_id].append(conn.bus_id)
        if conn.breaker_id:
            self._breaker_to_connection[conn.breaker_id] = conn

    def register_protection_zone(self, zone: ProtectionZone) -> None:
        self._protection_zones[zone.zone_id] = zone
        self._device_protection_zones.setdefault(zone.secondary_device_id, []).append(zone.zone_id)

    # -------------------------------------------------------------------------
    # 核心查询: 根据 secondary_device_id + signal_id 查找全链路
    # -------------------------------------------------------------------------

    def correlate_by_signal(self, secondary_device_id: str, signal_id: str) -> CorrelationResult:
        """
        根据二次设备ID和信号ID查找完整关系链

        返回: 所属厂站、间隔、一次设备、二次设备屏柜/装置
        """
        result = CorrelationResult()

        # 信号点
        signal = self._signal_points.get(signal_id)
        if signal:
            result.signal_point = signal

        # 二次设备
        sec_device = self._secondary_devices.get(secondary_device_id)
        if sec_device:
            result.secondary_device = sec_device
            bay_id = sec_device.bay_id
        elif signal and signal.secondary_device_id:
            sec_device = self._secondary_devices.get(signal.secondary_device_id)
            result.secondary_device = sec_device
            bay_id = sec_device.bay_id if sec_device else ""
        else:
            bay_id = ""

        # 间隔
        if bay_id:
            bay = self._bays.get(bay_id)
            if bay:
                result.bay = bay

                # 厂站
                station = self._stations.get(bay.station_id)
                if station:
                    result.station = station

                # 该间隔下的一次设备
                primary_ids = self._bay_primary_devices.get(bay_id, [])
                result.related_primary_devices = [
                    self._primary_devices[pid] for pid in primary_ids
                    if pid in self._primary_devices
                ]
                # 如果信号点指明了一次设备
                if signal and signal.primary_device_id:
                    pd = self._primary_devices.get(signal.primary_device_id)
                    if pd:
                        result.primary_device = pd
                elif result.related_primary_devices:
                    result.primary_device = result.related_primary_devices[0]

                # 该间隔下其他二次设备
                secondary_ids = self._bay_secondary_devices.get(bay_id, [])
                result.related_secondary_devices = [
                    self._secondary_devices[sid] for sid in secondary_ids
                    if sid in self._secondary_devices
                ]

        return result

    def correlate_by_device(self, device_id: str) -> CorrelationResult:
        """根据设备ID(一次或二次)查找关系"""
        result = CorrelationResult()

        # 先查一次设备
        if device_id in self._primary_devices:
            pd = self._primary_devices[device_id]
            result.primary_device = pd
            bay_id = pd.bay_id
        elif device_id in self._secondary_devices:
            sd = self._secondary_devices[device_id]
            result.secondary_device = sd
            bay_id = sd.bay_id
        else:
            return result

        if bay_id:
            bay = self._bays.get(bay_id)
            if bay:
                result.bay = bay
                station = self._stations.get(bay.station_id)
                if station:
                    result.station = station

            primary_ids = self._bay_primary_devices.get(bay_id, [])
            result.related_primary_devices = [
                self._primary_devices[pid] for pid in primary_ids
                if pid in self._primary_devices
            ]
            secondary_ids = self._bay_secondary_devices.get(bay_id, [])
            result.related_secondary_devices = [
                self._secondary_devices[sid] for sid in secondary_ids
                if sid in self._secondary_devices
            ]

        return result

    # -------------------------------------------------------------------------
    # 批量查询
    # -------------------------------------------------------------------------

    def get_station(self, station_id: str) -> Optional[Station]:
        return self._stations.get(station_id)

    def get_bay(self, bay_id: str) -> Optional[Bay]:
        return self._bays.get(bay_id)

    def get_primary_device(self, device_id: str) -> Optional[PrimaryDevice]:
        return self._primary_devices.get(device_id)

    def get_secondary_device(self, device_id: str) -> Optional[SecondaryDevice]:
        return self._secondary_devices.get(device_id)

    def list_stations(self) -> List[Station]:
        return list(self._stations.values())

    def list_bays(self, station_id: Optional[str] = None) -> List[Bay]:
        bays = list(self._bays.values())
        if station_id:
            bays = [b for b in bays if b.station_id == station_id]
        return bays

    def list_primary_devices(self, bay_id: Optional[str] = None) -> List[PrimaryDevice]:
        devices = list(self._primary_devices.values())
        if bay_id:
            devices = [d for d in devices if d.bay_id == bay_id]
        return devices

    def list_secondary_devices(self, bay_id: Optional[str] = None) -> List[SecondaryDevice]:
        devices = list(self._secondary_devices.values())
        if bay_id:
            devices = [d for d in devices if d.bay_id == bay_id]
        return devices

    # -------------------------------------------------------------------------
    # 母线拓扑查询
    # -------------------------------------------------------------------------

    def get_bus_section(self, bus_id: str) -> Optional[BusSection]:
        return self._bus_sections.get(bus_id)

    def list_bus_sections(self, station_id: Optional[str] = None) -> List[BusSection]:
        buses = list(self._bus_sections.values())
        if station_id:
            buses = [b for b in buses if b.station_id == station_id]
        return buses

    def get_bay_bus_connections(self, bay_id: Optional[str] = None, bus_id: Optional[str] = None) -> List[BayBusConnection]:
        conns = self._bay_bus_connections
        if bay_id:
            conns = [c for c in conns if c.bay_id == bay_id]
        if bus_id:
            conns = [c for c in conns if c.bus_id == bus_id]
        return conns

    def get_protection_zones(self, secondary_device_id: str) -> List[ProtectionZone]:
        """查询保护装置的保护区域"""
        zone_ids = self._device_protection_zones.get(secondary_device_id, [])
        return [self._protection_zones[zid] for zid in zone_ids if zid in self._protection_zones]

    def get_trip_breakers(self, secondary_device_id: str, protection_type: Optional[str] = None) -> List[str]:
        """查询保护装置跳闸控制的断路器ID列表"""
        zones = self.get_protection_zones(secondary_device_id)
        if protection_type:
            zones = [z for z in zones if z.protection_type == protection_type]
        breaker_ids = []
        for z in zones:
            for bid in z.trip_breaker_ids:
                if bid not in breaker_ids:
                    breaker_ids.append(bid)
        return breaker_ids

    def get_outage_scope(self, tripped_breaker_ids: List[str]) -> List[Dict[str, str]]:
        """
        根据跳开的断路器列表，推导失电/失压的间隔和母线。

        逻辑: 断路器跳开 → 对应的 bay-bus 连接断开 →
              检查该母线上是否还有其他未跳开断路器的电源路径 →
              若无 → 该母线及其下属间隔失电。
        """
        tripped_set = set(tripped_breaker_ids)
        outage: List[Dict[str, str]] = []
        checked_buses: set = set()

        for breaker_id in tripped_breaker_ids:
            conn = self._breaker_to_connection.get(breaker_id)
            if not conn:
                continue
            bus_id = conn.bus_id
            if bus_id in checked_buses:
                continue
            checked_buses.add(bus_id)

            # 检查该母线上是否还有未跳开的断路器提供电源
            bus_conns = self.get_bay_bus_connections(bus_id=bus_id)
            has_live_source = False
            for bc in bus_conns:
                if bc.breaker_id and bc.breaker_id not in tripped_set:
                    has_live_source = True
                    break

            if not has_live_source:
                # 该母线失电，收集母线上所有间隔
                bus = self._bus_sections.get(bus_id)
                bus_name = bus.bus_name if bus else bus_id
                voltage = bus.voltage_level if bus else ""
                bay_ids_on_bus = self._bus_to_bays.get(bus_id, [])
                for bay_id in bay_ids_on_bus:
                    bay = self._bays.get(bay_id)
                    outage.append({
                        "bay_id": bay_id,
                        "bay_name": bay.bay_name if bay else bay_id,
                        "voltage_level": voltage,
                        "outage_type": f"母线失电({bus_name})",
                    })

        # 对于每个跳开的断路器，其直属间隔也标记为受影响
        for breaker_id in tripped_breaker_ids:
            conn = self._breaker_to_connection.get(breaker_id)
            if not conn:
                continue
            bay_id = conn.bay_id
            # 避免重复
            if any(o["bay_id"] == bay_id for o in outage):
                continue
            bay = self._bays.get(bay_id)
            outage.append({
                "bay_id": bay_id,
                "bay_name": bay.bay_name if bay else bay_id,
                "voltage_level": bay.voltage_level if bay else "",
                "outage_type": "断路器跳开",
            })

        return outage

    # -------------------------------------------------------------------------
    # 配置加载
    # -------------------------------------------------------------------------

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载全部关系数据(支持嵌套和扁平两种格式)"""
        for s in data.get('stations', []):
            self.register_station(Station(**{k: v for k, v in s.items() if k in Station.__dataclass_fields__}))
        for b in data.get('bays', []):
            # 提取 bus_connections 嵌套数据
            bus_connections = b.pop('bus_connections', [])
            self.register_bay(Bay(**{k: v for k, v in b.items() if k in Bay.__dataclass_fields__}))
            for bc in bus_connections:
                bc.setdefault('bay_id', b.get('bay_id', ''))
                self.register_bay_bus_connection(BayBusConnection(
                    **{k: v for k, v in bc.items() if k in BayBusConnection.__dataclass_fields__}
                ))
        for d in data.get('primary_devices', []):
            self.register_primary_device(PrimaryDevice(**{k: v for k, v in d.items() if k in PrimaryDevice.__dataclass_fields__}))
        for d in data.get('secondary_devices', []):
            self.register_secondary_device(SecondaryDevice(**{k: v for k, v in d.items() if k in SecondaryDevice.__dataclass_fields__}))
        for s in data.get('signal_points', []):
            self.register_signal_point(SignalPoint(**{k: v for k, v in s.items() if k in SignalPoint.__dataclass_fields__}))
        for b in data.get('bus_sections', []):
            self.register_bus_section(BusSection(**{k: v for k, v in b.items() if k in BusSection.__dataclass_fields__}))
        # 扁平格式的 bay_bus_connections
        for c in data.get('bay_bus_connections', []):
            self.register_bay_bus_connection(BayBusConnection(
                **{k: v for k, v in c.items() if k in BayBusConnection.__dataclass_fields__}
            ))
        for z in data.get('protection_zones', []):
            self.register_protection_zone(ProtectionZone(
                **{k: v for k, v in z.items() if k in ProtectionZone.__dataclass_fields__}
            ))
        logger.info(
            f"关系数据加载完成: {len(self._stations)}厂站, "
            f"{len(self._bays)}间隔, {len(self._primary_devices)}一次设备, "
            f"{len(self._secondary_devices)}二次设备, {len(self._signal_points)}信号点, "
            f"{len(self._bus_sections)}母线段, {len(self._bay_bus_connections)}母线连接, "
            f"{len(self._protection_zones)}保护区域"
        )

    def load_from_json(self, filepath: str) -> None:
        """从JSON文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.load_from_dict(data)

    def load_from_yaml(self, filepath: str) -> None:
        """从YAML文件加载"""
        try:
            import yaml
        except ImportError:
            logger.error("需要安装 PyYAML: pip install pyyaml")
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # 支持单站 YAML 格式(station 为单个 dict 而非列表)
        if 'station' in data and 'stations' not in data:
            data['stations'] = [data.pop('station')]
        self.load_from_dict(data)

    def get_statistics(self) -> Dict[str, int]:
        return {
            "stations": len(self._stations),
            "bays": len(self._bays),
            "primary_devices": len(self._primary_devices),
            "secondary_devices": len(self._secondary_devices),
            "signal_points": len(self._signal_points),
            "bus_sections": len(self._bus_sections),
            "bay_bus_connections": len(self._bay_bus_connections),
            "protection_zones": len(self._protection_zones),
        }
