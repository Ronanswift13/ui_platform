#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区域配置加载器
==========================================

加载YAML区域配置，定义授权区、缓冲区、危险区和机柜扇区
基于《变电站多目标操作安全监测系统》方案实现

作者: G组 | 版本: 2.0.0
"""

from __future__ import annotations
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from .geometry import Point2D, Polygon, LineSegment

logger = logging.getLogger(__name__)


class ZoneType(str, Enum):
    """区域类型"""
    AUTHORIZED = "authorized"       # 授权区 - 允许人员进入的安全区域
    BUFFER = "buffer"               # 缓冲区 - 警告区域
    DANGER = "danger"               # 危险区 - 严禁进入
    CABINET = "cabinet"             # 机柜扇区
    PASSAGE = "passage"             # 通道区域
    YELLOW_LINE = "yellow_line"     # 黄线边界


class AlertLevel(str, Enum):
    """告警级别"""
    NORMAL = "normal"       # 正常 - 绿色
    ATTENTION = "attention" # 注意 - 蓝色
    WARNING = "warning"     # 警告 - 黄色
    ALARM = "alarm"         # 告警 - 橙色
    CRITICAL = "critical"   # 危急 - 红色


@dataclass
class ZoneDefinition:
    """区域定义"""
    zone_id: str
    name: str
    zone_type: ZoneType
    polygon: Polygon
    alert_level: AlertLevel = AlertLevel.NORMAL
    
    # 权限配置
    authorized_persons: Set[str] = field(default_factory=set)  # 授权人员ID
    max_occupancy: int = 1          # 最大允许人数
    require_authorization: bool = False  # 是否需要授权
    
    # 几何参数
    buffer_distance: float = 0.3    # 缓冲距离 (米)
    
    # 关联机柜
    cabinet_id: Optional[int] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def contains_point(self, point: Point2D) -> bool:
        """检查点是否在区域内"""
        from .geometry import point_in_polygon
        return point_in_polygon(point, self.polygon)
    
    def distance_to_boundary(self, point: Point2D) -> float:
        """计算点到区域边界的距离"""
        from .geometry import point_to_polygon_distance
        dist, _, _ = point_to_polygon_distance(point, self.polygon)
        return dist


@dataclass
class CabinetDefinition:
    """机柜定义"""
    cabinet_id: int
    name: str
    zone: ZoneDefinition
    
    # 位置信息
    x_start: float = 0.0        # 归一化X起始
    x_end: float = 0.0          # 归一化X结束
    y_position: float = 0.0     # Y位置 (米)
    
    # 状态
    is_authorized: bool = False  # 是否已授权操作
    current_occupants: List[str] = field(default_factory=list)
    status: str = "idle"         # idle, occupied, alert
    
    # 关联的黄线
    yellow_line: Optional[LineSegment] = None
    yellow_line_distance: float = 0.5  # 黄线距离 (米)


@dataclass
class YellowLineDefinition:
    """黄线定义"""
    line_id: str
    name: str
    line: LineSegment
    associated_cabinets: List[int] = field(default_factory=list)
    
    # 安全距离
    warning_distance: float = 0.5   # 警告距离
    danger_distance: float = 0.1    # 危险距离


@dataclass
class ZoneConfiguration:
    """完整区域配置"""
    config_id: str
    config_name: str
    version: str = "1.0.0"
    
    # 区域定义
    zones: Dict[str, ZoneDefinition] = field(default_factory=dict)
    cabinets: Dict[int, CabinetDefinition] = field(default_factory=dict)
    yellow_lines: Dict[str, YellowLineDefinition] = field(default_factory=dict)
    
    # 全局参数
    coordinate_system: str = "local"  # local, utm, pixel
    unit: str = "meter"               # meter, pixel
    
    # 场景边界
    scene_bounds: Optional[Polygon] = None
    
    # 默认告警配置
    default_alert_levels: Dict[ZoneType, AlertLevel] = field(default_factory=lambda: {
        ZoneType.AUTHORIZED: AlertLevel.NORMAL,
        ZoneType.BUFFER: AlertLevel.WARNING,
        ZoneType.DANGER: AlertLevel.CRITICAL,
        ZoneType.CABINET: AlertLevel.ATTENTION,
        ZoneType.PASSAGE: AlertLevel.NORMAL,
    })
    
    def get_zone_at_point(self, point: Point2D) -> Optional[ZoneDefinition]:
        """获取点所在的区域"""
        for zone in self.zones.values():
            if zone.contains_point(point):
                return zone
        return None
    
    def get_cabinet_at_point(self, point: Point2D) -> Optional[CabinetDefinition]:
        """获取点所在的机柜"""
        for cabinet in self.cabinets.values():
            if cabinet.zone.contains_point(point):
                return cabinet
        return None
    
    def get_zones_by_type(self, zone_type: ZoneType) -> List[ZoneDefinition]:
        """按类型获取区域"""
        return [z for z in self.zones.values() if z.zone_type == zone_type]
    
    def get_nearest_yellow_line(self, point: Point2D) -> Optional[tuple]:
        """获取最近的黄线及距离"""
        from .geometry import point_to_segment_distance
        
        nearest = None
        min_dist = float('inf')
        
        for yl in self.yellow_lines.values():
            dist, _ = point_to_segment_distance(point, yl.line)
            if dist < min_dist:
                min_dist = dist
                nearest = yl
        
        return (nearest, min_dist) if nearest else None


class ZoneConfigLoader:
    """区域配置加载器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self._config: Optional[ZoneConfiguration] = None
    
    def load(self, config_path: Optional[Path] = None) -> ZoneConfiguration:
        """
        加载区域配置
        
        Args:
            config_path: YAML配置文件路径
            
        Returns:
            ZoneConfiguration: 区域配置对象
        """
        path = config_path or self.config_path
        
        if path is None or not Path(path).exists():
            logger.warning(f"配置文件不存在: {path}, 使用默认配置")
            return self._create_default_config()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            return self._parse_config(data)
            
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return self._create_default_config()
    
    def _parse_config(self, data: Dict) -> ZoneConfiguration:
        """解析YAML配置数据"""
        config = ZoneConfiguration(
            config_id=data.get('id', 'default'),
            config_name=data.get('name', '默认配置'),
            version=data.get('version', '1.0.0'),
            coordinate_system=data.get('coordinate_system', 'local'),
            unit=data.get('unit', 'meter'),
        )
        
        # 解析场景边界
        if 'scene_bounds' in data:
            bounds_data = data['scene_bounds']
            config.scene_bounds = Polygon.from_list(bounds_data.get('vertices', []), 'scene_bounds')
        
        # 解析区域
        for zone_data in data.get('zones', []):
            zone = self._parse_zone(zone_data)
            config.zones[zone.zone_id] = zone
        
        # 解析机柜
        for cab_data in data.get('cabinets', []):
            cabinet = self._parse_cabinet(cab_data, config.zones)
            config.cabinets[cabinet.cabinet_id] = cabinet
        
        # 解析黄线
        for line_data in data.get('yellow_lines', []):
            yellow_line = self._parse_yellow_line(line_data)
            config.yellow_lines[yellow_line.line_id] = yellow_line
        
        logger.info(f"加载区域配置: {len(config.zones)}个区域, {len(config.cabinets)}个机柜, {len(config.yellow_lines)}条黄线")
        
        return config
    
    def _parse_zone(self, data: Dict) -> ZoneDefinition:
        """解析单个区域"""
        zone_type = ZoneType(data.get('type', 'authorized'))
        
        # 解析多边形顶点
        vertices = data.get('vertices', [])
        polygon = Polygon.from_list(vertices, data.get('name', ''))
        
        return ZoneDefinition(
            zone_id=data.get('id', ''),
            name=data.get('name', ''),
            zone_type=zone_type,
            polygon=polygon,
            alert_level=AlertLevel(data.get('alert_level', 'normal')),
            authorized_persons=set(data.get('authorized_persons', [])),
            max_occupancy=data.get('max_occupancy', 1),
            require_authorization=data.get('require_authorization', False),
            buffer_distance=data.get('buffer_distance', 0.3),
            cabinet_id=data.get('cabinet_id'),
            metadata=data.get('metadata', {}),
        )
    
    def _parse_cabinet(self, data: Dict, zones: Dict[str, ZoneDefinition]) -> CabinetDefinition:
        """解析机柜定义"""
        cabinet_id = data.get('id', 0)
        zone_id = data.get('zone_id', f'cabinet_{cabinet_id}')
        
        # 获取或创建关联区域
        if zone_id in zones:
            zone = zones[zone_id]
        else:
            # 从坐标创建区域
            vertices = data.get('vertices', [])
            polygon = Polygon.from_list(vertices, data.get('name', f'机柜-{cabinet_id}'))
            zone = ZoneDefinition(
                zone_id=zone_id,
                name=data.get('name', f'机柜-{cabinet_id}'),
                zone_type=ZoneType.CABINET,
                polygon=polygon,
                cabinet_id=cabinet_id,
            )
            zones[zone_id] = zone
        
        # 解析黄线
        yellow_line = None
        if 'yellow_line' in data:
            yl_data = data['yellow_line']
            start = Point2D(yl_data['start'][0], yl_data['start'][1])
            end = Point2D(yl_data['end'][0], yl_data['end'][1])
            yellow_line = LineSegment(start, end)
        
        return CabinetDefinition(
            cabinet_id=cabinet_id,
            name=data.get('name', f'机柜-{cabinet_id}'),
            zone=zone,
            x_start=data.get('x_start', 0.0),
            x_end=data.get('x_end', 0.0),
            y_position=data.get('y_position', 0.0),
            is_authorized=data.get('is_authorized', False),
            yellow_line=yellow_line,
            yellow_line_distance=data.get('yellow_line_distance', 0.5),
        )
    
    def _parse_yellow_line(self, data: Dict) -> YellowLineDefinition:
        """解析黄线定义"""
        start = Point2D(data['start'][0], data['start'][1])
        end = Point2D(data['end'][0], data['end'][1])
        
        return YellowLineDefinition(
            line_id=data.get('id', ''),
            name=data.get('name', ''),
            line=LineSegment(start, end),
            associated_cabinets=data.get('cabinets', []),
            warning_distance=data.get('warning_distance', 0.5),
            danger_distance=data.get('danger_distance', 0.1),
        )
    
    def _create_default_config(self) -> ZoneConfiguration:
        """创建默认配置 - 6机柜场景"""
        config = ZoneConfiguration(
            config_id='default',
            config_name='默认6机柜配置',
            version='1.0.0',
        )
        
        # 场景边界 (10m x 6m)
        config.scene_bounds = Polygon.from_list([
            (0, 0), (10, 0), (10, 6), (0, 6)
        ], 'scene')
        
        # 创建通道区域
        passage = ZoneDefinition(
            zone_id='passage_main',
            name='主通道',
            zone_type=ZoneType.PASSAGE,
            polygon=Polygon.from_list([(0, 0), (10, 0), (10, 2), (0, 2)], '主通道'),
            alert_level=AlertLevel.NORMAL,
        )
        config.zones['passage_main'] = passage
        
        # 创建6个机柜区域
        cabinet_width = 10.0 / 6
        for i in range(6):
            x_start = i * cabinet_width
            x_end = (i + 1) * cabinet_width
            
            cab_zone = ZoneDefinition(
                zone_id=f'cabinet_{i+1}',
                name=f'机柜-{i+1:02d}',
                zone_type=ZoneType.CABINET,
                polygon=Polygon.from_list([
                    (x_start, 2), (x_end, 2), (x_end, 6), (x_start, 6)
                ], f'机柜-{i+1:02d}'),
                alert_level=AlertLevel.ATTENTION,
                cabinet_id=i+1,
                max_occupancy=1,
                require_authorization=True,
            )
            config.zones[cab_zone.zone_id] = cab_zone
            
            # 黄线 (机柜前方0.5米)
            yellow_line = LineSegment(
                Point2D(x_start, 2.5),
                Point2D(x_end, 2.5)
            )
            
            cabinet = CabinetDefinition(
                cabinet_id=i+1,
                name=f'机柜-{i+1:02d}',
                zone=cab_zone,
                x_start=x_start / 10,
                x_end=x_end / 10,
                y_position=4.0,
                yellow_line=yellow_line,
                yellow_line_distance=0.5,
            )
            config.cabinets[i+1] = cabinet
            
            # 添加黄线定义
            config.yellow_lines[f'yl_{i+1}'] = YellowLineDefinition(
                line_id=f'yl_{i+1}',
                name=f'机柜{i+1}黄线',
                line=yellow_line,
                associated_cabinets=[i+1],
            )
        
        return config
    
    def save(self, config: ZoneConfiguration, path: Path) -> bool:
        """保存配置到YAML文件"""
        try:
            data = self._config_to_dict(config)
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            logger.info(f"配置已保存到: {path}")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    def _config_to_dict(self, config: ZoneConfiguration) -> Dict:
        """将配置转换为字典"""
        data = {
            'id': config.config_id,
            'name': config.config_name,
            'version': config.version,
            'coordinate_system': config.coordinate_system,
            'unit': config.unit,
            'zones': [],
            'cabinets': [],
            'yellow_lines': [],
        }
        
        if config.scene_bounds:
            data['scene_bounds'] = {
                'vertices': [v.to_tuple() for v in config.scene_bounds.vertices]
            }
        
        for zone in config.zones.values():
            data['zones'].append({
                'id': zone.zone_id,
                'name': zone.name,
                'type': zone.zone_type.value,
                'vertices': [v.to_tuple() for v in zone.polygon.vertices],
                'alert_level': zone.alert_level.value,
                'max_occupancy': zone.max_occupancy,
                'require_authorization': zone.require_authorization,
                'cabinet_id': zone.cabinet_id,
            })
        
        for cab in config.cabinets.values():
            cab_data = {
                'id': cab.cabinet_id,
                'name': cab.name,
                'zone_id': cab.zone.zone_id,
                'x_start': cab.x_start,
                'x_end': cab.x_end,
                'y_position': cab.y_position,
                'yellow_line_distance': cab.yellow_line_distance,
            }
            if cab.yellow_line:
                cab_data['yellow_line'] = {
                    'start': cab.yellow_line.start.to_tuple(),
                    'end': cab.yellow_line.end.to_tuple(),
                }
            data['cabinets'].append(cab_data)
        
        for yl in config.yellow_lines.values():
            data['yellow_lines'].append({
                'id': yl.line_id,
                'name': yl.name,
                'start': yl.line.start.to_tuple(),
                'end': yl.line.end.to_tuple(),
                'cabinets': yl.associated_cabinets,
                'warning_distance': yl.warning_distance,
                'danger_distance': yl.danger_distance,
            })
        
        return data


__all__ = [
    'ZoneType', 'AlertLevel', 'ZoneDefinition', 'CabinetDefinition', 
    'YellowLineDefinition', 'ZoneConfiguration', 'ZoneConfigLoader'
]
