#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态机模块
==========================================

基于人物坐标和区域配置计算状态:
- NORMAL: 正常状态
- ON_LINE: 压线状态
- CROSS_LINE: 越线状态
- MISPLACED: 错位状态
- HIGH_RISK: 高风险状态

按最坏情况生成全局报警

作者: G组 | 版本: 2.0.0
"""

from __future__ import annotations
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from .geometry import Point2D, point_to_segment_distance
from .zone_config import (
    ZoneConfiguration, ZoneDefinition, CabinetDefinition,
    ZoneType, AlertLevel
)

logger = logging.getLogger(__name__)


class PersonState(str, Enum):
    """人员状态枚举"""
    NORMAL = "normal"           # 正常 - 在授权区内
    ON_LINE = "on_line"         # 压线 - 接近黄线边界
    CROSS_LINE = "cross_line"   # 越线 - 已越过黄线
    MISPLACED = "misplaced"     # 错位 - 在未授权区域
    HIGH_RISK = "high_risk"     # 高风险 - 危险区域
    UNKNOWN = "unknown"         # 未知状态


class GlobalAlarmLevel(str, Enum):
    """全局告警级别"""
    GREEN = "green"     # 绿色 - 一切正常
    YELLOW = "yellow"   # 黄色 - 注意/警告
    RED = "red"         # 红色 - 危险/报警


@dataclass
class PersonStateResult:
    """人员状态评估结果"""
    person_id: str
    state: PersonState
    position: Point2D
    
    # 区域信息
    current_zone: Optional[str] = None
    current_cabinet: Optional[int] = None
    
    # 距离信息
    distance_to_yellow_line: float = float('inf')
    distance_to_nearest_boundary: float = float('inf')
    
    # 授权信息
    is_authorized: bool = False
    authorization_reason: str = ""
    
    # 告警信息
    alert_level: AlertLevel = AlertLevel.NORMAL
    alert_message: str = ""
    
    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 原始数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalStateResult:
    """全局状态评估结果"""
    alarm_level: GlobalAlarmLevel
    light_color: str  # green, yellow, red
    
    # 人员状态汇总
    total_persons: int = 0
    persons_normal: int = 0
    persons_on_line: int = 0
    persons_cross_line: int = 0
    persons_misplaced: int = 0
    persons_high_risk: int = 0
    
    # 详细状态
    person_states: List[PersonStateResult] = field(default_factory=list)
    
    # 活跃告警
    active_alarms: List[Dict[str, Any]] = field(default_factory=list)
    
    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 状态消息
    status_message: str = ""


class StateMachine:
    """
    状态机 - 计算人员状态并生成全局告警
    
    状态转换规则:
    1. 根据人员位置判断所在区域
    2. 检查授权状态
    3. 计算与黄线的距离
    4. 综合判断最终状态
    5. 按最坏情况聚合全局告警
    """
    
    # 状态优先级 (越高越严重)
    STATE_PRIORITY = {
        PersonState.NORMAL: 0,
        PersonState.ON_LINE: 1,
        PersonState.MISPLACED: 2,
        PersonState.CROSS_LINE: 3,
        PersonState.HIGH_RISK: 4,
        PersonState.UNKNOWN: 0,
    }
    
    # 状态到告警级别映射
    STATE_TO_ALERT = {
        PersonState.NORMAL: AlertLevel.NORMAL,
        PersonState.ON_LINE: AlertLevel.WARNING,
        PersonState.MISPLACED: AlertLevel.ALARM,
        PersonState.CROSS_LINE: AlertLevel.CRITICAL,
        PersonState.HIGH_RISK: AlertLevel.CRITICAL,
        PersonState.UNKNOWN: AlertLevel.ATTENTION,
    }
    
    # 告警级别到全局灯色映射
    ALERT_TO_LIGHT = {
        AlertLevel.NORMAL: GlobalAlarmLevel.GREEN,
        AlertLevel.ATTENTION: GlobalAlarmLevel.GREEN,
        AlertLevel.WARNING: GlobalAlarmLevel.YELLOW,
        AlertLevel.ALARM: GlobalAlarmLevel.YELLOW,
        AlertLevel.CRITICAL: GlobalAlarmLevel.RED,
    }
    
    def __init__(self, zone_config: ZoneConfiguration):
        """
        初始化状态机
        
        Args:
            zone_config: 区域配置
        """
        self.zone_config = zone_config
        
        # 阈值配置
        self.on_line_threshold = 0.3      # 压线判定阈值 (米)
        self.cross_line_threshold = 0.0   # 越线判定阈值 (米)
        
        # 状态历史 (用于防抖动)
        self._state_history: Dict[str, List[PersonState]] = {}
        self._history_length = 5  # 历史长度
        
        # 授权列表
        self._authorized_persons: Dict[str, set] = {}  # person_id -> cabinet_ids
        
        logger.info("状态机初始化完成")
    
    def set_authorization(self, person_id: str, cabinet_ids: List[int]):
        """
        设置人员授权
        
        Args:
            person_id: 人员ID
            cabinet_ids: 授权的机柜ID列表
        """
        self._authorized_persons[person_id] = set(cabinet_ids)
        logger.info(f"设置授权: {person_id} -> 机柜{cabinet_ids}")
    
    def clear_authorization(self, person_id: str):
        """清除人员授权"""
        if person_id in self._authorized_persons:
            del self._authorized_persons[person_id]
            logger.info(f"清除授权: {person_id}")
    
    def evaluate_person(
        self,
        person_id: str,
        position: Point2D,
        metadata: Optional[Dict] = None
    ) -> PersonStateResult:
        """
        评估单个人员的状态
        
        Args:
            person_id: 人员ID
            position: 人员位置 (地面坐标)
            metadata: 额外元数据
            
        Returns:
            PersonStateResult: 状态评估结果
        """
        result = PersonStateResult(
            person_id=person_id,
            state=PersonState.UNKNOWN,
            position=position,
            metadata=metadata or {},
        )
        
        # 1. 确定所在区域
        zone = self.zone_config.get_zone_at_point(position)
        cabinet = self.zone_config.get_cabinet_at_point(position)
        
        if zone:
            result.current_zone = zone.zone_id
        if cabinet:
            result.current_cabinet = cabinet.cabinet_id
        
        # 2. 计算到最近黄线的距离
        yellow_line_info = self.zone_config.get_nearest_yellow_line(position)
        if yellow_line_info:
            nearest_yl, distance = yellow_line_info
            result.distance_to_yellow_line = distance
        
        # 3. 检查授权状态
        authorized_cabinets = self._authorized_persons.get(person_id, set())
        if cabinet:
            result.is_authorized = cabinet.cabinet_id in authorized_cabinets
            if result.is_authorized:
                result.authorization_reason = f"已授权操作机柜{cabinet.cabinet_id}"
            else:
                result.authorization_reason = f"未授权操作机柜{cabinet.cabinet_id}"
        else:
            # 不在机柜区域
            result.is_authorized = True  # 在通道等公共区域默认授权
            result.authorization_reason = "在公共区域"
        
        # 4. 判断状态
        result.state = self._determine_state(result, zone, cabinet)
        
        # 5. 应用状态防抖
        result.state = self._apply_debounce(person_id, result.state)
        
        # 6. 设置告警级别和消息
        result.alert_level = self.STATE_TO_ALERT.get(result.state, AlertLevel.NORMAL)
        result.alert_message = self._generate_alert_message(result)
        
        return result
    
    def _determine_state(
        self,
        result: PersonStateResult,
        zone: Optional[ZoneDefinition],
        cabinet: Optional[CabinetDefinition]
    ) -> PersonState:
        """根据区域和距离判断状态"""
        
        # 危险区检查
        if zone and zone.zone_type == ZoneType.DANGER:
            return PersonState.HIGH_RISK
        
        # 越线检查 (距离黄线为负表示已越过)
        if result.distance_to_yellow_line <= self.cross_line_threshold:
            return PersonState.CROSS_LINE
        
        # 压线检查
        if result.distance_to_yellow_line <= self.on_line_threshold:
            return PersonState.ON_LINE
        
        # 授权检查
        if cabinet and not result.is_authorized:
            return PersonState.MISPLACED
        
        # 正常状态
        return PersonState.NORMAL
    
    def _apply_debounce(self, person_id: str, new_state: PersonState) -> PersonState:
        """
        应用状态防抖
        
        避免状态频繁切换，只有连续N帧相同状态才切换
        """
        if person_id not in self._state_history:
            self._state_history[person_id] = []
        
        history = self._state_history[person_id]
        history.append(new_state)
        
        # 保持固定长度
        if len(history) > self._history_length:
            history.pop(0)
        
        # 如果历史中全部相同，则确认状态
        if len(history) >= 3 and len(set(history[-3:])) == 1:
            return history[-1]
        
        # 否则保持之前的状态 (如果有)
        if len(history) > 1:
            # 取历史中优先级最高的状态
            return max(history[:-1], key=lambda s: self.STATE_PRIORITY.get(s, 0))
        
        return new_state
    
    def _generate_alert_message(self, result: PersonStateResult) -> str:
        """生成告警消息"""
        messages = {
            PersonState.NORMAL: f"人员{result.person_id}状态正常",
            PersonState.ON_LINE: f"人员{result.person_id}接近黄线 (距离{result.distance_to_yellow_line:.2f}m)",
            PersonState.CROSS_LINE: f"⚠️ 人员{result.person_id}越线! 请立即退回",
            PersonState.MISPLACED: f"人员{result.person_id}在未授权区域 (机柜{result.current_cabinet})",
            PersonState.HIGH_RISK: f"🚨 人员{result.person_id}进入危险区域!",
            PersonState.UNKNOWN: f"人员{result.person_id}状态未知",
        }
        return messages.get(result.state, "")
    
    def evaluate_global(
        self,
        person_positions: Dict[str, Point2D],
        metadata: Optional[Dict[str, Dict]] = None
    ) -> GlobalStateResult:
        """
        评估全局状态
        
        Args:
            person_positions: 人员位置字典 {person_id: position}
            metadata: 人员元数据字典
            
        Returns:
            GlobalStateResult: 全局状态结果
        """
        metadata = metadata or {}
        
        # 评估每个人员
        person_states = []
        for person_id, position in person_positions.items():
            state_result = self.evaluate_person(
                person_id,
                position,
                metadata.get(person_id)
            )
            person_states.append(state_result)
        
        # 统计各状态人数
        state_counts = {
            PersonState.NORMAL: 0,
            PersonState.ON_LINE: 0,
            PersonState.CROSS_LINE: 0,
            PersonState.MISPLACED: 0,
            PersonState.HIGH_RISK: 0,
        }
        
        for ps in person_states:
            if ps.state in state_counts:
                state_counts[ps.state] += 1
        
        # 按最坏情况确定全局告警级别
        worst_state = PersonState.NORMAL
        for ps in person_states:
            if self.STATE_PRIORITY.get(ps.state, 0) > self.STATE_PRIORITY.get(worst_state, 0):
                worst_state = ps.state
        
        worst_alert = self.STATE_TO_ALERT.get(worst_state, AlertLevel.NORMAL)
        global_alarm = self.ALERT_TO_LIGHT.get(worst_alert, GlobalAlarmLevel.GREEN)
        
        # 生成活跃告警
        active_alarms = []
        for ps in person_states:
            if ps.state not in (PersonState.NORMAL, PersonState.UNKNOWN):
                active_alarms.append({
                    'person_id': ps.person_id,
                    'state': ps.state.value,
                    'message': ps.alert_message,
                    'level': ps.alert_level.value,
                    'timestamp': ps.timestamp.isoformat(),
                })
        
        # 生成状态消息
        if global_alarm == GlobalAlarmLevel.GREEN:
            status_message = f"系统正常 - {len(person_positions)}人在监控区域"
        elif global_alarm == GlobalAlarmLevel.YELLOW:
            status_message = f"注意 - 有{state_counts[PersonState.ON_LINE] + state_counts[PersonState.MISPLACED]}人需要关注"
        else:
            status_message = f"⚠️ 警报 - {state_counts[PersonState.CROSS_LINE] + state_counts[PersonState.HIGH_RISK]}人越线/高风险"
        
        return GlobalStateResult(
            alarm_level=global_alarm,
            light_color=global_alarm.value,
            total_persons=len(person_positions),
            persons_normal=state_counts[PersonState.NORMAL],
            persons_on_line=state_counts[PersonState.ON_LINE],
            persons_cross_line=state_counts[PersonState.CROSS_LINE],
            persons_misplaced=state_counts[PersonState.MISPLACED],
            persons_high_risk=state_counts[PersonState.HIGH_RISK],
            person_states=person_states,
            active_alarms=active_alarms,
            status_message=status_message,
        )
    
    def check_multi_person_violation(
        self,
        person_states: List[PersonStateResult]
    ) -> List[Dict[str, Any]]:
        """
        检查同柜多人违规
        
        Args:
            person_states: 人员状态列表
            
        Returns:
            List[Dict]: 违规记录列表
        """
        violations = []
        
        # 按机柜分组
        cabinet_occupants: Dict[int, List[str]] = {}
        for ps in person_states:
            if ps.current_cabinet:
                if ps.current_cabinet not in cabinet_occupants:
                    cabinet_occupants[ps.current_cabinet] = []
                cabinet_occupants[ps.current_cabinet].append(ps.person_id)
        
        # 检查超员
        for cabinet_id, occupants in cabinet_occupants.items():
            cabinet = self.zone_config.cabinets.get(cabinet_id)
            if cabinet:
                max_occupancy = cabinet.zone.max_occupancy
                if len(occupants) > max_occupancy:
                    violations.append({
                        'type': 'multi_person',
                        'cabinet_id': cabinet_id,
                        'cabinet_name': cabinet.name,
                        'current_count': len(occupants),
                        'max_allowed': max_occupancy,
                        'persons': occupants,
                        'message': f"机柜{cabinet_id}人数超限: {len(occupants)}/{max_occupancy}",
                        'level': 'warning',
                    })
        
        return violations
    
    def reset(self):
        """重置状态机"""
        self._state_history.clear()
        logger.info("状态机已重置")


__all__ = [
    'PersonState', 'GlobalAlarmLevel', 'PersonStateResult', 
    'GlobalStateResult', 'StateMachine'
]
