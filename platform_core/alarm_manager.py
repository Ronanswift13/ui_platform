# -*- coding: utf-8 -*-
"""
报警管理器
统一处理、存储、推送各插件产生的报警
"""

import time
import logging
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from datetime import datetime
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class AlarmLevel(Enum):
    """报警级别"""
    INFO = "info"
    ATTENTION = "attention"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


class AlarmStatus(Enum):
    """报警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"


@dataclass
class Alarm:
    """报警数据结构"""
    alarm_id: str
    level: AlarmLevel
    source_plugin: str
    device_id: str
    alarm_type: str
    message: str
    timestamp: float
    confidence: float = 0.0
    details: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    status: AlarmStatus = AlarmStatus.ACTIVE
    acknowledged_by: str = ""
    acknowledged_at: float = 0.0
    resolved_at: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['level'] = self.level.value
        data['status'] = self.status.value
        return data


class AlarmManager:
    """
    报警管理器
    
    功能:
    1. 接收各插件的报警
    2. 报警去重与合并
    3. 报警级别升级
    4. 报警存储与查询
    5. 报警推送
    """
    
    # 报警级别优先级
    LEVEL_PRIORITY = {
        AlarmLevel.INFO: 0,
        AlarmLevel.ATTENTION: 1,
        AlarmLevel.WARNING: 2,
        AlarmLevel.ALARM: 3,
        AlarmLevel.CRITICAL: 4
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化报警管理器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 报警存储
        self.active_alarms: Dict[str, Alarm] = {}
        self.alarm_history: List[Alarm] = []
        
        # 配置
        self.max_history = self.config.get('max_history', 10000)
        self.dedup_window = self.config.get('dedup_window', 300)  # 5分钟去重窗口
        self.escalation_threshold = self.config.get('escalation_threshold', 3)  # 升级阈值
        
        # 报警计数(用于升级)
        self.alarm_counts: Dict[str, List[float]] = defaultdict(list)
        
        # 推送处理器
        self.push_handlers: List[Callable[[Alarm], None]] = []
        
        # 报警队列(异步处理)
        self.alarm_queue: Queue = Queue()
        self._processing = False
        self._process_thread: Optional[threading.Thread] = None
        
        # 统计
        self.stats = defaultdict(int)
        
        self._alarm_counter = 0
        self._lock = threading.Lock()
        
        logger.info("AlarmManager初始化完成")
    
    def start(self):
        """启动异步处理"""
        if self._processing:
            return
        
        self._processing = True
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        logger.info("报警处理线程已启动")
    
    def stop(self):
        """停止异步处理"""
        self._processing = False
        if self._process_thread:
            self._process_thread.join(timeout=5)
        logger.info("报警处理线程已停止")
    
    def _process_loop(self):
        """报警处理循环"""
        while self._processing:
            try:
                alarm_data = self.alarm_queue.get(timeout=1)
                self._process_alarm(alarm_data)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"处理报警异常: {str(e)}")
    
    def _generate_alarm_id(self) -> str:
        """生成报警ID"""
        with self._lock:
            self._alarm_counter += 1
            return f"ALM-{int(time.time())}-{self._alarm_counter:06d}"
    
    def _get_dedup_key(self, alarm: Alarm) -> str:
        """生成去重键"""
        return f"{alarm.source_plugin}:{alarm.device_id}:{alarm.alarm_type}"
    
    def add_alarm(
        self,
        level: str,
        source_plugin: str,
        device_id: str,
        alarm_type: str,
        message: str,
        confidence: float = 0.0,
        details: Optional[Dict] = None,
        recommendations: Optional[List[str]] = None
    ) -> str:
        """
        添加报警
        
        Args:
            level: 报警级别
            source_plugin: 来源插件
            device_id: 设备ID
            alarm_type: 报警类型
            message: 报警消息
            confidence: 置信度
            details: 详细信息
            recommendations: 处置建议
            
        Returns:
            报警ID
        """
        # 转换级别
        try:
            alarm_level = AlarmLevel(level.lower())
        except ValueError:
            alarm_level = AlarmLevel.WARNING
        
        alarm = Alarm(
            alarm_id=self._generate_alarm_id(),
            level=alarm_level,
            source_plugin=source_plugin,
            device_id=device_id,
            alarm_type=alarm_type,
            message=message,
            timestamp=time.time(),
            confidence=confidence,
            details=details or {},
            recommendations=recommendations or []
        )
        
        # 异步处理
        self.alarm_queue.put(alarm)
        
        return alarm.alarm_id
    
    def _process_alarm(self, alarm: Alarm):
        """处理报警"""
        dedup_key = self._get_dedup_key(alarm)
        
        # 检查去重
        if self._should_deduplicate(alarm, dedup_key):
            self.stats['deduplicated'] += 1
            logger.debug(f"报警去重: {alarm.alarm_id}")
            return
        
        # 检查升级
        alarm = self._check_escalation(alarm, dedup_key)
        
        # 存储
        self.active_alarms[alarm.alarm_id] = alarm
        self.alarm_history.append(alarm)
        
        # 限制历史长度
        if len(self.alarm_history) > self.max_history:
            self.alarm_history = self.alarm_history[-self.max_history:]
        
        # 更新统计
        self.stats['total'] += 1
        self.stats[f'level_{alarm.level.value}'] += 1
        
        # 推送
        self._push_alarm(alarm)
        
        logger.info(f"新报警: {alarm.alarm_id} [{alarm.level.value}] {alarm.message}")
    
    def _should_deduplicate(self, alarm: Alarm, dedup_key: str) -> bool:
        """检查是否应去重"""
        current_time = time.time()
        
        # 检查活跃报警
        for active_alarm in self.active_alarms.values():
            if self._get_dedup_key(active_alarm) == dedup_key:
                # 时间窗口内的相同报警
                if current_time - active_alarm.timestamp < self.dedup_window:
                    return True
        
        return False
    
    def _check_escalation(self, alarm: Alarm, dedup_key: str) -> Alarm:
        """检查报警升级"""
        current_time = time.time()
        
        # 清理过期计数
        self.alarm_counts[dedup_key] = [
            t for t in self.alarm_counts[dedup_key]
            if current_time - t < 3600  # 1小时内
        ]
        
        # 添加当前时间
        self.alarm_counts[dedup_key].append(current_time)
        
        # 检查是否需要升级
        count = len(self.alarm_counts[dedup_key])
        if count >= self.escalation_threshold:
            current_priority = self.LEVEL_PRIORITY.get(alarm.level, 0)
            
            # 升级到下一级别
            for level, priority in self.LEVEL_PRIORITY.items():
                if priority == current_priority + 1:
                    alarm.level = level
                    alarm.message = f"[升级] {alarm.message}"
                    alarm.details['escalated'] = True
                    alarm.details['escalation_count'] = count
                    self.stats['escalated'] += 1
                    logger.warning(f"报警升级: {alarm.alarm_id} -> {level.value}")
                    break
        
        return alarm
    
    def _push_alarm(self, alarm: Alarm):
        """推送报警"""
        for handler in self.push_handlers:
            try:
                handler(alarm)
            except Exception as e:
                logger.error(f"推送报警失败: {str(e)}")
    
    def register_push_handler(self, handler: Callable[[Alarm], None]):
        """
        注册推送处理器
        
        Args:
            handler: 处理函数，接收Alarm参数
        """
        self.push_handlers.append(handler)
        logger.info(f"注册推送处理器: {handler.__name__}")
    
    def acknowledge_alarm(self, alarm_id: str, user: str) -> bool:
        """
        确认报警
        
        Args:
            alarm_id: 报警ID
            user: 确认用户
            
        Returns:
            是否成功
        """
        if alarm_id not in self.active_alarms:
            return False
        
        alarm = self.active_alarms[alarm_id]
        alarm.status = AlarmStatus.ACKNOWLEDGED
        alarm.acknowledged_by = user
        alarm.acknowledged_at = time.time()
        
        logger.info(f"报警已确认: {alarm_id} by {user}")
        return True
    
    def resolve_alarm(self, alarm_id: str) -> bool:
        """
        解除报警
        
        Args:
            alarm_id: 报警ID
            
        Returns:
            是否成功
        """
        if alarm_id not in self.active_alarms:
            return False
        
        alarm = self.active_alarms[alarm_id]
        alarm.status = AlarmStatus.RESOLVED
        alarm.resolved_at = time.time()
        
        # 从活跃列表移除
        del self.active_alarms[alarm_id]
        
        logger.info(f"报警已解除: {alarm_id}")
        return True
    
    def get_active_alarms(
        self,
        device_id: Optional[str] = None,
        level: Optional[str] = None,
        source_plugin: Optional[str] = None
    ) -> List[Dict]:
        """
        获取活跃报警
        
        Args:
            device_id: 过滤设备ID
            level: 过滤级别
            source_plugin: 过滤来源插件
            
        Returns:
            报警列表
        """
        alarms = list(self.active_alarms.values())
        
        if device_id:
            alarms = [a for a in alarms if a.device_id == device_id]
        
        if level:
            try:
                filter_level = AlarmLevel(level.lower())
                alarms = [a for a in alarms if a.level == filter_level]
            except ValueError:
                pass
        
        if source_plugin:
            alarms = [a for a in alarms if a.source_plugin == source_plugin]
        
        # 按级别和时间排序
        alarms.sort(key=lambda a: (
            -self.LEVEL_PRIORITY.get(a.level, 0),
            -a.timestamp
        ))
        
        return [a.to_dict() for a in alarms]
    
    def get_alarm_history(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        device_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        获取报警历史
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            device_id: 设备ID
            limit: 返回数量限制
            
        Returns:
            报警历史列表
        """
        alarms = self.alarm_history.copy()
        
        if start_time:
            alarms = [a for a in alarms if a.timestamp >= start_time]
        
        if end_time:
            alarms = [a for a in alarms if a.timestamp <= end_time]
        
        if device_id:
            alarms = [a for a in alarms if a.device_id == device_id]
        
        # 按时间倒序
        alarms.sort(key=lambda a: -a.timestamp)
        
        return [a.to_dict() for a in alarms[:limit]]
    
    def get_alarm_statistics(self, hours: int = 24) -> Dict:
        """
        获取报警统计
        
        Args:
            hours: 统计时间范围(小时)
            
        Returns:
            统计信息
        """
        cutoff = time.time() - hours * 3600
        recent_alarms = [a for a in self.alarm_history if a.timestamp >= cutoff]
        
        # 按级别统计
        level_counts = defaultdict(int)
        for alarm in recent_alarms:
            level_counts[alarm.level.value] += 1
        
        # 按设备统计
        device_counts = defaultdict(int)
        for alarm in recent_alarms:
            device_counts[alarm.device_id] += 1
        
        # 按类型统计
        type_counts = defaultdict(int)
        for alarm in recent_alarms:
            type_counts[alarm.alarm_type] += 1
        
        # Top设备
        top_devices = sorted(device_counts.items(), key=lambda x: -x[1])[:10]
        
        return {
            'time_range_hours': hours,
            'total_alarms': len(recent_alarms),
            'active_alarms': len(self.active_alarms),
            'by_level': dict(level_counts),
            'by_type': dict(type_counts),
            'top_devices': dict(top_devices),
            'overall_stats': dict(self.stats)
        }
    
    def export_alarms(self, filepath: str, format: str = 'json'):
        """
        导出报警数据
        
        Args:
            filepath: 导出文件路径
            format: 格式 (json/csv)
        """
        alarms = [a.to_dict() for a in self.alarm_history]
        
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(alarms, f, ensure_ascii=False, indent=2)
        elif format == 'csv':
            import csv
            if alarms:
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=alarms[0].keys())
                    writer.writeheader()
                    writer.writerows(alarms)
        
        logger.info(f"报警数据已导出: {filepath}")
    
    def clear_resolved(self):
        """清理已解除的报警"""
        self.alarm_history = [
            a for a in self.alarm_history
            if a.status != AlarmStatus.RESOLVED
        ]
        logger.info("已清理已解除的报警")
