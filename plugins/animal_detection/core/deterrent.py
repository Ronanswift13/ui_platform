#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测 - 声光驱离控制器
====================================

管理声光驱离设备的联动与反馈:
- 声音驱离 (AudioAdapter): 超声波/高频声
- 灯光驱离 (LightAdapter): 频闪/强光
- 策略配置: 按动物种类调整
- 驱离效果反馈: 跟踪是否成功驱离

作者: G组 | 版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .event_schema import AnimalClass, RiskLevel, ANIMAL_RISK_MAP
from .tracker import AnimalTrack

logger = logging.getLogger(__name__)


class DeterrentType(str, Enum):
    AUDIO = "audio"
    LIGHT = "light"
    COMBINED = "combined"


class DeterrentResult(str, Enum):
    SUCCESS = "success"        # 动物已离开
    PARTIAL = "partial"        # 动物移动但未离开
    FAILED = "failed"          # 动物未反应
    PENDING = "pending"        # 等待评估
    ESCALATED = "escalated"    # 已升级人工处理


@dataclass
class DeterrentAction:
    """驱离动作记录"""
    action_id: str
    track_id: str
    animal_class: str
    deterrent_type: str
    triggered_at: float
    duration_s: float = 5.0
    result: str = DeterrentResult.PENDING
    evaluated_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "track_id": self.track_id,
            "animal_class": self.animal_class,
            "deterrent_type": self.deterrent_type,
            "triggered_at": self.triggered_at,
            "duration_s": self.duration_s,
            "result": self.result,
            "evaluated_at": self.evaluated_at,
        }


# 按动物种类的驱离策略
DETERRENT_STRATEGY: Dict[str, Dict[str, Any]] = {
    AnimalClass.MOUSE: {
        "type": DeterrentType.AUDIO,
        "frequency_hz": 20000,  # 超声波
        "duration_s": 10.0,
        "light_flash": False,
    },
    AnimalClass.CAT: {
        "type": DeterrentType.COMBINED,
        "frequency_hz": 18000,
        "duration_s": 8.0,
        "light_flash": True,
    },
    AnimalClass.SNAKE: {
        "type": DeterrentType.COMBINED,
        "frequency_hz": 300,  # 低频振动
        "duration_s": 15.0,
        "light_flash": True,
    },
    AnimalClass.BIRD: {
        "type": DeterrentType.AUDIO,
        "frequency_hz": 15000,
        "duration_s": 5.0,
        "light_flash": False,
    },
    AnimalClass.DOG: {
        "type": DeterrentType.COMBINED,
        "frequency_hz": 16000,
        "duration_s": 10.0,
        "light_flash": True,
    },
}


class DeterrentController:
    """
    声光驱离控制器

    集成 LightAdapter 和 AudioAdapter，
    根据动物类别和停留时间触发驱离，
    并追踪驱离效果。
    """

    def __init__(
        self,
        enabled: bool = False,
        stay_time_threshold_s: float = 30.0,
        cooldown_s: float = 60.0,
        max_retries: int = 3,
        escalation_threshold: int = 3,
        audio_enabled: bool = True,
        light_enabled: bool = True,
        light_adapter: Optional[Any] = None,   # LightAdapter实例
        audio_adapter: Optional[Any] = None,    # AudioAdapter实例
    ):
        self.enabled = enabled
        self.stay_time_threshold_s = stay_time_threshold_s
        self.cooldown_s = cooldown_s
        self.max_retries = max_retries
        self.escalation_threshold = escalation_threshold
        self.audio_enabled = audio_enabled
        self.light_enabled = light_enabled

        self._light_adapter = light_adapter
        self._audio_adapter = audio_adapter

        # 驱离记录
        self._actions: List[DeterrentAction] = []
        self._last_trigger_time: Dict[str, float] = {}  # track_id -> last_trigger_time
        self._retry_count: Dict[str, int] = {}           # track_id -> retry count
        self._action_counter: int = 0

        # 回调
        self._on_escalation: Optional[Callable] = None

    def set_light_adapter(self, adapter: Any):
        """注入灯光适配器"""
        self._light_adapter = adapter

    def set_audio_adapter(self, adapter: Any):
        """注入音频适配器"""
        self._audio_adapter = adapter

    def set_escalation_callback(self, callback: Callable):
        """设置升级回调（通知人工处理）"""
        self._on_escalation = callback

    def evaluate_and_trigger(
        self, overstayed_tracks: List[AnimalTrack]
    ) -> List[DeterrentAction]:
        """
        评估超时轨迹并触发驱离

        Args:
            overstayed_tracks: 停留时间超过阈值的轨迹列表

        Returns:
            本轮触发的驱离动作列表
        """
        if not self.enabled:
            return []

        now = time.time()
        triggered_actions = []

        for track in overstayed_tracks:
            # 检查冷却时间
            last_trigger = self._last_trigger_time.get(track.track_id, 0)
            if now - last_trigger < self.cooldown_s:
                continue

            # 检查重试次数
            retries = self._retry_count.get(track.track_id, 0)
            if retries >= self.escalation_threshold:
                # 升级告警
                self._escalate(track)
                continue

            # 获取驱离策略
            strategy = DETERRENT_STRATEGY.get(
                track.animal_class,
                {"type": DeterrentType.COMBINED, "frequency_hz": 18000,
                 "duration_s": 8.0, "light_flash": True}
            )

            # 触发驱离
            action = self._trigger_deterrent(track, strategy, now)
            triggered_actions.append(action)

            self._last_trigger_time[track.track_id] = now
            self._retry_count[track.track_id] = retries + 1

        return triggered_actions

    def evaluate_results(self, tracks: List[AnimalTrack]):
        """
        评估驱离结果

        通过检查轨迹是否已消失或移出来判断驱离是否成功
        """
        now = time.time()
        active_track_ids = {t.track_id for t in tracks}

        for action in self._actions:
            if action.result != DeterrentResult.PENDING:
                continue

            # 检查是否已经过了评估时间
            if now - action.triggered_at < action.duration_s + 5.0:
                continue

            if action.track_id not in active_track_ids:
                action.result = DeterrentResult.SUCCESS
                action.evaluated_at = now
                logger.info(f"驱离成功: track={action.track_id} class={action.animal_class}")
            else:
                track = None
                for t in tracks:
                    if t.track_id == action.track_id:
                        track = t
                        break

                if track and track.age > 10:
                    action.result = DeterrentResult.PARTIAL
                else:
                    action.result = DeterrentResult.FAILED
                action.evaluated_at = now
                logger.info(
                    f"驱离{'部分有效' if action.result == DeterrentResult.PARTIAL else '失败'}: "
                    f"track={action.track_id}"
                )

    def _trigger_deterrent(
        self, track: AnimalTrack, strategy: Dict[str, Any], now: float
    ) -> DeterrentAction:
        """执行驱离动作"""
        self._action_counter += 1
        action = DeterrentAction(
            action_id=f"DA-{self._action_counter:06d}",
            track_id=track.track_id,
            animal_class=track.animal_class,
            deterrent_type=strategy["type"],
            triggered_at=now,
            duration_s=strategy.get("duration_s", 8.0),
        )

        # 触发声音驱离
        if self.audio_enabled and self._audio_adapter:
            try:
                freq = strategy.get("frequency_hz", 18000)
                duration = strategy.get("duration_s", 8.0)
                self._audio_adapter.play_deterrent(frequency_hz=freq, duration_s=duration)
                logger.info(
                    f"声音驱离已触发: track={track.track_id} freq={freq}Hz dur={duration}s"
                )
            except Exception as e:
                logger.error(f"声音驱离失败: {e}")

        # 触发灯光驱离
        if self.light_enabled and strategy.get("light_flash", False) and self._light_adapter:
            try:
                # 使用现有的LightAdapter接口
                from plugins.indoor_fence.adapters.light_adapter import LightColor, LightMode
                self._light_adapter.set_color(LightColor.RED, LightMode.BLINK_FAST)
                logger.info(f"灯光驱离已触发: track={track.track_id} 红色快闪")
            except Exception as e:
                logger.error(f"灯光驱离失败: {e}")

        self._actions.append(action)
        track.deterrent_count += 1
        return action

    def _escalate(self, track: AnimalTrack):
        """升级告警 - 驱离多次失败"""
        logger.warning(
            f"驱离升级: track={track.track_id} class={track.animal_class} "
            f"retries={self._retry_count.get(track.track_id, 0)} "
            f"-> 通知运维人员人工处理"
        )

        # 记录升级动作
        self._action_counter += 1
        action = DeterrentAction(
            action_id=f"DA-{self._action_counter:06d}",
            track_id=track.track_id,
            animal_class=track.animal_class,
            deterrent_type="escalation",
            triggered_at=time.time(),
            result=DeterrentResult.ESCALATED,
        )
        self._actions.append(action)

        if self._on_escalation:
            try:
                self._on_escalation(track, action)
            except Exception as e:
                logger.error(f"升级回调失败: {e}")

    def get_recent_actions(self, limit: int = 50) -> List[Dict]:
        """获取最近的驱离记录"""
        return [a.to_dict() for a in self._actions[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        """获取驱离统计"""
        total = len(self._actions)
        success = sum(1 for a in self._actions if a.result == DeterrentResult.SUCCESS)
        failed = sum(1 for a in self._actions if a.result == DeterrentResult.FAILED)
        escalated = sum(1 for a in self._actions if a.result == DeterrentResult.ESCALATED)

        return {
            "enabled": self.enabled,
            "total_actions": total,
            "success_count": success,
            "failed_count": failed,
            "escalated_count": escalated,
            "success_rate": round(success / total, 4) if total > 0 else 0.0,
            "stay_threshold_s": self.stay_time_threshold_s,
            "cooldown_s": self.cooldown_s,
        }

    def reset(self):
        """重置"""
        self._actions.clear()
        self._last_trigger_time.clear()
        self._retry_count.clear()
        self._action_counter = 0
