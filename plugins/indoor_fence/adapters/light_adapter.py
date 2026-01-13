#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出适配器 - 报警灯和继电器控制
==========================================

基于全局告警级别控制三色灯:
- 绿灯: 系统正常
- 黄灯: 需要注意
- 红灯: 危险/报警

支持多种输出方式:
- GPIO直接控制
- RS485/Modbus
- HTTP API
- 模拟模式

作者: G组 | 版本: 2.0.0
"""

from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .base_adapter import BaseAdapter, AdapterStatus, AdapterConfig

logger = logging.getLogger(__name__)


class LightColor(str, Enum):
    """灯光颜色"""
    OFF = "off"
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    BLINK_GREEN = "blink_green"
    BLINK_YELLOW = "blink_yellow"
    BLINK_RED = "blink_red"


class LightMode(str, Enum):
    """灯光模式"""
    STATIC = "static"        # 静态常亮
    BLINK_SLOW = "slow"      # 慢闪 (1Hz)
    BLINK_FAST = "fast"      # 快闪 (2Hz)
    BREATH = "breath"        # 呼吸效果


@dataclass
class LightConfig(AdapterConfig):
    """灯光适配器配置"""
    output_type: str = "gpio"           # gpio, modbus, http, simulate
    
    # GPIO配置
    gpio_pin_green: int = 17
    gpio_pin_yellow: int = 27
    gpio_pin_red: int = 22
    gpio_active_high: bool = True
    
    # Modbus配置
    modbus_host: str = "127.0.0.1"
    modbus_port: int = 502
    modbus_unit_id: int = 1
    modbus_register_base: int = 0
    
    # HTTP配置
    http_url: str = "http://127.0.0.1:8080/api/light"
    http_timeout: float = 1.0
    
    # 灯光行为
    blink_interval_slow: float = 1.0    # 慢闪间隔 (秒)
    blink_interval_fast: float = 0.5    # 快闪间隔 (秒)


@dataclass
class LightState:
    """灯光状态"""
    color: LightColor = LightColor.OFF
    mode: LightMode = LightMode.STATIC
    
    # 各通道状态
    green_on: bool = False
    yellow_on: bool = False
    red_on: bool = False
    
    # 时间戳
    last_change: float = field(default_factory=time.time)
    
    # 关联的告警
    alarm_level: str = "normal"
    alarm_message: str = ""


class LightAdapter(BaseAdapter):
    """
    灯光输出适配器
    
    控制三色报警灯的开关和闪烁
    """
    
    def __init__(self, config: Optional[LightConfig] = None):
        self.config = config or LightConfig()
        super().__init__(self.config)
        
        self._state = LightState()
        self._driver: Optional[BaseLightDriver] = None
        
        logger.info(f"灯光适配器初始化: 类型={self.config.output_type}")
    
    def connect(self) -> bool:
        """连接到灯光输出设备"""
        self._status = AdapterStatus.CONNECTING
        self._stats['connect_attempts'] += 1
        
        try:
            # 根据配置选择驱动
            output_type = self.config.output_type.lower()
            
            if output_type == "gpio":
                self._driver = GPIOLightDriver(self.config)
            elif output_type == "modbus":
                self._driver = ModbusLightDriver(self.config)
            elif output_type == "http":
                self._driver = HTTPLightDriver(self.config)
            else:
                # 模拟模式
                self._driver = SimulatedLightDriver(self.config)
            
            # 尝试连接
            if self._driver.connect():
                self._status = AdapterStatus.CONNECTED
                # 初始化为绿灯
                self.set_color(LightColor.GREEN)
                logger.info(f"灯光适配器连接成功: {output_type}")
                return True
            else:
                self._status = AdapterStatus.SIMULATED
                self._driver = SimulatedLightDriver(self.config)
                self._driver.connect()
                logger.warning(f"灯光设备不可用, 使用模拟模式")
                return True
                
        except Exception as e:
            self._last_error = str(e)
            self._status = AdapterStatus.ERROR
            logger.error(f"灯光适配器连接失败: {e}")
            
            # 回退到模拟模式
            if self.config.simulate_if_unavailable:
                self._driver = SimulatedLightDriver(self.config)
                self._driver.connect()
                self._status = AdapterStatus.SIMULATED
                return True
            
            return False
    
    def disconnect(self) -> bool:
        """断开连接"""
        if self._driver:
            self._driver.set_all_off()
            self._driver.disconnect()
            self._driver = None
        
        self._status = AdapterStatus.DISCONNECTED
        return True
    
    def healthcheck(self) -> bool:
        """健康检查"""
        if not self._driver:
            return False
        return self._driver.healthcheck()
    
    def set_color(self, color: LightColor, mode: LightMode = LightMode.STATIC) -> bool:
        """
        设置灯光颜色
        
        Args:
            color: 灯光颜色
            mode: 灯光模式
            
        Returns:
            bool: 是否成功
        """
        if not self._driver:
            logger.warning("灯光驱动未初始化")
            return False
        
        try:
            # 更新状态
            self._state.color = color
            self._state.mode = mode
            self._state.last_change = time.time()
            
            # 重置所有灯
            self._state.green_on = False
            self._state.yellow_on = False
            self._state.red_on = False
            
            # 根据颜色设置
            if color == LightColor.GREEN or color == LightColor.BLINK_GREEN:
                self._state.green_on = True
            elif color == LightColor.YELLOW or color == LightColor.BLINK_YELLOW:
                self._state.yellow_on = True
            elif color == LightColor.RED or color == LightColor.BLINK_RED:
                self._state.red_on = True
            
            # 应用到驱动
            return self._driver.set_state(
                self._state.green_on,
                self._state.yellow_on,
                self._state.red_on
            )
            
        except Exception as e:
            logger.error(f"设置灯光颜色失败: {e}")
            return False
    
    def set_from_alarm_level(self, alarm_level: str, message: str = "") -> bool:
        """
        根据告警级别设置灯光
        
        Args:
            alarm_level: 告警级别 (green, yellow, red)
            message: 告警消息
            
        Returns:
            bool: 是否成功
        """
        self._state.alarm_level = alarm_level
        self._state.alarm_message = message
        
        color_map = {
            "green": LightColor.GREEN,
            "yellow": LightColor.YELLOW,
            "red": LightColor.RED,
        }
        
        color = color_map.get(alarm_level.lower(), LightColor.GREEN)
        
        # 红色告警使用快闪模式
        mode = LightMode.BLINK_FAST if color == LightColor.RED else LightMode.STATIC
        
        return self.set_color(color, mode)
    
    def get_state(self) -> LightState:
        """获取当前灯光状态"""
        return self._state
    
    def all_off(self) -> bool:
        """关闭所有灯"""
        return self.set_color(LightColor.OFF)


# ==============================================================================
# 灯光驱动基类
# ==============================================================================

class BaseLightDriver(ABC):
    """灯光驱动基类"""
    
    def __init__(self, config: LightConfig):
        self.config = config
        self._connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """连接设备"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        pass
    
    @abstractmethod
    def set_state(self, green: bool, yellow: bool, red: bool) -> bool:
        """设置三色灯状态"""
        pass
    
    def set_all_off(self) -> bool:
        """关闭所有灯"""
        return self.set_state(False, False, False)
    
    def healthcheck(self) -> bool:
        """健康检查"""
        return self._connected


class SimulatedLightDriver(BaseLightDriver):
    """模拟灯光驱动"""
    
    def __init__(self, config: LightConfig):
        super().__init__(config)
        self._green = False
        self._yellow = False
        self._red = False
    
    def connect(self) -> bool:
        self._connected = True
        logger.info("[模拟] 灯光驱动已连接")
        return True
    
    def disconnect(self) -> bool:
        self._connected = False
        logger.info("[模拟] 灯光驱动已断开")
        return True
    
    def set_state(self, green: bool, yellow: bool, red: bool) -> bool:
        self._green = green
        self._yellow = yellow
        self._red = red
        
        # 打印状态
        status = []
        if green: status.append("🟢绿")
        if yellow: status.append("🟡黄")
        if red: status.append("🔴红")
        if not status: status.append("⚫关")
        
        logger.debug(f"[模拟] 灯光状态: {' '.join(status)}")
        return True


class GPIOLightDriver(BaseLightDriver):
    """
    GPIO灯光驱动
    
    使用RPi.GPIO或lgpio控制GPIO引脚
    TODO: 真实GPIO控制需要在树莓派等设备上实现
    """
    
    def __init__(self, config: LightConfig):
        super().__init__(config)
        self._gpio = None
    
    def connect(self) -> bool:
        """连接GPIO"""
        try:
            # 尝试导入GPIO库
            # import RPi.GPIO as GPIO
            # self._gpio = GPIO
            # GPIO.setmode(GPIO.BCM)
            # GPIO.setup(self.config.gpio_pin_green, GPIO.OUT)
            # GPIO.setup(self.config.gpio_pin_yellow, GPIO.OUT)
            # GPIO.setup(self.config.gpio_pin_red, GPIO.OUT)
            
            # TODO: 真实实现
            logger.warning("GPIO驱动暂未实现, 回退到模拟模式")
            return False
            
        except Exception as e:
            logger.error(f"GPIO初始化失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        if self._gpio:
            # self._gpio.cleanup()
            pass
        self._connected = False
        return True
    
    def set_state(self, green: bool, yellow: bool, red: bool) -> bool:
        if not self._gpio:
            return False
        
        # active = self.config.gpio_active_high
        # self._gpio.output(self.config.gpio_pin_green, active if green else not active)
        # self._gpio.output(self.config.gpio_pin_yellow, active if yellow else not active)
        # self._gpio.output(self.config.gpio_pin_red, active if red else not active)
        
        return True


class ModbusLightDriver(BaseLightDriver):
    """
    Modbus灯光驱动
    
    通过Modbus TCP/RTU控制继电器
    TODO: 真实Modbus通信需要根据实际设备协议实现
    """
    
    def __init__(self, config: LightConfig):
        super().__init__(config)
        self._client = None
    
    def connect(self) -> bool:
        try:
            # from pymodbus.client import ModbusTcpClient
            # self._client = ModbusTcpClient(
            #     host=self.config.modbus_host,
            #     port=self.config.modbus_port,
            # )
            # return self._client.connect()
            
            logger.warning("Modbus驱动暂未实现, 回退到模拟模式")
            return False
            
        except Exception as e:
            logger.error(f"Modbus连接失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        if self._client:
            # self._client.close()
            pass
        self._connected = False
        return True
    
    def set_state(self, green: bool, yellow: bool, red: bool) -> bool:
        if not self._client:
            return False
        
        # 写入寄存器
        # values = [1 if green else 0, 1 if yellow else 0, 1 if red else 0]
        # self._client.write_registers(
        #     address=self.config.modbus_register_base,
        #     values=values,
        #     unit=self.config.modbus_unit_id
        # )
        
        return True


class HTTPLightDriver(BaseLightDriver):
    """
    HTTP API灯光驱动
    
    通过HTTP接口控制智能灯光设备
    TODO: 根据实际设备API实现
    """
    
    def __init__(self, config: LightConfig):
        super().__init__(config)
        self._session = None
    
    def connect(self) -> bool:
        try:
            import requests
            self._session = requests.Session()
            
            # 测试连接
            response = self._session.get(
                self.config.http_url,
                timeout=self.config.http_timeout
            )
            
            if response.status_code == 200:
                self._connected = True
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"HTTP灯光设备不可用: {e}")
            return False
    
    def disconnect(self) -> bool:
        if self._session:
            self._session.close()
        self._connected = False
        return True
    
    def set_state(self, green: bool, yellow: bool, red: bool) -> bool:
        if not self._session:
            return False
        
        try:
            payload = {
                "green": green,
                "yellow": yellow,
                "red": red,
            }
            
            response = self._session.post(
                self.config.http_url,
                json=payload,
                timeout=self.config.http_timeout
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"HTTP设置灯光失败: {e}")
            return False


__all__ = [
    'LightColor', 'LightMode', 'LightConfig', 'LightState', 'LightAdapter',
    'BaseLightDriver', 'SimulatedLightDriver', 'GPIOLightDriver',
    'ModbusLightDriver', 'HTTPLightDriver',
]
