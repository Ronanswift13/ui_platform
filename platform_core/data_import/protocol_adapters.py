"""
工业协议适配器
================

支持变电站常用的工业通信协议:
- Modbus RTU/TCP
- OPC UA
- MQTT
- HTTP/REST API
- IEC 61850 (MMS)

作者: 破夜绘明团队
版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
import json
import struct
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from queue import Queue
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 协议适配器基类
# =============================================================================
@dataclass
class ProtocolConfig:
    """协议配置基类"""
    host: str = "127.0.0.1"
    port: int = 502
    timeout_ms: int = 5000
    retry_count: int = 3
    reconnect_interval_ms: int = 5000


class ProtocolAdapter(ABC):
    """协议适配器基类"""

    def __init__(self, config: ProtocolConfig):
        self.config = config
        self._connected = False
        self._last_error = ""
        self._client = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def read(self, address: str, **kwargs) -> Any:
        pass

    @abstractmethod
    def write(self, address: str, value: Any, **kwargs) -> bool:
        pass


# =============================================================================
# Modbus 适配器
# =============================================================================
@dataclass
class ModbusConfig(ProtocolConfig):
    """Modbus配置"""
    mode: str = "tcp"  # tcp, rtu
    slave_id: int = 1
    # RTU专用
    serial_port: str = ""
    baudrate: int = 9600
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1


class ModbusAdapter(ProtocolAdapter):
    """
    Modbus协议适配器

    支持Modbus TCP和Modbus RTU
    常用于PLC、传感器、仪表等设备
    """

    def __init__(self, config: ModbusConfig):
        super().__init__(config)
        self.config: ModbusConfig = config

    def connect(self) -> bool:
        """连接Modbus设备"""
        try:
            if self.config.mode == "tcp":
                return self._connect_tcp()
            else:
                return self._connect_rtu()
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Modbus连接失败: {e}")
            return False

    def _connect_tcp(self) -> bool:
        """TCP连接"""
        try:
            from pymodbus.client import ModbusTcpClient

            self._client = ModbusTcpClient(
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.timeout_ms / 1000
            )

            if self._client.connect():
                self._connected = True
                logger.info(f"Modbus TCP连接成功: {self.config.host}:{self.config.port}")
                return True
            else:
                raise Exception("连接失败")

        except ImportError:
            logger.error("pymodbus未安装: pip install pymodbus")
            return False

    def _connect_rtu(self) -> bool:
        """RTU连接"""
        try:
            from pymodbus.client import ModbusSerialClient

            self._client = ModbusSerialClient(
                port=self.config.serial_port,
                baudrate=self.config.baudrate,
                bytesize=self.config.bytesize,
                parity=self.config.parity,
                stopbits=self.config.stopbits,
                timeout=self.config.timeout_ms / 1000
            )

            if self._client.connect():
                self._connected = True
                logger.info(f"Modbus RTU连接成功: {self.config.serial_port}")
                return True
            else:
                raise Exception("串口连接失败")

        except ImportError:
            logger.error("pymodbus未安装: pip install pymodbus")
            return False

    def disconnect(self) -> None:
        """断开连接"""
        if self._client:
            try:
                self._client.close()
            except:
                pass
            self._client = None
        self._connected = False

    def read(self, address: str, **kwargs) -> Any:
        """
        读取Modbus寄存器

        Args:
            address: 寄存器地址，格式: "HR:100" (保持寄存器100)
                    HR = 保持寄存器 (3x)
                    IR = 输入寄存器 (4x)
                    CO = 线圈 (0x)
                    DI = 离散输入 (1x)
            count: 读取数量
            slave: 从站地址
            data_type: 数据类型 (int16, int32, float32)

        Returns:
            读取的值
        """
        if not self._connected:
            raise Exception("未连接")

        count = kwargs.get("count", 1)
        slave = kwargs.get("slave", self.config.slave_id)
        data_type = kwargs.get("data_type", "int16")

        # 解析地址
        reg_type, reg_addr = self._parse_address(address)

        # 读取寄存器
        if reg_type == "HR":
            result = self._client.read_holding_registers(reg_addr, count, slave=slave)
        elif reg_type == "IR":
            result = self._client.read_input_registers(reg_addr, count, slave=slave)
        elif reg_type == "CO":
            result = self._client.read_coils(reg_addr, count, slave=slave)
        elif reg_type == "DI":
            result = self._client.read_discrete_inputs(reg_addr, count, slave=slave)
        else:
            raise ValueError(f"未知寄存器类型: {reg_type}")

        if result.isError():
            raise Exception(f"读取错误: {result}")

        # 解析数据
        if reg_type in ("CO", "DI"):
            return result.bits[:count]
        else:
            return self._decode_registers(result.registers, data_type)

    def write(self, address: str, value: Any, **kwargs) -> bool:
        """
        写入Modbus寄存器

        Args:
            address: 寄存器地址
            value: 要写入的值
            slave: 从站地址
            data_type: 数据类型

        Returns:
            是否成功
        """
        if not self._connected:
            raise Exception("未连接")

        slave = kwargs.get("slave", self.config.slave_id)
        data_type = kwargs.get("data_type", "int16")

        reg_type, reg_addr = self._parse_address(address)

        if reg_type == "HR":
            # 编码值
            registers = self._encode_value(value, data_type)
            result = self._client.write_registers(reg_addr, registers, slave=slave)
        elif reg_type == "CO":
            result = self._client.write_coil(reg_addr, bool(value), slave=slave)
        else:
            raise ValueError(f"不支持写入的寄存器类型: {reg_type}")

        if result.isError():
            raise Exception(f"写入错误: {result}")

        return True

    def _parse_address(self, address: str) -> Tuple[str, int]:
        """解析地址"""
        if ":" in address:
            reg_type, addr_str = address.split(":")
            return reg_type.upper(), int(addr_str)
        else:
            return "HR", int(address)

    def _decode_registers(self, registers: List[int], data_type: str) -> Any:
        """解码寄存器值"""
        if data_type == "int16":
            return registers[0]
        elif data_type == "uint16":
            return registers[0]
        elif data_type == "int32":
            return struct.unpack('>i', struct.pack('>HH', registers[0], registers[1]))[0]
        elif data_type == "uint32":
            return struct.unpack('>I', struct.pack('>HH', registers[0], registers[1]))[0]
        elif data_type == "float32":
            return struct.unpack('>f', struct.pack('>HH', registers[0], registers[1]))[0]
        elif data_type == "float64":
            return struct.unpack('>d', struct.pack('>HHHH', *registers[:4]))[0]
        else:
            return registers

    def _encode_value(self, value: Any, data_type: str) -> List[int]:
        """编码值为寄存器"""
        if data_type == "int16":
            return [int(value) & 0xFFFF]
        elif data_type == "uint16":
            return [int(value) & 0xFFFF]
        elif data_type == "int32":
            packed = struct.pack('>i', int(value))
            return list(struct.unpack('>HH', packed))
        elif data_type == "uint32":
            packed = struct.pack('>I', int(value))
            return list(struct.unpack('>HH', packed))
        elif data_type == "float32":
            packed = struct.pack('>f', float(value))
            return list(struct.unpack('>HH', packed))
        else:
            return [int(value)]

    # 便捷方法
    def read_holding_registers(self, address: int, count: int = 1) -> List[int]:
        """读取保持寄存器"""
        return self.read(f"HR:{address}", count=count)

    def read_input_registers(self, address: int, count: int = 1) -> List[int]:
        """读取输入寄存器"""
        return self.read(f"IR:{address}", count=count)

    def read_float(self, address: int) -> float:
        """读取浮点数"""
        return self.read(f"HR:{address}", count=2, data_type="float32")

    def write_float(self, address: int, value: float) -> bool:
        """写入浮点数"""
        return self.write(f"HR:{address}", value, data_type="float32")


# =============================================================================
# OPC UA 适配器
# =============================================================================
@dataclass
class OPCUAConfig(ProtocolConfig):
    """OPC UA配置"""
    endpoint: str = ""
    security_mode: str = "None"  # None, Sign, SignAndEncrypt
    security_policy: str = ""
    username: str = ""
    password: str = ""
    certificate_path: str = ""
    private_key_path: str = ""


class OPCUAAdapter(ProtocolAdapter):
    """
    OPC UA协议适配器

    用于与支持OPC UA的设备通信
    常见于工业控制系统、SCADA系统
    """

    def __init__(self, config: OPCUAConfig):
        super().__init__(config)
        self.config: OPCUAConfig = config
        self._subscription = None
        self._monitored_items: Dict[str, Any] = {}

    def connect(self) -> bool:
        """连接OPC UA服务器"""
        try:
            from opcua import Client, ua

            endpoint = self.config.endpoint or f"opc.tcp://{self.config.host}:{self.config.port}"

            self._client = Client(endpoint)
            self._client.set_security_string(
                f"{self.config.security_policy},{self.config.security_mode}"
            )

            if self.config.username:
                self._client.set_user(self.config.username)
                self._client.set_password(self.config.password)

            self._client.connect()
            self._connected = True
            logger.info(f"OPC UA连接成功: {endpoint}")
            return True

        except ImportError:
            logger.error("opcua未安装: pip install opcua")
            return False
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"OPC UA连接失败: {e}")
            return False

    def disconnect(self) -> None:
        """断开连接"""
        if self._subscription:
            try:
                self._subscription.delete()
            except:
                pass
            self._subscription = None

        if self._client:
            try:
                self._client.disconnect()
            except:
                pass
            self._client = None

        self._connected = False
        self._monitored_items.clear()

    def read(self, address: str, **kwargs) -> Any:
        """
        读取OPC UA节点

        Args:
            address: 节点ID，格式: "ns=2;s=Channel1.Device1.Tag1"
                    或 "ns=2;i=1001"

        Returns:
            节点值
        """
        if not self._connected:
            raise Exception("未连接")

        node = self._client.get_node(address)
        value = node.get_value()

        return value

    def write(self, address: str, value: Any, **kwargs) -> bool:
        """写入OPC UA节点"""
        if not self._connected:
            raise Exception("未连接")

        node = self._client.get_node(address)
        node.set_value(value)

        return True

    def browse(self, node_id: str = "i=85") -> List[Dict]:
        """浏览节点"""
        if not self._connected:
            raise Exception("未连接")

        node = self._client.get_node(node_id)
        children = node.get_children()

        return [
            {
                "node_id": str(child.nodeid),
                "name": child.get_browse_name().Name,
                "node_class": str(child.get_node_class()),
            }
            for child in children
        ]

    def subscribe(self, node_id: str, callback: Callable[[str, Any], None],
                  interval_ms: int = 1000) -> bool:
        """
        订阅节点变化

        Args:
            node_id: 节点ID
            callback: 回调函数 (node_id, value) -> None
            interval_ms: 采样间隔

        Returns:
            是否成功
        """
        if not self._connected:
            raise Exception("未连接")

        try:
            from opcua import ua

            # 创建订阅
            if self._subscription is None:
                handler = _OPCUASubscriptionHandler(callback)
                self._subscription = self._client.create_subscription(interval_ms, handler)

            # 添加监控项
            node = self._client.get_node(node_id)
            handle = self._subscription.subscribe_data_change(node)
            self._monitored_items[node_id] = handle

            logger.info(f"订阅节点: {node_id}")
            return True

        except Exception as e:
            logger.error(f"订阅失败: {e}")
            return False

    def unsubscribe(self, node_id: str) -> bool:
        """取消订阅"""
        if node_id in self._monitored_items:
            try:
                self._subscription.unsubscribe(self._monitored_items[node_id])
                del self._monitored_items[node_id]
                return True
            except:
                pass
        return False


class _OPCUASubscriptionHandler:
    """OPC UA订阅处理器"""

    def __init__(self, callback: Callable[[str, Any], None]):
        self.callback = callback

    def datachange_notification(self, node, val, data):
        try:
            self.callback(str(node.nodeid), val)
        except Exception as e:
            logger.error(f"订阅回调异常: {e}")


# =============================================================================
# MQTT 适配器
# =============================================================================
@dataclass
class MQTTConfig(ProtocolConfig):
    """MQTT配置"""
    port: int = 1883
    client_id: str = ""
    username: str = ""
    password: str = ""
    use_tls: bool = False
    ca_certs: str = ""
    keepalive: int = 60
    qos: int = 1


class MQTTAdapter(ProtocolAdapter):
    """
    MQTT协议适配器

    用于IoT设备通信，支持发布/订阅模式
    """

    def __init__(self, config: MQTTConfig):
        super().__init__(config)
        self.config: MQTTConfig = config
        self._subscriptions: Dict[str, Callable] = {}
        self._message_queue: Queue = Queue(maxsize=1000)

    def connect(self) -> bool:
        """连接MQTT Broker"""
        try:
            import paho.mqtt.client as mqtt

            client_id = self.config.client_id or f"sensor_adapter_{time.time()}"
            self._client = mqtt.Client(client_id=client_id)

            # 设置回调
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            self._client.on_disconnect = self._on_disconnect

            # 认证
            if self.config.username:
                self._client.username_pw_set(self.config.username, self.config.password)

            # TLS
            if self.config.use_tls:
                self._client.tls_set(ca_certs=self.config.ca_certs)

            # 连接
            self._client.connect(
                self.config.host,
                self.config.port,
                keepalive=self.config.keepalive
            )

            # 启动网络循环
            self._client.loop_start()

            # 等待连接完成
            timeout = self.config.timeout_ms / 1000
            start = time.time()
            while not self._connected and time.time() - start < timeout:
                time.sleep(0.1)

            if not self._connected:
                raise Exception("连接超时")

            logger.info(f"MQTT连接成功: {self.config.host}:{self.config.port}")
            return True

        except ImportError:
            logger.error("paho-mqtt未安装: pip install paho-mqtt")
            return False
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"MQTT连接失败: {e}")
            return False

    def disconnect(self) -> None:
        """断开连接"""
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except:
                pass
            self._client = None

        self._connected = False
        self._subscriptions.clear()

    def read(self, address: str, **kwargs) -> Any:
        """
        读取MQTT消息 (从队列获取)

        Args:
            address: 主题名称
            timeout: 超时时间(秒)

        Returns:
            消息内容
        """
        timeout = kwargs.get("timeout", 5.0)

        try:
            message = self._message_queue.get(timeout=timeout)
            if message["topic"] == address or address == "#":
                return message["payload"]
        except:
            pass

        return None

    def write(self, address: str, value: Any, **kwargs) -> bool:
        """
        发布MQTT消息

        Args:
            address: 主题名称
            value: 消息内容
            qos: QoS级别
            retain: 是否保留

        Returns:
            是否成功
        """
        if not self._connected:
            raise Exception("未连接")

        qos = kwargs.get("qos", self.config.qos)
        retain = kwargs.get("retain", False)

        # 序列化消息
        if isinstance(value, (dict, list)):
            payload = json.dumps(value)
        else:
            payload = str(value)

        result = self._client.publish(address, payload, qos=qos, retain=retain)

        return result.rc == 0

    def subscribe(self, topic: str, callback: Callable[[str, Any], None],
                  qos: int = None) -> bool:
        """
        订阅主题

        Args:
            topic: 主题名称
            callback: 回调函数 (topic, payload) -> None
            qos: QoS级别

        Returns:
            是否成功
        """
        if not self._connected:
            raise Exception("未连接")

        qos = qos if qos is not None else self.config.qos

        self._subscriptions[topic] = callback
        result = self._client.subscribe(topic, qos=qos)

        if result[0] == 0:
            logger.info(f"订阅主题: {topic}")
            return True
        else:
            logger.error(f"订阅失败: {topic}")
            return False

    def unsubscribe(self, topic: str) -> bool:
        """取消订阅"""
        if topic in self._subscriptions:
            self._client.unsubscribe(topic)
            del self._subscriptions[topic]
            return True
        return False

    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self._connected = True
            logger.info("MQTT已连接")

            # 重新订阅
            for topic in self._subscriptions:
                client.subscribe(topic, qos=self.config.qos)
        else:
            logger.error(f"MQTT连接失败: rc={rc}")

    def _on_message(self, client, userdata, msg):
        """消息回调"""
        try:
            # 解析消息
            try:
                payload = json.loads(msg.payload.decode())
            except:
                payload = msg.payload.decode()

            # 加入队列
            try:
                self._message_queue.put_nowait({
                    "topic": msg.topic,
                    "payload": payload,
                    "timestamp": time.time()
                })
            except:
                pass

            # 触发回调
            for topic, callback in self._subscriptions.items():
                if self._topic_matches(topic, msg.topic):
                    try:
                        callback(msg.topic, payload)
                    except Exception as e:
                        logger.error(f"MQTT回调异常: {e}")

        except Exception as e:
            logger.error(f"消息处理异常: {e}")

    def _on_disconnect(self, client, userdata, rc):
        """断开回调"""
        self._connected = False
        if rc != 0:
            logger.warning(f"MQTT意外断开: rc={rc}")

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """检查主题是否匹配"""
        if pattern == "#":
            return True

        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        for i, p in enumerate(pattern_parts):
            if p == "#":
                return True
            if p == "+":
                continue
            if i >= len(topic_parts) or p != topic_parts[i]:
                return False

        return len(pattern_parts) == len(topic_parts)


# =============================================================================
# HTTP 适配器
# =============================================================================
@dataclass
class HTTPConfig(ProtocolConfig):
    """HTTP配置"""
    port: int = 80
    base_url: str = ""
    use_https: bool = False
    api_key: str = ""
    headers: Dict[str, str] = field(default_factory=dict)


class HTTPAdapter(ProtocolAdapter):
    """
    HTTP/REST API适配器

    用于与Web API通信
    """

    def __init__(self, config: HTTPConfig):
        super().__init__(config)
        self.config: HTTPConfig = config
        self._session = None

    def connect(self) -> bool:
        """初始化HTTP会话"""
        try:
            import requests

            self._session = requests.Session()

            # 设置默认头
            self._session.headers.update(self.config.headers)
            if self.config.api_key:
                self._session.headers["Authorization"] = f"Bearer {self.config.api_key}"

            # 测试连接
            base_url = self._get_base_url()
            response = self._session.get(base_url, timeout=self.config.timeout_ms / 1000)

            self._connected = True
            logger.info(f"HTTP连接成功: {base_url}")
            return True

        except ImportError:
            logger.error("requests未安装: pip install requests")
            return False
        except Exception as e:
            # HTTP适配器允许连接失败后继续使用
            self._connected = True
            logger.warning(f"HTTP初始化警告: {e}")
            return True

    def disconnect(self) -> None:
        """关闭会话"""
        if self._session:
            self._session.close()
            self._session = None
        self._connected = False

    def read(self, address: str, **kwargs) -> Any:
        """
        GET请求

        Args:
            address: API路径
            params: 查询参数

        Returns:
            响应数据
        """
        if not self._session:
            raise Exception("未初始化")

        url = self._get_url(address)
        params = kwargs.get("params", {})

        response = self._session.get(
            url,
            params=params,
            timeout=self.config.timeout_ms / 1000
        )
        response.raise_for_status()

        return response.json()

    def write(self, address: str, value: Any, **kwargs) -> bool:
        """
        POST请求

        Args:
            address: API路径
            value: 请求体数据
            method: HTTP方法 (POST, PUT, PATCH)

        Returns:
            是否成功
        """
        if not self._session:
            raise Exception("未初始化")

        url = self._get_url(address)
        method = kwargs.get("method", "POST").upper()

        if method == "POST":
            response = self._session.post(url, json=value, timeout=self.config.timeout_ms / 1000)
        elif method == "PUT":
            response = self._session.put(url, json=value, timeout=self.config.timeout_ms / 1000)
        elif method == "PATCH":
            response = self._session.patch(url, json=value, timeout=self.config.timeout_ms / 1000)
        else:
            raise ValueError(f"不支持的方法: {method}")

        response.raise_for_status()
        return True

    def _get_base_url(self) -> str:
        """获取基础URL"""
        if self.config.base_url:
            return self.config.base_url

        protocol = "https" if self.config.use_https else "http"
        return f"{protocol}://{self.config.host}:{self.config.port}"

    def _get_url(self, path: str) -> str:
        """获取完整URL"""
        base = self._get_base_url()
        if path.startswith("/"):
            return f"{base}{path}"
        else:
            return f"{base}/{path}"


# =============================================================================
# 协议工厂
# =============================================================================
class ProtocolFactory:
    """协议工厂"""

    _registry = {
        "modbus": (ModbusAdapter, ModbusConfig),
        "opcua": (OPCUAAdapter, OPCUAConfig),
        "mqtt": (MQTTAdapter, MQTTConfig),
        "http": (HTTPAdapter, HTTPConfig),
    }

    @classmethod
    def create(cls, protocol: str, **kwargs) -> ProtocolAdapter:
        """创建协议适配器"""
        if protocol not in cls._registry:
            raise ValueError(f"不支持的协议: {protocol}")

        adapter_class, config_class = cls._registry[protocol]
        config = config_class(**kwargs)

        return adapter_class(config)

    @classmethod
    def register(cls, protocol: str, adapter_class: type, config_class: type) -> None:
        """注册协议"""
        cls._registry[protocol] = (adapter_class, config_class)
