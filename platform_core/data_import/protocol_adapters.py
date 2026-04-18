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
# IEC 60870-5-104 适配器
# =============================================================================
@dataclass
class IEC104Config(ProtocolConfig):
    """IEC 104 配置"""
    port: int = 2404
    # 站地址 (ASDU Common Address)
    common_address: int = 1
    # 原点地址 (Originator Address)
    originator_address: int = 0
    # k / w 参数 (滑动窗口)
    k: int = 12
    w: int = 8
    # t1/t2/t3 超时 (秒)
    t1_timeout: float = 15.0
    t2_timeout: float = 10.0
    t3_timeout: float = 20.0
    # 总召唤周期 (秒, 0=不自动)
    general_interrogation_interval: float = 0.0
    # 时钟同步周期 (秒, 0=不自动)
    clock_sync_interval: float = 0.0


class IEC104Adapter(ProtocolAdapter):
    """
    IEC 60870-5-104 协议适配器

    用于与保护信息子站、远动装置、故障录波器等二次设备通信。
    支持:
    - 总召唤 (General Interrogation)
    - SOE (带时标的单/双点遥信)
    - 遥测 (Measured Values)
    - 遥控 (Single/Double Command)
    - 时钟同步

    依赖: pip install iec104-python   (或 lib60870-python)
    无依赖时以模拟模式运行, 不阻塞平台启动。
    """

    def __init__(self, config: IEC104Config):
        super().__init__(config)
        self.config: IEC104Config = config
        self._subscriptions: Dict[str, Callable] = {}
        self._soe_callbacks: List[Callable] = []
        self._gi_callbacks: List[Callable] = []
        self._message_queue: Queue = Queue(maxsize=10000)
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        self._gi_thread: Optional[threading.Thread] = None
        # 统计
        self._stats = {"rx_asdu": 0, "tx_asdu": 0, "gi_count": 0, "soe_count": 0}
        # 缓存: IOA -> 最新值
        self._point_cache: Dict[int, Dict[str, Any]] = {}
        # 模拟模式标志
        self._simulated = False

    def connect(self) -> bool:
        """连接 IEC 104 主站 -> 被控站"""
        try:
            import iec104  # type: ignore
            self._client = iec104.IEC104Client(
                host=self.config.host,
                port=self.config.port,
                originator_address=self.config.originator_address,
                k=self.config.k,
                w=self.config.w,
                t1=self.config.t1_timeout,
                t2=self.config.t2_timeout,
                t3=self.config.t3_timeout,
            )
            self._client.on_receive = self._on_asdu_received
            self._client.connect()
            self._connected = True
            self._running = True
            self._start_background_threads()
            logger.info(f"IEC104 连接成功: {self.config.host}:{self.config.port}")
            return True
        except ImportError:
            logger.warning(
                "iec104-python 未安装, 以模拟模式运行 (pip install iec104-python)"
            )
            return self._connect_simulated()
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"IEC104 连接失败: {e}")
            return False

    def _connect_simulated(self) -> bool:
        """模拟模式: 不需要真实连接即可调试全链路"""
        self._simulated = True
        self._connected = True
        self._running = True
        logger.info("IEC104 以模拟模式运行")
        return True

    def disconnect(self) -> None:
        self._running = False
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=3)
        if self._gi_thread and self._gi_thread.is_alive():
            self._gi_thread.join(timeout=3)
        if self._client and not self._simulated:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
        self._connected = False
        self._subscriptions.clear()

    def read(self, address: str, **kwargs) -> Any:
        """
        读取 IEC 104 信息对象

        Args:
            address: IOA 地址, 格式: "1001" 或 "SP:1001" (单点), "DP:2001" (双点),
                     "MV:4001" (遥测), "SOE:1001" (带时标单点)
        Returns:
            缓存中的最新值, 若无缓存返回 None
        """
        ioa = self._parse_ioa(address)
        cached = self._point_cache.get(ioa)
        if cached is not None:
            return cached
        # 如果缓存为空, 尝试发总召
        if not self._simulated and self._client:
            self.general_interrogation()
        return self._point_cache.get(ioa)

    def write(self, address: str, value: Any, **kwargs) -> bool:
        """
        发送遥控命令 (Single/Double Command)

        Args:
            address: IOA, 格式同 read
            value: bool(单点) 或 int(双点 0/1/2/3)
            qualifier: 命令限定词, 默认 0
            select: 是否为选择 (SBO), 默认 False
        """
        if self._simulated:
            logger.info(f"IEC104 模拟遥控: IOA={address}, value={value}")
            return True

        if not self._connected:
            raise Exception("IEC104 未连接")

        ioa = self._parse_ioa(address)
        qualifier = kwargs.get("qualifier", 0)
        select = kwargs.get("select", False)

        try:
            self._client.send_command(
                common_address=self.config.common_address,
                ioa=ioa,
                value=value,
                qualifier=qualifier,
                select=select,
            )
            self._stats["tx_asdu"] += 1
            return True
        except Exception as e:
            logger.error(f"IEC104 遥控失败: {e}")
            return False

    # ---------- 总召唤 ----------

    def general_interrogation(self) -> bool:
        """发送总召唤"""
        if self._simulated:
            logger.debug("IEC104 模拟总召唤")
            self._stats["gi_count"] += 1
            for cb in self._gi_callbacks:
                try:
                    cb(self._point_cache)
                except Exception as e:
                    logger.error(f"GI 回调异常: {e}")
            return True

        if not self._connected:
            return False
        try:
            self._client.send_general_interrogation(self.config.common_address)
            self._stats["gi_count"] += 1
            return True
        except Exception as e:
            logger.error(f"总召唤失败: {e}")
            return False

    # ---------- 订阅 ----------

    def subscribe(self, address: str, callback: Callable[[str, Any], None],
                  interval_ms: int = 0) -> bool:
        """
        订阅信息对象值变化

        当 IOA 对应值发生变化 (遥信变位/SOE) 时触发 callback(address, value_dict)。
        """
        self._subscriptions[address] = callback
        logger.info(f"IEC104 订阅: {address}")
        return True

    def subscribe_soe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """订阅所有 SOE 事件 (带时标单/双点)"""
        self._soe_callbacks.append(callback)

    def subscribe_gi(self, callback: Callable[[Dict[int, Dict]], None]) -> None:
        """订阅总召唤完成回调"""
        self._gi_callbacks.append(callback)

    def unsubscribe(self, address: str) -> bool:
        if address in self._subscriptions:
            del self._subscriptions[address]
            return True
        return False

    # ---------- 模拟数据注入 ----------

    def inject_simulated_point(self, ioa: int, value: Any, quality: int = 0,
                               timestamp: str = "", type_id: str = "SP") -> None:
        """
        注入模拟数据点 (用于无真实设备时的全链路调试)

        Args:
            ioa: 信息对象地址
            value: 值
            quality: 品质描述词
            timestamp: ISO8601 时间戳
            type_id: SP(单点)/DP(双点)/MV(遥测)/SOE(带时标)
        """
        import time as _time
        ts = timestamp or __import__('datetime').datetime.now().isoformat()
        point_data = {
            "ioa": ioa,
            "value": value,
            "quality": quality,
            "timestamp": ts,
            "type_id": type_id,
            "common_address": self.config.common_address,
        }
        old = self._point_cache.get(ioa)
        self._point_cache[ioa] = point_data

        # 值变化 -> 触发订阅
        addr_key = str(ioa)
        if addr_key in self._subscriptions:
            try:
                self._subscriptions[addr_key](addr_key, point_data)
            except Exception as e:
                logger.error(f"IEC104 订阅回调异常: {e}")

        # SOE 回调
        if type_id == "SOE":
            self._stats["soe_count"] += 1
            for cb in self._soe_callbacks:
                try:
                    cb(point_data)
                except Exception as e:
                    logger.error(f"SOE 回调异常: {e}")

    # ---------- 内部方法 ----------

    def _on_asdu_received(self, asdu: Any) -> None:
        """接收 ASDU 回调 (真实连接模式)"""
        self._stats["rx_asdu"] += 1
        try:
            for io in asdu.information_objects:
                ioa = io.ioa
                value = io.value
                quality = getattr(io, "quality", 0)
                ts_raw = getattr(io, "timestamp", None)
                ts = ts_raw.isoformat() if ts_raw else ""
                type_id = self._classify_type_id(asdu.type_id)

                point_data = {
                    "ioa": ioa,
                    "value": value,
                    "quality": quality,
                    "timestamp": ts,
                    "type_id": type_id,
                    "common_address": asdu.common_address,
                }
                old = self._point_cache.get(ioa)
                self._point_cache[ioa] = point_data

                # 值变化 -> 订阅回调
                addr_key = str(ioa)
                if addr_key in self._subscriptions:
                    self._subscriptions[addr_key](addr_key, point_data)

                # SOE
                if type_id == "SOE":
                    self._stats["soe_count"] += 1
                    for cb in self._soe_callbacks:
                        cb(point_data)

        except Exception as e:
            logger.error(f"ASDU 解析异常: {e}")

    def _classify_type_id(self, raw_type_id: int) -> str:
        """将 IEC104 TypeID 数值分类为简写"""
        # M_SP_NA_1(1), M_SP_TA_1(2), M_SP_TB_1(30) -> 单点
        # M_DP_NA_1(3), M_DP_TA_1(4), M_DP_TB_1(31) -> 双点
        # M_ME_NA_1(9)..M_ME_TF_1(36) -> 遥测
        # 带 CP56Time2a 的为 SOE
        soe_types = {2, 4, 30, 31, 33, 34, 35, 36}
        sp_types = {1, 2, 30}
        dp_types = {3, 4, 31}
        if raw_type_id in soe_types:
            return "SOE"
        if raw_type_id in sp_types:
            return "SP"
        if raw_type_id in dp_types:
            return "DP"
        return "MV"

    def _parse_ioa(self, address: str) -> int:
        """解析 IOA 地址"""
        if ":" in address:
            _, ioa_str = address.split(":", 1)
            return int(ioa_str)
        return int(address)

    def _start_background_threads(self) -> None:
        """启动后台线程"""
        # 总召唤定时线程
        gi_interval = self.config.general_interrogation_interval
        if gi_interval > 0:
            self._gi_thread = threading.Thread(
                target=self._gi_loop, args=(gi_interval,), daemon=True
            )
            self._gi_thread.start()

    def _gi_loop(self, interval: float) -> None:
        """定时总召唤"""
        while self._running:
            time.sleep(interval)
            if self._running:
                self.general_interrogation()

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def get_point_cache(self) -> Dict[int, Dict[str, Any]]:
        return dict(self._point_cache)


# =============================================================================
# IEC 61850 (MMS/GOOSE) 适配器
# =============================================================================
@dataclass
class IEC61850Config(ProtocolConfig):
    """IEC 61850 配置"""
    port: int = 102                  # MMS 默认端口
    # IED 名称
    ied_name: str = ""
    # 访问点
    access_point: str = "S1"
    # 数据集路径 (用于 Report / GOOSE 订阅)
    dataset_ref: str = ""
    # Report Control Block 引用
    rcb_ref: str = ""
    # 是否启用 GOOSE 订阅
    goose_enabled: bool = False
    # GOOSE AppID (十六进制字符串)
    goose_appid: str = ""
    # 网络接口 (GOOSE 抓包用)
    goose_interface: str = "eth0"
    # 认证
    tls_enabled: bool = False
    certificate_path: str = ""
    private_key_path: str = ""


class IEC61850Adapter(ProtocolAdapter):
    """
    IEC 61850 协议适配器

    支持:
    - MMS 客户端: 读写数据属性 (DA), 读取数据集 (DataSet)
    - Reporting: 订阅 Buffered/Unbuffered Report
    - GOOSE 订阅: 接收 GOOSE 报文 (跳闸/保护动作)
    - 控制操作: SBO / Direct-Operate

    适用设备: 保护装置(IED)、合并单元(MU)、智能终端、
              备自投装置、故障录波器等

    依赖: pip install libiec61850-python   (或 iec61850)
    无依赖时以模拟模式运行。
    """

    def __init__(self, config: IEC61850Config):
        super().__init__(config)
        self.config: IEC61850Config = config
        self._subscriptions: Dict[str, Callable] = {}
        self._report_callbacks: List[Callable] = []
        self._goose_callbacks: List[Callable] = []
        self._message_queue: Queue = Queue(maxsize=10000)
        self._running = False
        self._goose_thread: Optional[threading.Thread] = None
        self._simulated = False
        # 缓存: 数据引用 -> 最新值
        self._data_cache: Dict[str, Dict[str, Any]] = {}
        # 统计
        self._stats = {"mms_read": 0, "mms_write": 0, "report_rx": 0, "goose_rx": 0}

    def connect(self) -> bool:
        """连接 IEC 61850 MMS 服务器"""
        try:
            import iec61850  # type: ignore
            self._client = iec61850.IedConnection()
            error = self._client.connect(self.config.host, self.config.port)
            if error:
                raise Exception(f"MMS 连接错误: {error}")
            self._connected = True
            self._running = True

            # 启用 Reporting
            if self.config.rcb_ref:
                self._enable_reporting()

            # 启用 GOOSE
            if self.config.goose_enabled:
                self._start_goose_listener()

            logger.info(
                f"IEC61850 MMS 连接成功: {self.config.host}:{self.config.port} "
                f"IED={self.config.ied_name}"
            )
            return True
        except ImportError:
            logger.warning(
                "libiec61850-python 未安装, 以模拟模式运行 "
                "(pip install libiec61850-python)"
            )
            return self._connect_simulated()
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"IEC61850 连接失败: {e}")
            return False

    def _connect_simulated(self) -> bool:
        self._simulated = True
        self._connected = True
        self._running = True
        logger.info("IEC61850 以模拟模式运行")
        return True

    def disconnect(self) -> None:
        self._running = False
        if self._goose_thread and self._goose_thread.is_alive():
            self._goose_thread.join(timeout=3)
        if self._client and not self._simulated:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._connected = False
        self._subscriptions.clear()

    def read(self, address: str, **kwargs) -> Any:
        """
        读取 IEC 61850 数据属性

        Args:
            address: 对象引用, 格式:
                "IED1/LLN0$BR$brcbProtection"  (Report)
                "IED1/PTRC1$ST$Tr$general"     (保护跳闸)
                "IED1/MMXU1$MX$TotW$mag$f"    (有功功率)
                "IED1/XCBR1$ST$Pos$stVal"      (断路器位置)
                "IED1/RSYN1$ST$Rel$stVal"      (重合闸状态)

        Returns:
            数据值字典 {"value": ..., "quality": ..., "timestamp": ...}
        """
        # 先查缓存
        if address in self._data_cache:
            cached = self._data_cache[address]
            self._stats["mms_read"] += 1
            return cached

        if self._simulated:
            return None

        if not self._connected:
            raise Exception("IEC61850 未连接")

        try:
            value = self._client.read_value(address)
            result = {
                "value": value.value if hasattr(value, "value") else value,
                "quality": getattr(value, "quality", 0),
                "timestamp": getattr(value, "timestamp", ""),
                "ref": address,
            }
            self._data_cache[address] = result
            self._stats["mms_read"] += 1
            return result
        except Exception as e:
            logger.error(f"MMS 读取失败 [{address}]: {e}")
            return None

    def write(self, address: str, value: Any, **kwargs) -> bool:
        """
        写入/控制操作

        Args:
            address: 对象引用
            value: 写入值
            control_model: "direct" 或 "sbo" (Select-Before-Operate)
        """
        if self._simulated:
            logger.info(f"IEC61850 模拟写入: {address} = {value}")
            return True

        if not self._connected:
            raise Exception("IEC61850 未连接")

        control_model = kwargs.get("control_model", "direct")
        try:
            if control_model == "sbo":
                selected = self._client.select(address)
                if not selected:
                    logger.error(f"SBO 选择失败: {address}")
                    return False
            self._client.operate(address, value)
            self._stats["mms_write"] += 1
            return True
        except Exception as e:
            logger.error(f"IEC61850 控制失败 [{address}]: {e}")
            return False

    # ---------- Report 订阅 ----------

    def subscribe(self, address: str, callback: Callable[[str, Any], None],
                  interval_ms: int = 0) -> bool:
        """
        订阅数据属性变化

        address 可以是:
        - 具体 DA 引用: 当值变化时触发
        - "*": 接收所有 Report 数据变化
        """
        self._subscriptions[address] = callback
        logger.info(f"IEC61850 订阅: {address}")
        return True

    def subscribe_report(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """订阅所有 Report 回调"""
        self._report_callbacks.append(callback)

    def subscribe_goose(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """订阅 GOOSE 报文回调"""
        self._goose_callbacks.append(callback)

    def unsubscribe(self, address: str) -> bool:
        if address in self._subscriptions:
            del self._subscriptions[address]
            return True
        return False

    # ---------- 模拟数据注入 ----------

    def inject_simulated_data(self, ref: str, value: Any, quality: int = 0,
                              timestamp: str = "", data_type: str = "boolean") -> None:
        """
        注入模拟数据 (用于无真实 IED 时调试)

        Args:
            ref: 数据属性引用 (如 "IED1/PTRC1$ST$Tr$general")
            value: 值
            quality: 品质
            timestamp: ISO8601
            data_type: boolean/integer/float/bitstring/enumerated
        """
        ts = timestamp or __import__('datetime').datetime.now().isoformat()
        data = {
            "ref": ref,
            "value": value,
            "quality": quality,
            "timestamp": ts,
            "data_type": data_type,
        }
        old = self._data_cache.get(ref)
        self._data_cache[ref] = data

        # 触发订阅
        if ref in self._subscriptions:
            try:
                self._subscriptions[ref](ref, data)
            except Exception as e:
                logger.error(f"IEC61850 订阅回调异常: {e}")

        # 通配符订阅
        if "*" in self._subscriptions:
            try:
                self._subscriptions["*"](ref, data)
            except Exception as e:
                logger.error(f"IEC61850 通配回调异常: {e}")

        # Report 回调
        for cb in self._report_callbacks:
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Report 回调异常: {e}")

    # ---------- GOOSE ----------

    def inject_simulated_goose(self, appid: str, data_set: List[Dict[str, Any]],
                               st_num: int = 1, sq_num: int = 0) -> None:
        """注入模拟 GOOSE 报文"""
        ts = __import__('datetime').datetime.now().isoformat()
        goose_msg = {
            "appid": appid,
            "st_num": st_num,
            "sq_num": sq_num,
            "timestamp": ts,
            "data_set": data_set,
        }
        self._stats["goose_rx"] += 1
        for cb in self._goose_callbacks:
            try:
                cb(goose_msg)
            except Exception as e:
                logger.error(f"GOOSE 回调异常: {e}")

    # ---------- 内部方法 ----------

    def _enable_reporting(self) -> None:
        """启用 Report Control Block"""
        if self._simulated or not self._client:
            return
        try:
            rcb_ref = self.config.rcb_ref
            self._client.enable_reporting(rcb_ref, self._on_report_received)
            logger.info(f"IEC61850 Reporting 已启用: {rcb_ref}")
        except Exception as e:
            logger.error(f"Reporting 启用失败: {e}")

    def _on_report_received(self, report: Any) -> None:
        """Report 回调 (真实连接)"""
        self._stats["report_rx"] += 1
        try:
            for entry in report.entries:
                ref = entry.reference
                value = entry.value
                quality = getattr(entry, "quality", 0)
                ts = getattr(entry, "timestamp", "")
                data = {
                    "ref": ref,
                    "value": value.value if hasattr(value, "value") else value,
                    "quality": quality,
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                }
                self._data_cache[ref] = data

                # 触发订阅
                if ref in self._subscriptions:
                    self._subscriptions[ref](ref, data)
                if "*" in self._subscriptions:
                    self._subscriptions["*"](ref, data)

            # Report 回调
            for cb in self._report_callbacks:
                cb({"entries": [e.__dict__ for e in report.entries]})

        except Exception as e:
            logger.error(f"Report 解析异常: {e}")

    def _start_goose_listener(self) -> None:
        """启动 GOOSE 监听线程"""
        if self._simulated:
            return
        self._goose_thread = threading.Thread(
            target=self._goose_listen_loop, daemon=True
        )
        self._goose_thread.start()

    def _goose_listen_loop(self) -> None:
        """GOOSE 监听循环"""
        try:
            import iec61850  # type: ignore
            receiver = iec61850.GooseReceiver()
            receiver.set_interface(self.config.goose_interface)
            if self.config.goose_appid:
                receiver.add_appid(int(self.config.goose_appid, 16))
            receiver.start()
            while self._running:
                msg = receiver.get_next_message(timeout_ms=1000)
                if msg:
                    self._stats["goose_rx"] += 1
                    goose_data = {
                        "appid": msg.appid,
                        "st_num": msg.st_num,
                        "sq_num": msg.sq_num,
                        "timestamp": msg.timestamp,
                        "data_set": msg.data_set,
                    }
                    for cb in self._goose_callbacks:
                        try:
                            cb(goose_data)
                        except Exception as e:
                            logger.error(f"GOOSE 回调异常: {e}")
            receiver.stop()
        except Exception as e:
            logger.error(f"GOOSE 监听异常: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def get_data_cache(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._data_cache)

    # ---------- 便捷方法: 变电站二次设备常用数据引用 ----------

    def read_breaker_position(self, ied: str, xcbr_ln: str = "XCBR1") -> Any:
        """读取断路器位置 (stVal: 0=中间/1=分/2=合)"""
        return self.read(f"{ied}/{xcbr_ln}$ST$Pos$stVal")

    def read_protection_trip(self, ied: str, ptrc_ln: str = "PTRC1") -> Any:
        """读取保护跳闸信号"""
        return self.read(f"{ied}/{ptrc_ln}$ST$Tr$general")

    def read_recloser_status(self, ied: str, rrec_ln: str = "RREC1") -> Any:
        """读取重合闸状态"""
        return self.read(f"{ied}/{rrec_ln}$ST$Op$general")

    def read_measured_value(self, ied: str, mmxu_ln: str = "MMXU1",
                           measurement: str = "TotW") -> Any:
        """读取遥测值 (有功/无功/电压/电流)"""
        return self.read(f"{ied}/{mmxu_ln}$MX${measurement}$mag$f")

    def operate_breaker(self, ied: str, xcbr_ln: str = "XCBR1",
                        open: bool = True, **kwargs) -> bool:
        """操作断路器 (分/合)"""
        ref = f"{ied}/{xcbr_ln}$CO$Pos"
        return self.write(ref, not open, **kwargs)  # Pos: True=合, False=分


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
        "iec104": (IEC104Adapter, IEC104Config),
        "iec61850": (IEC61850Adapter, IEC61850Config),
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
