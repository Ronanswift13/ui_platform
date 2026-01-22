"""
微服务架构模块
输变电激光星芒破夜绘明监测平台

基于UPDATE.md中期迭代计划:
- 各插件封装为独立的FastAPI/gRPC服务
- 统一API网关路由
- Celery/Kafka异步任务队列
- WebSocket实时告警推送
- 移动端支持

功能:
- 服务注册与发现
- API网关路由
- 异步任务调度
- WebSocket实时通信
- 负载均衡

版本: 1.0.0
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import time
import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import PriorityQueue, Empty
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举定义
# =============================================================================

class ServiceStatus(Enum):
    """服务状态"""
    STARTING = "starting"           # 启动中
    RUNNING = "running"             # 运行中
    STOPPING = "stopping"           # 停止中
    STOPPED = "stopped"             # 已停止
    UNHEALTHY = "unhealthy"         # 不健康
    DEGRADED = "degraded"           # 降级


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"             # 待执行
    QUEUED = "queued"               # 已入队
    RUNNING = "running"             # 执行中
    COMPLETED = "completed"         # 已完成
    FAILED = "failed"               # 失败
    CANCELLED = "cancelled"         # 已取消
    RETRYING = "retrying"           # 重试中
    TIMEOUT = "timeout"             # 超时


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 0                    # 紧急
    HIGH = 1                        # 高
    NORMAL = 2                      # 普通
    LOW = 3                         # 低
    BACKGROUND = 4                  # 后台


class MessageType(Enum):
    """消息类型"""
    ALARM = "alarm"                 # 告警
    STATUS = "status"               # 状态更新
    PROGRESS = "progress"           # 进度通知
    RESULT = "result"               # 结果
    HEARTBEAT = "heartbeat"         # 心跳
    COMMAND = "command"             # 命令
    BROADCAST = "broadcast"         # 广播


class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"     # 轮询
    LEAST_CONNECTIONS = "least_connections"  # 最少连接
    RANDOM = "random"               # 随机
    WEIGHTED = "weighted"           # 加权
    RESPONSE_TIME = "response_time" # 响应时间


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class ServiceInfo:
    """服务信息"""
    service_id: str
    name: str
    version: str
    host: str
    port: int
    protocol: str = "http"          # http, grpc, websocket
    status: ServiceStatus = ServiceStatus.STOPPED
    health_endpoint: str = "/health"
    weight: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 运行时信息
    registered_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    active_connections: int = 0
    avg_response_time: float = 0.0

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "name": self.name,
            "version": self.version,
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "status": self.status.value,
            "health_endpoint": self.health_endpoint,
            "weight": self.weight,
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "active_connections": self.active_connections,
            "avg_response_time": self.avg_response_time,
        }


@dataclass
class RouteConfig:
    """路由配置"""
    path_pattern: str               # 路径模式 (支持通配符)
    service_name: str               # 目标服务名
    methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    timeout: float = 30.0           # 超时时间(秒)
    retry_count: int = 3            # 重试次数
    rate_limit: Optional[int] = None  # 速率限制(请求/分钟)
    auth_required: bool = False     # 是否需要认证
    strip_prefix: bool = True       # 是否移除前缀

    def matches(self, path: str, method: str) -> bool:
        """检查是否匹配路由"""
        if method.upper() not in self.methods:
            return False

        # 简单通配符匹配
        import fnmatch
        return fnmatch.fnmatch(path, self.path_pattern)


@dataclass
class AsyncTask:
    """异步任务"""
    task_id: str
    name: str
    func_name: str
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING

    # 调度配置
    timeout: float = 300.0          # 超时时间(秒)
    max_retries: int = 3            # 最大重试次数
    retry_delay: float = 5.0        # 重试延迟(秒)
    retry_count: int = 0            # 当前重试次数

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 结果
    result: Any = None
    error: Optional[str] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """优先级比较"""
        return self.priority.value < other.priority.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "func_name": self.func_name,
            "priority": self.priority.value,
            "status": self.status.value,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class WebSocketMessage:
    """WebSocket消息"""
    message_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    target: Optional[str] = None    # None表示广播

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "target": self.target,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class WebSocketClient:
    """WebSocket客户端"""
    client_id: str
    user_id: Optional[str] = None
    device_type: str = "web"        # web, mobile, desktop
    subscriptions: Set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 连接对象 (实际使用时替换为真实的WebSocket连接)
    connection: Any = None


# =============================================================================
# 服务注册与发现
# =============================================================================

class ServiceRegistry:
    """服务注册中心"""

    def __init__(
        self,
        heartbeat_interval: float = 10.0,
        health_check_interval: float = 30.0,
        unhealthy_threshold: int = 3
    ):
        self.heartbeat_interval = heartbeat_interval
        self.health_check_interval = health_check_interval
        self.unhealthy_threshold = unhealthy_threshold

        # 服务注册表: {service_name: {service_id: ServiceInfo}}
        self._services: Dict[str, Dict[str, ServiceInfo]] = defaultdict(dict)

        # 健康检查失败计数: {(service_name, service_id): failure_count}
        self._health_failures: Dict[Tuple[str, str], int] = defaultdict(int)

        # 服务变更回调
        self._callbacks: List[Callable[[str, ServiceInfo, str], None]] = []

        # 锁
        self._lock = threading.RLock()

        # 健康检查线程
        self._running = False
        self._health_thread: Optional[threading.Thread] = None

        logger.info("服务注册中心初始化完成")

    def start(self):
        """启动服务注册中心"""
        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_thread.start()
        logger.info("服务注册中心启动")

    def stop(self):
        """停止服务注册中心"""
        self._running = False
        if self._health_thread:
            self._health_thread.join(timeout=5.0)
        logger.info("服务注册中心停止")

    def register(self, service: ServiceInfo) -> bool:
        """
        注册服务

        Args:
            service: 服务信息

        Returns:
            是否注册成功
        """
        with self._lock:
            service.registered_at = datetime.now()
            service.last_heartbeat = datetime.now()
            service.status = ServiceStatus.RUNNING

            self._services[service.name][service.service_id] = service
            self._notify_callbacks(service.name, service, "register")

            logger.info(f"服务注册: {service.name}@{service.service_id}")
            return True

    def deregister(self, service_name: str, service_id: str) -> bool:
        """
        注销服务

        Args:
            service_name: 服务名
            service_id: 服务ID

        Returns:
            是否注销成功
        """
        with self._lock:
            if service_name in self._services:
                if service_id in self._services[service_name]:
                    service = self._services[service_name].pop(service_id)
                    service.status = ServiceStatus.STOPPED
                    self._notify_callbacks(service_name, service, "deregister")
                    logger.info(f"服务注销: {service_name}@{service_id}")
                    return True
            return False

    def heartbeat(self, service_name: str, service_id: str) -> bool:
        """
        服务心跳

        Args:
            service_name: 服务名
            service_id: 服务ID

        Returns:
            是否成功
        """
        with self._lock:
            if service_name in self._services:
                if service_id in self._services[service_name]:
                    service = self._services[service_name][service_id]
                    service.last_heartbeat = datetime.now()
                    self._health_failures[(service_name, service_id)] = 0

                    if service.status == ServiceStatus.UNHEALTHY:
                        service.status = ServiceStatus.RUNNING
                        self._notify_callbacks(service_name, service, "recovered")
                    return True
            return False

    def get_services(self, service_name: str) -> List[ServiceInfo]:
        """
        获取服务实例列表

        Args:
            service_name: 服务名

        Returns:
            健康的服务实例列表
        """
        with self._lock:
            if service_name not in self._services:
                return []

            return [
                s for s in self._services[service_name].values()
                if s.status == ServiceStatus.RUNNING
            ]

    def get_service(self, service_name: str, service_id: str) -> Optional[ServiceInfo]:
        """获取指定服务实例"""
        with self._lock:
            if service_name in self._services:
                return self._services[service_name].get(service_id)
            return None

    def get_all_services(self) -> Dict[str, List[ServiceInfo]]:
        """获取所有服务"""
        with self._lock:
            return {
                name: list(instances.values())
                for name, instances in self._services.items()
            }

    def on_service_change(self, callback: Callable[[str, ServiceInfo, str], None]):
        """
        注册服务变更回调

        Args:
            callback: 回调函数(service_name, service_info, event_type)
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self, service_name: str, service: ServiceInfo, event: str):
        """通知回调"""
        for callback in self._callbacks:
            try:
                callback(service_name, service, event)
            except Exception as e:
                logger.error(f"服务变更回调异常: {e}")

    def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                self._check_services_health()
            except Exception as e:
                logger.error(f"健康检查异常: {e}")

            time.sleep(self.health_check_interval)

    def _check_services_health(self):
        """检查所有服务健康状态"""
        now = datetime.now()

        with self._lock:
            for service_name, instances in self._services.items():
                for service_id, service in list(instances.items()):
                    if service.last_heartbeat:
                        elapsed = (now - service.last_heartbeat).total_seconds()

                        if elapsed > self.heartbeat_interval * 2:
                            # 心跳超时
                            key = (service_name, service_id)
                            self._health_failures[key] += 1

                            if self._health_failures[key] >= self.unhealthy_threshold:
                                service.status = ServiceStatus.UNHEALTHY
                                self._notify_callbacks(
                                    service_name, service, "unhealthy"
                                )
                                logger.warning(
                                    f"服务不健康: {service_name}@{service_id}"
                                )


# =============================================================================
# API网关
# =============================================================================

class APIGateway:
    """API网关"""

    def __init__(
        self,
        registry: ServiceRegistry,
        load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    ):
        self.registry = registry
        self.load_balance_strategy = load_balance_strategy

        # 路由配置
        self._routes: List[RouteConfig] = []

        # 轮询索引
        self._rr_index: Dict[str, int] = defaultdict(int)

        # 速率限制
        self._rate_limits: Dict[str, List[datetime]] = defaultdict(list)

        # 锁
        self._lock = threading.RLock()

        logger.info("API网关初始化完成")

    def add_route(self, route: RouteConfig):
        """添加路由配置"""
        with self._lock:
            self._routes.append(route)
            logger.info(f"添加路由: {route.path_pattern} -> {route.service_name}")

    def remove_route(self, path_pattern: str):
        """移除路由配置"""
        with self._lock:
            self._routes = [r for r in self._routes if r.path_pattern != path_pattern]

    def get_routes(self) -> List[RouteConfig]:
        """获取所有路由"""
        return self._routes.copy()

    def route_request(
        self,
        path: str,
        method: str,
        client_id: Optional[str] = None
    ) -> Optional[Tuple[ServiceInfo, RouteConfig]]:
        """
        路由请求

        Args:
            path: 请求路径
            method: HTTP方法
            client_id: 客户端ID (用于速率限制)

        Returns:
            (目标服务, 路由配置) 或 None
        """
        with self._lock:
            # 查找匹配的路由
            for route in self._routes:
                if route.matches(path, method):
                    # 检查速率限制
                    if not self._check_rate_limit(route, client_id):
                        logger.warning(f"速率限制: {path}")
                        return None

                    # 获取服务实例
                    service = self._select_service(route.service_name)
                    if service:
                        return service, route
                    else:
                        logger.warning(f"无可用服务: {route.service_name}")
                        return None

            logger.debug(f"未找到路由: {method} {path}")
            return None

    def _check_rate_limit(
        self,
        route: RouteConfig,
        client_id: Optional[str]
    ) -> bool:
        """检查速率限制"""
        if not route.rate_limit or not client_id:
            return True

        key = f"{client_id}:{route.path_pattern}"
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # 清理过期记录
        self._rate_limits[key] = [
            t for t in self._rate_limits[key] if t > minute_ago
        ]

        # 检查是否超限
        if len(self._rate_limits[key]) >= route.rate_limit:
            return False

        # 记录请求
        self._rate_limits[key].append(now)
        return True

    def _select_service(self, service_name: str) -> Optional[ServiceInfo]:
        """选择服务实例"""
        services = self.registry.get_services(service_name)
        if not services:
            return None

        if self.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin(service_name, services)
        elif self.load_balance_strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections(services)
        elif self.load_balance_strategy == LoadBalanceStrategy.RANDOM:
            return self._random(services)
        elif self.load_balance_strategy == LoadBalanceStrategy.WEIGHTED:
            return self._weighted(services)
        elif self.load_balance_strategy == LoadBalanceStrategy.RESPONSE_TIME:
            return self._response_time(services)
        else:
            return services[0]

    def _round_robin(
        self,
        service_name: str,
        services: List[ServiceInfo]
    ) -> ServiceInfo:
        """轮询选择"""
        idx = self._rr_index[service_name]
        service = services[idx % len(services)]
        self._rr_index[service_name] = idx + 1
        return service

    def _least_connections(self, services: List[ServiceInfo]) -> ServiceInfo:
        """最少连接选择"""
        return min(services, key=lambda s: s.active_connections)

    def _random(self, services: List[ServiceInfo]) -> ServiceInfo:
        """随机选择"""
        import random
        return random.choice(services)

    def _weighted(self, services: List[ServiceInfo]) -> ServiceInfo:
        """加权选择"""
        import random
        total_weight = sum(s.weight for s in services)
        rand = random.randint(1, total_weight)

        cumulative = 0
        for service in services:
            cumulative += service.weight
            if rand <= cumulative:
                return service

        return services[-1]

    def _response_time(self, services: List[ServiceInfo]) -> ServiceInfo:
        """响应时间选择"""
        return min(services, key=lambda s: s.avg_response_time)


# =============================================================================
# 异步任务队列
# =============================================================================

class TaskQueue:
    """异步任务队列"""

    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 10000,
        task_timeout: float = 300.0
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.task_timeout = task_timeout

        # 任务队列
        self._queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)

        # 任务注册表
        self._tasks: Dict[str, AsyncTask] = {}

        # 任务处理函数
        self._handlers: Dict[str, Callable] = {}

        # 工作线程池
        self._executor: Optional[ThreadPoolExecutor] = None

        # 结果回调
        self._callbacks: List[Callable[[AsyncTask], None]] = []

        # 运行状态
        self._running = False

        # 锁
        self._lock = threading.RLock()

        # 调度线程
        self._scheduler_thread: Optional[threading.Thread] = None

        logger.info(f"任务队列初始化完成, workers={max_workers}")

    def start(self):
        """启动任务队列"""
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self._scheduler_thread = threading.Thread(
            target=self._schedule_loop,
            daemon=True
        )
        self._scheduler_thread.start()

        logger.info("任务队列启动")

    def stop(self, wait: bool = True):
        """停止任务队列"""
        self._running = False

        if self._executor:
            self._executor.shutdown(wait=wait)

        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)

        logger.info("任务队列停止")

    def register_handler(self, func_name: str, handler: Callable):
        """
        注册任务处理函数

        Args:
            func_name: 函数名
            handler: 处理函数
        """
        self._handlers[func_name] = handler
        logger.info(f"注册任务处理器: {func_name}")

    def submit(
        self,
        name: str,
        func_name: str,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        提交任务

        Args:
            name: 任务名称
            func_name: 处理函数名
            args: 位置参数
            kwargs: 关键字参数
            priority: 优先级
            timeout: 超时时间
            metadata: 元数据

        Returns:
            任务ID
        """
        if func_name not in self._handlers:
            raise ValueError(f"未注册的处理器: {func_name}")

        task_id = str(uuid.uuid4())

        task = AsyncTask(
            task_id=task_id,
            name=name,
            func_name=func_name,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            timeout=timeout or self.task_timeout,
            metadata=metadata or {},
        )

        with self._lock:
            self._tasks[task_id] = task
            self._queue.put((priority.value, time.time(), task))
            task.status = TaskStatus.QUEUED

        logger.info(f"任务提交: {name}[{task_id}]")
        return task_id

    def cancel(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            是否取消成功
        """
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if task.status in (TaskStatus.PENDING, TaskStatus.QUEUED):
                    task.status = TaskStatus.CANCELLED
                    logger.info(f"任务取消: {task_id}")
                    return True
        return False

    def get_task(self, task_id: str) -> Optional[AsyncTask]:
        """获取任务"""
        return self._tasks.get(task_id)

    def get_pending_tasks(self) -> List[AsyncTask]:
        """获取待执行任务"""
        with self._lock:
            return [
                t for t in self._tasks.values()
                if t.status in (TaskStatus.PENDING, TaskStatus.QUEUED)
            ]

    def get_running_tasks(self) -> List[AsyncTask]:
        """获取执行中任务"""
        with self._lock:
            return [
                t for t in self._tasks.values()
                if t.status == TaskStatus.RUNNING
            ]

    def on_task_complete(self, callback: Callable[[AsyncTask], None]):
        """注册任务完成回调"""
        self._callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """获取队列统计"""
        with self._lock:
            status_counts = defaultdict(int)
            for task in self._tasks.values():
                status_counts[task.status.value] += 1

            return {
                "queue_size": self._queue.qsize(),
                "total_tasks": len(self._tasks),
                "status_counts": dict(status_counts),
                "handlers": list(self._handlers.keys()),
            }

    def _schedule_loop(self):
        """调度循环"""
        while self._running:
            try:
                # 获取任务
                try:
                    _, _, task = self._queue.get(timeout=1.0)
                except Empty:
                    continue

                # 检查任务状态
                if task.status == TaskStatus.CANCELLED:
                    continue

                # 提交到线程池执行
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()

                if self._executor:
                    self._executor.submit(self._execute_task, task)

            except Exception as e:
                logger.error(f"调度异常: {e}")

    def _execute_task(self, task: AsyncTask):
        """执行任务"""
        try:
            handler = self._handlers.get(task.func_name)
            if not handler:
                raise ValueError(f"处理器不存在: {task.func_name}")

            # 执行
            result = handler(*task.args, **task.kwargs)

            # 更新任务
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            logger.info(f"任务完成: {task.name}[{task.task_id}]")

        except Exception as e:
            logger.error(f"任务失败: {task.name}[{task.task_id}]: {e}")

            task.error = str(e)
            task.retry_count += 1

            # 检查是否重试
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRYING
                time.sleep(task.retry_delay)

                # 重新入队
                with self._lock:
                    self._queue.put((
                        task.priority.value,
                        time.time(),
                        task
                    ))
                    task.status = TaskStatus.QUEUED
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()

        finally:
            # 通知回调
            for callback in self._callbacks:
                try:
                    callback(task)
                except Exception as e:
                    logger.error(f"任务回调异常: {e}")


# =============================================================================
# WebSocket管理器
# =============================================================================

class WebSocketManager:
    """WebSocket连接管理器"""

    def __init__(
        self,
        ping_interval: float = 30.0,
        inactive_timeout: float = 300.0
    ):
        self.ping_interval = ping_interval
        self.inactive_timeout = inactive_timeout

        # 客户端连接: {client_id: WebSocketClient}
        self._clients: Dict[str, WebSocketClient] = {}

        # 订阅映射: {topic: {client_ids}}
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)

        # 消息历史 (用于重连恢复)
        self._message_history: Dict[str, List[WebSocketMessage]] = defaultdict(list)
        self._history_max_size = 100

        # 运行状态
        self._running = False

        # 锁
        self._lock = threading.RLock()

        # 心跳线程
        self._heartbeat_thread: Optional[threading.Thread] = None

        # 消息发送队列
        self._send_queue: Optional[asyncio.Queue] = None

        logger.info("WebSocket管理器初始化完成")

    def start(self):
        """启动WebSocket管理器"""
        self._running = True

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self._heartbeat_thread.start()

        logger.info("WebSocket管理器启动")

    def stop(self):
        """停止WebSocket管理器"""
        self._running = False

        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)

        logger.info("WebSocket管理器停止")

    def connect(
        self,
        client_id: str,
        connection: Any,
        user_id: Optional[str] = None,
        device_type: str = "web",
        metadata: Optional[Dict[str, Any]] = None
    ) -> WebSocketClient:
        """
        注册客户端连接

        Args:
            client_id: 客户端ID
            connection: WebSocket连接对象
            user_id: 用户ID
            device_type: 设备类型
            metadata: 元数据

        Returns:
            客户端对象
        """
        client = WebSocketClient(
            client_id=client_id,
            user_id=user_id,
            device_type=device_type,
            connection=connection,
            metadata=metadata or {},
        )

        with self._lock:
            self._clients[client_id] = client

        logger.info(f"客户端连接: {client_id} ({device_type})")
        return client

    def disconnect(self, client_id: str):
        """
        断开客户端连接

        Args:
            client_id: 客户端ID
        """
        with self._lock:
            if client_id in self._clients:
                client = self._clients.pop(client_id)

                # 清理订阅
                for topic in list(client.subscriptions):
                    if topic in self._subscriptions:
                        self._subscriptions[topic].discard(client_id)

                logger.info(f"客户端断开: {client_id}")

    def subscribe(self, client_id: str, topics: List[str]):
        """
        订阅主题

        Args:
            client_id: 客户端ID
            topics: 主题列表
        """
        with self._lock:
            if client_id in self._clients:
                client = self._clients[client_id]

                for topic in topics:
                    client.subscriptions.add(topic)
                    self._subscriptions[topic].add(client_id)

                logger.debug(f"客户端订阅: {client_id} -> {topics}")

    def unsubscribe(self, client_id: str, topics: List[str]):
        """
        取消订阅

        Args:
            client_id: 客户端ID
            topics: 主题列表
        """
        with self._lock:
            if client_id in self._clients:
                client = self._clients[client_id]

                for topic in topics:
                    client.subscriptions.discard(topic)
                    if topic in self._subscriptions:
                        self._subscriptions[topic].discard(client_id)

    async def send(
        self,
        client_id: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> bool:
        """
        发送消息给指定客户端

        Args:
            client_id: 客户端ID
            message_type: 消息类型
            payload: 消息内容

        Returns:
            是否发送成功
        """
        with self._lock:
            if client_id not in self._clients:
                return False

            client = self._clients[client_id]

        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            payload=payload,
            target=client_id,
        )

        try:
            # 实际发送 (模拟)
            if client.connection:
                # await client.connection.send(message.to_json())
                logger.debug(f"发送消息: {client_id} <- {message_type.value}")

            client.last_activity = datetime.now()
            return True

        except Exception as e:
            logger.error(f"消息发送失败: {client_id}: {e}")
            return False

    async def broadcast(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        topic: Optional[str] = None,
        exclude: Optional[List[str]] = None
    ) -> int:
        """
        广播消息

        Args:
            message_type: 消息类型
            payload: 消息内容
            topic: 主题 (None表示全部)
            exclude: 排除的客户端ID列表

        Returns:
            成功发送的数量
        """
        exclude = exclude or []

        with self._lock:
            if topic:
                # 发送给订阅了该主题的客户端
                client_ids = self._subscriptions.get(topic, set()) - set(exclude)
            else:
                # 发送给所有客户端
                client_ids = set(self._clients.keys()) - set(exclude)

        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            payload=payload,
        )

        # 保存到历史
        if topic:
            self._save_to_history(topic, message)

        # 发送消息
        success_count = 0
        for client_id in client_ids:
            if await self.send(client_id, message_type, payload):
                success_count += 1

        logger.info(
            f"广播消息: {message_type.value} -> {success_count}/{len(client_ids)}个客户端"
        )
        return success_count

    async def send_alarm(
        self,
        alarm_id: str,
        alarm_type: str,
        severity: str,
        message: str,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        topic: str = "alarms"
    ):
        """
        发送告警消息

        Args:
            alarm_id: 告警ID
            alarm_type: 告警类型
            severity: 严重程度
            message: 告警消息
            source: 来源
            data: 附加数据
            topic: 主题
        """
        payload = {
            "alarm_id": alarm_id,
            "alarm_type": alarm_type,
            "severity": severity,
            "message": message,
            "source": source,
            "data": data or {},
            "timestamp": datetime.now().isoformat(),
        }

        await self.broadcast(MessageType.ALARM, payload, topic=topic)

    async def send_progress(
        self,
        task_id: str,
        progress: float,
        message: str,
        client_id: Optional[str] = None
    ):
        """
        发送进度通知

        Args:
            task_id: 任务ID
            progress: 进度 (0-100)
            message: 消息
            client_id: 目标客户端 (None表示广播)
        """
        payload = {
            "task_id": task_id,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }

        if client_id:
            await self.send(client_id, MessageType.PROGRESS, payload)
        else:
            await self.broadcast(MessageType.PROGRESS, payload, topic="tasks")

    def get_client(self, client_id: str) -> Optional[WebSocketClient]:
        """获取客户端"""
        return self._clients.get(client_id)

    def get_clients(self, topic: Optional[str] = None) -> List[WebSocketClient]:
        """获取客户端列表"""
        with self._lock:
            if topic:
                client_ids = self._subscriptions.get(topic, set())
                return [
                    self._clients[cid]
                    for cid in client_ids
                    if cid in self._clients
                ]
            else:
                return list(self._clients.values())

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            device_counts = defaultdict(int)
            for client in self._clients.values():
                device_counts[client.device_type] += 1

            return {
                "total_clients": len(self._clients),
                "device_counts": dict(device_counts),
                "subscriptions": {
                    topic: len(clients)
                    for topic, clients in self._subscriptions.items()
                },
            }

    def _save_to_history(self, topic: str, message: WebSocketMessage):
        """保存消息到历史"""
        history = self._message_history[topic]
        history.append(message)

        # 限制历史大小
        while len(history) > self._history_max_size:
            history.pop(0)

    def get_history(
        self,
        topic: str,
        since: Optional[datetime] = None
    ) -> List[WebSocketMessage]:
        """获取历史消息"""
        history = self._message_history.get(topic, [])

        if since:
            return [m for m in history if m.timestamp > since]

        return history.copy()

    def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            try:
                self._check_clients()
            except Exception as e:
                logger.error(f"心跳检查异常: {e}")

            time.sleep(self.ping_interval)

    def _check_clients(self):
        """检查客户端状态"""
        now = datetime.now()
        timeout = timedelta(seconds=self.inactive_timeout)

        with self._lock:
            inactive_clients = []

            for client_id, client in self._clients.items():
                if now - client.last_activity > timeout:
                    inactive_clients.append(client_id)

            for client_id in inactive_clients:
                logger.warning(f"客户端超时断开: {client_id}")
                self.disconnect(client_id)


# =============================================================================
# 插件服务包装器
# =============================================================================

class PluginServiceWrapper:
    """插件服务包装器 - 将插件封装为微服务"""

    def __init__(
        self,
        plugin_name: str,
        plugin_instance: Any,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        self.plugin_name = plugin_name
        self.plugin_instance = plugin_instance
        self.host = host
        self.port = port

        # 服务信息
        self.service_info: Optional[ServiceInfo] = None

        # 注册中心引用
        self.registry: Optional[ServiceRegistry] = None

        # 运行状态
        self._running = False

        logger.info(f"插件服务包装器创建: {plugin_name}")

    def register_to(self, registry: ServiceRegistry):
        """注册到服务注册中心"""
        self.registry = registry

        self.service_info = ServiceInfo(
            service_id=str(uuid.uuid4()),
            name=f"plugin_{self.plugin_name}",
            version="1.0.0",
            host=self.host,
            port=self.port,
            protocol="http",
            metadata={
                "plugin_name": self.plugin_name,
                "capabilities": self._get_capabilities(),
            }
        )

        registry.register(self.service_info)

    def _get_capabilities(self) -> List[str]:
        """获取插件能力列表"""
        capabilities = []

        # 检查常见方法
        if hasattr(self.plugin_instance, "process"):
            capabilities.append("process")
        if hasattr(self.plugin_instance, "analyze"):
            capabilities.append("analyze")
        if hasattr(self.plugin_instance, "detect"):
            capabilities.append("detect")
        if hasattr(self.plugin_instance, "recognize"):
            capabilities.append("recognize")

        return capabilities

    def create_routes(self) -> List[RouteConfig]:
        """创建路由配置"""
        routes = []

        # 基础路由
        base_path = f"/api/plugins/{self.plugin_name}"

        # 健康检查
        routes.append(RouteConfig(
            path_pattern=f"{base_path}/health",
            service_name=f"plugin_{self.plugin_name}",
            methods=["GET"],
        ))

        # 处理接口
        if hasattr(self.plugin_instance, "process"):
            routes.append(RouteConfig(
                path_pattern=f"{base_path}/process",
                service_name=f"plugin_{self.plugin_name}",
                methods=["POST"],
                timeout=60.0,
            ))

        # 分析接口
        if hasattr(self.plugin_instance, "analyze"):
            routes.append(RouteConfig(
                path_pattern=f"{base_path}/analyze",
                service_name=f"plugin_{self.plugin_name}",
                methods=["POST"],
                timeout=120.0,
            ))

        return routes

    async def handle_request(
        self,
        endpoint: str,
        method: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理请求

        Args:
            endpoint: 端点
            method: HTTP方法
            data: 请求数据

        Returns:
            响应数据
        """
        try:
            if endpoint == "health":
                return {"status": "healthy", "plugin": self.plugin_name}

            elif endpoint == "process" and hasattr(self.plugin_instance, "process"):
                result = self.plugin_instance.process(data)
                return {"status": "success", "result": result}

            elif endpoint == "analyze" and hasattr(self.plugin_instance, "analyze"):
                result = self.plugin_instance.analyze(data)
                return {"status": "success", "result": result}

            else:
                return {"status": "error", "message": f"未知端点: {endpoint}"}

        except Exception as e:
            logger.error(f"请求处理失败: {endpoint}: {e}")
            return {"status": "error", "message": str(e)}


# =============================================================================
# 微服务协调器
# =============================================================================

class MicroserviceOrchestrator:
    """微服务协调器 - 整合所有微服务组件"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        config = config or {}

        # 服务注册中心
        self.registry = ServiceRegistry(
            heartbeat_interval=config.get("heartbeat_interval", 10.0),
            health_check_interval=config.get("health_check_interval", 30.0),
        )

        # API网关
        self.gateway = APIGateway(
            registry=self.registry,
            load_balance_strategy=LoadBalanceStrategy(
                config.get("load_balance", "round_robin")
            ),
        )

        # 任务队列
        self.task_queue = TaskQueue(
            max_workers=config.get("max_workers", 4),
            max_queue_size=config.get("max_queue_size", 10000),
        )

        # WebSocket管理器
        self.websocket = WebSocketManager(
            ping_interval=config.get("ping_interval", 30.0),
            inactive_timeout=config.get("inactive_timeout", 300.0),
        )

        # 插件服务
        self._plugin_services: Dict[str, PluginServiceWrapper] = {}

        # 运行状态
        self._running = False

        logger.info("微服务协调器初始化完成")

    def start(self):
        """启动所有服务"""
        self._running = True

        self.registry.start()
        self.task_queue.start()
        self.websocket.start()

        logger.info("微服务协调器启动")

    def stop(self):
        """停止所有服务"""
        self._running = False

        self.websocket.stop()
        self.task_queue.stop()
        self.registry.stop()

        logger.info("微服务协调器停止")

    def register_plugin(
        self,
        plugin_name: str,
        plugin_instance: Any,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        """
        注册插件为微服务

        Args:
            plugin_name: 插件名称
            plugin_instance: 插件实例
            host: 主机地址
            port: 端口
        """
        wrapper = PluginServiceWrapper(
            plugin_name=plugin_name,
            plugin_instance=plugin_instance,
            host=host,
            port=port,
        )

        # 注册到服务中心
        wrapper.register_to(self.registry)

        # 创建路由
        for route in wrapper.create_routes():
            self.gateway.add_route(route)

        self._plugin_services[plugin_name] = wrapper

        logger.info(f"插件注册为微服务: {plugin_name}")

    def register_task_handler(self, func_name: str, handler: Callable):
        """注册任务处理器"""
        self.task_queue.register_handler(func_name, handler)

    def submit_task(
        self,
        name: str,
        func_name: str,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """提交异步任务"""
        return self.task_queue.submit(
            name=name,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
        )

    async def route_request(
        self,
        path: str,
        method: str,
        data: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        路由请求

        Args:
            path: 请求路径
            method: HTTP方法
            data: 请求数据
            client_id: 客户端ID

        Returns:
            响应数据
        """
        result = self.gateway.route_request(path, method, client_id)

        if not result:
            return {"status": "error", "message": "路由失败"}

        service, route = result

        # 查找对应的插件服务
        plugin_name = service.metadata.get("plugin_name")
        if plugin_name and plugin_name in self._plugin_services:
            wrapper = self._plugin_services[plugin_name]

            # 提取端点
            endpoint = path.split("/")[-1]

            return await wrapper.handle_request(endpoint, method, data)

        return {"status": "error", "message": "服务不可用"}

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "registry": {
                "services": {
                    name: [s.to_dict() for s in instances]
                    for name, instances in self.registry.get_all_services().items()
                },
            },
            "gateway": {
                "routes": len(self.gateway.get_routes()),
                "load_balance": self.gateway.load_balance_strategy.value,
            },
            "task_queue": self.task_queue.get_statistics(),
            "websocket": self.websocket.get_statistics(),
        }


# =============================================================================
# 便捷函数
# =============================================================================

# 全局协调器实例
_orchestrator: Optional[MicroserviceOrchestrator] = None


def get_orchestrator() -> MicroserviceOrchestrator:
    """获取全局微服务协调器"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MicroserviceOrchestrator()
    return _orchestrator


def init_orchestrator(config: Optional[Dict[str, Any]] = None) -> MicroserviceOrchestrator:
    """初始化全局微服务协调器"""
    global _orchestrator
    _orchestrator = MicroserviceOrchestrator(config)
    return _orchestrator


# =============================================================================
# FastAPI集成示例
# =============================================================================

def create_fastapi_app():
    """
    创建FastAPI应用示例

    示例代码，展示如何集成微服务架构
    """
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        logger.warning("FastAPI未安装，跳过应用创建")
        return None

    app = FastAPI(
        title="破夜绘明监测平台API",
        description="输变电激光星芒监测平台微服务API",
        version="1.0.0",
    )

    # CORS配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 获取协调器
    orchestrator = get_orchestrator()

    # 请求模型
    class TaskRequest(BaseModel):
        name: str
        func_name: str
        args: list = []
        kwargs: dict = {}
        priority: str = "normal"

    # 系统状态
    @app.get("/api/status")
    async def get_status():
        return orchestrator.get_system_status()

    # 服务列表
    @app.get("/api/services")
    async def get_services():
        return orchestrator.registry.get_all_services()

    # 提交任务
    @app.post("/api/tasks")
    async def submit_task(request: TaskRequest):
        try:
            priority = TaskPriority[request.priority.upper()]
        except KeyError:
            priority = TaskPriority.NORMAL

        task_id = orchestrator.submit_task(
            name=request.name,
            func_name=request.func_name,
            args=tuple(request.args),
            kwargs=request.kwargs,
            priority=priority,
        )

        return {"task_id": task_id}

    # 获取任务状态
    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str):
        task = orchestrator.task_queue.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        return task.to_dict()

    # WebSocket连接
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        await websocket.accept()

        # 注册客户端
        client = orchestrator.websocket.connect(
            client_id=client_id,
            connection=websocket,
            device_type="web",
        )

        try:
            while True:
                # 接收消息
                data = await websocket.receive_json()

                # 处理订阅
                if data.get("type") == "subscribe":
                    topics = data.get("topics", [])
                    orchestrator.websocket.subscribe(client_id, topics)

                # 处理取消订阅
                elif data.get("type") == "unsubscribe":
                    topics = data.get("topics", [])
                    orchestrator.websocket.unsubscribe(client_id, topics)

                # 更新活动时间
                client.last_activity = datetime.now()

        except WebSocketDisconnect:
            orchestrator.websocket.disconnect(client_id)

    return app


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # 创建协调器
    orchestrator = init_orchestrator({
        "max_workers": 4,
        "heartbeat_interval": 5.0,
    })

    # 启动服务
    orchestrator.start()

    # 注册示例任务处理器
    def sample_task(data: Dict) -> Dict:
        """示例任务"""
        time.sleep(1)
        return {"processed": True, "data": data}

    orchestrator.register_task_handler("sample_task", sample_task)

    # 提交任务
    task_id = orchestrator.submit_task(
        name="测试任务",
        func_name="sample_task",
        kwargs={"input": "test"},
    )

    print(f"提交任务: {task_id}")

    # 等待任务完成
    time.sleep(3)

    # 获取任务状态
    task = orchestrator.task_queue.get_task(task_id)
    if task:
        print(f"任务状态: {task.status.value}")
        print(f"任务结果: {task.result}")

    # 打印系统状态
    print("\n系统状态:")
    print(json.dumps(orchestrator.get_system_status(), indent=2, ensure_ascii=False))

    # 停止服务
    orchestrator.stop()
