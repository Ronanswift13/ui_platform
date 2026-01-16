"""
异步插件基础架构
================

基于UPDATE.md优化方案实现:
- 微服务插件化架构
- 异步任务处理
- 统一数据接口
- 健康检查和日志追踪

适用于: 所有检测插件的统一基类
"""

from __future__ import annotations
import asyncio
import hashlib
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PluginStatus(str, Enum):
    """插件状态"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AlarmLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class UnifiedResult:
    """
    统一输出结构

    所有插件必须返回此格式,便于前端解析和跨模块融合
    """
    # 基础信息
    plugin_id: str
    plugin_version: str
    task_id: str
    timestamp: str

    # 状态
    success: bool
    status: str                           # normal, warning, alarm, error

    # 检测结果
    detections: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0

    # 告警
    alarms: List[Dict[str, Any]] = field(default_factory=list)

    # 性能指标
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "success": self.success,
            "status": self.status,
            "detections": self.detections,
            "confidence": self.confidence,
            "alarms": self.alarms,
            "inference_time_ms": self.inference_time_ms,
            "total_time_ms": self.total_time_ms,
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


@dataclass
class TaskContext:
    """任务上下文"""
    task_id: str
    site_id: str = ""
    device_id: str = ""
    component_id: str = ""
    position_id: str = ""
    timestamp: float = field(default_factory=time.time)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """健康状态"""
    healthy: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# 异步任务队列
# =============================================================================
@dataclass
class AsyncTask(Generic[T]):
    """异步任务"""
    task_id: str
    plugin_id: str
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[T] = None
    error: Optional[str] = None
    progress: float = 0.0


class TaskQueue:
    """
    异步任务队列

    支持:
    - 任务排队
    - 优先级调度
    - 并发控制
    - 超时处理
    """

    def __init__(self, max_workers: int = 4, max_queue_size: int = 100):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._tasks: Dict[str, AsyncTask] = {}
        self._workers: List[asyncio.Task] = []
        self._running = False

    async def start(self):
        """启动任务队列"""
        self._running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        logger.info(f"[TaskQueue] 已启动 {self.max_workers} 个工作线程")

    async def stop(self):
        """停止任务队列"""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()
        logger.info("[TaskQueue] 已停止")

    async def submit(
        self,
        task_id: str,
        plugin_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> str:
        """提交任务"""
        task = AsyncTask(
            task_id=task_id,
            plugin_id=plugin_id,
            status=TaskStatus.PENDING,
            created_at=time.time(),
        )
        self._tasks[task_id] = task

        await self._queue.put((task_id, func, args, kwargs))
        logger.debug(f"[TaskQueue] 任务已提交: {task_id}")
        return task_id

    def get_task(self, task_id: str) -> Optional[AsyncTask]:
        """获取任务状态"""
        return self._tasks.get(task_id)

    async def wait_for_task(self, task_id: str, timeout: float = 60.0) -> Optional[Any]:
        """等待任务完成"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            task = self._tasks.get(task_id)
            if task and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return task.result
            await asyncio.sleep(0.1)
        return None

    async def _worker(self, worker_id: int):
        """工作线程"""
        while self._running:
            try:
                task_id, func, args, kwargs = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )

                task = self._tasks.get(task_id)
                if task is None:
                    continue

                task.status = TaskStatus.RUNNING
                task.started_at = time.time()

                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    task.result = result
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    logger.error(f"[TaskQueue] 任务失败 {task_id}: {e}")
                finally:
                    task.completed_at = time.time()
                    self._queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


# =============================================================================
# 增强版插件基类
# =============================================================================
class EnhancedBasePlugin(ABC):
    """
    增强版插件基类

    新增特性:
    1. 异步推理支持
    2. 统一输出格式
    3. 深度学习模型集成
    4. 健康检查和指标
    5. 日志追踪
    """

    def __init__(self, plugin_id: str, plugin_dir: Path):
        self.plugin_id = plugin_id
        self.plugin_dir = Path(plugin_dir)

        # 状态
        self._status = PluginStatus.UNLOADED
        self._last_error: str = ""
        self._initialized = False

        # 配置
        self._config: Dict[str, Any] = {}

        # 统计
        self._inference_count = 0
        self._error_count = 0
        self._total_inference_time = 0.0
        self._last_inference_time: Optional[datetime] = None

        # 深度学习模型
        self._dl_detector = None
        self._use_deep_learning = True

        # 计算代码哈希
        self._code_hash = self._compute_code_hash()

    @property
    def id(self) -> str:
        return self.plugin_id

    @property
    def version(self) -> str:
        return getattr(self, 'PLUGIN_VERSION', '1.0.0')

    @property
    def code_hash(self) -> str:
        return self._code_hash

    @property
    def status(self) -> PluginStatus:
        return self._status

    @status.setter
    def status(self, value: PluginStatus):
        self._status = value

    def _compute_code_hash(self) -> str:
        """计算代码哈希"""
        plugin_file = self.plugin_dir / "plugin.py"
        if plugin_file.exists():
            content = plugin_file.read_bytes()
            return f"sha256:{hashlib.sha256(content).hexdigest()[:12]}"
        return "unknown"

    # =========================================================================
    # 抽象方法 (子类必须实现)
    # =========================================================================

    @abstractmethod
    def init(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        pass

    @abstractmethod
    def process(self, inputs: Dict[str, Any], context: TaskContext) -> UnifiedResult:
        """
        同步处理 (核心方法)

        Args:
            inputs: 输入数据 (图像、音频、传感器数据等)
            context: 任务上下文

        Returns:
            统一格式的结果
        """
        pass

    # =========================================================================
    # 异步处理方法
    # =========================================================================

    async def process_async(
        self,
        inputs: Dict[str, Any],
        context: TaskContext,
    ) -> UnifiedResult:
        """
        异步处理

        默认实现: 在线程池中运行同步方法
        子类可以重写以实现真正的异步处理
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.process, inputs, context
        )

    async def batch_process_async(
        self,
        batch_inputs: List[Dict[str, Any]],
        context: TaskContext,
    ) -> List[UnifiedResult]:
        """批量异步处理"""
        tasks = [
            self.process_async(inputs, context)
            for inputs in batch_inputs
        ]
        return await asyncio.gather(*tasks)

    # =========================================================================
    # 深度学习集成
    # =========================================================================

    def init_deep_learning(self, model_type: str, **kwargs) -> bool:
        """
        初始化深度学习模型

        Args:
            model_type: 模型类型 (yolov8_vit, gl_translstm, acoustic_cnn_lstm, etc.)
            **kwargs: 模型配置参数

        Returns:
            是否成功
        """
        try:
            if model_type == "yolov8_vit":
                from ai_models.deep_learning.yolov8_vit import YOLOv8ViTDetector, YOLOv8ViTConfig
                config = YOLOv8ViTConfig(**kwargs)
                self._dl_detector = YOLOv8ViTDetector(config)
                self._dl_detector.load()

            elif model_type == "gl_translstm":
                from ai_models.deep_learning.gl_translstm import GLTransLSTM, GLTransLSTMConfig
                config = GLTransLSTMConfig(**kwargs)
                self._dl_detector = GLTransLSTM(config)
                self._dl_detector.initialize()

            elif model_type == "acoustic_cnn_lstm":
                from ai_models.deep_learning.acoustic_cnn_lstm import AcousticCNNLSTM, AcousticModelConfig
                config = AcousticModelConfig(**kwargs)
                self._dl_detector = AcousticCNNLSTM(config)
                self._dl_detector.load()

            elif model_type == "keypoint":
                from ai_models.deep_learning.keypoint_detector import KeypointDetector, KeypointConfig
                config = KeypointConfig(**kwargs)
                self._dl_detector = KeypointDetector(config)
                self._dl_detector.load()

            logger.info(f"[{self.plugin_id}] 深度学习模型初始化成功: {model_type}")
            return True

        except Exception as e:
            logger.error(f"[{self.plugin_id}] 深度学习模型初始化失败: {e}")
            return False

    def detect_with_dl(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用深度学习模型检测"""
        if self._dl_detector is None:
            return []

        try:
            if hasattr(self._dl_detector, 'detect'):
                result = self._dl_detector.detect(image)
                return [
                    {
                        "bbox": d.bbox,
                        "label": d.class_name,
                        "confidence": d.confidence,
                        "class_id": d.class_id,
                    }
                    for d in result.detections
                ]
            elif hasattr(self._dl_detector, 'analyze'):
                result = self._dl_detector.analyze(image)
                return result.anomalies if hasattr(result, 'anomalies') else []
            else:
                return []
        except Exception as e:
            logger.error(f"[{self.plugin_id}] 深度学习检测失败: {e}")
            return []

    # =========================================================================
    # 结果构建辅助方法
    # =========================================================================

    def create_result(
        self,
        task_id: str,
        success: bool,
        status: str = "normal",
        detections: Optional[List[Dict]] = None,
        alarms: Optional[List[Dict]] = None,
        confidence: float = 0.0,
        inference_time_ms: float = 0.0,
        metadata: Optional[Dict] = None,
        error_message: Optional[str] = None,
    ) -> UnifiedResult:
        """创建统一结果"""
        return UnifiedResult(
            plugin_id=self.plugin_id,
            plugin_version=self.version,
            task_id=task_id,
            timestamp=datetime.now().isoformat(),
            success=success,
            status=status,
            detections=detections or [],
            confidence=confidence,
            alarms=alarms or [],
            inference_time_ms=inference_time_ms,
            total_time_ms=inference_time_ms,
            metadata=metadata or {},
            error_message=error_message,
        )

    def create_alarm(
        self,
        level: AlarmLevel,
        title: str,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """创建告警"""
        return {
            "level": level.value,
            "title": title,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "plugin_id": self.plugin_id,
            **kwargs
        }

    def create_detection(
        self,
        label: str,
        confidence: float,
        bbox: Optional[Dict[str, float]] = None,
        value: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """创建检测结果"""
        det = {
            "label": label,
            "confidence": confidence,
            "plugin_id": self.plugin_id,
            "model_version": self.version,
            "code_hash": self.code_hash,
        }
        if bbox:
            det["bbox"] = bbox
        if value is not None:
            det["value"] = value
        det.update(kwargs)
        return det

    # =========================================================================
    # 健康检查和统计
    # =========================================================================

    def healthcheck(self) -> HealthStatus:
        """健康检查"""
        if not self._initialized:
            return HealthStatus(
                healthy=False,
                message="插件未初始化",
                details={"status": self._status.value}
            )

        return HealthStatus(
            healthy=True,
            message="插件运行正常",
            details={
                "status": self._status.value,
                "inference_count": self._inference_count,
                "error_count": self._error_count,
                "avg_inference_time_ms": self.avg_inference_time,
                "last_inference": self._last_inference_time.isoformat() if self._last_inference_time else None,
                "deep_learning_enabled": self._dl_detector is not None,
            }
        )

    @property
    def avg_inference_time(self) -> float:
        """平均推理时间"""
        if self._inference_count == 0:
            return 0.0
        return self._total_inference_time / self._inference_count

    def record_inference(self, time_ms: float, success: bool = True):
        """记录推理统计"""
        self._inference_count += 1
        self._total_inference_time += time_ms
        self._last_inference_time = datetime.now()
        if not success:
            self._error_count += 1

    # =========================================================================
    # 生命周期方法
    # =========================================================================

    def shutdown(self) -> bool:
        """关闭插件"""
        try:
            if self._dl_detector and hasattr(self._dl_detector, 'cleanup'):
                self._dl_detector.cleanup()
            self._initialized = False
            self._status = PluginStatus.UNLOADED
            logger.info(f"[{self.plugin_id}] 插件已关闭")
            return True
        except Exception as e:
            logger.error(f"[{self.plugin_id}] 关闭失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        self.shutdown()


# =============================================================================
# 插件管理器扩展
# =============================================================================
class AsyncPluginManager:
    """
    异步插件管理器

    支持:
    - 插件动态加载/卸载
    - 异步任务调度
    - 负载均衡
    - 故障恢复
    """

    def __init__(self, plugins_dir: Path, max_workers: int = 4):
        self.plugins_dir = Path(plugins_dir)
        self._plugins: Dict[str, EnhancedBasePlugin] = {}
        self._task_queue = TaskQueue(max_workers=max_workers)

    async def start(self):
        """启动管理器"""
        await self._task_queue.start()
        logger.info("[AsyncPluginManager] 已启动")

    async def stop(self):
        """停止管理器"""
        await self._task_queue.stop()
        for plugin in self._plugins.values():
            plugin.shutdown()
        logger.info("[AsyncPluginManager] 已停止")

    def register_plugin(self, plugin: EnhancedBasePlugin):
        """注册插件"""
        self._plugins[plugin.id] = plugin
        logger.info(f"[AsyncPluginManager] 注册插件: {plugin.id}")

    def get_plugin(self, plugin_id: str) -> Optional[EnhancedBasePlugin]:
        """获取插件"""
        return self._plugins.get(plugin_id)

    async def submit_task(
        self,
        plugin_id: str,
        inputs: Dict[str, Any],
        context: TaskContext,
    ) -> str:
        """提交异步任务"""
        plugin = self._plugins.get(plugin_id)
        if plugin is None:
            raise ValueError(f"插件不存在: {plugin_id}")

        task_id = context.task_id
        await self._task_queue.submit(
            task_id, plugin_id,
            plugin.process_async, inputs, context
        )
        return task_id

    async def get_task_result(
        self,
        task_id: str,
        timeout: float = 60.0
    ) -> Optional[UnifiedResult]:
        """获取任务结果"""
        return await self._task_queue.wait_for_task(task_id, timeout)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有插件"""
        return [
            {
                "id": p.id,
                "version": p.version,
                "status": p.status.value,
                "healthy": p.healthcheck().healthy,
            }
            for p in self._plugins.values()
        ]


# =============================================================================
# 便捷函数
# =============================================================================

def create_task_context(
    site_id: str = "",
    device_id: str = "",
    **kwargs
) -> TaskContext:
    """创建任务上下文"""
    import uuid
    return TaskContext(
        task_id=str(uuid.uuid4()),
        site_id=site_id,
        device_id=device_id,
        **kwargs
    )
