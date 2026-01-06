"""
任务引擎

核心调度引擎,负责:
- 任务生命周期管理
- 任务与插件的串联
- 失败重试策略
- 任务优先级调度
"""


from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from uuid import uuid4

import numpy as np

from platform_core.config import get_config
from platform_core.evidence.manager import EvidenceManager, get_evidence_manager
from platform_core.exceptions import TaskError, TaskTimeoutError
from platform_core.logging import get_logger, TaskLogger
from platform_core.plugin_manager import PluginManager
from platform_core.plugin_manager.base import PluginContext
from platform_core.schema.models import (
    AlarmRule,
    PluginOutput,
    ROI,
    Task,
    TaskStatus,
)

logger = get_logger(__name__)


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    run_id: str
    success: bool
    output: Optional[PluginOutput] = None
    error_message: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    retry_count: int = 0

    @property
    def duration_ms(self) -> float:
        if self.completed_at is None:
            return 0
        return (self.completed_at - self.started_at).total_seconds() * 1000


class TaskEngine:
    """
    任务引擎

    管理任务的完整执行流程:
    点位 → 任务 → 插件 → ROI → 规则 → 结果
    """

    _instance: Optional["TaskEngine"] = None

    def __new__(cls) -> "TaskEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.config = get_config()
        self.plugin_manager = PluginManager()
        self.evidence_manager = get_evidence_manager()

        self._executor = ThreadPoolExecutor(
            max_workers=self.config.scheduler.max_workers,
            thread_name_prefix="task_worker_",
        )
        self._running_tasks: dict[str, Task] = {}
        self._task_results: dict[str, TaskResult] = {}
        self._callbacks: dict[str, list[Callable]] = {
            "on_start": [],
            "on_complete": [],
            "on_error": [],
        }

        self._initialized = True
        logger.info("任务引擎初始化完成")

    def register_callback(self, event: str, callback: Callable) -> None:
        """注册回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args, **kwargs) -> None:
        """触发回调"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"回调执行失败 [{event}]: {e}")

    async def execute_task(
        self,
        task: Task,
        frame: np.ndarray,
        rois: list[ROI],
        rules: list[AlarmRule] | None = None,
    ) -> TaskResult:
        """
        异步执行任务

        Args:
            task: 任务对象
            frame: 输入图像帧
            rois: ROI列表
            rules: 告警规则列表

        Returns:
            任务执行结果
        """
        result = TaskResult(
            task_id=task.id,
            run_id=str(uuid4()),
            success=False,  # 初始化为False，成功后设置为True
        )

        try:
            # 更新任务状态
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self._running_tasks[task.id] = task
            self._emit("on_start", task)

            # 创建证据会话
            plugin = self.plugin_manager.get_plugin(task.plugin_id)
            if plugin is None:
                plugin = self.plugin_manager.load_plugin(task.plugin_id)

            with self.evidence_manager.create_session(
                task_id=task.id,
                plugin_id=task.plugin_id,
                plugin_version=plugin.version,
                code_hash=plugin.code_hash,
                site_id=task.site_id,
                device_id=task.device_id,
            ) as evidence:
                result.run_id = evidence.run_id

                # 保存原图
                evidence.save_raw_image(frame, "input")

                # 创建上下文
                context = PluginContext(
                    task_id=task.id,
                    site_id=task.site_id,
                    device_id=task.device_id,
                    config=task.config,
                )

                # 执行插件 (在线程池中)
                loop = asyncio.get_event_loop()
                output = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        self._execute_plugin,
                        task.plugin_id,
                        frame,
                        rois,
                        context,
                        rules,
                    ),
                    timeout=self.config.scheduler.task_timeout,
                )

                # 保存结果
                evidence.save_result(output)

                # 保存标注图 (如果有)
                if output.results:
                    annotated = self._draw_annotations(frame, output)
                    evidence.save_annotated_image(annotated, "annotated")

                result.output = output
                result.success = output.success

        except asyncio.TimeoutError:
            result.success = False
            result.error_message = f"任务超时 ({self.config.scheduler.task_timeout}s)"
            task.status = TaskStatus.FAILED
            self._emit("on_error", task, result.error_message)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            task.status = TaskStatus.FAILED
            logger.exception(f"任务执行失败 [{task.id}]: {e}")
            self._emit("on_error", task, str(e))

        finally:
            result.completed_at = datetime.now()
            task.completed_at = result.completed_at

            if result.success:
                task.status = TaskStatus.COMPLETED
            else:
                task.status = TaskStatus.FAILED
                task.error_message = result.error_message

            self._running_tasks.pop(task.id, None)
            self._task_results[task.id] = result
            self._emit("on_complete", task, result)

        return result

    def execute_task_sync(
        self,
        task: Task,
        frame: np.ndarray,
        rois: list[ROI],
        rules: list[AlarmRule] | None = None,
    ) -> TaskResult:
        """同步执行任务"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.execute_task(task, frame, rois, rules)
            )
        finally:
            loop.close()

    def _execute_plugin(
        self,
        plugin_id: str,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
        rules: list[AlarmRule] | None,
    ) -> PluginOutput:
        """在线程池中执行插件"""
        return self.plugin_manager.execute_plugin(
            plugin_id=plugin_id,
            frame=frame,
            rois=rois,
            context=context,
            rules=rules,
        )

    def _draw_annotations(self, frame: np.ndarray, output: PluginOutput) -> np.ndarray:
        """在图像上绘制标注"""
        import cv2

        annotated = frame.copy()
        h, w = frame.shape[:2]

        for result in output.results:
            bbox = result.bbox
            x1 = int(bbox.x * w)
            y1 = int(bbox.y * h)
            x2 = int((bbox.x + bbox.width) * w)
            y2 = int((bbox.y + bbox.height) * h)

            # 根据置信度选择颜色
            if result.confidence >= 0.8:
                color = (0, 255, 0)  # 绿色
            elif result.confidence >= 0.5:
                color = (0, 255, 255)  # 黄色
            else:
                color = (0, 0, 255)  # 红色

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{result.label}: {result.confidence:.2f}"
            if result.value is not None:
                label += f" ({result.value})"

            cv2.putText(
                annotated, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1,
            )

        return annotated

    def get_running_tasks(self) -> list[Task]:
        """获取正在运行的任务"""
        return list(self._running_tasks.values())

    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        return self._task_results.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            self._running_tasks.pop(task_id, None)
            logger.info(f"任务已取消: {task_id}")
            return True
        return False

    def shutdown(self) -> None:
        """关闭任务引擎"""
        self._executor.shutdown(wait=True)
        logger.info("任务引擎已关闭")


def get_task_engine() -> TaskEngine:
    """获取任务引擎单例"""
    return TaskEngine()
