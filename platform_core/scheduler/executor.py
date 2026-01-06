"""
任务执行器

封装单个任务的执行逻辑
"""


from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np

from platform_core.logging import get_logger
from platform_core.plugin_manager.base import PluginContext
from platform_core.schema.models import AlarmRule, PluginOutput, ROI, Task

logger = get_logger(__name__)


@dataclass
class ExecutionContext:
    """执行上下文"""
    task: Task
    frame: np.ndarray
    rois: list[ROI]
    rules: list[AlarmRule]
    config: dict[str, Any]
    started_at: Optional[datetime] = None

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now()


class TaskExecutor:
    """
    任务执行器

    负责单个任务的执行流程
    """

    def __init__(self, plugin_manager, evidence_manager):
        self.plugin_manager = plugin_manager
        self.evidence_manager = evidence_manager

    def execute(self, ctx: ExecutionContext) -> PluginOutput:
        """
        执行任务

        Args:
            ctx: 执行上下文

        Returns:
            插件输出
        """
        task = ctx.task

        # 加载插件
        plugin = self.plugin_manager.get_plugin(task.plugin_id)
        if plugin is None:
            plugin = self.plugin_manager.load_plugin(task.plugin_id)

        # 创建插件上下文
        plugin_context = PluginContext(
            task_id=task.id,
            site_id=task.site_id,
            device_id=task.device_id,
            config=ctx.config,
        )

        # 执行推理
        results = plugin.infer(ctx.frame, ctx.rois, plugin_context)

        # 后处理
        alarms = []
        if ctx.rules:
            alarms = plugin.postprocess(results, ctx.rules)

        # 创建输出
        return plugin.create_output(
            task_id=task.id,
            results=results,
            alarms=alarms,
        )


class BatchExecutor:
    """
    批量执行器

    支持多帧批处理
    """

    def __init__(self, plugin_manager, evidence_manager, batch_size: int = 4):
        self.plugin_manager = plugin_manager
        self.evidence_manager = evidence_manager
        self.batch_size = batch_size

    def execute_batch(
        self,
        task: Task,
        frames: list[np.ndarray],
        rois: list[ROI],
        rules: list[AlarmRule] | None = None,
    ) -> list[PluginOutput]:
        """
        批量执行

        Args:
            task: 任务对象
            frames: 图像帧列表
            rois: ROI列表
            rules: 告警规则

        Returns:
            输出列表
        """
        outputs = []

        # 分批处理
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]

            for frame in batch_frames:
                ctx = ExecutionContext(
                    task=task,
                    frame=frame,
                    rois=rois,
                    rules=rules or [],
                    config=task.config,
                )
                executor = TaskExecutor(self.plugin_manager, self.evidence_manager)
                output = executor.execute(ctx)
                outputs.append(output)

        return outputs
