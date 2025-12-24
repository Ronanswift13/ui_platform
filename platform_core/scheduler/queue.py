"""
任务队列

支持优先级调度的任务队列
"""


from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from threading import Lock
from typing import Optional

from platform_core.schema.models import Task


class TaskPriority(IntEnum):
    """任务优先级"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass(order=True)
class PriorityTask:
    """带优先级的任务包装"""
    priority: int
    created_at: datetime = field(compare=False)
    task: Task = field(compare=False)

    def __init__(self, task: Task, priority: TaskPriority = TaskPriority.NORMAL):
        self.priority = priority.value
        self.created_at = datetime.now()
        self.task = task


class TaskQueue:
    """
    任务队列

    线程安全的优先级队列
    """

    def __init__(self, max_size: int = 1000):
        self._queue: list[PriorityTask] = []
        self._lock = Lock()
        self._max_size = max_size
        self._task_ids: set[str] = set()

    def put(self, task: Task, priority: TaskPriority = TaskPriority.NORMAL) -> bool:
        """
        添加任务到队列

        Args:
            task: 任务对象
            priority: 优先级

        Returns:
            是否添加成功
        """
        with self._lock:
            if len(self._queue) >= self._max_size:
                return False

            if task.id in self._task_ids:
                return False

            priority_task = PriorityTask(task, priority)
            heapq.heappush(self._queue, priority_task)
            self._task_ids.add(task.id)
            return True

    def get(self) -> Optional[Task]:
        """获取下一个任务"""
        with self._lock:
            if not self._queue:
                return None

            priority_task = heapq.heappop(self._queue)
            self._task_ids.discard(priority_task.task.id)
            return priority_task.task

    def peek(self) -> Optional[Task]:
        """查看下一个任务但不移除"""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0].task

    def remove(self, task_id: str) -> bool:
        """从队列中移除任务"""
        with self._lock:
            if task_id not in self._task_ids:
                return False

            self._queue = [pt for pt in self._queue if pt.task.id != task_id]
            heapq.heapify(self._queue)
            self._task_ids.discard(task_id)
            return True

    def contains(self, task_id: str) -> bool:
        """检查任务是否在队列中"""
        return task_id in self._task_ids

    def size(self) -> int:
        """获取队列大小"""
        return len(self._queue)

    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return len(self._queue) == 0

    def is_full(self) -> bool:
        """检查队列是否已满"""
        return len(self._queue) >= self._max_size

    def clear(self) -> int:
        """清空队列"""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            self._task_ids.clear()
            return count

    def list_tasks(self) -> list[Task]:
        """列出所有待处理任务"""
        with self._lock:
            return [pt.task for pt in sorted(self._queue)]
