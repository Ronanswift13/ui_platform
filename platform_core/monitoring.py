"""
系统监控模块
输变电激光星芒破夜绘明监测平台

基于UPDATE.md中期迭代计划:
- Prometheus指标导出
- ELK日志追踪
- 分布式追踪
- 性能监控
- 告警规则

功能:
- 指标收集与导出
- 结构化日志
- 请求追踪
- 性能分析
- 健康检查

版本: 1.0.0
"""

from __future__ import annotations
import functools
import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举定义
# =============================================================================

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"             # 计数器 (只增)
    GAUGE = "gauge"                 # 仪表 (可增可减)
    HISTOGRAM = "histogram"         # 直方图
    SUMMARY = "summary"             # 摘要


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class MetricValue:
    """指标值"""
    name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

    def to_prometheus(self) -> str:
        """转换为Prometheus格式"""
        # 标签字符串
        if self.labels:
            label_str = ",".join(
                f'{k}="{v}"' for k, v in self.labels.items()
            )
            return f'{self.name}{{{label_str}}} {self.value}'
        else:
            return f'{self.name} {self.value}'


@dataclass
class HistogramBucket:
    """直方图桶"""
    le: float                       # 上限
    count: int = 0                  # 计数


@dataclass
class HistogramValue:
    """直方图值"""
    name: str
    buckets: List[HistogramBucket] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0
    labels: Dict[str, str] = field(default_factory=dict)

    def observe(self, value: float):
        """记录观测值"""
        self.sum_value += value
        self.count += 1

        for bucket in self.buckets:
            if value <= bucket.le:
                bucket.count += 1

    def to_prometheus(self) -> List[str]:
        """转换为Prometheus格式"""
        lines = []
        label_str = ""
        if self.labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())

        for bucket in self.buckets:
            if label_str:
                lines.append(
                    f'{self.name}_bucket{{le="{bucket.le}",{label_str}}} {bucket.count}'
                )
            else:
                lines.append(
                    f'{self.name}_bucket{{le="{bucket.le}"}} {bucket.count}'
                )

        # +Inf桶
        if label_str:
            lines.append(f'{self.name}_bucket{{le="+Inf",{label_str}}} {self.count}')
        else:
            lines.append(f'{self.name}_bucket{{le="+Inf"}} {self.count}')

        # sum和count
        if label_str:
            lines.append(f'{self.name}_sum{{{label_str}}} {self.sum_value}')
            lines.append(f'{self.name}_count{{{label_str}}} {self.count}')
        else:
            lines.append(f'{self.name}_sum {self.sum_value}')
            lines.append(f'{self.name}_count {self.count}')

        return lines


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str = "root"
    module: str = ""
    function: str = ""
    line_number: int = 0

    # 追踪信息
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    # 上下文
    context: Dict[str, Any] = field(default_factory=dict)

    # 异常信息
    exception: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (ELK格式)"""
        return {
            "@timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "logger": self.logger_name,
            "module": self.module,
            "function": self.function,
            "line": self.line_number,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "context": self.context,
            "exception": self.exception,
            "stack_trace": self.stack_trace,
        }

    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Span:
    """追踪跨度"""
    span_id: str
    trace_id: str
    name: str
    service_name: str
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "ok"
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """持续时间(毫秒)"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0

    def finish(self, status: str = "ok"):
        """结束跨度"""
        self.end_time = datetime.now()
        self.status = status

    def add_tag(self, key: str, value: str):
        """添加标签"""
        self.tags[key] = value

    def add_log(self, message: str, **kwargs):
        """添加日志"""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            **kwargs,
        })

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "service_name": self.service_name,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs,
        }


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str                  # e.g., "> 0.9", "< 100"
    threshold: float
    severity: AlertSeverity
    duration: float = 60.0          # 持续时间(秒)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    # 运行时状态
    last_evaluated: Optional[datetime] = None
    last_triggered: Optional[datetime] = None
    firing: bool = False
    firing_since: Optional[datetime] = None


@dataclass
class Alert:
    """告警"""
    alert_id: str
    rule: AlertRule
    value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule.rule_id,
            "name": self.rule.name,
            "severity": self.rule.severity.value,
            "value": self.value,
            "threshold": self.rule.threshold,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "labels": self.labels,
            "annotations": self.annotations,
        }


@dataclass
class HealthCheck:
    """健康检查结果"""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# 指标收集器
# =============================================================================

class MetricsCollector:
    """Prometheus风格指标收集器"""

    # 默认直方图桶
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self, prefix: str = "poyehuiming"):
        self.prefix = prefix

        # 指标存储
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[str, HistogramValue]] = defaultdict(dict)

        # 指标元数据
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # 锁
        self._lock = threading.RLock()

        logger.info(f"指标收集器初始化完成, prefix={prefix}")

    def _make_key(self, labels: Dict[str, str]) -> str:
        """生成标签键"""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _full_name(self, name: str) -> str:
        """完整指标名"""
        return f"{self.prefix}_{name}"

    # -------------------------------------------------------------------------
    # 计数器
    # -------------------------------------------------------------------------

    def counter_inc(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        description: str = ""
    ):
        """计数器递增"""
        full_name = self._full_name(name)
        key = self._make_key(labels or {})

        with self._lock:
            self._counters[full_name][key] += value

            if full_name not in self._metadata:
                self._metadata[full_name] = {
                    "type": MetricType.COUNTER,
                    "description": description,
                }

    def counter_get(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> float:
        """获取计数器值"""
        full_name = self._full_name(name)
        key = self._make_key(labels or {})

        with self._lock:
            return self._counters[full_name].get(key, 0.0)

    # -------------------------------------------------------------------------
    # 仪表
    # -------------------------------------------------------------------------

    def gauge_set(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        description: str = ""
    ):
        """设置仪表值"""
        full_name = self._full_name(name)
        key = self._make_key(labels or {})

        with self._lock:
            self._gauges[full_name][key] = value

            if full_name not in self._metadata:
                self._metadata[full_name] = {
                    "type": MetricType.GAUGE,
                    "description": description,
                }

    def gauge_inc(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ):
        """仪表递增"""
        full_name = self._full_name(name)
        key = self._make_key(labels or {})

        with self._lock:
            self._gauges[full_name][key] += value

    def gauge_dec(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ):
        """仪表递减"""
        full_name = self._full_name(name)
        key = self._make_key(labels or {})

        with self._lock:
            self._gauges[full_name][key] -= value

    def gauge_get(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> float:
        """获取仪表值"""
        full_name = self._full_name(name)
        key = self._make_key(labels or {})

        with self._lock:
            return self._gauges[full_name].get(key, 0.0)

    # -------------------------------------------------------------------------
    # 直方图
    # -------------------------------------------------------------------------

    def histogram_observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
        description: str = ""
    ):
        """记录直方图观测值"""
        full_name = self._full_name(name)
        key = self._make_key(labels or {})

        with self._lock:
            if key not in self._histograms[full_name]:
                bucket_values = buckets or self.DEFAULT_BUCKETS
                self._histograms[full_name][key] = HistogramValue(
                    name=full_name,
                    buckets=[HistogramBucket(le=b) for b in bucket_values],
                    labels=labels or {},
                )

                if full_name not in self._metadata:
                    self._metadata[full_name] = {
                        "type": MetricType.HISTOGRAM,
                        "description": description,
                    }

            self._histograms[full_name][key].observe(value)

    # -------------------------------------------------------------------------
    # 导出
    # -------------------------------------------------------------------------

    def export_prometheus(self) -> str:
        """导出Prometheus格式"""
        lines = []

        with self._lock:
            # 计数器
            for name, values in self._counters.items():
                meta = self._metadata.get(name, {})
                if meta.get("description"):
                    lines.append(f"# HELP {name} {meta['description']}")
                lines.append(f"# TYPE {name} counter")

                for key, value in values.items():
                    if key:
                        lines.append(f"{name}{{{key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            # 仪表
            for name, values in self._gauges.items():
                meta = self._metadata.get(name, {})
                if meta.get("description"):
                    lines.append(f"# HELP {name} {meta['description']}")
                lines.append(f"# TYPE {name} gauge")

                for key, value in values.items():
                    if key:
                        lines.append(f"{name}{{{key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            # 直方图
            for name, histograms in self._histograms.items():
                meta = self._metadata.get(name, {})
                if meta.get("description"):
                    lines.append(f"# HELP {name} {meta['description']}")
                lines.append(f"# TYPE {name} histogram")

                for histogram in histograms.values():
                    lines.extend(histogram.to_prometheus())

        return "\n".join(lines)

    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self._lock:
            return {
                "counters": {
                    name: dict(values)
                    for name, values in self._counters.items()
                },
                "gauges": {
                    name: dict(values)
                    for name, values in self._gauges.items()
                },
                "histograms": {
                    name: {
                        key: {
                            "count": h.count,
                            "sum": h.sum_value,
                        }
                        for key, h in histograms.items()
                    }
                    for name, histograms in self._histograms.items()
                },
            }

    @contextmanager
    def timer(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        description: str = ""
    ):
        """
        计时器上下文管理器

        使用示例:
            with metrics.timer("request_duration", {"endpoint": "/api/test"}):
                process_request()
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.histogram_observe(name, duration, labels, description=description)


# =============================================================================
# 结构化日志
# =============================================================================

class StructuredLogger:
    """结构化日志记录器 (ELK兼容)"""

    def __init__(
        self,
        name: str = "poyehuiming",
        level: LogLevel = LogLevel.INFO,
        output_json: bool = True,
        log_file: Optional[str] = None,
        max_queue_size: int = 10000
    ):
        self.name = name
        self.level = level
        self.output_json = output_json
        self.log_file = log_file

        # 上下文线程本地存储
        self._context = threading.local()

        # 日志队列 (异步写入)
        self._queue: Queue = Queue(maxsize=max_queue_size)

        # 写入线程
        self._running = False
        self._writer_thread: Optional[threading.Thread] = None

        # 日志处理器
        self._handlers: List[Callable[[LogEntry], None]] = []

        # 文件句柄
        self._file_handle = None

        logger.info(f"结构化日志记录器初始化: {name}")

    def start(self):
        """启动日志写入线程"""
        self._running = True

        # 打开日志文件
        if self.log_file:
            self._file_handle = open(self.log_file, "a", encoding="utf-8")

        self._writer_thread = threading.Thread(
            target=self._write_loop,
            daemon=True
        )
        self._writer_thread.start()

    def stop(self):
        """停止日志写入"""
        self._running = False

        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)

        if self._file_handle:
            self._file_handle.close()

    def add_handler(self, handler: Callable[[LogEntry], None]):
        """添加日志处理器"""
        self._handlers.append(handler)

    def set_context(self, **kwargs):
        """设置上下文"""
        if not hasattr(self._context, "data"):
            self._context.data = {}
        self._context.data.update(kwargs)

    def clear_context(self):
        """清除上下文"""
        self._context.data = {}

    def get_context(self) -> Dict[str, Any]:
        """获取上下文"""
        return getattr(self._context, "data", {})

    def _should_log(self, level: LogLevel) -> bool:
        """检查是否应该记录"""
        level_order = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]
        return level_order.index(level) >= level_order.index(self.level)

    def _create_entry(
        self,
        level: LogLevel,
        message: str,
        exc_info: bool = False,
        **kwargs
    ) -> LogEntry:
        """创建日志条目"""
        # 获取调用信息
        frame = sys._getframe(3)
        module = frame.f_globals.get("__name__", "")
        function = frame.f_code.co_name
        line_number = frame.f_lineno

        # 合并上下文
        context = {**self.get_context(), **kwargs}

        # 获取追踪信息
        trace_id = context.pop("trace_id", None)
        span_id = context.pop("span_id", None)
        parent_span_id = context.pop("parent_span_id", None)

        # 异常信息
        exception = None
        stack_trace = None
        if exc_info:
            exc_type, exc_value, exc_tb = sys.exc_info()
            if exc_type:
                exception = f"{exc_type.__name__}: {exc_value}"
                stack_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

        return LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            logger_name=self.name,
            module=module,
            function=function,
            line_number=line_number,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            context=context,
            exception=exception,
            stack_trace=stack_trace,
        )

    def _log(self, level: LogLevel, message: str, exc_info: bool = False, **kwargs):
        """记录日志"""
        if not self._should_log(level):
            return

        entry = self._create_entry(level, message, exc_info, **kwargs)

        # 入队
        try:
            self._queue.put_nowait(entry)
        except Exception:
            # 队列满，直接输出
            self._output_entry(entry)

    def debug(self, message: str, **kwargs):
        """DEBUG日志"""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """INFO日志"""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """WARNING日志"""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """ERROR日志"""
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """CRITICAL日志"""
        self._log(LogLevel.CRITICAL, message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs):
        """异常日志"""
        self.error(message, exc_info=True, **kwargs)

    def _write_loop(self):
        """写入循环"""
        while self._running:
            try:
                entry = self._queue.get(timeout=1.0)
                self._output_entry(entry)

                # 调用处理器
                for handler in self._handlers:
                    try:
                        handler(entry)
                    except Exception as e:
                        print(f"日志处理器异常: {e}", file=sys.stderr)

            except Empty:
                continue
            except Exception as e:
                print(f"日志写入异常: {e}", file=sys.stderr)

    def _output_entry(self, entry: LogEntry):
        """输出日志条目"""
        if self.output_json:
            output = entry.to_json()
        else:
            # 传统格式
            output = (
                f"{entry.timestamp.isoformat()} [{entry.level.value.upper()}] "
                f"{entry.logger_name}: {entry.message}"
            )
            if entry.context:
                output += f" | {entry.context}"
            if entry.exception:
                output += f" | {entry.exception}"

        # 输出到控制台
        print(output)

        # 输出到文件
        if self._file_handle:
            self._file_handle.write(output + "\n")
            self._file_handle.flush()


# =============================================================================
# 分布式追踪
# =============================================================================

class Tracer:
    """分布式追踪器"""

    def __init__(
        self,
        service_name: str = "poyehuiming",
        sample_rate: float = 1.0
    ):
        self.service_name = service_name
        self.sample_rate = sample_rate

        # 追踪上下文
        self._context = threading.local()

        # 跨度存储
        self._spans: Dict[str, Span] = {}

        # 完成的跨度 (用于导出)
        self._finished_spans: List[Span] = []
        self._max_finished_spans = 10000

        # 锁
        self._lock = threading.RLock()

        logger.info(f"追踪器初始化: {service_name}")

    def _should_sample(self) -> bool:
        """是否采样"""
        import random
        return random.random() < self.sample_rate

    def start_span(
        self,
        name: str,
        parent: Optional[Span] = None,
        trace_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Span:
        """
        开始新跨度

        Args:
            name: 跨度名称
            parent: 父跨度
            trace_id: 追踪ID
            tags: 标签

        Returns:
            新跨度
        """
        # 确定追踪ID
        if trace_id:
            tid = trace_id
        elif parent:
            tid = parent.trace_id
        elif hasattr(self._context, "trace_id"):
            tid = self._context.trace_id
        else:
            tid = str(uuid.uuid4())

        # 创建跨度
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=tid,
            name=name,
            service_name=self.service_name,
            parent_span_id=parent.span_id if parent else None,
            tags=tags or {},
        )

        # 保存到上下文
        self._context.trace_id = tid
        self._context.current_span = span

        with self._lock:
            self._spans[span.span_id] = span

        return span

    def finish_span(self, span: Span, status: str = "ok"):
        """
        结束跨度

        Args:
            span: 跨度
            status: 状态
        """
        span.finish(status)

        with self._lock:
            if span.span_id in self._spans:
                del self._spans[span.span_id]

            self._finished_spans.append(span)

            # 限制大小
            while len(self._finished_spans) > self._max_finished_spans:
                self._finished_spans.pop(0)

    def get_current_span(self) -> Optional[Span]:
        """获取当前跨度"""
        return getattr(self._context, "current_span", None)

    def get_current_trace_id(self) -> Optional[str]:
        """获取当前追踪ID"""
        return getattr(self._context, "trace_id", None)

    @contextmanager
    def trace(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        追踪上下文管理器

        使用示例:
            with tracer.trace("process_request", {"endpoint": "/api/test"}):
                process()
        """
        parent = self.get_current_span()
        span = self.start_span(name, parent=parent, tags=tags)

        try:
            yield span
            self.finish_span(span, "ok")
        except Exception as e:
            span.add_tag("error", str(e))
            span.add_log("exception", error=str(e))
            self.finish_span(span, "error")
            raise

    def inject_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        注入追踪头 (用于HTTP请求传播)

        Args:
            headers: 原始头

        Returns:
            包含追踪信息的头
        """
        span = self.get_current_span()
        if span:
            headers["X-Trace-ID"] = span.trace_id
            headers["X-Span-ID"] = span.span_id

        return headers

    def extract_headers(self, headers: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
        """
        提取追踪头

        Args:
            headers: HTTP头

        Returns:
            (trace_id, parent_span_id)
        """
        trace_id = headers.get("X-Trace-ID")
        parent_span_id = headers.get("X-Span-ID")
        return trace_id, parent_span_id

    def get_finished_spans(
        self,
        trace_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Span]:
        """获取完成的跨度"""
        with self._lock:
            if trace_id:
                spans = [s for s in self._finished_spans if s.trace_id == trace_id]
            else:
                spans = self._finished_spans.copy()

            return spans[-limit:]

    def export_jaeger(self) -> Dict[str, Any]:
        """导出Jaeger格式"""
        with self._lock:
            return {
                "data": [
                    {
                        "traceID": span.trace_id,
                        "spans": [span.to_dict()],
                        "processes": {
                            "p1": {
                                "serviceName": self.service_name,
                            }
                        },
                    }
                    for span in self._finished_spans[-100:]
                ],
            }


# =============================================================================
# 告警管理器
# =============================================================================

class AlertManager:
    """告警管理器"""

    def __init__(
        self,
        metrics: MetricsCollector,
        evaluation_interval: float = 60.0
    ):
        self.metrics = metrics
        self.evaluation_interval = evaluation_interval

        # 告警规则
        self._rules: Dict[str, AlertRule] = {}

        # 活跃告警
        self._active_alerts: Dict[str, Alert] = {}

        # 告警历史
        self._alert_history: List[Alert] = []
        self._max_history = 1000

        # 告警通知回调
        self._callbacks: List[Callable[[Alert, str], None]] = []

        # 运行状态
        self._running = False
        self._eval_thread: Optional[threading.Thread] = None

        # 锁
        self._lock = threading.RLock()

        logger.info("告警管理器初始化完成")

    def start(self):
        """启动告警管理器"""
        self._running = True

        self._eval_thread = threading.Thread(
            target=self._evaluation_loop,
            daemon=True
        )
        self._eval_thread.start()

        logger.info("告警管理器启动")

    def stop(self):
        """停止告警管理器"""
        self._running = False

        if self._eval_thread:
            self._eval_thread.join(timeout=5.0)

        logger.info("告警管理器停止")

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self._lock:
            self._rules[rule.rule_id] = rule
            logger.info(f"添加告警规则: {rule.name}")

    def remove_rule(self, rule_id: str):
        """移除告警规则"""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]

    def on_alert(self, callback: Callable[[Alert, str], None]):
        """
        注册告警回调

        Args:
            callback: 回调函数(alert, event_type)
        """
        self._callbacks.append(callback)

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        with self._lock:
            return self._alert_history[-limit:]

    def _evaluation_loop(self):
        """评估循环"""
        while self._running:
            try:
                self._evaluate_rules()
            except Exception as e:
                logger.error(f"告警评估异常: {e}")

            time.sleep(self.evaluation_interval)

    def _evaluate_rules(self):
        """评估所有规则"""
        now = datetime.now()

        with self._lock:
            for rule in self._rules.values():
                if not rule.enabled:
                    continue

                try:
                    self._evaluate_rule(rule, now)
                except Exception as e:
                    logger.error(f"规则评估失败 {rule.name}: {e}")

    def _evaluate_rule(self, rule: AlertRule, now: datetime):
        """评估单个规则"""
        # 获取指标值
        metric_value = self._get_metric_value(rule.metric_name, rule.labels)

        if metric_value is None:
            return

        # 评估条件
        is_firing = self._check_condition(metric_value, rule.condition, rule.threshold)

        rule.last_evaluated = now

        if is_firing:
            if not rule.firing:
                # 开始触发
                rule.firing = True
                rule.firing_since = now

            elif rule.firing_since:
                # 检查持续时间
                duration = (now - rule.firing_since).total_seconds()

                if duration >= rule.duration and rule.rule_id not in self._active_alerts:
                    # 创建告警
                    self._create_alert(rule, metric_value, now)

        else:
            if rule.firing:
                # 恢复
                rule.firing = False
                rule.firing_since = None

                # 解决告警
                if rule.rule_id in self._active_alerts:
                    self._resolve_alert(rule.rule_id, now)

    def _get_metric_value(
        self,
        metric_name: str,
        labels: Dict[str, str]
    ) -> Optional[float]:
        """获取指标值"""
        # 尝试从仪表获取
        value = self.metrics.gauge_get(metric_name, labels)
        if value != 0.0:
            return value

        # 尝试从计数器获取
        value = self.metrics.counter_get(metric_name, labels)
        return value if value != 0.0 else None

    def _check_condition(
        self,
        value: float,
        condition: str,
        threshold: float
    ) -> bool:
        """检查条件"""
        if condition.startswith(">"):
            if condition.startswith(">="):
                return value >= threshold
            return value > threshold
        elif condition.startswith("<"):
            if condition.startswith("<="):
                return value <= threshold
            return value < threshold
        elif condition.startswith("=="):
            return abs(value - threshold) < 1e-9
        elif condition.startswith("!="):
            return abs(value - threshold) >= 1e-9

        return False

    def _create_alert(self, rule: AlertRule, value: float, now: datetime):
        """创建告警"""
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            rule=rule,
            value=value,
            triggered_at=now,
            labels={**rule.labels},
            annotations={**rule.annotations},
        )

        self._active_alerts[rule.rule_id] = alert
        rule.last_triggered = now

        logger.warning(f"告警触发: {rule.name} (value={value}, threshold={rule.threshold})")

        # 通知回调
        self._notify_callbacks(alert, "firing")

    def _resolve_alert(self, rule_id: str, now: datetime):
        """解决告警"""
        if rule_id in self._active_alerts:
            alert = self._active_alerts.pop(rule_id)
            alert.resolved_at = now

            # 添加到历史
            self._alert_history.append(alert)
            while len(self._alert_history) > self._max_history:
                self._alert_history.pop(0)

            logger.info(f"告警解决: {alert.rule.name}")

            # 通知回调
            self._notify_callbacks(alert, "resolved")

    def _notify_callbacks(self, alert: Alert, event: str):
        """通知回调"""
        for callback in self._callbacks:
            try:
                callback(alert, event)
            except Exception as e:
                logger.error(f"告警回调异常: {e}")


# =============================================================================
# 健康检查管理器
# =============================================================================

class HealthCheckManager:
    """健康检查管理器"""

    def __init__(self):
        # 检查函数
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}

        # 检查结果缓存
        self._results: Dict[str, HealthCheck] = {}

        # 锁
        self._lock = threading.RLock()

        logger.info("健康检查管理器初始化完成")

    def register(self, name: str, check_func: Callable[[], HealthCheck]):
        """
        注册健康检查

        Args:
            name: 检查名称
            check_func: 检查函数
        """
        self._checks[name] = check_func
        logger.info(f"注册健康检查: {name}")

    def unregister(self, name: str):
        """注销健康检查"""
        if name in self._checks:
            del self._checks[name]

    def check(self, name: str) -> HealthCheck:
        """执行单个健康检查"""
        if name not in self._checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="检查不存在",
            )

        try:
            start = time.time()
            result = self._checks[name]()
            result.latency_ms = (time.time() - start) * 1000
            result.timestamp = datetime.now()

            with self._lock:
                self._results[name] = result

            return result

        except Exception as e:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"检查异常: {e}",
            )

    def check_all(self) -> Dict[str, HealthCheck]:
        """执行所有健康检查"""
        results = {}

        for name in self._checks:
            results[name] = self.check(name)

        return results

    def get_overall_status(self) -> Tuple[HealthStatus, Dict[str, HealthCheck]]:
        """
        获取整体健康状态

        Returns:
            (整体状态, 检查结果)
        """
        results = self.check_all()

        if not results:
            return HealthStatus.HEALTHY, results

        # 确定整体状态
        statuses = [r.status for r in results.values()]

        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return overall, results

    def get_cached_results(self) -> Dict[str, HealthCheck]:
        """获取缓存的检查结果"""
        with self._lock:
            return self._results.copy()


# =============================================================================
# 监控系统
# =============================================================================

class MonitoringSystem:
    """监控系统 - 整合所有监控组件"""

    def __init__(
        self,
        service_name: str = "poyehuiming",
        config: Optional[Dict[str, Any]] = None
    ):
        config = config or {}

        self.service_name = service_name

        # 指标收集器
        self.metrics = MetricsCollector(
            prefix=config.get("metrics_prefix", service_name)
        )

        # 结构化日志
        self.logger = StructuredLogger(
            name=service_name,
            level=LogLevel(config.get("log_level", "info")),
            output_json=config.get("log_json", True),
            log_file=config.get("log_file"),
        )

        # 分布式追踪
        self.tracer = Tracer(
            service_name=service_name,
            sample_rate=config.get("trace_sample_rate", 1.0),
        )

        # 告警管理器
        self.alerts = AlertManager(
            metrics=self.metrics,
            evaluation_interval=config.get("alert_interval", 60.0),
        )

        # 健康检查
        self.health = HealthCheckManager()

        # 注册默认健康检查
        self._register_default_health_checks()

        # 注册默认告警规则
        self._register_default_alert_rules()

        logger.info(f"监控系统初始化完成: {service_name}")

    def start(self):
        """启动监控系统"""
        self.logger.start()
        self.alerts.start()

        logger.info("监控系统启动")

    def stop(self):
        """停止监控系统"""
        self.alerts.stop()
        self.logger.stop()

        logger.info("监控系统停止")

    def _register_default_health_checks(self):
        """注册默认健康检查"""

        # 系统资源检查
        def check_system():
            try:
                import psutil
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()

                status = HealthStatus.HEALTHY
                message = "系统正常"

                if cpu_percent > 90 or memory.percent > 90:
                    status = HealthStatus.DEGRADED
                    message = f"资源紧张: CPU={cpu_percent}%, 内存={memory.percent}%"

                return HealthCheck(
                    name="system",
                    status=status,
                    message=message,
                    details={
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_available_gb": memory.available / (1024**3),
                    },
                )
            except ImportError:
                return HealthCheck(
                    name="system",
                    status=HealthStatus.HEALTHY,
                    message="psutil未安装,跳过系统检查",
                )

        self.health.register("system", check_system)

        # 磁盘检查
        def check_disk():
            try:
                import psutil
                disk = psutil.disk_usage("/")

                status = HealthStatus.HEALTHY
                message = "磁盘正常"

                if disk.percent > 90:
                    status = HealthStatus.DEGRADED
                    message = f"磁盘使用率高: {disk.percent}%"
                elif disk.percent > 95:
                    status = HealthStatus.UNHEALTHY
                    message = f"磁盘空间不足: {disk.percent}%"

                return HealthCheck(
                    name="disk",
                    status=status,
                    message=message,
                    details={
                        "disk_percent": disk.percent,
                        "disk_free_gb": disk.free / (1024**3),
                    },
                )
            except ImportError:
                return HealthCheck(
                    name="disk",
                    status=HealthStatus.HEALTHY,
                    message="psutil未安装,跳过磁盘检查",
                )

        self.health.register("disk", check_disk)

    def _register_default_alert_rules(self):
        """注册默认告警规则"""

        # CPU使用率告警
        self.alerts.add_rule(AlertRule(
            rule_id="high_cpu",
            name="CPU使用率过高",
            description="CPU使用率超过90%",
            metric_name="system_cpu_percent",
            condition=">",
            threshold=90.0,
            severity=AlertSeverity.WARNING,
            duration=60.0,
        ))

        # 内存使用率告警
        self.alerts.add_rule(AlertRule(
            rule_id="high_memory",
            name="内存使用率过高",
            description="内存使用率超过90%",
            metric_name="system_memory_percent",
            condition=">",
            threshold=90.0,
            severity=AlertSeverity.WARNING,
            duration=60.0,
        ))

        # 错误率告警
        self.alerts.add_rule(AlertRule(
            rule_id="high_error_rate",
            name="错误率过高",
            description="错误率超过5%",
            metric_name="request_error_rate",
            condition=">",
            threshold=0.05,
            severity=AlertSeverity.ERROR,
            duration=120.0,
        ))

    # -------------------------------------------------------------------------
    # 便捷方法
    # -------------------------------------------------------------------------

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        error: Optional[str] = None
    ):
        """记录请求指标"""
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code),
        }

        # 计数器
        self.metrics.counter_inc("http_requests_total", 1, labels)

        # 持续时间
        self.metrics.histogram_observe(
            "http_request_duration_seconds",
            duration,
            {"endpoint": endpoint, "method": method},
        )

        # 错误
        if error or status_code >= 400:
            self.metrics.counter_inc(
                "http_requests_errors_total",
                1,
                {"endpoint": endpoint, "error_type": error or "http_error"},
            )

    def record_inference(
        self,
        model_name: str,
        duration: float,
        success: bool = True
    ):
        """记录推理指标"""
        labels = {"model": model_name}

        self.metrics.counter_inc("inference_total", 1, labels)
        self.metrics.histogram_observe("inference_duration_seconds", duration, labels)

        if not success:
            self.metrics.counter_inc("inference_errors_total", 1, labels)

    def update_system_metrics(self):
        """更新系统指标"""
        try:
            import psutil

            # CPU
            cpu_percent = psutil.cpu_percent()
            self.metrics.gauge_set("system_cpu_percent", cpu_percent)

            # 内存
            memory = psutil.virtual_memory()
            self.metrics.gauge_set("system_memory_percent", memory.percent)
            self.metrics.gauge_set(
                "system_memory_available_bytes",
                float(memory.available)
            )

            # 磁盘
            disk = psutil.disk_usage("/")
            self.metrics.gauge_set("system_disk_percent", disk.percent)
            self.metrics.gauge_set("system_disk_free_bytes", float(disk.free))

        except ImportError:
            pass

    def get_prometheus_metrics(self) -> str:
        """获取Prometheus格式指标"""
        return self.metrics.export_prometheus()

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        overall, checks = self.health.get_overall_status()

        return {
            "status": overall.value,
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "latency_ms": check.latency_ms,
                    "details": check.details,
                }
                for name, check in checks.items()
            },
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# 装饰器
# =============================================================================

def trace(name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    追踪装饰器

    使用示例:
        @trace("process_image", {"type": "detection"})
        def process_image(image):
            ...
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitoring = get_monitoring_system()
            if monitoring:
                with monitoring.tracer.trace(span_name, tags):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def timed(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    计时装饰器

    使用示例:
        @timed("function_duration", {"function": "process"})
        def process():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitoring = get_monitoring_system()
            if monitoring:
                with monitoring.metrics.timer(metric_name, labels):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# 便捷函数
# =============================================================================

# 全局监控系统实例
_monitoring: Optional[MonitoringSystem] = None


def get_monitoring_system() -> Optional[MonitoringSystem]:
    """获取全局监控系统"""
    return _monitoring


def init_monitoring(
    service_name: str = "poyehuiming",
    config: Optional[Dict[str, Any]] = None
) -> MonitoringSystem:
    """初始化全局监控系统"""
    global _monitoring
    _monitoring = MonitoringSystem(service_name, config)
    return _monitoring


# =============================================================================
# FastAPI集成
# =============================================================================

def create_monitoring_routes():
    """
    创建监控路由

    示例代码，展示如何集成监控API
    """
    try:
        from fastapi import APIRouter, Response
    except ImportError:
        logger.warning("FastAPI未安装，跳过路由创建")
        return None

    router = APIRouter(prefix="/monitoring", tags=["监控"])

    @router.get("/health")
    async def health():
        """健康检查"""
        monitoring = get_monitoring_system()
        if monitoring:
            return monitoring.get_health_status()
        return {"status": "unknown"}

    @router.get("/metrics")
    async def metrics():
        """Prometheus指标"""
        monitoring = get_monitoring_system()
        if monitoring:
            return Response(
                content=monitoring.get_prometheus_metrics(),
                media_type="text/plain",
            )
        return Response(content="", media_type="text/plain")

    @router.get("/alerts")
    async def alerts():
        """活跃告警"""
        monitoring = get_monitoring_system()
        if monitoring:
            return [a.to_dict() for a in monitoring.alerts.get_active_alerts()]
        return []

    @router.get("/traces")
    async def traces(trace_id: Optional[str] = None, limit: int = 100):
        """追踪数据"""
        monitoring = get_monitoring_system()
        if monitoring:
            spans = monitoring.tracer.get_finished_spans(trace_id, limit)
            return [s.to_dict() for s in spans]
        return []

    return router


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # 初始化监控系统
    monitoring = init_monitoring("test_service", {
        "log_level": "info",
        "log_json": False,
        "trace_sample_rate": 1.0,
    })

    # 启动
    monitoring.start()

    # 记录一些指标
    monitoring.metrics.counter_inc("test_counter", 1)
    monitoring.metrics.gauge_set("test_gauge", 42.0)
    monitoring.metrics.histogram_observe("test_histogram", 0.5)

    # 记录请求
    monitoring.record_request("/api/test", "GET", 200, 0.1)
    monitoring.record_request("/api/test", "POST", 500, 0.5, "internal_error")

    # 记录推理
    monitoring.record_inference("yolov8", 0.05, True)

    # 追踪示例
    with monitoring.tracer.trace("main_process"):
        with monitoring.tracer.trace("sub_process_1"):
            time.sleep(0.1)

        with monitoring.tracer.trace("sub_process_2"):
            time.sleep(0.2)

    # 日志示例
    monitoring.logger.set_context(user_id="123", request_id="abc")
    monitoring.logger.info("处理请求", endpoint="/api/test")
    monitoring.logger.warning("性能下降", latency_ms=500)

    try:
        raise ValueError("测试异常")
    except Exception:
        monitoring.logger.exception("发生异常")

    # 健康检查
    print("\n健康状态:")
    print(json.dumps(monitoring.get_health_status(), indent=2, ensure_ascii=False))

    # Prometheus指标
    print("\nPrometheus指标:")
    print(monitoring.get_prometheus_metrics())

    # 追踪数据
    print("\n追踪数据:")
    for span in monitoring.tracer.get_finished_spans():
        print(f"  {span.name}: {span.duration_ms:.2f}ms")

    # 等待一下让日志写入
    time.sleep(1)

    # 停止
    monitoring.stop()
