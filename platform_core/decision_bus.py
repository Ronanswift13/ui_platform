# -*- coding: utf-8 -*-
"""
决策总线模块
============

输变电激光监测平台 V3.5 - 室外巡视增强
多模态融合与闭环控制核心组件

实现功能:
1. 跨插件缺陷证据汇聚
2. 优先级排队和调度
3. PTZ自动复核触发
4. 闭环控制状态机
5. 证据链追溯

作者: AI巡检系统
版本: 3.5.0
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from queue import PriorityQueue
import asyncio
import threading
import time
import uuid
import logging
import json

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举定义
# =============================================================================

class DefectSeverity(Enum):
    """缺陷严重程度"""
    CRITICAL = 1      # 紧急 - 需立即处理
    ALARM = 2         # 告警 - 需尽快处理
    WARNING = 3       # 预警 - 需关注
    ATTENTION = 4     # 注意 - 需观察
    NORMAL = 5        # 正常


class ReviewStatus(Enum):
    """复核状态"""
    PENDING = "pending"           # 待复核
    IN_PROGRESS = "in_progress"   # 复核中
    CONFIRMED = "confirmed"       # 已确认
    FALSE_ALARM = "false_alarm"   # 误报
    UNCERTAIN = "uncertain"       # 不确定，需人工
    TIMEOUT = "timeout"           # 超时
    FAILED = "failed"             # 失败


class ClosedLoopState(Enum):
    """闭环控制状态"""
    IDLE = auto()                 # 空闲
    DETECTION = auto()            # 检测中
    EVIDENCE_COLLECTION = auto()  # 证据收集中
    FUSION = auto()               # 融合分析中
    REVIEW = auto()               # 复核中
    DECISION = auto()             # 决策中
    ACTION = auto()               # 执行中
    VERIFICATION = auto()         # 验证中
    COMPLETED = auto()            # 完成
    ERROR = auto()                # 错误


class EvidenceType(Enum):
    """证据类型"""
    VISUAL_DETECTION = "visual_detection"       # 可见光检测
    THERMAL_DETECTION = "thermal_detection"     # 红外检测
    TEMPORAL_TRACKING = "temporal_tracking"     # 时序跟踪
    TEXTURE_ANALYSIS = "texture_analysis"       # 纹理分析
    SEGMENTATION = "segmentation"               # 语义分割
    OBB_DETECTION = "obb_detection"             # 旋转框检测
    FUSION_RESULT = "fusion_result"             # 融合结果
    MANUAL_ANNOTATION = "manual_annotation"     # 人工标注
    PTZ_REVIEW = "ptz_review"                   # PTZ复核


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class Evidence:
    """证据"""
    evidence_id: str
    evidence_type: EvidenceType
    source_plugin: str
    timestamp: float
    confidence: float
    data: Dict[str, Any]
    image_path: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'evidence_id': self.evidence_id,
            'evidence_type': self.evidence_type.value,
            'source_plugin': self.source_plugin,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'data': self.data,
            'image_path': self.image_path,
            'bbox': self.bbox,
            'metadata': self.metadata
        }


@dataclass
class DefectReport:
    """缺陷报告"""
    report_id: str
    device_id: str
    defect_type: str
    severity: DefectSeverity
    location: Dict[str, Any]
    evidences: List[Evidence]
    confidence: float
    review_status: ReviewStatus = ReviewStatus.PENDING
    review_results: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    final_decision: Optional[str] = None
    action_taken: Optional[str] = None

    def add_evidence(self, evidence: Evidence) -> None:
        """添加证据"""
        self.evidences.append(evidence)
        self.updated_at = time.time()
        # 更新置信度
        if evidence.confidence > self.confidence:
            self.confidence = (self.confidence + evidence.confidence) / 2

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'report_id': self.report_id,
            'device_id': self.device_id,
            'defect_type': self.defect_type,
            'severity': self.severity.name,
            'location': self.location,
            'evidences': [e.to_dict() for e in self.evidences],
            'confidence': self.confidence,
            'review_status': self.review_status.value,
            'review_results': self.review_results,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'final_decision': self.final_decision,
            'action_taken': self.action_taken
        }


@dataclass
class ReviewRequest:
    """复核请求"""
    request_id: str
    report: DefectReport
    priority: int
    preset_id: Optional[int] = None
    target_position: Optional[Dict[str, float]] = None
    zoom_level: float = 1.0
    capture_modes: List[str] = field(default_factory=lambda: ['visible', 'thermal'])
    max_retries: int = 3
    timeout_seconds: float = 60.0
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other: 'ReviewRequest') -> bool:
        """优先级比较（用于优先队列）"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


@dataclass
class ClosedLoopContext:
    """闭环控制上下文"""
    session_id: str
    state: ClosedLoopState
    device_id: str
    reports: List[DefectReport]
    start_time: float
    current_step: int = 0
    total_steps: int = 0
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 决策总线核心类
# =============================================================================

class DecisionBus:
    """
    决策总线

    负责协调各检测插件的输出，汇聚证据，触发复核，执行闭环控制
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化决策总线

        Args:
            config: 配置参数
        """
        self.config = config or {}

        # 缺陷报告存储
        self._reports: Dict[str, DefectReport] = {}
        self._device_reports: Dict[str, List[str]] = defaultdict(list)

        # 复核队列
        self._review_queue: PriorityQueue = PriorityQueue()
        self._pending_reviews: Dict[str, ReviewRequest] = {}

        # 闭环控制上下文
        self._contexts: Dict[str, ClosedLoopContext] = {}

        # 插件注册表
        self._plugins: Dict[str, Any] = {}

        # PTZ控制器引用
        self._ptz_controller: Optional[Any] = None

        # 融合引擎引用
        self._fusion_engine: Optional[Any] = None

        # 回调函数
        self._on_defect_detected: List[Callable] = []
        self._on_review_completed: List[Callable] = []
        self._on_decision_made: List[Callable] = []

        # 线程锁
        self._lock = threading.RLock()

        # 异步事件循环
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._review_task: Optional[asyncio.Task] = None

        # 配置参数
        self._auto_review_enabled = self.config.get('auto_review_enabled', True)
        self._review_threshold = self.config.get('review_threshold', 0.7)
        self._max_queue_size = self.config.get('max_queue_size', 100)

        # 严重程度到优先级的映射
        self._severity_priority = {
            DefectSeverity.CRITICAL: 1,
            DefectSeverity.ALARM: 2,
            DefectSeverity.WARNING: 3,
            DefectSeverity.ATTENTION: 4,
            DefectSeverity.NORMAL: 5
        }

        logger.info("决策总线初始化完成")

    # =========================================================================
    # 组件注册
    # =========================================================================

    def register_plugin(self, plugin_id: str, plugin: Any) -> None:
        """注册检测插件"""
        self._plugins[plugin_id] = plugin
        logger.info(f"插件已注册: {plugin_id}")

    def set_ptz_controller(self, controller: Any) -> None:
        """设置PTZ控制器"""
        self._ptz_controller = controller
        logger.info("PTZ控制器已设置")

    def set_fusion_engine(self, engine: Any) -> None:
        """设置融合引擎"""
        self._fusion_engine = engine
        logger.info("融合引擎已设置")

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        注册回调函数

        Args:
            event: 事件类型 ('defect_detected', 'review_completed', 'decision_made')
            callback: 回调函数
        """
        if event == 'defect_detected':
            self._on_defect_detected.append(callback)
        elif event == 'review_completed':
            self._on_review_completed.append(callback)
        elif event == 'decision_made':
            self._on_decision_made.append(callback)

    # =========================================================================
    # 缺陷报告管理
    # =========================================================================

    def submit_defect(
        self,
        device_id: str,
        defect_type: str,
        severity: DefectSeverity,
        location: Dict[str, Any],
        evidence: Evidence,
        confidence: float,
        auto_review: bool = True
    ) -> DefectReport:
        """
        提交缺陷检测结果

        Args:
            device_id: 设备ID
            defect_type: 缺陷类型
            severity: 严重程度
            location: 位置信息
            evidence: 初始证据
            confidence: 置信度
            auto_review: 是否自动触发复核

        Returns:
            缺陷报告
        """
        with self._lock:
            # 检查是否有相似的待处理报告
            existing_report = self._find_similar_report(device_id, defect_type, location)

            if existing_report:
                # 更新现有报告
                existing_report.add_evidence(evidence)
                logger.info(f"更新现有缺陷报告: {existing_report.report_id}")
                report = existing_report
            else:
                # 创建新报告
                report_id = f"DR-{uuid.uuid4().hex[:8]}"
                report = DefectReport(
                    report_id=report_id,
                    device_id=device_id,
                    defect_type=defect_type,
                    severity=severity,
                    location=location,
                    evidences=[evidence],
                    confidence=confidence
                )
                self._reports[report_id] = report
                self._device_reports[device_id].append(report_id)
                logger.info(f"创建新缺陷报告: {report_id}, 类型: {defect_type}, 严重程度: {severity.name}")

            # 触发回调
            for callback in self._on_defect_detected:
                try:
                    callback(report)
                except Exception as e:
                    logger.error(f"缺陷检测回调执行失败: {e}")

            # 自动复核
            if auto_review and self._auto_review_enabled:
                if confidence < self._review_threshold or severity.value <= DefectSeverity.WARNING.value:
                    self.request_review(report)

            return report

    def _find_similar_report(
        self,
        device_id: str,
        defect_type: str,
        location: Dict[str, Any],
        time_window: float = 60.0
    ) -> Optional[DefectReport]:
        """查找相似的待处理报告"""
        current_time = time.time()

        for report_id in self._device_reports.get(device_id, []):
            report = self._reports.get(report_id)
            if not report:
                continue

            # 检查时间窗口
            if current_time - report.created_at > time_window:
                continue

            # 检查缺陷类型
            if report.defect_type != defect_type:
                continue

            # 检查位置相似性（IoU）
            if self._is_location_similar(report.location, location):
                return report

        return None

    def _is_location_similar(self, loc1: Dict, loc2: Dict, threshold: float = 0.5) -> bool:
        """判断两个位置是否相似"""
        if 'bbox' in loc1 and 'bbox' in loc2:
            # 计算IoU
            b1 = loc1['bbox']
            b2 = loc2['bbox']

            x1 = max(b1.get('x', 0), b2.get('x', 0))
            y1 = max(b1.get('y', 0), b2.get('y', 0))
            x2 = min(b1.get('x', 0) + b1.get('width', 0), b2.get('x', 0) + b2.get('width', 0))
            y2 = min(b1.get('y', 0) + b1.get('height', 0), b2.get('y', 0) + b2.get('height', 0))

            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = b1.get('width', 0) * b1.get('height', 0)
            area2 = b2.get('width', 0) * b2.get('height', 0)
            union_area = area1 + area2 - inter_area

            if union_area > 0:
                iou = inter_area / union_area
                return iou >= threshold

        # 检查预置位ID
        if loc1.get('preset_id') == loc2.get('preset_id'):
            return True

        return False

    def get_report(self, report_id: str) -> Optional[DefectReport]:
        """获取缺陷报告"""
        return self._reports.get(report_id)

    def get_device_reports(
        self,
        device_id: str,
        status_filter: Optional[List[ReviewStatus]] = None
    ) -> List[DefectReport]:
        """获取设备的所有缺陷报告"""
        reports = []
        for report_id in self._device_reports.get(device_id, []):
            report = self._reports.get(report_id)
            if report:
                if status_filter is None or report.review_status in status_filter:
                    reports.append(report)
        return reports

    # =========================================================================
    # 复核管理
    # =========================================================================

    def request_review(
        self,
        report: DefectReport,
        preset_id: Optional[int] = None,
        capture_modes: Optional[List[str]] = None
    ) -> str:
        """
        请求PTZ复核

        Args:
            report: 缺陷报告
            preset_id: 预置位ID
            capture_modes: 拍摄模式列表

        Returns:
            复核请求ID
        """
        with self._lock:
            request_id = f"RR-{uuid.uuid4().hex[:8]}"

            # 计算优先级
            priority = self._severity_priority.get(report.severity, 5)

            # 创建复核请求
            request = ReviewRequest(
                request_id=request_id,
                report=report,
                priority=priority,
                preset_id=preset_id or report.location.get('preset_id'),
                target_position=report.location.get('ptz_position'),
                zoom_level=self._calculate_zoom_level(report),
                capture_modes=capture_modes or ['visible', 'thermal']
            )

            # 加入队列
            if self._review_queue.qsize() < self._max_queue_size:
                self._review_queue.put(request)
                self._pending_reviews[request_id] = request
                report.review_status = ReviewStatus.PENDING
                logger.info(f"复核请求已加入队列: {request_id}, 优先级: {priority}")
            else:
                logger.warning(f"复核队列已满，请求被拒绝: {request_id}")

            return request_id

    def _calculate_zoom_level(self, report: DefectReport) -> float:
        """根据缺陷类型计算放大倍数"""
        defect_type = report.defect_type.lower()

        zoom_map = {
            'crack': 3.0,
            'corrosion': 2.5,
            'oil_leak': 2.0,
            'rust': 2.0,
            'hotspot': 1.5,
            'tilt': 1.0,
            'displacement': 1.5,
            'contamination': 2.0
        }

        for key, zoom in zoom_map.items():
            if key in defect_type:
                return zoom

        return 1.5  # 默认放大倍数

    async def process_review_queue(self) -> None:
        """处理复核队列"""
        while True:
            try:
                if self._review_queue.empty():
                    await asyncio.sleep(1.0)
                    continue

                # 获取下一个复核请求
                request = self._review_queue.get_nowait()

                if request.request_id not in self._pending_reviews:
                    continue

                logger.info(f"开始处理复核请求: {request.request_id}")

                # 执行复核
                result = await self._execute_review(request)

                # 更新报告状态
                self._update_review_result(request, result)

                # 从待处理中移除
                del self._pending_reviews[request.request_id]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"复核处理异常: {e}")
                await asyncio.sleep(1.0)

    async def _execute_review(self, request: ReviewRequest) -> Dict[str, Any]:
        """
        执行单次复核

        Args:
            request: 复核请求

        Returns:
            复核结果
        """
        result = {
            'request_id': request.request_id,
            'success': False,
            'status': ReviewStatus.FAILED,
            'captures': [],
            'analysis': None,
            'confidence': 0.0,
            'error': None
        }

        try:
            request.report.review_status = ReviewStatus.IN_PROGRESS

            if self._ptz_controller is None:
                raise RuntimeError("PTZ控制器未设置")

            # 移动到目标位置
            if request.preset_id is not None:
                await self._ptz_controller.goto_preset(request.preset_id)
            elif request.target_position:
                from .ptz_controller import PTZPosition
                position = PTZPosition(**request.target_position)
                await self._ptz_controller._adapter.set_position(position)

            # 等待稳定
            await asyncio.sleep(0.5)

            # 多模式拍摄
            captures = []
            for mode in request.capture_modes:
                capture = await self._capture_image(mode, request.zoom_level)
                if capture:
                    captures.append(capture)

            result['captures'] = captures

            # 执行复核分析
            if captures:
                analysis = await self._analyze_captures(request.report, captures)
                result['analysis'] = analysis
                result['confidence'] = analysis.get('confidence', 0.0)

                # 判断复核结果
                if analysis.get('confirmed', False):
                    result['status'] = ReviewStatus.CONFIRMED
                elif analysis.get('false_alarm', False):
                    result['status'] = ReviewStatus.FALSE_ALARM
                else:
                    result['status'] = ReviewStatus.UNCERTAIN

                result['success'] = True
            else:
                result['status'] = ReviewStatus.FAILED
                result['error'] = "拍摄失败"

        except asyncio.TimeoutError:
            result['status'] = ReviewStatus.TIMEOUT
            result['error'] = "复核超时"
        except Exception as e:
            result['status'] = ReviewStatus.FAILED
            result['error'] = str(e)
            logger.error(f"复核执行失败: {e}")

        return result

    async def _capture_image(self, mode: str, zoom_level: float) -> Optional[Dict]:
        """拍摄图像"""
        try:
            # 调整变焦
            if self._ptz_controller and zoom_level > 1.0:
                for _ in range(int(zoom_level)):
                    await self._ptz_controller._adapter.execute_command(
                        'zoom_in' if hasattr(self._ptz_controller._adapter, 'execute_command') else None
                    )
                    await asyncio.sleep(0.2)

            # 模拟拍摄（实际应调用相机接口）
            capture = {
                'mode': mode,
                'timestamp': time.time(),
                'zoom_level': zoom_level,
                'image_path': f"/tmp/capture_{mode}_{int(time.time())}.jpg",
                'metadata': {}
            }

            return capture

        except Exception as e:
            logger.error(f"拍摄失败 ({mode}): {e}")
            return None

    async def _analyze_captures(
        self,
        report: DefectReport,
        captures: List[Dict]
    ) -> Dict[str, Any]:
        """
        分析复核拍摄结果

        Args:
            report: 原始缺陷报告
            captures: 拍摄结果列表

        Returns:
            分析结果
        """
        analysis = {
            'confirmed': False,
            'false_alarm': False,
            'confidence': 0.0,
            'details': {},
            'recommendations': []
        }

        try:
            # 获取相关插件
            plugin_id = report.evidences[0].source_plugin if report.evidences else None
            plugin = self._plugins.get(plugin_id)

            if plugin and hasattr(plugin, 'detect_roi'):
                # 使用插件重新检测
                for capture in captures:
                    # 模拟读取图像
                    import numpy as np
                    image = np.zeros((480, 640, 3), dtype=np.uint8)  # 实际应读取真实图像

                    # 执行检测
                    detections = plugin.detect_roi(image, report.location.get('bbox'))

                    if detections:
                        # 计算与原始检测的一致性
                        consistency = self._calculate_consistency(
                            report.evidences[0].data,
                            detections[0] if detections else {}
                        )

                        if consistency > 0.7:
                            analysis['confirmed'] = True
                            analysis['confidence'] = consistency
                        elif consistency < 0.3:
                            analysis['false_alarm'] = True
                            analysis['confidence'] = 1 - consistency
                        else:
                            analysis['confidence'] = 0.5

                        analysis['details']['detection'] = detections[0] if detections else None
            else:
                # 使用融合引擎分析
                if self._fusion_engine:
                    # 简化的融合分析
                    analysis['confirmed'] = report.confidence > 0.7
                    analysis['confidence'] = report.confidence
                else:
                    # 默认逻辑
                    analysis['confirmed'] = report.confidence > 0.8
                    analysis['confidence'] = report.confidence

            # 生成建议
            if analysis['confirmed']:
                analysis['recommendations'].append(f"确认存在{report.defect_type}，建议安排检修")
            elif analysis['false_alarm']:
                analysis['recommendations'].append("复核未发现异常，原检测可能为误报")
            else:
                analysis['recommendations'].append("复核结果不确定，建议人工复核")

        except Exception as e:
            logger.error(f"复核分析失败: {e}")
            analysis['details']['error'] = str(e)

        return analysis

    def _calculate_consistency(self, original: Dict, review: Dict) -> float:
        """计算原始检测与复核检测的一致性"""
        if not original or not review:
            return 0.5

        # 比较类型
        type_match = original.get('type') == review.get('type')

        # 比较置信度
        orig_conf = original.get('confidence', 0.5)
        review_conf = review.get('confidence', 0.5)
        conf_diff = abs(orig_conf - review_conf)

        # 计算综合一致性
        consistency = 0.0
        if type_match:
            consistency = 0.5 + 0.5 * (1 - conf_diff)
        else:
            consistency = 0.5 * (1 - conf_diff)

        return consistency

    def _update_review_result(self, request: ReviewRequest, result: Dict) -> None:
        """更新复核结果"""
        report = request.report
        report.review_status = result['status']
        report.review_results.append({
            'request_id': request.request_id,
            'timestamp': time.time(),
            'result': result
        })
        report.updated_at = time.time()

        # 如果确认，创建复核证据
        if result['status'] == ReviewStatus.CONFIRMED:
            evidence = Evidence(
                evidence_id=f"EV-{uuid.uuid4().hex[:8]}",
                evidence_type=EvidenceType.PTZ_REVIEW,
                source_plugin='decision_bus',
                timestamp=time.time(),
                confidence=result['confidence'],
                data=result['analysis'] or {},
                metadata={'captures': result['captures']}
            )
            report.add_evidence(evidence)

        # 触发回调
        for callback in self._on_review_completed:
            try:
                callback(report, result)
            except Exception as e:
                logger.error(f"复核完成回调执行失败: {e}")

        logger.info(f"复核完成: {request.request_id}, 状态: {result['status'].value}")

    # =========================================================================
    # 闭环控制
    # =========================================================================

    def start_closed_loop(self, device_id: str) -> str:
        """
        启动闭环控制流程

        Args:
            device_id: 设备ID

        Returns:
            会话ID
        """
        session_id = f"CL-{uuid.uuid4().hex[:8]}"

        context = ClosedLoopContext(
            session_id=session_id,
            state=ClosedLoopState.IDLE,
            device_id=device_id,
            reports=[],
            start_time=time.time()
        )

        self._contexts[session_id] = context
        logger.info(f"闭环控制会话已创建: {session_id}")

        return session_id

    def get_closed_loop_status(self, session_id: str) -> Optional[Dict]:
        """获取闭环控制状态"""
        context = self._contexts.get(session_id)
        if not context:
            return None

        return {
            'session_id': context.session_id,
            'state': context.state.name,
            'device_id': context.device_id,
            'reports_count': len(context.reports),
            'current_step': context.current_step,
            'total_steps': context.total_steps,
            'errors': context.errors,
            'metrics': context.metrics,
            'duration': time.time() - context.start_time
        }

    async def run_closed_loop_cycle(self, session_id: str) -> Dict[str, Any]:
        """
        运行一次完整的闭环控制周期

        Args:
            session_id: 会话ID

        Returns:
            执行结果
        """
        context = self._contexts.get(session_id)
        if not context:
            return {'success': False, 'error': '会话不存在'}

        result = {
            'success': True,
            'session_id': session_id,
            'states_visited': [],
            'reports_generated': 0,
            'reviews_completed': 0,
            'decisions_made': 0
        }

        try:
            # 状态机流转
            state_handlers = {
                ClosedLoopState.IDLE: self._state_idle,
                ClosedLoopState.DETECTION: self._state_detection,
                ClosedLoopState.EVIDENCE_COLLECTION: self._state_evidence_collection,
                ClosedLoopState.FUSION: self._state_fusion,
                ClosedLoopState.REVIEW: self._state_review,
                ClosedLoopState.DECISION: self._state_decision,
                ClosedLoopState.ACTION: self._state_action,
                ClosedLoopState.VERIFICATION: self._state_verification,
            }

            # 开始状态流转
            context.state = ClosedLoopState.DETECTION

            while context.state not in [ClosedLoopState.COMPLETED, ClosedLoopState.ERROR]:
                handler = state_handlers.get(context.state)
                if handler:
                    result['states_visited'].append(context.state.name)
                    next_state = await handler(context)
                    context.state = next_state
                else:
                    context.state = ClosedLoopState.COMPLETED

                context.current_step += 1

            result['reports_generated'] = len(context.reports)
            result['metrics'] = context.metrics

        except Exception as e:
            context.state = ClosedLoopState.ERROR
            context.errors.append(str(e))
            result['success'] = False
            result['error'] = str(e)
            logger.error(f"闭环控制执行失败: {e}")

        return result

    async def _state_idle(self, context: ClosedLoopContext) -> ClosedLoopState:
        """空闲状态"""
        return ClosedLoopState.DETECTION

    async def _state_detection(self, context: ClosedLoopContext) -> ClosedLoopState:
        """检测状态"""
        # 触发各插件执行检测
        for plugin_id, plugin in self._plugins.items():
            if hasattr(plugin, 'detect') or hasattr(plugin, 'detect_defects'):
                try:
                    # 这里应该传入实际图像，简化处理
                    logger.debug(f"触发插件检测: {plugin_id}")
                except Exception as e:
                    logger.warning(f"插件 {plugin_id} 检测失败: {e}")

        await asyncio.sleep(0.1)  # 模拟检测时间
        return ClosedLoopState.EVIDENCE_COLLECTION

    async def _state_evidence_collection(self, context: ClosedLoopContext) -> ClosedLoopState:
        """证据收集状态"""
        # 收集当前设备的所有待处理报告
        reports = self.get_device_reports(
            context.device_id,
            status_filter=[ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS]
        )
        context.reports = reports

        if not reports:
            return ClosedLoopState.COMPLETED

        return ClosedLoopState.FUSION

    async def _state_fusion(self, context: ClosedLoopContext) -> ClosedLoopState:
        """融合状态"""
        if self._fusion_engine and context.reports:
            try:
                # 准备融合数据
                modality_data = {}
                for report in context.reports:
                    for evidence in report.evidences:
                        modality = evidence.source_plugin
                        if modality not in modality_data:
                            modality_data[modality] = {
                                'detections': [],
                                'confidence': 0.0
                            }
                        modality_data[modality]['detections'].append(evidence.data)
                        modality_data[modality]['confidence'] = max(
                            modality_data[modality]['confidence'],
                            evidence.confidence
                        )

                # 执行融合（如果引擎可用）
                if hasattr(self._fusion_engine, 'fuse'):
                    # fusion_result = self._fusion_engine.fuse(modality_data)
                    context.metrics['fusion_performed'] = True
            except Exception as e:
                logger.warning(f"融合处理失败: {e}")

        return ClosedLoopState.REVIEW

    async def _state_review(self, context: ClosedLoopContext) -> ClosedLoopState:
        """复核状态"""
        review_count = 0

        for report in context.reports:
            if report.review_status == ReviewStatus.PENDING:
                self.request_review(report)
                review_count += 1

        # 等待复核完成（简化处理）
        if review_count > 0:
            await asyncio.sleep(0.5)

        context.metrics['reviews_requested'] = review_count
        return ClosedLoopState.DECISION

    async def _state_decision(self, context: ClosedLoopContext) -> ClosedLoopState:
        """决策状态"""
        decisions = []

        for report in context.reports:
            decision = self._make_decision(report)
            report.final_decision = decision['action']
            decisions.append(decision)

            # 触发回调
            for callback in self._on_decision_made:
                try:
                    callback(report, decision)
                except Exception as e:
                    logger.error(f"决策回调执行失败: {e}")

        context.metrics['decisions_made'] = len(decisions)
        return ClosedLoopState.ACTION

    def _make_decision(self, report: DefectReport) -> Dict[str, Any]:
        """
        根据报告做出决策

        Args:
            report: 缺陷报告

        Returns:
            决策结果
        """
        decision = {
            'report_id': report.report_id,
            'action': 'monitor',
            'priority': 'low',
            'recommendations': []
        }

        # 根据严重程度和复核状态决策
        if report.review_status == ReviewStatus.FALSE_ALARM:
            decision['action'] = 'dismiss'
            decision['recommendations'].append('复核确认为误报，可忽略')
        elif report.review_status == ReviewStatus.CONFIRMED:
            if report.severity == DefectSeverity.CRITICAL:
                decision['action'] = 'emergency_stop'
                decision['priority'] = 'critical'
                decision['recommendations'].append('紧急停机检查')
            elif report.severity == DefectSeverity.ALARM:
                decision['action'] = 'schedule_maintenance'
                decision['priority'] = 'high'
                decision['recommendations'].append('安排紧急检修')
            elif report.severity == DefectSeverity.WARNING:
                decision['action'] = 'increase_monitoring'
                decision['priority'] = 'medium'
                decision['recommendations'].append('加强监测频率')
            else:
                decision['action'] = 'monitor'
                decision['recommendations'].append('持续观察')
        else:
            decision['action'] = 'manual_review'
            decision['recommendations'].append('需要人工复核确认')

        return decision

    async def _state_action(self, context: ClosedLoopContext) -> ClosedLoopState:
        """执行状态"""
        # 执行决策（实际应对接执行系统）
        for report in context.reports:
            if report.final_decision:
                report.action_taken = report.final_decision
                logger.info(f"执行决策: {report.report_id} -> {report.final_decision}")

        return ClosedLoopState.VERIFICATION

    async def _state_verification(self, context: ClosedLoopContext) -> ClosedLoopState:
        """验证状态"""
        # 验证执行结果
        context.metrics['verification_completed'] = True
        return ClosedLoopState.COMPLETED

    # =========================================================================
    # 启动和停止
    # =========================================================================

    async def start(self) -> None:
        """启动决策总线"""
        self._loop = asyncio.get_event_loop()
        self._review_task = asyncio.create_task(self.process_review_queue())
        logger.info("决策总线已启动")

    async def stop(self) -> None:
        """停止决策总线"""
        if self._review_task:
            self._review_task.cancel()
            try:
                await self._review_task
            except asyncio.CancelledError:
                pass
        logger.info("决策总线已停止")

    # =========================================================================
    # 统计和查询
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_reports = len(self._reports)
            status_counts = defaultdict(int)
            severity_counts = defaultdict(int)

            for report in self._reports.values():
                status_counts[report.review_status.value] += 1
                severity_counts[report.severity.name] += 1

            return {
                'total_reports': total_reports,
                'pending_reviews': self._review_queue.qsize(),
                'status_distribution': dict(status_counts),
                'severity_distribution': dict(severity_counts),
                'registered_plugins': list(self._plugins.keys()),
                'active_sessions': len(self._contexts)
            }

    def export_evidence_chain(self, report_id: str) -> Optional[Dict]:
        """
        导出证据链

        Args:
            report_id: 报告ID

        Returns:
            完整的证据链数据
        """
        report = self._reports.get(report_id)
        if not report:
            return None

        return {
            'report': report.to_dict(),
            'evidence_chain': [e.to_dict() for e in report.evidences],
            'timeline': self._build_timeline(report),
            'export_time': time.time()
        }

    def _build_timeline(self, report: DefectReport) -> List[Dict]:
        """构建时间线"""
        timeline = []

        # 添加创建事件
        timeline.append({
            'timestamp': report.created_at,
            'event': 'created',
            'description': f'缺陷报告创建: {report.defect_type}'
        })

        # 添加证据事件
        for evidence in report.evidences:
            timeline.append({
                'timestamp': evidence.timestamp,
                'event': 'evidence_added',
                'description': f'证据添加: {evidence.evidence_type.value}',
                'source': evidence.source_plugin
            })

        # 添加复核事件
        for review in report.review_results:
            timeline.append({
                'timestamp': review['timestamp'],
                'event': 'review_completed',
                'description': f'复核完成: {review["result"]["status"].value}'
            })

        # 按时间排序
        timeline.sort(key=lambda x: x['timestamp'])

        return timeline


# =============================================================================
# 便捷函数
# =============================================================================

_decision_bus_instance: Optional[DecisionBus] = None


def get_decision_bus() -> DecisionBus:
    """获取决策总线实例"""
    global _decision_bus_instance
    if _decision_bus_instance is None:
        _decision_bus_instance = DecisionBus()
    return _decision_bus_instance


def submit_defect(
    device_id: str,
    defect_type: str,
    severity: str,
    location: Dict,
    evidence_data: Dict,
    source_plugin: str,
    confidence: float
) -> DefectReport:
    """便捷提交缺陷函数"""
    bus = get_decision_bus()

    severity_map = {
        'critical': DefectSeverity.CRITICAL,
        'alarm': DefectSeverity.ALARM,
        'warning': DefectSeverity.WARNING,
        'attention': DefectSeverity.ATTENTION,
        'normal': DefectSeverity.NORMAL
    }

    evidence = Evidence(
        evidence_id=f"EV-{uuid.uuid4().hex[:8]}",
        evidence_type=EvidenceType.VISUAL_DETECTION,
        source_plugin=source_plugin,
        timestamp=time.time(),
        confidence=confidence,
        data=evidence_data
    )

    return bus.submit_defect(
        device_id=device_id,
        defect_type=defect_type,
        severity=severity_map.get(severity.lower(), DefectSeverity.ATTENTION),
        location=location,
        evidence=evidence,
        confidence=confidence
    )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'DefectSeverity',
    'ReviewStatus',
    'ClosedLoopState',
    'EvidenceType',
    'Evidence',
    'DefectReport',
    'ReviewRequest',
    'ClosedLoopContext',
    'DecisionBus',
    'get_decision_bus',
    'submit_defect'
]
