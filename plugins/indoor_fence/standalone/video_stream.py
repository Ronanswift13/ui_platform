#!/usr/bin/env python3
"""
视频流服务 - 后台推理循环 + MJPEG 帧输出 + 帧标注渲染

提供:
- 后台线程持续从 CameraAdapter 获取帧并执行推理
- MJPEG 流生成器供 HTTP StreamingResponse 使用
- 帧标注 (检测框、区域边界、统计信息)
- 模拟帧渲染 (仿真模式下生成可视化画面)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入 cv2, 缺失时标记不可用
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("FALLBACK: cv2 不可用, 视频流使用纯 numpy 编码")


# =============================================================================
# 颜色常量 (BGR 格式)
# =============================================================================
COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 200, 255)
COLOR_BLUE = (255, 150, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (80, 80, 80)
COLOR_DARK_BG = (30, 26, 26)      # #1a1a1e
COLOR_GRID = (50, 45, 45)          # 网格线
COLOR_SIM_PERSON = (0, 255, 200)   # 模拟人形
COLOR_ZONE_BORDER = (255, 180, 0)  # 区域边界
COLOR_FENCE_LINE = (0, 100, 255)   # 围栏线


class VideoStreamService:
    """
    视频流服务

    负责:
    1. 后台推理循环 (守护线程)
    2. 帧标注绘制
    3. MJPEG 异步生成器
    """

    def __init__(self, plugin: Any, fps: int = 15):
        """
        Args:
            plugin: IndoorFencePlugin 实例
            fps: 目标帧率
        """
        self._plugin = plugin
        self._fps = fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # 统计
        self._frame_count = 0
        self._detection_count = 0
        self._actual_fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0

    # -----------------------------------------------------------------
    # 生命周期
    # -----------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """启动后台推理线程"""
        if self._running:
            return

        self._running = True
        self._frame_count = 0
        self._detection_count = 0
        self._thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="video-stream-inference",
        )
        self._thread.start()
        logger.info("视频流服务已启动 (fps=%d)", self._fps)

    def stop(self) -> None:
        """停止后台推理线程"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

        with self._frame_lock:
            self._latest_annotated_frame = None

        logger.info("视频流服务已停止")

    # -----------------------------------------------------------------
    # 后台推理循环
    # -----------------------------------------------------------------

    def _inference_loop(self) -> None:
        """后台推理循环 (在守护线程中运行)"""
        interval = 1.0 / self._fps

        while self._running:
            loop_start = time.time()

            try:
                frame = self._acquire_frame()
                if frame is None:
                    time.sleep(interval)
                    continue

                # 执行检测
                detections = self._run_detection(frame)

                # 标注帧
                annotated = self._annotate_frame(frame, detections)

                # 更新缓冲
                with self._frame_lock:
                    self._latest_annotated_frame = annotated

                # 更新统计
                self._frame_count += 1
                self._detection_count = len(detections)
                self._update_fps()

            except Exception as e:
                logger.error("视频流推理异常: %s", e, exc_info=True)

            # 帧率控制
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _acquire_frame(self) -> Optional[np.ndarray]:
        """从相机适配器获取帧"""
        camera = getattr(self._plugin, '_camera_adapter', None)
        if camera is None:
            return self._fallback_frame()

        frame = camera.get_frame()
        if frame is None:
            return self._fallback_frame()

        return frame

    def _fallback_frame(self) -> np.ndarray:
        """当相机不可用时的后备帧"""
        return self._render_simulation_frame([], "NO CAMERA")

    def _run_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """执行检测并返回结果列表"""
        detections = []
        camera = getattr(self._plugin, '_camera_adapter', None)

        if camera is None:
            return detections

        try:
            # 使用相机适配器的检测功能
            if hasattr(camera, 'is_connected') and (camera.is_connected or
               getattr(camera, '_simulated', False)):
                person_dets = camera.get_person_detections(frame)
                if person_dets:
                    for pd in person_dets:
                        det_dict = {
                            'detection_id': pd.detection_id,
                            'bbox': pd.bbox,
                            'confidence': pd.confidence,
                            'track_id': getattr(pd, 'track_id', None),
                            'foot_pixel': getattr(pd, 'foot_pixel', None),
                        }
                        detections.append(det_dict)
        except Exception as e:
            logger.debug("检测执行异常: %s", e)

        return detections

    # -----------------------------------------------------------------
    # 帧标注
    # -----------------------------------------------------------------

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """在帧上绘制检测结果和信息叠加"""
        if not CV2_AVAILABLE:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. 绘制区域边界
        self._draw_zones(annotated)

        # 2. 绘制检测框
        for det in detections:
            self._draw_detection(annotated, det, w, h)

        # 3. 绘制信息栏
        self._draw_info_bar(annotated, len(detections))

        return annotated

    def _draw_detection(
        self,
        frame: np.ndarray,
        det: Dict[str, Any],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """绘制单个检测框"""
        bbox = det.get('bbox', {})
        if isinstance(bbox, dict):
            x = int(bbox.get('x', 0) * frame_w)
            y = int(bbox.get('y', 0) * frame_h)
            bw = int(bbox.get('width', 0) * frame_w)
            bh = int(bbox.get('height', 0) * frame_h)
        else:
            return

        conf = det.get('confidence', 0)
        track_id = det.get('track_id', '')

        # 检测框
        color = COLOR_GREEN if conf > 0.7 else COLOR_YELLOW
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

        # 标签
        label = f"{track_id or 'Person'} {conf:.0%}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            frame,
            (x, y - label_size[1] - 6),
            (x + label_size[0] + 4, y),
            color,
            -1,
        )
        cv2.putText(
            frame, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )

    def _draw_zones(self, frame: np.ndarray) -> None:
        """绘制区域边界和围栏线"""
        zone_config = getattr(self._plugin, '_zone_config', None)
        if zone_config is None:
            return

        h, w = frame.shape[:2]

        # 绘制区域
        for zone in getattr(zone_config, 'zones', {}).values():
            points = getattr(zone, 'points', [])
            if len(points) >= 3:
                pts = np.array(
                    [(int(p[0] * w), int(p[1] * h)) for p in points],
                    np.int32,
                )
                cv2.polylines(frame, [pts], True, COLOR_ZONE_BORDER, 1)

        # 绘制围栏线
        for line in getattr(zone_config, 'fence_lines', []):
            start = getattr(line, 'start', None)
            end = getattr(line, 'end', None)
            if start and end:
                p1 = (int(start[0] * w), int(start[1] * h))
                p2 = (int(end[0] * w), int(end[1] * h))
                cv2.line(frame, p1, p2, COLOR_FENCE_LINE, 2)

    def _draw_info_bar(self, frame: np.ndarray, det_count: int) -> None:
        """绘制顶部信息栏"""
        h, w = frame.shape[:2]

        # 半透明背景条
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 信息文本
        now_str = datetime.now().strftime("%H:%M:%S")
        info = (
            f"FPS: {self._actual_fps:.1f} | "
            f"Frame: {self._frame_count} | "
            f"Detected: {det_count} | "
            f"{now_str}"
        )
        cv2.putText(
            frame, info, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1,
        )

        # 模拟模式标记
        camera = getattr(self._plugin, '_camera_adapter', None)
        if camera and getattr(camera, '_simulated', False):
            cv2.putText(
                frame, "SIM", (w - 50, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 2,
            )

    # -----------------------------------------------------------------
    # 模拟帧渲染
    # -----------------------------------------------------------------

    def _render_simulation_frame(
        self,
        detections: Optional[List[Dict]] = None,
        watermark: str = "SIMULATION",
    ) -> np.ndarray:
        """生成模拟可视化帧"""
        w, h = 640, 480
        camera = getattr(self._plugin, '_camera_adapter', None)
        if camera and hasattr(camera, 'config'):
            cfg_res = getattr(camera.config, 'resolution', (640, 480))
            w, h = cfg_res[0], cfg_res[1]

        # 深灰背景
        frame = np.full((h, w, 3), 26, dtype=np.uint8)

        if not CV2_AVAILABLE:
            return frame

        # 网格线
        grid_spacing = 40
        for gx in range(0, w, grid_spacing):
            cv2.line(frame, (gx, 0), (gx, h), COLOR_GRID, 1)
        for gy in range(0, h, grid_spacing):
            cv2.line(frame, (0, gy), (w, gy), COLOR_GRID, 1)

        # 水印
        text_size = cv2.getTextSize(
            watermark, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2,
        )[0]
        tx = (w - text_size[0]) // 2
        ty = (h + text_size[1]) // 2
        cv2.putText(
            frame, watermark, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 55, 55), 2,
        )

        return frame

    # -----------------------------------------------------------------
    # MJPEG 生成器
    # -----------------------------------------------------------------

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新标注帧 (线程安全)"""
        with self._frame_lock:
            if self._latest_annotated_frame is not None:
                return self._latest_annotated_frame.copy()
        return None

    async def generate_mjpeg(self) -> AsyncGenerator[bytes, None]:
        """异步 MJPEG 帧生成器"""
        if not CV2_AVAILABLE:
            logger.error("MJPEG 流不可用: cv2 未安装")
            return

        interval = 1.0 / self._fps
        consecutive_empty = 0

        while self._running:
            frame = self.get_latest_frame()

            if frame is not None:
                consecutive_empty = 0
                ret, jpeg_buf = cv2.imencode(
                    '.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                if ret:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + jpeg_buf.tobytes()
                        + b'\r\n'
                    )
            else:
                consecutive_empty += 1
                # 超过 2 秒无帧, 发送占位帧
                if consecutive_empty > self._fps * 2:
                    placeholder = self._render_simulation_frame(
                        watermark="WAITING...",
                    )
                    ret, jpeg_buf = cv2.imencode('.jpg', placeholder)
                    if ret:
                        yield (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n'
                            + jpeg_buf.tobytes()
                            + b'\r\n'
                        )
                    consecutive_empty = 0

            await asyncio.sleep(interval)

    def get_snapshot_jpeg(self) -> Optional[bytes]:
        """获取单帧 JPEG 快照"""
        frame = self.get_latest_frame()
        if frame is None:
            frame = self._render_simulation_frame(watermark="SNAPSHOT")

        if not CV2_AVAILABLE:
            return None

        ret, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return jpeg_buf.tobytes() if ret else None

    # -----------------------------------------------------------------
    # 内部工具
    # -----------------------------------------------------------------

    def _update_fps(self) -> None:
        """更新实际帧率统计"""
        self._fps_frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._actual_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_time = now

    @property
    def stats(self) -> Dict[str, Any]:
        """返回流服务统计"""
        return {
            "running": self._running,
            "frame_count": self._frame_count,
            "detection_count": self._detection_count,
            "fps": round(self._actual_fps, 1),
            "target_fps": self._fps,
        }


__all__ = ['VideoStreamService']
