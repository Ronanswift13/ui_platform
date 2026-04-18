"""
表计读数 - 视频流服务
提供 MJPEG 实时流、帧标注、推理循环

支持两种数据源:
- simulated: 从 MeterSimulator 获取合成帧
- camera/rtsp: 从摄像头或 RTSP 流获取帧
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class StreamStats:
    """流统计"""
    running: bool = False
    source: str = "idle"
    frame_count: int = 0
    fps: float = 0.0
    last_reading: Optional[dict] = None
    avg_inference_ms: float = 0.0


class VideoStreamService:
    """MJPEG 视频流服务: 帧获取 → 推理 → 标注 → 编码"""

    def __init__(self, simulator: Any = None, detector: Any = None):
        self._simulator = simulator
        self._detector = detector
        self._running = False
        self._source = "idle"
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 帧缓冲
        self._current_frame: Optional[bytes] = None  # JPEG bytes
        self._frame_count = 0
        self._fps = 0.0
        self._fps_timestamps: deque[float] = deque(maxlen=30)
        self._inference_times: deque[float] = deque(maxlen=30)

        # 最近读数结果
        self._recent_readings: deque[dict] = deque(maxlen=50)
        self._last_reading: Optional[dict] = None

        # 事件日志
        self._events: deque[dict] = deque(maxlen=200)

    def start(self, source: str = "simulated") -> dict[str, Any]:
        """启动视频流"""
        if self._running:
            self.stop()

        self._source = source
        self._running = True
        self._frame_count = 0
        self._fps_timestamps.clear()
        self._inference_times.clear()

        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

        return {"status": "started", "source": source}

    def stop(self) -> dict[str, Any]:
        """停止视频流"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self._source = "idle"
        return {"status": "stopped"}

    def get_stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "source": self._source,
            "frame_count": self._frame_count,
            "fps": round(self._fps, 1),
            "avg_inference_ms": round(
                sum(self._inference_times) / max(1, len(self._inference_times)) * 1000, 2
            ) if self._inference_times else 0.0,
            "last_reading": self._last_reading,
        }

    def get_recent_readings(self, limit: int = 20) -> list[dict]:
        return list(self._recent_readings)[-limit:]

    def get_events(self, limit: int = 50) -> list[dict]:
        return list(self._events)[-limit:]

    def get_snapshot_jpeg(self) -> Optional[bytes]:
        """获取当前帧 JPEG"""
        return self._current_frame

    def generate_mjpeg(self):
        """MJPEG 流生成器 (用于 StreamingResponse)"""
        while self._running:
            frame_bytes = self._current_frame
            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame_bytes +
                    b"\r\n"
                )
            time.sleep(0.03)  # ~30 FPS cap

    # ── 推理循环 ───────────────────────────────────────────

    def _inference_loop(self) -> None:
        """后台推理线程"""
        while self._running:
            t0 = time.time()

            try:
                frame = self._acquire_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                # 运行检测
                readings = self._run_detection(frame)

                # 标注帧
                annotated = self._annotate_frame(frame, readings)

                # 编码 JPEG
                if cv2 is not None:
                    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self._current_frame = buf.tobytes()
                else:
                    self._current_frame = self._encode_placeholder(annotated)

                # 更新统计
                elapsed = time.time() - t0
                self._inference_times.append(elapsed)
                self._frame_count += 1
                self._fps_timestamps.append(time.time())
                self._update_fps()

                # 记录读数
                if readings:
                    for r in readings:
                        self._recent_readings.append(r)
                    self._last_reading = readings[-1] if readings else None

            except Exception as e:
                self._add_event("error", f"推理异常: {e}")
                time.sleep(0.5)

            # 控制帧率
            dt = time.time() - t0
            sleep_time = max(0, 0.1 - dt)  # ~10 FPS target for simulation
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _acquire_frame(self) -> Optional[np.ndarray]:
        """从数据源获取帧"""
        if self._source == "simulated" and self._simulator is not None:
            self._simulator.step()
            return self._simulator.render_frame()
        return None

    def _run_detection(self, frame: np.ndarray) -> list[dict]:
        """对帧运行检测"""
        if self._detector is None:
            return self._mock_detection()

        try:
            reading = self._detector.read_meter(frame, "pressure_gauge", "sim_roi")
            if reading is not None and reading.value is not None:
                import datetime
                return [{
                    "meter_id": "sim_roi",
                    "meter_type": reading.meter_type.value if hasattr(reading.meter_type, 'value') else str(reading.meter_type),
                    "value": reading.value,
                    "unit": reading.unit or "",
                    "confidence": reading.confidence,
                    "status": reading.status.value if hasattr(reading.status, 'value') else str(reading.status),
                    "timestamp": datetime.datetime.now().isoformat(),
                }]
        except Exception:
            pass
        return self._mock_detection()

    def _mock_detection(self) -> list[dict]:
        """仿真模式下用 Simulator 状态作为检测结果"""
        if self._simulator is None:
            return []

        import datetime
        state = self._simulator.get_state()
        if not state.get("active"):
            return []

        results = []
        for m in state.get("meters", []):
            results.append({
                "meter_id": m["meter_id"],
                "meter_type": m["meter_type"],
                "label": m.get("label", ""),
                "value": m["value"],
                "unit": m.get("unit", ""),
                "confidence": 0.92,
                "status": "success",
                "range_min": m.get("range_min", 0),
                "range_max": m.get("range_max", 1),
                "led_color": m.get("led_color"),
                "timestamp": datetime.datetime.now().isoformat(),
            })

            # 超限检测
            if m["meter_type"] != "led":
                rng = m.get("range_max", 1) - m.get("range_min", 0)
                frac = (m["value"] - m.get("range_min", 0)) / max(rng, 0.001)
                if frac > 0.9 or frac < 0.05:
                    self._add_event("warning", f"{m.get('label', m['meter_id'])} 读数 {m['value']} {m.get('unit', '')} 接近量程极限")

        return results

    def _annotate_frame(self, frame: np.ndarray, readings: list[dict]) -> np.ndarray:
        """在帧上标注检测结果"""
        if cv2 is None:
            return frame

        img = frame.copy()
        h, w = img.shape[:2]

        # 信息栏背景
        cv2.rectangle(img, (0, 0), (w, 30), (20, 20, 25), -1)
        info_text = f"FPS: {self._fps:.1f} | Frames: {self._frame_count} | Readings: {len(readings)}"
        cv2.putText(img, info_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 180), 1)

        # SIM 水印
        if self._source == "simulated":
            cv2.putText(img, "SIM", (w - 50, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 200), 2)

        # 读数标注 (右侧叠加)
        y_offset = 50
        for r in readings[:6]:
            label = r.get("label", r.get("meter_type", "meter"))
            if r.get("meter_type") == "led":
                val_text = f"{label}: {r.get('led_color', '?')}"
            else:
                val_text = f"{label}: {r.get('value', '?')} {r.get('unit', '')}"
            conf = r.get("confidence", 0)
            conf_text = f"conf: {conf:.2f}"

            color = (80, 255, 80) if conf >= 0.6 else (80, 200, 255) if conf >= 0.5 else (80, 80, 255)
            cv2.putText(img, val_text, (w - 250, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(img, conf_text, (w - 250, y_offset + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 140), 1)
            y_offset += 35

        return img

    def _update_fps(self) -> None:
        ts = self._fps_timestamps
        if len(ts) >= 2:
            dt = ts[-1] - ts[0]
            if dt > 0:
                self._fps = (len(ts) - 1) / dt

    def _add_event(self, level: str, message: str) -> None:
        import datetime
        self._events.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "stream",
            "data": {"level": level, "message": message},
        })

    @staticmethod
    def _encode_placeholder(frame: np.ndarray) -> bytes:
        """无 cv2 时的简单编码"""
        return b""
