"""
回放录制器

录制任务执行过程,用于后续回放
"""


from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from platform_core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RecordingFrame:
    """录制帧"""
    index: int
    timestamp: datetime
    image: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class ReplayRecorder:
    """
    回放录制器

    记录任务执行过程的所有数据
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._frames: list[RecordingFrame] = []
        self._recording = False
        self._start_time: Optional[datetime] = None
        self._metadata: dict[str, Any] = {}

    def start(self, metadata: Optional[dict[str, Any]] = None) -> None:
        """开始录制"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "frames").mkdir(exist_ok=True)

        self._frames.clear()
        self._recording = True
        self._start_time = datetime.now()
        self._metadata = metadata or {}

        logger.info(f"开始录制: {self.output_dir}")

    def stop(self) -> dict[str, Any]:
        """停止录制"""
        self._recording = False
        end_time = datetime.now()

        # 保存录制信息
        recording_info = {
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": end_time.isoformat(),
            "frame_count": len(self._frames),
            "metadata": self._metadata,
        }

        info_path = self.output_dir / "recording.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(recording_info, f, ensure_ascii=False, indent=2)

        logger.info(f"录制完成: {len(self._frames)} 帧")
        return recording_info

    def add_frame(
        self,
        image: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """
        添加一帧

        Args:
            image: 图像数据
            metadata: 帧元数据

        Returns:
            帧索引
        """
        if not self._recording:
            raise RuntimeError("未开始录制")

        index = len(self._frames)
        frame = RecordingFrame(
            index=index,
            timestamp=datetime.now(),
            image=image,
            metadata=metadata or {},
        )
        self._frames.append(frame)

        # 保存图像
        frame_path = self.output_dir / "frames" / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(frame_path), image)

        return index

    def add_annotated_frame(
        self,
        index: int,
        annotated_image: np.ndarray,
    ) -> None:
        """添加标注后的帧"""
        annotated_dir = self.output_dir / "annotated"
        annotated_dir.mkdir(exist_ok=True)

        frame_path = annotated_dir / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(frame_path), annotated_image)

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self._frames)
