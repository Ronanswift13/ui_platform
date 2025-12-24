"""
回放播放器

实现确定性回放功能:
- 从证据目录加载历史运行数据
- 重现任务执行过程
- 对比新旧结果
"""


from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

import cv2
import numpy as np

from platform_core.evidence.manager import EvidenceManager, EvidenceMetadata, get_evidence_manager
from platform_core.logging import get_logger
from platform_core.schema.models import PluginOutput, RecognitionResult

logger = get_logger(__name__)


@dataclass
class ReplayFrame:
    """回放帧"""
    index: int
    timestamp: datetime
    raw_image: np.ndarray
    annotated_image: Optional[np.ndarray] = None
    results: list[RecognitionResult] = field(default_factory=list)


@dataclass
class ReplayComparison:
    """回放对比结果"""
    original_output: PluginOutput
    replay_output: PluginOutput
    match: bool
    differences: list[dict[str, Any]]


class ReplaySession:
    """
    回放会话

    管理单次回放的上下文
    """

    def __init__(self, run_id: str, evidence_dir: Path):
        self.run_id = run_id
        self.evidence_dir = evidence_dir
        self._metadata: Optional[EvidenceMetadata] = None
        self._original_output: Optional[PluginOutput] = None
        self._frames: list[ReplayFrame] = []
        self._current_index = 0

    def load(self) -> bool:
        """加载回放数据"""
        try:
            # 加载元数据
            meta_path = self.evidence_dir / "meta.json"
            if not meta_path.exists():
                logger.error(f"元数据不存在: {meta_path}")
                return False

            with open(meta_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)

            self._metadata = EvidenceMetadata(
                run_id=meta_dict["run_id"],
                task_id=meta_dict["task_id"],
                plugin_id=meta_dict["plugin_id"],
                plugin_version=meta_dict["plugin_version"],
                code_hash=meta_dict["code_hash"],
                site_id=meta_dict["site_id"],
                device_id=meta_dict["device_id"],
                started_at=datetime.fromisoformat(meta_dict["started_at"]),
            )

            # 加载原始输出
            output_path = self.evidence_dir / "results" / "output.json"
            if output_path.exists():
                with open(output_path, "r", encoding="utf-8") as f:
                    output_dict = json.load(f)
                self._original_output = PluginOutput.model_validate(output_dict)

            # 加载图像帧
            self._load_frames()

            logger.info(f"回放会话已加载: {self.run_id}, {len(self._frames)} 帧")
            return True

        except Exception as e:
            logger.error(f"加载回放数据失败: {e}")
            return False

    def _load_frames(self) -> None:
        """加载所有图像帧"""
        raw_dir = self.evidence_dir / "raw"
        annotated_dir = self.evidence_dir / "annotated"

        if not raw_dir.exists():
            return

        for i, raw_path in enumerate(sorted(raw_dir.glob("*.jpg")) + sorted(raw_dir.glob("*.png"))):
            raw_image = cv2.imread(str(raw_path))
            if raw_image is None:
                continue

            annotated_image = None
            annotated_path = annotated_dir / raw_path.name
            if annotated_path.exists():
                annotated_image = cv2.imread(str(annotated_path))

            frame = ReplayFrame(
                index=i,
                timestamp=datetime.fromtimestamp(raw_path.stat().st_mtime),
                raw_image=raw_image,
                annotated_image=annotated_image,
            )
            self._frames.append(frame)

    @property
    def metadata(self) -> Optional[EvidenceMetadata]:
        return self._metadata

    @property
    def original_output(self) -> Optional[PluginOutput]:
        return self._original_output

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def get_frame(self, index: int) -> Optional[ReplayFrame]:
        """获取指定帧"""
        if 0 <= index < len(self._frames):
            return self._frames[index]
        return None

    def iter_frames(self) -> Generator[ReplayFrame, None, None]:
        """迭代所有帧"""
        for frame in self._frames:
            yield frame

    def seek(self, index: int) -> bool:
        """跳转到指定帧"""
        if 0 <= index < len(self._frames):
            self._current_index = index
            return True
        return False

    def next_frame(self) -> Optional[ReplayFrame]:
        """获取下一帧"""
        if self._current_index < len(self._frames):
            frame = self._frames[self._current_index]
            self._current_index += 1
            return frame
        return None

    def reset(self) -> None:
        """重置到开始"""
        self._current_index = 0


class ReplayPlayer:
    """
    回放播放器

    支持从证据目录进行确定性回放
    """

    def __init__(self):
        self.evidence_manager = get_evidence_manager()
        self._current_session: Optional[ReplaySession] = None

    def open_session(self, run_id: str) -> Optional[ReplaySession]:
        """
        打开回放会话

        Args:
            run_id: 运行ID

        Returns:
            回放会话
        """
        evidence_dir = self.evidence_manager.runs_dir / run_id
        if not evidence_dir.exists():
            logger.error(f"证据目录不存在: {evidence_dir}")
            return None

        session = ReplaySession(run_id, evidence_dir)
        if session.load():
            self._current_session = session
            return session
        return None

    def close_session(self) -> None:
        """关闭当前会话"""
        self._current_session = None

    @property
    def current_session(self) -> Optional[ReplaySession]:
        return self._current_session

    def replay_and_compare(
        self,
        run_id: str,
        plugin_executor,
    ) -> Optional[ReplayComparison]:
        """
        回放并对比结果

        Args:
            run_id: 运行ID
            plugin_executor: 插件执行器

        Returns:
            对比结果
        """
        session = self.open_session(run_id)
        if session is None:
            return None

        if session.original_output is None:
            logger.error("原始输出不存在,无法对比")
            return None

        # TODO: 重新执行插件并对比结果
        # 这里需要根据实际情况实现

        return None

    def list_available_runs(
        self,
        plugin_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[EvidenceMetadata]:
        """列出可回放的运行记录"""
        return self.evidence_manager.list_runs(plugin_id=plugin_id, limit=limit)


def compare_outputs(
    original: PluginOutput,
    replay: PluginOutput,
) -> ReplayComparison:
    """
    对比两次执行的输出

    Args:
        original: 原始输出
        replay: 回放输出

    Returns:
        对比结果
    """
    differences = []

    # 对比结果数量
    if len(original.results) != len(replay.results):
        differences.append({
            "type": "result_count",
            "original": len(original.results),
            "replay": len(replay.results),
        })

    # 对比每个结果
    for i, (orig, rep) in enumerate(zip(original.results, replay.results)):
        if orig.label != rep.label:
            differences.append({
                "type": "label_mismatch",
                "index": i,
                "original": orig.label,
                "replay": rep.label,
            })

        if abs(orig.confidence - rep.confidence) > 0.01:
            differences.append({
                "type": "confidence_diff",
                "index": i,
                "original": orig.confidence,
                "replay": rep.confidence,
            })

    match = len(differences) == 0

    return ReplayComparison(
        original_output=original,
        replay_output=replay,
        match=match,
        differences=differences,
    )
