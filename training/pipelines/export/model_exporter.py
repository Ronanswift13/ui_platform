"""
统一模型导出器

将训练产出的模型导出为部署格式:
- ONNX (所有插件)
- TensorRT (可选，GPU 部署)

导出按 plugin_id + task_type 组织:
  exports/{plugin_id}/{task_type}/model.onnx
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """导出结果"""

    plugin_id: str
    task_type: str
    format: str
    output_path: Path
    file_size_mb: float = 0.0
    sha256: str = ""
    success: bool = True
    message: str = ""


class ModelExporter:
    """统一模型导出器"""

    def __init__(self, exports_root: Path | None = None):
        self.exports_root = exports_root or Path(__file__).parent.parent.parent / "exports"

    def export_onnx(
        self,
        model_path: Path,
        plugin_id: str,
        task_type: str,
        input_shape: tuple[int, ...] = (1, 3, 640, 640),
        opset_version: int = 11,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
    ) -> ExportResult:
        """
        导出为 ONNX 格式

        Args:
            model_path: PyTorch 模型文件 (.pt / .pth)
            plugin_id: 插件 ID
            task_type: 任务类型
            input_shape: 输入张量形状
            opset_version: ONNX opset 版本
            dynamic_axes: 动态轴配置

        Returns:
            ExportResult
        """
        output_dir = self.exports_root / plugin_id / task_type
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model.onnx"

        try:
            import torch

            # 加载模型
            if not model_path.exists():
                return ExportResult(
                    plugin_id=plugin_id,
                    task_type=task_type,
                    format="onnx",
                    output_path=output_path,
                    success=False,
                    message=f"模型文件不存在: {model_path}",
                )

            model = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(model, dict) and "model" in model:
                model = model["model"]

            if hasattr(model, "eval"):
                model.eval()

            # 导出
            dummy_input = torch.randn(*input_shape)
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                opset_version=opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes or {"input": {0: "batch"}, "output": {0: "batch"}},
            )

            # 计算文件信息
            file_size = output_path.stat().st_size / (1024 * 1024)
            sha256 = self._compute_sha256(output_path)

            # 保存元数据
            meta = {
                "plugin_id": plugin_id,
                "task_type": task_type,
                "format": "onnx",
                "opset_version": opset_version,
                "input_shape": list(input_shape),
                "file_size_mb": round(file_size, 2),
                "sha256": sha256,
            }
            with open(output_dir / "export_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info(f"ONNX 导出成功: {output_path} ({file_size:.1f} MB)")
            return ExportResult(
                plugin_id=plugin_id,
                task_type=task_type,
                format="onnx",
                output_path=output_path,
                file_size_mb=file_size,
                sha256=sha256,
                success=True,
                message="导出成功",
            )

        except ImportError:
            return ExportResult(
                plugin_id=plugin_id,
                task_type=task_type,
                format="onnx",
                output_path=output_path,
                success=False,
                message="PyTorch 未安装",
            )
        except Exception as e:
            return ExportResult(
                plugin_id=plugin_id,
                task_type=task_type,
                format="onnx",
                output_path=output_path,
                success=False,
                message=f"导出失败: {e}",
            )

    def _compute_sha256(self, path: Path) -> str:
        """计算文件 SHA256"""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def export_bundle(
        self,
        *,
        plugin_id: str,
        task_type: str,
        version: str,
        model_path: Path,
        modality: str,
        metrics: dict[str, Any] | None = None,
        compatible_runtime: list[dict[str, Any]] | None = None,
        labels: list[str] | None = None,
        target_schema: dict[str, Any] | None = None,
        preprocess_config_path: Path | None = None,
        postprocess_config_path: Path | None = None,
        model_role: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ExportResult:
        """
        以统一 bundle 结构导出模型资产。

        输出目录:
          exports/{plugin_id}/{task_type}/{version}/
            - model.*
            - bundle.json
            - label_map.json (可选)
            - preprocess.*
            - postprocess.*
        """
        output_dir = self.exports_root / plugin_id / task_type / version
        output_dir.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            return ExportResult(
                plugin_id=plugin_id,
                task_type=task_type,
                format=model_path.suffix.lstrip(".") or "artifact",
                output_path=output_dir / model_path.name,
                success=False,
                message=f"模型文件不存在: {model_path}",
            )

        model_dst = output_dir / f"model{model_path.suffix}"
        shutil.copy2(model_path, model_dst)

        if preprocess_config_path and preprocess_config_path.exists():
            shutil.copy2(
                preprocess_config_path,
                output_dir / f"preprocess{preprocess_config_path.suffix}",
            )
        if postprocess_config_path and postprocess_config_path.exists():
            shutil.copy2(
                postprocess_config_path,
                output_dir / f"postprocess{postprocess_config_path.suffix}",
            )

        if labels:
            label_map = {str(idx): label for idx, label in enumerate(labels)}
            with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
                json.dump(label_map, f, ensure_ascii=False, indent=2)

        bundle = {
            "plugin_id": plugin_id,
            "task_type": task_type,
            "version": version,
            "modality": modality,
            "model_role": model_role,
            "metrics": metrics or {},
            "labels": labels or [],
            "target_schema": target_schema or {},
            "compatible_runtime": compatible_runtime or [],
            "export_path": str(output_dir),
        }
        if extra_metadata:
            bundle.update(extra_metadata)
        with open(output_dir / "bundle.json", "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)

        return ExportResult(
            plugin_id=plugin_id,
            task_type=task_type,
            format=model_path.suffix.lstrip(".") or "artifact",
            output_path=model_dst,
            file_size_mb=model_dst.stat().st_size / (1024 * 1024),
            sha256=self._compute_sha256(model_dst),
            success=True,
            message=f"bundle 导出成功: {output_dir}",
        )
