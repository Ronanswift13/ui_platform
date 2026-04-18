"""
数据摄入流水线

职责:
1. 校验上传数据包的 manifest.json (ManifestValidator — 快速校验)
2. 完整六维上传校验 (UploadValidator — 深度校验)
3. 根据 task_type 路由数据到对应的预处理流水线 (DataRouter)
"""

from .manifest_validator import ManifestValidator
from .upload_validator import UploadValidator
from .data_router import DataRouter

__all__ = ["ManifestValidator", "UploadValidator", "DataRouter"]
