#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据上传 API 路由 (Data Upload API V2.0)
===========================================

实现修复方案中的后端 API 接口, 替代原有演示版本.

端点清单:
  POST   /api/training/data/upload            直接上传文件 (策略A)
  POST   /api/training/data/validate          校验已上传的数据集
  GET    /api/training/data/list              列出所有数据集
  DELETE /api/training/data/{dataset_id}      删除数据集
  GET    /api/training/data/status/{id}       获取数据集处理状态
  
  POST   /api/training/data/chunk/init        初始化分片上传
  POST   /api/training/data/chunk/upload      上传单个分片
  POST   /api/training/data/chunk/merge       合并分片
  
  POST   /api/training/data/import/local      服务器端扫描导入 (策略B)
  POST   /api/training/data/aggregate         聚合数据集生成训练YAML
  POST   /api/training/data/prepare-split     准备 train/val/test 拆分
  
  GET    /api/training/data/voltage-levels    获取电压等级列表
  GET    /api/training/data/plugin-types      获取插件类型列表

集成方式:
  在主应用 (app.py / main.py) 中调用:
    from apps.data_upload_api import integrate_data_upload_routes
    integrate_data_upload_routes(app)

作者: 破夜绘明团队
日期: 2026-02-01
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import BaseModel, Field

# 导入核心数据管理器
import sys
import os

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from platform_core.data_manager import DataManager

logger = logging.getLogger(__name__)

# =============================================================================
# 请求/响应模型 (Pydantic)
# =============================================================================

class ChunkInitRequest(BaseModel):
    """分片上传初始化请求"""
    file_name: str = Field(..., description="文件名")
    file_size: int = Field(..., description="文件总大小(bytes)")
    total_chunks: int = Field(..., description="总分片数")
    file_md5: str = Field("", description="文件MD5 (可选)")
    voltage_level: str = Field(..., description="电压等级")
    plugin_type: str = Field(..., description="插件类型")
    uploader: str = Field("anonymous", description="上传者")


class ChunkUploadRequest(BaseModel):
    """分片上传 - 单个分片元数据 (文件数据通过 multipart 发送)"""
    session_id: str
    chunk_index: int
    chunk_md5: Optional[str] = None


class ChunkMergeRequest(BaseModel):
    """分片合并请求"""
    session_id: str


class LocalImportRequest(BaseModel):
    """服务器端导入请求"""
    voltage_level: str
    plugin_type: str
    uploader: str = "server_import"


class AggregateRequest(BaseModel):
    """数据聚合请求"""
    voltage_level: str
    plugin_type: str


class PrepareSplitRequest(BaseModel):
    """训练拆分请求"""
    voltage_level: str
    plugin_type: str
    train_ratio: float = 0.8


class ValidateRequest(BaseModel):
    """校验请求"""
    dataset_id: str


# =============================================================================
# API 路由
# =============================================================================

router = APIRouter(prefix="/api/training/data", tags=["数据上传管理"])

# 懒加载 DataManager 单例
_data_manager: Optional[DataManager] = None


def get_data_manager() -> DataManager:
    """获取 DataManager 单例"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager.get_instance()
    return _data_manager


# ---------- 1. 直接上传 (策略A: Web端 < 2GB) ----------

@router.post("/upload")
async def upload_data(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    voltage_level: str = Form(...),
    plugins: str = Form(...),               # JSON 数组字符串 或 单个值
    uploader: str = Form("anonymous"),
    dataset_name: str = Form("upload"),
):
    """
    直接上传文件到结构化数据仓库

    支持:
    - 多文件上传 (图片 + 标注)
    - 压缩包上传 (.zip, .tar.gz) -> 自动解压
    - 后台异步校验

    前端通过 FormData 发送:
      formData.append('files', file1)
      formData.append('files', file2)
      formData.append('voltage_level', 'HV_220kV')
      formData.append('plugins', JSON.stringify(['transformer']))
      formData.append('uploader', 'StudentA')
    """
    logger.info(f"[DataUploadAPI] 收到上传请求: voltage={voltage_level}, plugins={plugins}, files={len(files)}")

    dm = get_data_manager()

    # 解析插件列表
    try:
        plugin_list = json.loads(plugins)
        if isinstance(plugin_list, str):
            plugin_list = [plugin_list]
    except (json.JSONDecodeError, TypeError):
        plugin_list = [plugins]

    logger.info(f"[DataUploadAPI] 解析插件列表: {plugin_list}")

    if not plugin_list:
        logger.error("[DataUploadAPI] 未选择插件类型")
        raise HTTPException(status_code=400, detail="至少选择一个插件类型")

    # 对每个插件类型创建数据集
    results = []
    for plugin_type in plugin_list:
        try:
            logger.info(f"[DataUploadAPI] 开始处理插件: {plugin_type}")

            upload_result = dm.upload_files(
                files=files,
                voltage_level=voltage_level,
                plugin_type=plugin_type,
                uploader=uploader,
                dataset_name=dataset_name,
            )

            dataset_id = upload_result["dataset_id"]
            dataset_path = upload_result["dataset_path"]

            logger.info(f"[DataUploadAPI] 文件已保存: dataset_id={dataset_id}, path={dataset_path}")

            # 后台处理: 解压 + 校验
            background_tasks.add_task(
                dm.process_dataset_background,
                dataset_id,
                dataset_path,
            )

            results.append({
                "plugin_type": plugin_type,
                "dataset_id": dataset_id,
                "status": "processing",
                "message": "文件已接收, 后台处理中...",
            })

            # 重置文件指针以便下一个插件类型复用
            for f in files:
                f.file.seek(0)

        except ValueError as e:
            logger.error(f"[DataUploadAPI] 处理插件 {plugin_type} 失败: {e}", exc_info=True)
            results.append({
                "plugin_type": plugin_type,
                "dataset_id": None,
                "status": "error",
                "message": str(e),
            })
        except Exception as e:
            logger.error(f"[DataUploadAPI] 未知错误: {e}", exc_info=True)
            results.append({
                "plugin_type": plugin_type,
                "dataset_id": None,
                "status": "error",
                "message": f"上传失败: {str(e)}",
            })

    success = any(r["status"] != "error" for r in results)
    file_count = len(files)

    logger.info(f"[DataUploadAPI] 上传完成: success={success}, file_count={file_count}")

    return {
        "success": success,
        "message": f"已接收 {file_count} 个文件" if success else "上传失败",
        "file_count": file_count,
        "results": results,
    }


# ---------- 2. 校验数据集 ----------

@router.post("/validate")
async def validate_dataset(request: ValidateRequest):
    """
    校验指定数据集
    
    调用 DataGatekeeper:
    - 垃圾文件清洗
    - 图片-标签对齐检查
    - 格式探针 (YOLO/COCO/VOC)
    """
    dm = get_data_manager()
    report = dm.validate_dataset(request.dataset_id)

    return {
        "success": True,
        "valid": report.get("valid", False),
        "dataset_id": request.dataset_id,
        "report": report,
        "message": "校验通过" if report.get("valid") else "校验发现问题",
    }


@router.post("/validate-preview")
async def validate_preview(
    voltage_level: str = Form(""),
    plugins: str = Form("[]"),
    files: str = Form("[]"),
):
    """
    上传前预验证 (不实际上传文件, 只检查文件名/数量/类型)
    兼容原有前端的验证接口
    """
    try:
        file_list = json.loads(files) if isinstance(files, str) else files
    except json.JSONDecodeError:
        file_list = []

    try:
        plugin_list = json.loads(plugins) if isinstance(plugins, str) else plugins
    except json.JSONDecodeError:
        plugin_list = []

    # 分类统计
    images = 0
    labels = 0
    archives = 0
    other = 0

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    label_exts = {'.txt', '.xml', '.json'}
    archive_exts = {'.zip', '.tar', '.gz', '.tgz', '.rar'}

    for f_info in file_list:
        name = f_info.get("name", "")
        ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext in image_exts:
            images += 1
        elif ext in label_exts:
            labels += 1
        elif ext in archive_exts:
            archives += 1
        else:
            other += 1

    warnings = []
    if images > 0 and labels == 0 and archives == 0:
        warnings.append("未发现标注文件, 如果在压缩包中请忽略此警告")
    if images == 0 and archives == 0:
        warnings.append("未发现图片文件或压缩包")
    if not voltage_level:
        warnings.append("未选择电压等级")
    if not plugin_list:
        warnings.append("未选择插件类型")

    valid = len(warnings) == 0 or archives > 0

    return {
        "valid": valid,
        "message": "数据验证通过, 可以上传" if valid else "; ".join(warnings),
        "summary": {
            "images": images,
            "labels": labels,
            "archives": archives,
            "other": other,
            "total": len(file_list),
        },
        "warnings": warnings,
    }


# ---------- 3. 列出数据集 ----------

@router.get("/list")
async def list_datasets(
    voltage_level: Optional[str] = Query(None),
    plugin_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    """
    列出所有数据集, 支持过滤
    
    返回格式兼容前端 loadImportHistory():
      { records: [{ created_at, voltage_level, plugins, file_count, status, id }] }
    """
    dm = get_data_manager()
    datasets = dm.list_datasets(
        voltage_level=voltage_level,
        plugin_type=plugin_type,
        status=status,
    )

    # 转为前端兼容格式
    records = []
    for meta in datasets:
        records.append({
            "id": meta.get("dataset_id", ""),
            "created_at": meta.get("upload_time", ""),
            "voltage_level": meta.get("voltage_level", ""),
            "plugins": [meta.get("plugin_type", "")],
            "file_count": meta.get("image_count", 0) + meta.get("label_count", 0),
            "image_count": meta.get("image_count", 0),
            "label_count": meta.get("label_count", 0),
            "status": meta.get("status", "pending"),
            "format": meta.get("format", "UNKNOWN"),
            "uploader": meta.get("uploader", ""),
            "dataset_name": meta.get("dataset_name", ""),
            "validation_report": meta.get("validation_report"),
        })

    return {
        "success": True,
        "records": records,
        "total": len(records),
    }


# ---------- 4. 删除数据集 ----------

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """删除指定数据集"""
    dm = get_data_manager()
    success = dm.delete_dataset(dataset_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"数据集不存在: {dataset_id}")

    return {
        "success": True,
        "message": f"数据集 {dataset_id} 已删除",
    }


# ---------- 5. 获取处理状态 ----------

@router.get("/status/{dataset_id}")
async def get_dataset_status(dataset_id: str):
    """获取数据集处理状态 (解压/校验进度)"""
    dm = get_data_manager()

    # 先查异步任务状态
    task_status = dm.get_task_status(dataset_id)

    # 再查元数据
    meta = dm.warehouse.get_metadata(dataset_id)

    return {
        "success": True,
        "dataset_id": dataset_id,
        "task_status": task_status,
        "metadata": meta,
    }


# ---------- 6. 分片断点续传 ----------

@router.post("/chunk/init")
async def chunk_init(request: ChunkInitRequest):
    """初始化分片上传会话"""
    dm = get_data_manager()
    session_id = dm.ingestion.init_chunk_upload(
        file_name=request.file_name,
        file_size=request.file_size,
        total_chunks=request.total_chunks,
        file_md5=request.file_md5,
        voltage_level=request.voltage_level,
        plugin_type=request.plugin_type,
        uploader=request.uploader,
    )
    return {
        "success": True,
        "session_id": session_id,
        "message": f"分片上传会话已创建, 共 {request.total_chunks} 个分片",
    }


@router.post("/chunk/upload")
async def chunk_upload(
    session_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk_md5: str = Form(""),
    chunk: UploadFile = File(...),
):
    """上传单个分片"""
    dm = get_data_manager()

    chunk_data = await chunk.read()

    try:
        result = dm.ingestion.receive_chunk(
            session_id=session_id,
            chunk_index=chunk_index,
            chunk_data=chunk_data,
            chunk_md5=chunk_md5 or None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "success": True,
        **result,
    }


@router.post("/chunk/merge")
async def chunk_merge(
    request: ChunkMergeRequest,
    background_tasks: BackgroundTasks,
):
    """合并所有分片, 触发后台解压+校验"""
    dm = get_data_manager()

    try:
        dataset_id, dataset_path = dm.ingestion.merge_chunks(request.session_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 后台处理
    background_tasks.add_task(
        dm.process_dataset_background,
        dataset_id,
        str(dataset_path),
    )

    return {
        "success": True,
        "dataset_id": dataset_id,
        "message": "分片合并完成, 后台处理中...",
    }


# ---------- 7. 服务器端扫描导入 (策略B) ----------

@router.post("/import/local")
async def import_local(
    request: LocalImportRequest,
    background_tasks: BackgroundTasks,
):
    """
    扫描服务器 temp_upload/ 目录, 导入压缩包到结构化仓库
    
    适用于 > 2GB 超大数据集:
    1. 学生通过 U盘/FTP 将数据拷贝至服务器 training/data/temp_upload/ 目录
    2. 调用此接口触发扫描导入
    3. 后台自动解压 + 校验
    """
    dm = get_data_manager()

    import_results = dm.scan_and_import(
        voltage_level=request.voltage_level,
        plugin_type=request.plugin_type,
        uploader=request.uploader,
    )

    # 为每个成功导入的数据集安排后台处理
    for result in import_results:
        if result["status"] == "queued" and result["dataset_id"]:
            meta = dm.warehouse.get_metadata(result["dataset_id"])
            if meta:
                background_tasks.add_task(
                    dm.process_dataset_background,
                    result["dataset_id"],
                    meta["data_path"],
                )

    return {
        "success": True,
        "imported_count": len([r for r in import_results if r["status"] == "queued"]),
        "results": import_results,
        "message": f"扫描完成, 发现 {len(import_results)} 个文件",
    }


# ---------- 8. 数据聚合 (模块四) ----------

@router.post("/aggregate")
async def aggregate_datasets(request: AggregateRequest):
    """
    聚合指定 电压+插件 下的所有有效数据集
    生成 YOLO 训练用的 aggregated_data.yaml
    """
    dm = get_data_manager()

    result = dm.aggregate_for_training(
        voltage_level=request.voltage_level,
        plugin_type=request.plugin_type,
    )

    return {
        "success": True,
        "message": f"聚合 {result['dataset_count']} 个数据集, "
                   f"{result['total_images']} 张图像",
        **result,
    }


@router.post("/prepare-split")
async def prepare_training_split(request: PrepareSplitRequest):
    """
    准备训练拆分: 将所有有效数据集的图片/标签按比例拆分到
    processed/{voltage}/{plugin}/images/{train,val,test}
    """
    dm = get_data_manager()

    result = dm.prepare_training_split(
        voltage_level=request.voltage_level,
        plugin_type=request.plugin_type,
        train_ratio=request.train_ratio,
    )

    if "error" in result:
        return {
            "success": False,
            "message": result["error"],
        }

    return {
        "success": True,
        "message": f"拆分完成: train={result['splits']['train']}, "
                   f"val={result['splits']['val']}, test={result['splits']['test']}",
        **result,
    }


# ---------- 9. 枚举值接口 ----------

@router.get("/voltage-levels")
async def get_voltage_levels():
    """获取所有电压等级选项"""
    dm = get_data_manager()
    return {
        "success": True,
        "voltage_levels": dm.voltage_levels,
    }


@router.get("/plugin-types")
async def get_plugin_types():
    """获取所有插件类型选项"""
    dm = get_data_manager()
    return {
        "success": True,
        "plugin_types": dm.plugin_types,
    }


# =============================================================================
# 集成函数 (挂载到主应用)
# =============================================================================

def integrate_data_upload_routes(app):
    """
    将数据上传路由集成到主 FastAPI 应用
    
    用法:
        from apps.data_upload_api import integrate_data_upload_routes
        integrate_data_upload_routes(app)
    """
    app.include_router(router)
    logger.info("[DataUploadAPI] 数据上传路由已集成")


# =============================================================================
# 独立运行测试
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="数据上传API测试")
    integrate_data_upload_routes(app)

    uvicorn.run(app, host="127.0.0.1", port=8082)
