#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training 内部独立数据上传 API。

兼容旧的 /api/training/data/* 路由，同时提供批次详情与标准化能力。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from training.services.upload_batch_service import (
    PLUGIN_CLASSES,
    SUPPORTED_FORMATS,
    batch_service,
    record_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training/data", tags=["训练数据上传"])


class ValidateRequest(BaseModel):
    voltage_level: str
    plugins: list[str]
    files: list[dict[str, Any]]


class ImportLocalRequest(BaseModel):
    source_path: str
    voltage_level: str
    plugins: list[str]
    task_type: str | None = None
    dataset_name: str | None = None
    auto_standardize: bool = False


def _parse_plugin_list(raw_plugins: str) -> list[str]:
    try:
        parsed = json.loads(raw_plugins)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in raw_plugins.split(",") if item.strip()]


@router.get("/voltage-levels")
async def get_voltage_levels():
    return {
        "success": True,
        "levels": [
            {"id": "UHV_1000kV_AC", "name": "特高压1000kV交流", "display": "1000kV"},
            {"id": "UHV_800kV_DC", "name": "特高压800kV直流", "display": "800kV"},
            {"id": "EHV_500kV", "name": "超高压500kV", "display": "500kV"},
            {"id": "EHV_330kV", "name": "超高压330kV", "display": "330kV"},
            {"id": "HV_220kV", "name": "高压220kV", "display": "220kV"},
            {"id": "HV_110kV", "name": "高压110kV", "display": "110kV"},
            {"id": "MV_35kV", "name": "中压35kV", "display": "35kV"},
            {"id": "LV_10kV", "name": "低压10kV", "display": "10kV"},
        ],
    }


@router.get("/plugin-types")
async def get_plugin_types():
    return {"success": True, "plugins": batch_service.list_plugin_types()}


@router.get("/supported-formats")
async def get_supported_formats():
    return {"success": True, "formats": SUPPORTED_FORMATS}


@router.post("/validate")
async def validate_upload(request: ValidateRequest):
    result = batch_service.preflight_validate(
        voltage_level=request.voltage_level,
        plugins=request.plugins,
        files=request.files,
    )
    return result


@router.post("/upload")
async def upload_training_data(
    background_tasks: BackgroundTasks,
    voltage_level: str = Form(...),
    plugins: str = Form(...),
    task_type: str | None = Form(None),
    dataset_name: str | None = Form(None),
    auto_standardize: bool = Form(False),
    files: list[UploadFile] = File(...),
):
    plugin_list = _parse_plugin_list(plugins)
    if not plugin_list:
        raise HTTPException(status_code=400, detail="至少需要一个目标插件")
    if not files:
        raise HTTPException(status_code=400, detail="没有收到上传文件")

    try:
        batch = batch_service.create_batch_from_uploaded_files(
            files=files,
            voltage_level=voltage_level,
            plugins=plugin_list,
            task_type=task_type,
            dataset_name=dataset_name,
        )
        background_tasks.add_task(
            batch_service.process_uploaded_batch,
            batch.batch_id,
            auto_standardize=auto_standardize,
        )
        return {
            "success": True,
            "message": "数据已接收，正在后台解包与校验",
            "upload_id": batch.batch_id,
            "batch_id": batch.batch_id,
            "status": batch.status,
            "dataset_name": batch.dataset_name,
            "plugin_id": batch.plugin_id,
            "task_type": batch.task_type,
        }
    except Exception as exc:
        logger.error("上传失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/list")
async def list_upload_records(limit: int = 100):
    records = [batch.to_dict() for batch in batch_service.list_batches(limit=limit)]
    return {"success": True, "count": len(records), "records": records}


@router.get("/batches")
async def list_batches(limit: int = 100):
    records = [batch.to_dict() for batch in batch_service.list_batches(limit=limit)]
    return {"success": True, "count": len(records), "batches": records}


@router.get("/record/{upload_id}")
async def get_upload_record(upload_id: str):
    batch = batch_service.get_batch(upload_id)
    if not batch or batch.deleted:
        raise HTTPException(status_code=404, detail="上传批次不存在")
    return {"success": True, "record": batch.to_dict()}


@router.get("/batches/{batch_id}")
async def get_batch_detail(batch_id: str):
    batch = batch_service.get_batch(batch_id)
    if not batch or batch.deleted:
        raise HTTPException(status_code=404, detail="上传批次不存在")
    return {"success": True, "batch": batch.to_dict()}


@router.post("/standardize/{batch_id}")
async def standardize_batch(batch_id: str):
    try:
        batch = batch_service.standardize_batch(batch_id)
        return {
            "success": True,
            "message": batch.message,
            "batch": batch.to_dict(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("标准化失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/record/{upload_id}")
async def delete_upload_record(upload_id: str, delete_files: bool = False):
    try:
        batch = batch_service.soft_delete_failed_batch(upload_id, purge_files=delete_files)
        return {
            "success": True,
            "message": batch.message,
            "record": batch.to_dict(),
            "files_deleted": delete_files,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/import/local")
async def import_local_data(
    request: ImportLocalRequest,
    background_tasks: BackgroundTasks,
):
    try:
        batch = batch_service.import_local_directory(
            source_path=Path(request.source_path),
            voltage_level=request.voltage_level,
            plugins=request.plugins,
            task_type=request.task_type,
            dataset_name=request.dataset_name,
        )
        background_tasks.add_task(
            batch_service.process_uploaded_batch,
            batch.batch_id,
            auto_standardize=request.auto_standardize,
        )
        return {
            "success": True,
            "message": "本地数据已登记，正在后台解包与校验",
            "upload_id": batch.batch_id,
            "batch_id": batch.batch_id,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("本地导入失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stats")
async def get_upload_stats():
    return {"success": True, "stats": batch_service.get_stats()}


def integrate_data_upload_routes(app) -> bool:
    try:
        app.include_router(router)
        logger.info("✓ training 内部数据上传 API 已集成")
        return True
    except Exception as exc:
        logger.error("集成 training 内部数据上传 API 失败: %s", exc)
        return False

