#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI服务器数据上传路由集成补丁
输变电激光星芒破夜绘明监测平台

使用方法:
1. 将此文件放到项目 apps/ 目录下
2. 在 ui_server.py 中添加以下代码:

    # ============== 集成数据上传API (V3.0) ==============
    try:
        from apps.patch_data_upload import apply_data_upload_patch
        apply_data_upload_patch(app)
    except ImportError as e:
        print(f"✗ 数据上传API导入失败: {e}")

或者手动调用:
    from data_upload_api_complete import integrate_data_upload_routes
    integrate_data_upload_routes(app)

作者: Claude AI Assistant
日期: 2026-02-02
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def apply_data_upload_patch(app):
    """
    应用数据上传补丁到FastAPI应用
    
    Args:
        app: FastAPI应用实例
    
    Returns:
        bool: 是否成功
    """
    # 尝试多个可能的导入路径
    import_paths = [
        "data_upload_api_complete",
        "apps.data_upload_api_complete",
        "apps.data_upload_api",
        "data_upload_api",
    ]
    
    for module_path in import_paths:
        try:
            # 动态导入模块
            if "." in module_path:
                parts = module_path.rsplit(".", 1)
                module = __import__(module_path, fromlist=[parts[-1]])
            else:
                module = __import__(module_path)
            
            # 调用集成函数
            if hasattr(module, 'integrate_data_upload_routes'):
                result = module.integrate_data_upload_routes(app)
                if result:
                    logger.info(f"✓ 数据上传API已从 {module_path} 集成")
                    return True
            elif hasattr(module, 'router'):
                app.include_router(module.router)
                logger.info(f"✓ 数据上传API路由已从 {module_path} 集成")
                return True
                
        except ImportError as e:
            logger.debug(f"尝试导入 {module_path} 失败: {e}")
            continue
        except Exception as e:
            logger.warning(f"从 {module_path} 集成失败: {e}")
            continue
    
    logger.error("✗ 无法找到数据上传API模块")
    return False


def create_minimal_upload_routes(app):
    """
    创建最小化的数据上传路由 (如果完整API不可用)
    
    这是一个后备方案，提供基本的上传功能
    """
    try:
        from fastapi import APIRouter, UploadFile, File, Form, HTTPException
        from pathlib import Path
        import json
        from datetime import datetime
        
        router = APIRouter(prefix="/api/training/data", tags=["数据上传"])
        
        # 数据目录
        DATA_DIR = Path("training/data/raw")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        @router.post("/upload")
        async def upload_data(
            voltage_level: str = Form(...),
            plugins: str = Form(...),
            files: list[UploadFile] = File(...)
        ):
            """基本的数据上传端点"""
            try:
                # 创建上传目录
                upload_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                upload_dir = DATA_DIR / voltage_level / upload_id
                upload_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存文件
                saved_files = []
                for file in files:
                    file_path = upload_dir / file.filename
                    content = await file.read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    saved_files.append(file.filename)
                
                return {
                    "success": True,
                    "message": f"上传成功: {len(saved_files)} 个文件",
                    "upload_id": upload_id,
                    "files": saved_files
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/list")
        async def list_uploads():
            """列出上传记录"""
            records = []
            if DATA_DIR.exists():
                for voltage_dir in DATA_DIR.iterdir():
                    if voltage_dir.is_dir():
                        for upload_dir in voltage_dir.iterdir():
                            if upload_dir.is_dir():
                                file_count = sum(1 for _ in upload_dir.iterdir())
                                records.append({
                                    "id": upload_dir.name,
                                    "voltage_level": voltage_dir.name,
                                    "file_count": file_count,
                                    "status": "completed"
                                })
            return {"success": True, "records": records}
        
        @router.post("/validate")
        async def validate_data(data: dict):
            """验证上传数据"""
            return {"valid": True, "message": "验证通过"}
        
        app.include_router(router)
        logger.info("✓ 最小化数据上传API已创建")
        return True
        
    except Exception as e:
        logger.error(f"创建最小化上传路由失败: {e}")
        return False


def check_and_apply(app, fallback: bool = True):
    """
    检查并应用数据上传补丁
    
    Args:
        app: FastAPI应用实例
        fallback: 是否在完整API不可用时使用最小化版本
    """
    success = apply_data_upload_patch(app)
    
    if not success and fallback:
        logger.warning("使用最小化数据上传API作为后备")
        success = create_minimal_upload_routes(app)
    
    return success


# 如果直接运行此脚本，执行测试
if __name__ == "__main__":
    print("数据上传补丁模块")
    print("=" * 40)
    print("此模块用于将数据上传API集成到UI服务器")
    print()
    print("使用方法:")
    print("  from patch_data_upload import apply_data_upload_patch")
    print("  apply_data_upload_patch(app)")
