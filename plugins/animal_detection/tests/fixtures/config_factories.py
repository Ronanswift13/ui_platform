"""
配置数据工厂
============

生成测试用配置字典，模拟不同运行模式。
所有配置结构对齐 configs/default.yaml。
"""

from typing import Any, Dict


def default_config(**overrides: Any) -> Dict[str, Any]:
    """默认配置（仿真模式开启）"""
    config = {
        "model": {
            "path": "models/animal_yolov8n.onnx",
            "device": "cpu",
            "input_size": [640, 640],
        },
        "inference": {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.45,
            "max_detections": 50,
            "multi_scale": False,
        },
        "thermal": {
            "enabled": False,
            "heat_diff_threshold_c": 2.0,
            "background_update_rate": 0.01,
        },
        "tracking": {
            "enabled": True,
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3,
        },
        "deterrent": {
            "enabled": False,
            "stay_time_threshold_s": 30,
            "cooldown_s": 60,
            "audio_enabled": True,
            "light_enabled": True,
        },
        "simulation": {
            "enabled": True,
        },
    }
    config.update(overrides)
    return config


def simulation_config() -> Dict[str, Any]:
    """纯仿真模式配置"""
    return default_config(simulation={"enabled": True})


def production_config(model_path: str = "models/animal_yolov8n.onnx") -> Dict[str, Any]:
    """生产模式配置（仿真关闭）"""
    return default_config(
        simulation={"enabled": False},
        model={"path": model_path, "device": "cpu", "input_size": [640, 640]},
    )


def thermal_enabled_config() -> Dict[str, Any]:
    """热成像启用配置"""
    return default_config(
        thermal={"enabled": True, "heat_diff_threshold_c": 2.0, "background_update_rate": 0.01},
    )


def deterrent_enabled_config() -> Dict[str, Any]:
    """驱离功能启用配置"""
    return default_config(
        deterrent={
            "enabled": True,
            "stay_time_threshold_s": 30,
            "cooldown_s": 60,
            "audio_enabled": True,
            "light_enabled": True,
        },
    )
