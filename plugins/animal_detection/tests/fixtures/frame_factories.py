"""
帧数据工厂
==========

生成各种测试用图像帧：空白帧、噪声帧、热成像帧等。
所有帧均为 numpy ndarray，符合 core/ 模块的输入契约。
"""

import numpy as np


def blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """全黑 BGR 帧"""
    return np.zeros((height, width, 3), dtype=np.uint8)


def noise_frame(width: int = 640, height: int = 480, seed: int = 42) -> np.ndarray:
    """随机噪声 BGR 帧（可固定种子确保可复现）"""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (height, width, 3), dtype=np.uint8)


def white_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """全白 BGR 帧"""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def single_pixel_frame() -> np.ndarray:
    """1x1 像素帧（边界测试）"""
    return np.zeros((1, 1, 3), dtype=np.uint8)


def large_frame(width: int = 4096, height: int = 4096) -> np.ndarray:
    """最大尺寸帧（4096x4096，输入上限）"""
    return np.zeros((height, width, 3), dtype=np.uint8)


def grayscale_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """单通道灰度帧（非法输入测试）"""
    return np.zeros((height, width), dtype=np.uint8)


# --- 热成像帧 ---

def uniform_thermal(width: int = 100, height: int = 100, temp_c: float = 25.0) -> np.ndarray:
    """均匀温度热成像帧（float32 摄氏度）"""
    return np.full((height, width), temp_c, dtype=np.float32)


def thermal_with_hotspot(
    width: int = 100,
    height: int = 100,
    bg_temp_c: float = 25.0,
    hotspot_temp_c: float = 30.0,
    hotspot_region: tuple = (10, 10, 30, 30),
) -> np.ndarray:
    """带热点区域的热成像帧

    Args:
        hotspot_region: (y1, x1, y2, x2) 热点像素坐标
    """
    thermal = np.full((height, width), bg_temp_c, dtype=np.float32)
    y1, x1, y2, x2 = hotspot_region
    thermal[y1:y2, x1:x2] = hotspot_temp_c
    return thermal
