"""
SLAM建图与自主导航插件
SLAM Mapping and Autonomous Navigation Plugin

功能:
- 3D LiDAR SLAM实时定位与建图
- 点云语义分割
- 自主路径规划
- 巡检路线优化
"""

from .plugin import (
    # 核心类
    LiDARSLAM,
    PointCloudFeatureExtractor,
    PointCloudRegistration,
    PathPlanner,
    
    # 插件接口
    SLAMMappingPlugin,
    SLAMDetectorEnhanced,
)

__all__ = [
    'LiDARSLAM',
    'PointCloudFeatureExtractor',
    'PointCloudRegistration',
    'PathPlanner',
    'SLAMMappingPlugin',
    'SLAMDetectorEnhanced',
]

__version__ = '1.0.0'
