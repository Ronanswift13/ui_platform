"""
SLAM三维建图插件
SLAM and 3D Mapping Plugin

功能：
- LiDAR/雷达点云处理
- 三维地图构建与更新（占据栅格地图）
- ICP点云配准与位姿估计
- 设备定位与追踪
- A*路径规划
- 地面沉降监测
- 障碍物检测
"""

from .plugin import (
    SLAMMappingPlugin,
    Point3D,
    Pose,
    MapCell,
    SubsidencePoint,
    PointCloudProcessor,
    OccupancyGridMap,
    ICPMatcher,
    PathPlanner,
    SubsidenceMonitor
)

__all__ = [
    'SLAMMappingPlugin',
    'Point3D',
    'Pose',
    'MapCell',
    'SubsidencePoint',
    'PointCloudProcessor',
    'OccupancyGridMap',
    'ICPMatcher',
    'PathPlanner',
    'SubsidenceMonitor'
]

__version__ = '1.0.0'
__author__ = 'Substation AI Team'
