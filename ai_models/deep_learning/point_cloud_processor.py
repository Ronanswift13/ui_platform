"""
点云处理与3D感知模块
输变电激光星芒破夜绘明监测平台

基于UPDATE.md中期迭代计划:
- 3D深度摄像头/毫米波雷达数据处理
- 点云语义分割
- 点云与图像融合
- 3D目标检测与跟踪

支持:
- 深度图像转点云
- 毫米波雷达点云处理
- 多传感器时空对齐
- PointNet++/PointTransformer推理

版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    o3d = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


# =============================================================================
# 枚举定义
# =============================================================================

class PointCloudFormat(Enum):
    """点云格式"""
    XYZ = "xyz"             # 仅坐标
    XYZRGB = "xyzrgb"       # 坐标+颜色
    XYZI = "xyzi"           # 坐标+强度
    XYZRGBI = "xyzrgbi"     # 坐标+颜色+强度


class SensorType(Enum):
    """传感器类型"""
    DEPTH_CAMERA = "depth_camera"       # 深度摄像头 (RealSense/Kinect/ToF)
    STEREO_CAMERA = "stereo_camera"     # 双目立体相机
    LIDAR_2D = "lidar_2d"               # 2D激光雷达
    LIDAR_3D = "lidar_3d"               # 3D激光雷达
    RADAR_MMWAVE = "radar_mmwave"       # 毫米波雷达
    STRUCTURED_LIGHT = "structured_light"  # 结构光


class SemanticLabel(Enum):
    """语义标签"""
    GROUND = 0              # 地面
    PERSON = 1              # 人员
    EQUIPMENT = 2           # 设备
    CABINET = 3             # 机柜
    WALL = 4                # 墙壁
    FENCE = 5               # 围栏
    CABLE = 6               # 电缆
    OTHER = 255             # 其他


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class CameraIntrinsics:
    """相机内参"""
    fx: float               # X方向焦距
    fy: float               # Y方向焦距
    cx: float               # 主点X
    cy: float               # 主点Y
    width: int              # 图像宽度
    height: int             # 图像高度
    depth_scale: float = 1000.0  # 深度缩放因子 (mm -> m)

    def to_matrix(self) -> np.ndarray:
        """转换为3x3内参矩阵"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass
class ExtrinsicTransform:
    """外参变换 (传感器到世界坐标系)"""
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3旋转矩阵
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 平移向量

    def to_matrix(self) -> np.ndarray:
        """转换为4x4变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> 'ExtrinsicTransform':
        """从4x4矩阵创建"""
        return cls(
            rotation=T[:3, :3].copy(),
            translation=T[:3, 3].copy()
        )


@dataclass
class PointCloud:
    """点云数据结构"""
    points: np.ndarray                          # (N, 3) XYZ坐标
    colors: Optional[np.ndarray] = None         # (N, 3) RGB颜色 [0-1]
    intensities: Optional[np.ndarray] = None    # (N,) 强度
    normals: Optional[np.ndarray] = None        # (N, 3) 法向量
    labels: Optional[np.ndarray] = None         # (N,) 语义标签
    timestamps: Optional[np.ndarray] = None     # (N,) 时间戳
    confidence: Optional[np.ndarray] = None     # (N,) 置信度
    sensor_type: SensorType = SensorType.DEPTH_CAMERA
    frame_id: str = "world"
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """点数量"""
        return len(self.points)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取边界框"""
        if self.size == 0:
            return np.zeros(3), np.zeros(3)
        return np.min(self.points, axis=0), np.max(self.points, axis=0)

    def transform(self, T: np.ndarray) -> 'PointCloud':
        """应用变换矩阵"""
        points_h = np.hstack([self.points, np.ones((self.size, 1))])
        transformed = (T @ points_h.T).T[:, :3]

        # 法向量只需旋转
        normals = None
        if self.normals is not None:
            R = T[:3, :3]
            normals = (R @ self.normals.T).T

        return PointCloud(
            points=transformed,
            colors=self.colors.copy() if self.colors is not None else None,
            intensities=self.intensities.copy() if self.intensities is not None else None,
            normals=normals,
            labels=self.labels.copy() if self.labels is not None else None,
            timestamps=self.timestamps.copy() if self.timestamps is not None else None,
            confidence=self.confidence.copy() if self.confidence is not None else None,
            sensor_type=self.sensor_type,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            metadata=self.metadata.copy(),
        )

    def downsample(self, voxel_size: float) -> 'PointCloud':
        """体素下采样"""
        if self.size == 0:
            return self

        # 计算体素索引
        voxel_indices = np.floor(self.points / voxel_size).astype(np.int32)

        # 使用字典去重
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)

        # 取每个体素的中心点
        new_points = []
        new_colors = [] if self.colors is not None else None
        new_intensities = [] if self.intensities is not None else None
        new_labels = [] if self.labels is not None else None

        for indices in voxel_dict.values():
            new_points.append(np.mean(self.points[indices], axis=0))
            if self.colors is not None:
                new_colors.append(np.mean(self.colors[indices], axis=0))
            if self.intensities is not None:
                new_intensities.append(np.mean(self.intensities[indices]))
            if self.labels is not None:
                # 取众数
                labels = self.labels[indices]
                unique, counts = np.unique(labels, return_counts=True)
                new_labels.append(unique[np.argmax(counts)])

        return PointCloud(
            points=np.array(new_points),
            colors=np.array(new_colors) if new_colors else None,
            intensities=np.array(new_intensities) if new_intensities else None,
            labels=np.array(new_labels) if new_labels else None,
            sensor_type=self.sensor_type,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            metadata=self.metadata.copy(),
        )

    def filter_by_range(
        self,
        min_range: float = 0.0,
        max_range: float = float('inf'),
        axis: Optional[int] = None
    ) -> 'PointCloud':
        """按距离过滤"""
        if axis is not None:
            distances = np.abs(self.points[:, axis])
        else:
            distances = np.linalg.norm(self.points, axis=1)

        mask = (distances >= min_range) & (distances <= max_range)
        return self._filter_by_mask(mask)

    def filter_by_bbox(
        self,
        min_point: Tuple[float, float, float],
        max_point: Tuple[float, float, float]
    ) -> 'PointCloud':
        """按边界框过滤"""
        min_p = np.array(min_point)
        max_p = np.array(max_point)
        mask = np.all((self.points >= min_p) & (self.points <= max_p), axis=1)
        return self._filter_by_mask(mask)

    def _filter_by_mask(self, mask: np.ndarray) -> 'PointCloud':
        """按掩码过滤"""
        return PointCloud(
            points=self.points[mask],
            colors=self.colors[mask] if self.colors is not None else None,
            intensities=self.intensities[mask] if self.intensities is not None else None,
            normals=self.normals[mask] if self.normals is not None else None,
            labels=self.labels[mask] if self.labels is not None else None,
            timestamps=self.timestamps[mask] if self.timestamps is not None else None,
            confidence=self.confidence[mask] if self.confidence is not None else None,
            sensor_type=self.sensor_type,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            metadata=self.metadata.copy(),
        )


@dataclass
class Detection3D:
    """3D检测结果"""
    class_id: int
    class_name: str
    confidence: float
    center: np.ndarray          # (3,) 中心点
    dimensions: np.ndarray      # (3,) 长宽高
    rotation: float = 0.0       # Z轴旋转角度
    velocity: Optional[np.ndarray] = None  # (3,) 速度
    track_id: int = -1
    points_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_corners(self) -> np.ndarray:
        """获取8个角点"""
        l, w, h = self.dimensions
        corners = np.array([
            [-l/2, -w/2, -h/2],
            [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2],
            [l/2, -w/2, h/2],
            [l/2, w/2, h/2],
            [-l/2, w/2, h/2],
        ])

        # 旋转
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners = (R @ corners.T).T

        # 平移
        return corners + self.center


@dataclass
class SegmentationResult:
    """语义分割结果"""
    point_cloud: PointCloud
    labels: np.ndarray          # (N,) 语义标签
    probabilities: Optional[np.ndarray] = None  # (N, num_classes)
    class_names: Dict[int, str] = field(default_factory=dict)
    inference_time_ms: float = 0.0


# =============================================================================
# 点云处理器配置
# =============================================================================

@dataclass
class PointCloudProcessorConfig:
    """点云处理器配置"""
    # 深度图转点云
    depth_scale: float = 1000.0             # 深度缩放
    depth_min: float = 0.1                  # 最小深度 (m)
    depth_max: float = 10.0                 # 最大深度 (m)

    # 下采样
    voxel_size: float = 0.05                # 体素大小 (m)

    # 滤波
    statistical_filter_k: int = 20          # 统计滤波邻域点数
    statistical_filter_std: float = 2.0     # 标准差倍数
    radius_filter_radius: float = 0.1       # 半径滤波半径
    radius_filter_min_neighbors: int = 5    # 最小邻居数

    # 地面分割
    ground_max_distance: float = 0.1        # 地面最大距离
    ground_normal_threshold: float = 0.9    # 地面法向量阈值

    # 聚类
    cluster_eps: float = 0.3                # DBSCAN eps
    cluster_min_points: int = 10            # 最小点数

    # 配准
    icp_max_iterations: int = 50            # ICP最大迭代
    icp_max_correspondence_distance: float = 0.1  # 最大对应距离

    # 模型
    model_path: Optional[str] = None        # 语义分割模型路径
    device: str = "cpu"


# =============================================================================
# 深度图到点云转换器
# =============================================================================

class DepthToPointCloud:
    """
    深度图到点云转换器

    支持:
    - RealSense深度图
    - Kinect深度图
    - 结构光深度图
    - 双目视差图
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics
        self._u_map = None
        self._v_map = None
        self._init_projection_maps()

    def _init_projection_maps(self):
        """初始化投影映射表"""
        u = np.arange(self.intrinsics.width)
        v = np.arange(self.intrinsics.height)
        self._u_map, self._v_map = np.meshgrid(u, v)

    def convert(
        self,
        depth_image: np.ndarray,
        color_image: Optional[np.ndarray] = None,
        extrinsic: Optional[ExtrinsicTransform] = None,
        depth_scale: Optional[float] = None,
    ) -> PointCloud:
        """
        将深度图转换为点云

        Args:
            depth_image: 深度图像 (H, W) uint16/float32
            color_image: 彩色图像 (H, W, 3) uint8
            extrinsic: 外参变换
            depth_scale: 深度缩放因子

        Returns:
            PointCloud: 转换后的点云
        """
        if depth_scale is None:
            depth_scale = self.intrinsics.depth_scale

        # 深度转换为米
        if depth_image.dtype == np.uint16:
            depth = depth_image.astype(np.float32) / depth_scale
        else:
            depth = depth_image.astype(np.float32)

        # 有效深度掩码
        valid_mask = depth > 0

        # 计算3D坐标
        z = depth
        x = (self._u_map - self.intrinsics.cx) * z / self.intrinsics.fx
        y = (self._v_map - self.intrinsics.cy) * z / self.intrinsics.fy

        # 提取有效点
        points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)

        # 提取颜色
        colors = None
        if color_image is not None:
            if len(color_image.shape) == 3:
                colors = color_image[valid_mask].astype(np.float32) / 255.0
            else:
                # 灰度图
                gray = color_image[valid_mask].astype(np.float32) / 255.0
                colors = np.stack([gray, gray, gray], axis=1)

        # 创建点云
        pc = PointCloud(
            points=points,
            colors=colors,
            sensor_type=SensorType.DEPTH_CAMERA,
            frame_id="camera",
            timestamp=time.time(),
        )

        # 应用外参变换
        if extrinsic is not None:
            T = extrinsic.to_matrix()
            pc = pc.transform(T)
            pc.frame_id = "world"

        return pc


# =============================================================================
# 点云处理器
# =============================================================================

class PointCloudProcessor:
    """
    点云处理器

    功能:
    - 滤波降噪
    - 下采样
    - 地面分割
    - 聚类
    - 语义分割 (需要模型)
    """

    def __init__(self, config: PointCloudProcessorConfig):
        self.config = config
        self._segmentation_model = None

        # 加载语义分割模型
        if config.model_path:
            self._load_segmentation_model()

    def _load_segmentation_model(self):
        """加载语义分割模型"""
        try:
            import onnxruntime as ort

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if self.config.device == "cuda" else ['CPUExecutionProvider']

            self._segmentation_model = ort.InferenceSession(
                self.config.model_path,
                providers=providers
            )
            logger.info(f"[PointCloudProcessor] 语义分割模型加载成功")
        except Exception as e:
            logger.warning(f"[PointCloudProcessor] 语义分割模型加载失败: {e}")

    def filter_statistical(self, pc: PointCloud) -> PointCloud:
        """统计滤波去噪"""
        if pc.size < self.config.statistical_filter_k:
            return pc

        if O3D_AVAILABLE:
            # 使用Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc.points)

            _, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=self.config.statistical_filter_k,
                std_ratio=self.config.statistical_filter_std
            )

            mask = np.zeros(pc.size, dtype=bool)
            mask[inlier_indices] = True
            return pc._filter_by_mask(mask)
        else:
            # 简单实现: 基于邻居距离
            return self._filter_statistical_simple(pc)

    def _filter_statistical_simple(self, pc: PointCloud) -> PointCloud:
        """简单统计滤波实现"""
        from scipy.spatial import KDTree

        tree = KDTree(pc.points)
        k = min(self.config.statistical_filter_k, pc.size - 1)

        distances, _ = tree.query(pc.points, k=k + 1)
        mean_distances = np.mean(distances[:, 1:], axis=1)

        mean_dist = np.mean(mean_distances)
        std_dist = np.std(mean_distances)
        threshold = mean_dist + self.config.statistical_filter_std * std_dist

        mask = mean_distances < threshold
        return pc._filter_by_mask(mask)

    def filter_radius(self, pc: PointCloud) -> PointCloud:
        """半径滤波"""
        if pc.size < self.config.radius_filter_min_neighbors:
            return pc

        if O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc.points)

            _, inlier_indices = pcd.remove_radius_outlier(
                nb_points=self.config.radius_filter_min_neighbors,
                radius=self.config.radius_filter_radius
            )

            mask = np.zeros(pc.size, dtype=bool)
            mask[inlier_indices] = True
            return pc._filter_by_mask(mask)
        else:
            return pc  # 无Open3D时跳过

    def segment_ground(self, pc: PointCloud) -> Tuple[PointCloud, PointCloud]:
        """
        地面分割

        Returns:
            (ground_points, non_ground_points)
        """
        if pc.size == 0:
            return pc, pc

        if O3D_AVAILABLE:
            return self._segment_ground_ransac(pc)
        else:
            return self._segment_ground_simple(pc)

    def _segment_ground_ransac(self, pc: PointCloud) -> Tuple[PointCloud, PointCloud]:
        """RANSAC地面分割"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points)

        # RANSAC平面拟合
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.config.ground_max_distance,
            ransac_n=3,
            num_iterations=100
        )

        # 检查是否为水平面
        normal = np.array(plane_model[:3])
        if abs(normal[2]) < self.config.ground_normal_threshold:
            # 不是地面，返回空地面
            empty = PointCloud(points=np.zeros((0, 3)), sensor_type=pc.sensor_type)
            return empty, pc

        ground_mask = np.zeros(pc.size, dtype=bool)
        ground_mask[inliers] = True

        return pc._filter_by_mask(ground_mask), pc._filter_by_mask(~ground_mask)

    def _segment_ground_simple(self, pc: PointCloud) -> Tuple[PointCloud, PointCloud]:
        """简单地面分割 (基于高度)"""
        # 假设Z轴向上，取最低10%点的平均高度作为地面
        z_values = pc.points[:, 2]
        z_threshold = np.percentile(z_values, 10) + self.config.ground_max_distance

        ground_mask = z_values < z_threshold
        return pc._filter_by_mask(ground_mask), pc._filter_by_mask(~ground_mask)

    def cluster_dbscan(self, pc: PointCloud) -> List[PointCloud]:
        """DBSCAN聚类"""
        if pc.size < self.config.cluster_min_points:
            return []

        if O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc.points)

            labels = np.array(pcd.cluster_dbscan(
                eps=self.config.cluster_eps,
                min_points=self.config.cluster_min_points
            ))
        else:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(
                eps=self.config.cluster_eps,
                min_samples=self.config.cluster_min_points
            ).fit(pc.points)
            labels = clustering.labels_

        # 提取各个聚类
        clusters = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # 移除噪声标签

        for label in unique_labels:
            mask = labels == label
            cluster_pc = pc._filter_by_mask(mask)
            cluster_pc.metadata["cluster_id"] = int(label)
            clusters.append(cluster_pc)

        return clusters

    def compute_normals(self, pc: PointCloud, radius: float = 0.1) -> PointCloud:
        """计算法向量"""
        if pc.size == 0:
            return pc

        if O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc.points)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=30
                )
            )
            normals = np.asarray(pcd.normals)
        else:
            # 简化实现
            normals = np.zeros_like(pc.points)
            normals[:, 2] = 1.0  # 默认向上

        return PointCloud(
            points=pc.points.copy(),
            colors=pc.colors.copy() if pc.colors is not None else None,
            intensities=pc.intensities.copy() if pc.intensities is not None else None,
            normals=normals,
            labels=pc.labels.copy() if pc.labels is not None else None,
            sensor_type=pc.sensor_type,
            frame_id=pc.frame_id,
            timestamp=pc.timestamp,
            metadata=pc.metadata.copy(),
        )

    def semantic_segmentation(self, pc: PointCloud) -> SegmentationResult:
        """语义分割"""
        if self._segmentation_model is None:
            # 返回模拟结果
            labels = np.zeros(pc.size, dtype=np.int32)
            return SegmentationResult(
                point_cloud=pc,
                labels=labels,
                class_names={0: "unknown"},
            )

        start_time = time.time()

        # 预处理
        points = pc.points.astype(np.float32)

        # 正规化
        centroid = np.mean(points, axis=0)
        points_normalized = points - centroid
        max_dist = np.max(np.linalg.norm(points_normalized, axis=1))
        if max_dist > 0:
            points_normalized /= max_dist

        # 运行模型
        input_name = self._segmentation_model.get_inputs()[0].name
        outputs = self._segmentation_model.run(
            None,
            {input_name: points_normalized[np.newaxis, ...]}
        )

        labels = outputs[0].squeeze().astype(np.int32)
        probs = outputs[1].squeeze() if len(outputs) > 1 else None

        inference_time = (time.time() - start_time) * 1000

        return SegmentationResult(
            point_cloud=pc,
            labels=labels,
            probabilities=probs,
            class_names={
                0: "ground",
                1: "person",
                2: "equipment",
                3: "cabinet",
                4: "wall",
                5: "fence",
                6: "cable",
                255: "other",
            },
            inference_time_ms=inference_time,
        )


# =============================================================================
# 多传感器融合
# =============================================================================

class MultiSensorFusion:
    """
    多传感器点云融合

    支持:
    - 深度相机 + RGB相机
    - 深度相机 + 2D激光雷达
    - 多深度相机融合
    - 毫米波雷达 + 视觉
    """

    def __init__(self):
        self._sensor_transforms: Dict[str, ExtrinsicTransform] = {}
        self._intrinsics: Dict[str, CameraIntrinsics] = {}

    def register_sensor(
        self,
        sensor_id: str,
        extrinsic: ExtrinsicTransform,
        intrinsics: Optional[CameraIntrinsics] = None
    ):
        """注册传感器"""
        self._sensor_transforms[sensor_id] = extrinsic
        if intrinsics:
            self._intrinsics[sensor_id] = intrinsics

    def fuse_point_clouds(
        self,
        point_clouds: List[Tuple[str, PointCloud]],
        voxel_size: float = 0.05
    ) -> PointCloud:
        """
        融合多个点云

        Args:
            point_clouds: [(sensor_id, point_cloud), ...]
            voxel_size: 下采样体素大小

        Returns:
            融合后的点云
        """
        all_points = []
        all_colors = []
        all_intensities = []

        for sensor_id, pc in point_clouds:
            # 变换到世界坐标系
            if sensor_id in self._sensor_transforms:
                T = self._sensor_transforms[sensor_id].to_matrix()
                pc = pc.transform(T)

            all_points.append(pc.points)
            if pc.colors is not None:
                all_colors.append(pc.colors)
            if pc.intensities is not None:
                all_intensities.append(pc.intensities)

        # 合并
        fused = PointCloud(
            points=np.vstack(all_points) if all_points else np.zeros((0, 3)),
            colors=np.vstack(all_colors) if all_colors else None,
            intensities=np.hstack(all_intensities) if all_intensities else None,
            sensor_type=SensorType.DEPTH_CAMERA,
            frame_id="world",
            timestamp=time.time(),
        )

        # 下采样去除重复点
        if voxel_size > 0:
            fused = fused.downsample(voxel_size)

        return fused

    def project_to_image(
        self,
        pc: PointCloud,
        sensor_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将点云投影到图像平面

        Returns:
            (uv_coords, valid_mask)
        """
        if sensor_id not in self._intrinsics:
            raise ValueError(f"未注册传感器内参: {sensor_id}")

        intrinsics = self._intrinsics[sensor_id]

        # 变换到相机坐标系
        if sensor_id in self._sensor_transforms:
            T = self._sensor_transforms[sensor_id].to_matrix()
            T_inv = np.linalg.inv(T)
            points_h = np.hstack([pc.points, np.ones((pc.size, 1))])
            points_cam = (T_inv @ points_h.T).T[:, :3]
        else:
            points_cam = pc.points

        # 投影
        z = points_cam[:, 2]
        valid_mask = z > 0.001

        u = (points_cam[:, 0] * intrinsics.fx / z + intrinsics.cx)
        v = (points_cam[:, 1] * intrinsics.fy / z + intrinsics.cy)

        # 边界检查
        valid_mask &= (u >= 0) & (u < intrinsics.width)
        valid_mask &= (v >= 0) & (v < intrinsics.height)

        uv = np.stack([u, v], axis=1)
        return uv, valid_mask


# =============================================================================
# 3D人员检测器
# =============================================================================

class Person3DDetector:
    """
    3D人员检测器

    基于点云的人员检测和跟踪
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 人员尺寸约束
        self.person_height_min = self.config.get("person_height_min", 1.4)
        self.person_height_max = self.config.get("person_height_max", 2.2)
        self.person_width_max = self.config.get("person_width_max", 0.8)
        self.person_depth_max = self.config.get("person_depth_max", 0.6)

        # 点云处理器
        self.processor = PointCloudProcessor(PointCloudProcessorConfig(
            voxel_size=0.03,
            cluster_eps=0.2,
            cluster_min_points=50,
        ))

    def detect(self, pc: PointCloud) -> List[Detection3D]:
        """
        检测人员

        Args:
            pc: 输入点云

        Returns:
            人员检测结果列表
        """
        detections = []

        # 1. 地面分割
        _, non_ground = self.processor.segment_ground(pc)

        if non_ground.size < 50:
            return detections

        # 2. 聚类
        clusters = self.processor.cluster_dbscan(non_ground)

        # 3. 分析每个聚类
        for i, cluster in enumerate(clusters):
            min_pt, max_pt = cluster.get_bounds()
            dimensions = max_pt - min_pt
            center = (min_pt + max_pt) / 2

            height = dimensions[2]
            width = max(dimensions[0], dimensions[1])
            depth = min(dimensions[0], dimensions[1])

            # 检查是否符合人员尺寸
            if (self.person_height_min <= height <= self.person_height_max and
                width <= self.person_width_max and
                depth <= self.person_depth_max):

                confidence = self._compute_person_confidence(cluster, dimensions)

                detections.append(Detection3D(
                    class_id=1,
                    class_name="person",
                    confidence=confidence,
                    center=center,
                    dimensions=dimensions,
                    points_count=cluster.size,
                    metadata={"cluster_id": i},
                ))

        return detections

    def _compute_person_confidence(
        self,
        cluster: PointCloud,
        dimensions: np.ndarray
    ) -> float:
        """计算人员置信度"""
        # 基于尺寸的置信度
        height = dimensions[2]
        typical_height = 1.7
        height_score = 1.0 - abs(height - typical_height) / typical_height

        # 基于点数的置信度
        point_density = cluster.size / np.prod(dimensions)
        density_score = min(1.0, point_density / 1000)

        # 基于形状的置信度 (人应该是细长的)
        aspect_ratio = height / max(dimensions[0], dimensions[1])
        shape_score = min(1.0, aspect_ratio / 3.0)

        confidence = 0.4 * height_score + 0.3 * density_score + 0.3 * shape_score
        return max(0.0, min(1.0, confidence))


# =============================================================================
# 便捷函数
# =============================================================================

def depth_to_point_cloud(
    depth_image: np.ndarray,
    intrinsics: CameraIntrinsics,
    color_image: Optional[np.ndarray] = None,
) -> PointCloud:
    """深度图转点云便捷函数"""
    converter = DepthToPointCloud(intrinsics)
    return converter.convert(depth_image, color_image)


def create_processor(config: Optional[Dict] = None) -> PointCloudProcessor:
    """创建点云处理器便捷函数"""
    proc_config = PointCloudProcessorConfig()
    if config:
        for key, value in config.items():
            if hasattr(proc_config, key):
                setattr(proc_config, key, value)
    return PointCloudProcessor(proc_config)
