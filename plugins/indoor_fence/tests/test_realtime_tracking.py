"""
Tests for Realtime Multi-Person Tracking and Pipeline
======================================================

测试覆盖：
- RealtimeMultiPersonTracker 轨迹管理
- 距离计算（人与人、人与围栏）
- 轨迹预测
- 告警生成
- SimulationRenderer 渲染方法
- RealtimeMultiPersonPipeline 集成
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import math
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from plugins.indoor_fence.core.tracking.realtime_tracker import (
    DistanceInfo,
    DistanceLevel,
    PersonTrack,
    RealtimeMultiPersonTracker,
    RealtimeTrackingResult,
    TrackStatus,
    TrajectoryPoint,
)


class TestRealtimeMultiPersonTracker:
    """RealtimeMultiPersonTracker 单元测试"""

    def test_init_default_params(self):
        """测试默认参数初始化"""
        tracker = RealtimeMultiPersonTracker()
        assert tracker.fence_line_y == 3.0
        assert tracker._max_age == 30
        assert tracker._min_hits == 3
        assert len(tracker._tracks) == 0

    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        tracker = RealtimeMultiPersonTracker(
            fence_line_y=5.0,
            max_age=50,
            min_hits=5,
            prediction_horizon=20,
        )
        assert tracker.fence_line_y == 5.0
        assert tracker._max_age == 50
        assert tracker._min_hits == 5
        assert tracker._prediction_horizon == 20

    def test_update_creates_track(self):
        """测试更新创建新轨迹"""
        tracker = RealtimeMultiPersonTracker()
        detections = [
            {"position": (1.0, 2.0, 0.0), "velocity": (0.1, 0.0, 0.0), "confidence": 0.9}
        ]

        result = tracker.update(detections)

        assert len(tracker._tracks) == 1
        assert result.stats["total_tracks"] == 1

    def test_update_multiple_detections(self):
        """测试多检测更新"""
        tracker = RealtimeMultiPersonTracker()
        detections = [
            {"position": (1.0, 1.0, 0.0), "confidence": 0.9},
            {"position": (5.0, 2.0, 0.0), "confidence": 0.8},
            {"position": (8.0, 1.5, 0.0), "confidence": 0.85},
        ]

        result = tracker.update(detections)

        assert len(tracker._tracks) == 3

    def test_track_confirmation(self):
        """测试轨迹确认（需要 min_hits 次更新）"""
        tracker = RealtimeMultiPersonTracker(min_hits=3)
        detection = {"position": (2.0, 2.0, 0.0), "confidence": 0.9}

        # 第一次更新 - tentative
        tracker.update([detection])
        track = list(tracker._tracks.values())[0]
        assert track.status == TrackStatus.TENTATIVE

        # 第二次更新 - still tentative
        tracker.update([{"position": (2.1, 2.0, 0.0), "confidence": 0.9}])
        assert track.status == TrackStatus.TENTATIVE

        # 第三次更新 - confirmed
        tracker.update([{"position": (2.2, 2.0, 0.0), "confidence": 0.9}])
        assert track.status == TrackStatus.CONFIRMED

    def test_track_association(self):
        """测试轨迹关联（最近邻匹配）"""
        tracker = RealtimeMultiPersonTracker()

        # 创建初始轨迹
        tracker.update([
            {"position": (1.0, 1.0, 0.0)},
            {"position": (5.0, 1.0, 0.0)},
        ])

        # 更新位置（应该关联到最近的轨迹）
        tracker.update([
            {"position": (1.2, 1.1, 0.0)},  # 接近第一个
            {"position": (5.1, 1.0, 0.0)},  # 接近第二个
        ])

        # 轨迹数量不变
        assert len(tracker._tracks) == 2

    def test_track_pruning(self):
        """测试轨迹删除（超时）"""
        tracker = RealtimeMultiPersonTracker(max_age=2)

        # 创建轨迹
        tracker.update([{"position": (1.0, 1.0, 0.0)}])
        assert len(tracker._tracks) == 1

        # 多次更新无检测
        for _ in range(5):
            tracker.update([])

        # 轨迹应被删除
        assert len(tracker._tracks) == 0

    def test_fence_distance_calculation(self):
        """测试围栏距离计算"""
        tracker = RealtimeMultiPersonTracker(fence_line_y=3.0, min_hits=1)

        # 人在 y=2.0，距离围栏 1.0m
        tracker.update([{"position": (1.0, 2.0, 0.0)}])
        track = list(tracker._tracks.values())[0]

        assert abs(track.distance_to_fence - 1.0) < 0.01
        assert track.distance_level == DistanceLevel.SAFE

    def test_fence_distance_warning(self):
        """测试围栏距离警告"""
        tracker = RealtimeMultiPersonTracker(fence_line_y=3.0, min_hits=1)

        # 人在 y=2.7，距离围栏 0.3m（警告区）
        tracker.update([{"position": (1.0, 2.7, 0.0)}])
        track = list(tracker._tracks.values())[0]

        assert abs(track.distance_to_fence - 0.3) < 0.01
        assert track.distance_level == DistanceLevel.WARNING

    def test_fence_distance_danger(self):
        """测试围栏距离危险"""
        tracker = RealtimeMultiPersonTracker(fence_line_y=3.0, min_hits=1)

        # 人在 y=2.9，距离围栏 0.1m（危险区）
        tracker.update([{"position": (1.0, 2.9, 0.0)}])
        track = list(tracker._tracks.values())[0]

        assert abs(track.distance_to_fence - 0.1) < 0.01
        assert track.distance_level == DistanceLevel.DANGER

    def test_fence_distance_critical(self):
        """测试围栏距离临界（越线）"""
        tracker = RealtimeMultiPersonTracker(fence_line_y=3.0, min_hits=1)

        # 人在 y=3.1，已越线
        tracker.update([{"position": (1.0, 3.1, 0.0)}])
        track = list(tracker._tracks.values())[0]

        assert track.distance_to_fence < 0
        assert track.distance_level == DistanceLevel.CRITICAL

    def test_person_distance_calculation(self):
        """测试人与人距离计算"""
        tracker = RealtimeMultiPersonTracker(min_hits=1)

        # 两人相距 2.0m
        for _ in range(3):  # 确认轨迹
            tracker.update([
                {"position": (1.0, 1.0, 0.0)},
                {"position": (3.0, 1.0, 0.0)},
            ])

        result = tracker.update([
            {"position": (1.0, 1.0, 0.0)},
            {"position": (3.0, 1.0, 0.0)},
        ])

        # 查找人与人距离
        person_distances = [d for d in result.distances if d.to_track_id is not None]
        assert len(person_distances) == 1
        assert abs(person_distances[0].distance - 2.0) < 0.01
        assert person_distances[0].level == DistanceLevel.SAFE

    def test_person_distance_warning(self):
        """测试人与人距离警告"""
        tracker = RealtimeMultiPersonTracker(min_hits=1)

        # 两人相距 1.2m（警告区）
        for _ in range(3):
            tracker.update([
                {"position": (1.0, 1.0, 0.0)},
                {"position": (2.2, 1.0, 0.0)},
            ])

        result = tracker.update([
            {"position": (1.0, 1.0, 0.0)},
            {"position": (2.2, 1.0, 0.0)},
        ])

        person_distances = [d for d in result.distances if d.to_track_id is not None]
        assert person_distances[0].level == DistanceLevel.WARNING

    def test_trajectory_prediction(self):
        """测试轨迹预测"""
        tracker = RealtimeMultiPersonTracker(min_hits=1, prediction_horizon=5)

        # 创建有速度的轨迹
        for i in range(3):
            tracker.update([
                {"position": (1.0 + i * 0.1, 1.0, 0.0), "velocity": (1.0, 0.0, 0.0)}
            ])

        track = list(tracker._tracks.values())[0]

        # 应该有预测位置
        assert len(track.predicted_positions) == 5

        # 预测位置应该沿速度方向
        if track.predicted_positions:
            pred = track.predicted_positions[0]
            assert pred[0] > track.position[0]  # x 增加

    def test_alert_generation_fence_breach(self):
        """测试围栏越线告警生成"""
        tracker = RealtimeMultiPersonTracker(fence_line_y=3.0, min_hits=1)

        # 确认轨迹
        for _ in range(3):
            tracker.update([{"position": (1.0, 2.9, 0.0)}])

        # 越线
        result = tracker.update([{"position": (1.0, 3.1, 0.0)}])

        # 应该有告警
        fence_alerts = [a for a in result.alerts if a["type"] == "fence_breach"]
        assert len(fence_alerts) >= 1
        assert fence_alerts[0]["level"] == "critical"

    def test_alert_generation_proximity(self):
        """测试人员接近告警生成"""
        tracker = RealtimeMultiPersonTracker(min_hits=1)

        # 确认两个轨迹
        for _ in range(3):
            tracker.update([
                {"position": (1.0, 1.0, 0.0)},
                {"position": (1.5, 1.0, 0.0)},  # 相距 0.5m（危险）
            ])

        result = tracker.update([
            {"position": (1.0, 1.0, 0.0)},
            {"position": (1.5, 1.0, 0.0)},
        ])

        proximity_alerts = [a for a in result.alerts if a["type"] == "proximity"]
        assert len(proximity_alerts) >= 1

    def test_get_confirmed_tracks(self):
        """测试获取已确认轨迹"""
        tracker = RealtimeMultiPersonTracker(min_hits=2)

        # 创建轨迹
        tracker.update([{"position": (1.0, 1.0, 0.0)}])
        assert len(tracker.get_confirmed_tracks()) == 0

        # 确认轨迹
        tracker.update([{"position": (1.1, 1.0, 0.0)}])
        assert len(tracker.get_confirmed_tracks()) == 1

    def test_get_trajectory_data(self):
        """测试获取轨迹历史数据"""
        tracker = RealtimeMultiPersonTracker()

        # 创建轨迹
        tracker.update([{"position": (1.0, 1.0, 0.0)}])
        tracker.update([{"position": (1.1, 1.0, 0.0)}])
        tracker.update([{"position": (1.2, 1.0, 0.0)}])

        track_id = list(tracker._tracks.keys())[0]
        data = tracker.get_trajectory_data(track_id)

        assert len(data) == 3
        assert "x" in data[0]
        assert "y" in data[0]
        assert "timestamp" in data[0]

    def test_reset(self):
        """测试重置"""
        tracker = RealtimeMultiPersonTracker()

        # 创建轨迹
        tracker.update([{"position": (1.0, 1.0, 0.0)}])
        assert len(tracker._tracks) == 1

        # 重置
        tracker.reset()
        assert len(tracker._tracks) == 0
        assert tracker._frame_count == 0


class TestSimulationRenderer:
    """SimulationRenderer 单元测试"""

    @pytest.fixture
    def renderer(self):
        """创建渲染器实例"""
        try:
            from plugins.indoor_fence.standalone.simulation_renderer import SimulationRenderer
            return SimulationRenderer(
                scene_bounds=(0.0, 0.0, 10.0, 6.0),
                resolution=(640, 480),
                yellow_line_y=3.0,
            )
        except ImportError:
            pytest.skip("cv2 not available")

    def test_init(self, renderer):
        """测试初始化"""
        assert renderer._w == 640
        assert renderer._h == 480
        assert renderer._yellow_line_y == 3.0

    def test_world_to_pixel(self, renderer):
        """测试坐标转换"""
        # 场景中心
        px, py = renderer.world_to_pixel(5.0, 3.0)
        assert 200 < px < 440
        assert 200 < py < 280

        # 左下角
        px, py = renderer.world_to_pixel(0.0, 0.0)
        assert px == renderer._margin
        assert py == renderer._margin

    def test_render_empty(self, renderer):
        """测试空场景渲染"""
        frame = renderer.render([], step_count=0)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

    def test_render_single_person(self, renderer):
        """测试单人渲染"""
        persons = [{"id": 0, "x": 5.0, "y": 2.0, "z": 0.0}]
        frame = renderer.render(persons, step_count=1)
        assert frame.shape == (480, 640, 3)

    def test_render_multiple_persons(self, renderer):
        """测试多人渲染"""
        persons = [
            {"id": 0, "x": 2.0, "y": 1.0, "z": 0.0},
            {"id": 1, "x": 5.0, "y": 2.0, "z": 0.0},
            {"id": 2, "x": 8.0, "y": 1.5, "z": 0.0},
        ]
        frame = renderer.render(persons, step_count=10)
        assert frame.shape == (480, 640, 3)

    def test_render_with_velocity(self, renderer):
        """测试带速度渲染"""
        persons = [
            {"id": 0, "x": 5.0, "y": 2.0, "z": 0.0, "vx": 1.0, "vy": 0.5},
        ]
        frame = renderer.render(persons, step_count=1)
        assert frame.shape == (480, 640, 3)

    def test_trail_accumulation(self, renderer):
        """测试轨迹累积"""
        for i in range(10):
            persons = [{"id": 0, "x": 2.0 + i * 0.1, "y": 2.0, "z": 0.0}]
            renderer.render(persons, step_count=i)

        assert 0 in renderer._trails
        assert len(renderer._trails[0]) == 10

    def test_reset_trails(self, renderer):
        """测试轨迹重置"""
        # 累积轨迹
        for i in range(5):
            renderer.render([{"id": 0, "x": 2.0 + i * 0.1, "y": 2.0, "z": 0.0}], i)

        assert len(renderer._trails) > 0

        # 重置
        renderer.reset_trails()
        assert len(renderer._trails) == 0

    def test_set_display_options(self, renderer):
        """测试显示选项设置"""
        renderer.set_display_options(
            show_distances=False,
            show_predictions=False,
            show_velocities=False,
        )
        assert renderer._show_distances is False
        assert renderer._show_predictions is False
        assert renderer._show_velocities is False

    def test_set_alerts(self, renderer):
        """测试告警设置"""
        alerts = [
            {"type": "fence_breach", "level": "critical", "message": "P0 越线"},
        ]
        renderer.set_alerts(alerts)
        assert len(renderer._alerts) == 1

    def test_scenario_name(self, renderer):
        """测试场景名称"""
        renderer.scenario_name = "test_scenario"
        assert renderer.scenario_name == "test_scenario"


class TestRealtimeMultiPersonPipeline:
    """RealtimeMultiPersonPipeline 集成测试"""

    def test_init_default(self):
        """测试默认初始化"""
        from plugins.indoor_fence.standalone.realtime_pipeline import (
            PipelineConfig,
            PipelineMode,
            RealtimeMultiPersonPipeline,
        )

        pipeline = RealtimeMultiPersonPipeline()
        assert pipeline.config.mode == PipelineMode.SIMULATION
        assert pipeline.state.is_running is False

    def test_init_custom_config(self):
        """测试自定义配置初始化"""
        from plugins.indoor_fence.standalone.realtime_pipeline import (
            PipelineConfig,
            PipelineMode,
            RealtimeMultiPersonPipeline,
        )

        config = PipelineConfig(
            mode=PipelineMode.SIMULATION,
            fence_line_y=4.0,
            update_rate=20.0,
        )
        pipeline = RealtimeMultiPersonPipeline(config=config)

        assert pipeline.config.fence_line_y == 4.0
        assert pipeline.config.update_rate == 20.0

    def test_initialize_simulation(self):
        """测试仿真模式初始化"""
        from plugins.indoor_fence.standalone.realtime_pipeline import (
            PipelineConfig,
            PipelineMode,
            RealtimeMultiPersonPipeline,
        )

        config = PipelineConfig(mode=PipelineMode.SIMULATION)
        pipeline = RealtimeMultiPersonPipeline(config=config)

        result = pipeline.initialize()
        assert result is True
        assert pipeline._renderer is not None
        assert pipeline._simulator is not None

    def test_start_stop(self):
        """测试启动和停止"""
        from plugins.indoor_fence.standalone.realtime_pipeline import RealtimeMultiPersonPipeline

        pipeline = RealtimeMultiPersonPipeline()
        pipeline.initialize()

        # 启动
        assert pipeline.start() is True
        assert pipeline.state.is_running is True

        # 等待几帧
        time.sleep(0.2)

        # 停止
        pipeline.stop()
        assert pipeline.state.is_running is False

    def test_get_tracking_data(self):
        """测试获取跟踪数据"""
        from plugins.indoor_fence.standalone.realtime_pipeline import RealtimeMultiPersonPipeline

        pipeline = RealtimeMultiPersonPipeline()
        pipeline.initialize()
        pipeline.start()

        time.sleep(0.3)  # 等待几帧

        data = pipeline.get_tracking_data()

        assert "timestamp" in data
        assert "frame_count" in data
        assert "persons" in data
        assert "stats" in data

        pipeline.stop()

    def test_reset(self):
        """测试重置"""
        from plugins.indoor_fence.standalone.realtime_pipeline import RealtimeMultiPersonPipeline

        pipeline = RealtimeMultiPersonPipeline()
        pipeline.initialize()
        pipeline.start()

        time.sleep(0.2)

        pipeline.reset()

        assert pipeline.state.frame_count == 0
        assert len(pipeline._tracker._tracks) == 0

        pipeline.stop()

    def test_callbacks(self):
        """测试回调函数"""
        from plugins.indoor_fence.standalone.realtime_pipeline import RealtimeMultiPersonPipeline

        frames_received = []
        tracking_received = []

        def on_frame(frame):
            frames_received.append(frame)

        def on_tracking(result):
            tracking_received.append(result)

        pipeline = RealtimeMultiPersonPipeline(
            on_frame_callback=on_frame,
            on_tracking_callback=on_tracking,
        )
        pipeline.initialize()
        pipeline.start()

        time.sleep(0.3)

        pipeline.stop()

        assert len(frames_received) > 0
        assert len(tracking_received) > 0


class TestHungarianAssociation:
    """Hungarian 算法关联测试"""

    def test_optimal_assignment_no_crossing(self):
        """测试最优分配避免交叉匹配"""
        tracker = RealtimeMultiPersonTracker(min_hits=1)

        # 创建两个初始轨迹
        tracker.update([
            {"position": (1.0, 1.0, 0.0)},
            {"position": (5.0, 1.0, 0.0)},
        ])

        # 新的检测靠近各自轨迹（贪婪可能交叉，Hungarian 不会）
        result = tracker.update([
            {"position": (4.8, 1.0, 0.0)},  # 靠近轨迹 2
            {"position": (1.2, 1.0, 0.0)},  # 靠近轨迹 1
        ])

        assert len(tracker._tracks) == 2  # 不应创建新轨迹

    def test_3d_distance_in_cost_matrix(self):
        """测试 3D 距离用于代价矩阵"""
        tracker = RealtimeMultiPersonTracker(min_hits=1)

        # 创建 3D 轨迹
        tracker.update([
            {"position": (1.0, 1.0, 0.0)},
            {"position": (1.0, 1.0, 2.0)},  # 同 x,y 不同 z
        ])

        # 更新只有 z 不同
        result = tracker.update([
            {"position": (1.0, 1.0, 0.1)},  # 靠近 z=0 的轨迹
            {"position": (1.0, 1.0, 1.9)},  # 靠近 z=2 的轨迹
        ])

        # 两个轨迹应正确匹配（不应增加）
        assert len(tracker._tracks) == 2

    def test_many_targets_assignment(self):
        """测试多目标分配"""
        tracker = RealtimeMultiPersonTracker(min_hits=1)

        # 创建 5 个轨迹
        initial = [{"position": (float(i * 2), 1.0, 0.0)} for i in range(5)]
        tracker.update(initial)

        # 稍微偏移后更新
        updated = [{"position": (float(i * 2) + 0.1, 1.05, 0.0)} for i in range(5)]
        result = tracker.update(updated)

        assert len(tracker._tracks) == 5

    def test_unmatched_detection_creates_new_track(self):
        """测试未匹配检测创建新轨迹"""
        tracker = RealtimeMultiPersonTracker(min_hits=1)

        tracker.update([{"position": (1.0, 1.0, 0.0)}])

        # 新检测距已有轨迹 > 2m
        result = tracker.update([
            {"position": (1.1, 1.0, 0.0)},  # 匹配已有
            {"position": (8.0, 4.0, 0.0)},  # 新目标
        ])

        assert len(tracker._tracks) == 2

    def test_empty_detections(self):
        """测试空检测列表"""
        tracker = RealtimeMultiPersonTracker()
        result = tracker.update([])
        assert result.stats["total_tracks"] == 0

    def test_empty_tracks_new_detections(self):
        """测试无轨迹时创建新轨迹"""
        tracker = RealtimeMultiPersonTracker()
        result = tracker.update([
            {"position": (1.0, 1.0, 0.0)},
            {"position": (5.0, 5.0, 0.0)},
        ])
        assert len(tracker._tracks) == 2


class TestDistanceCalculations:
    """距离计算边界测试"""

    def test_euclidean_distance(self):
        """测试欧几里得距离"""
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        assert abs(dist - 5.0) < 0.001

    def test_fence_distance_positive(self):
        """测试正向围栏距离"""
        fence_y = 3.0
        person_y = 2.0
        dist = fence_y - person_y
        assert dist == 1.0

    def test_fence_distance_negative(self):
        """测试负向围栏距离（越线）"""
        fence_y = 3.0
        person_y = 3.5
        dist = fence_y - person_y
        assert dist == -0.5

    def test_distance_level_thresholds(self):
        """测试距离等级阈值"""
        tracker = RealtimeMultiPersonTracker()

        # 安全
        assert tracker._classify_fence_distance(1.0) == DistanceLevel.SAFE

        # 警告
        assert tracker._classify_fence_distance(0.4) == DistanceLevel.WARNING

        # 危险
        assert tracker._classify_fence_distance(0.15) == DistanceLevel.DANGER

        # 临界
        assert tracker._classify_fence_distance(-0.1) == DistanceLevel.CRITICAL


class TestTrajectoryPrediction:
    """轨迹预测测试"""

    def test_linear_prediction(self):
        """测试线性预测"""
        x, y = 1.0, 1.0
        vx, vy = 1.0, 0.5
        dt = 0.1

        # 预测 10 步
        predictions = []
        for i in range(1, 11):
            px = x + vx * dt * i
            py = y + vy * dt * i
            predictions.append((px, py))

        # 验证预测
        assert len(predictions) == 10
        assert abs(predictions[9][0] - 2.0) < 0.001  # x = 1 + 1*0.1*10 = 2
        assert abs(predictions[9][1] - 1.5) < 0.001  # y = 1 + 0.5*0.1*10 = 1.5

    def test_stationary_no_prediction(self):
        """测试静止目标无预测"""
        tracker = RealtimeMultiPersonTracker(min_hits=1)

        # 静止目标
        for _ in range(3):
            tracker.update([{"position": (1.0, 1.0, 0.0), "velocity": (0.0, 0.0, 0.0)}])

        track = list(tracker._tracks.values())[0]

        # 速度为 0，预测位置应该不变或为空
        if track.predicted_positions:
            for pred in track.predicted_positions:
                assert abs(pred[0] - 1.0) < 0.5
                assert abs(pred[1] - 1.0) < 0.5


class TestMultiSensorFusionPipeline:
    """多传感器融合管道集成测试"""

    def test_pipeline_with_multi_sensor_simulation(self):
        """测试多传感器仿真模式管道"""
        from plugins.indoor_fence.standalone.realtime_pipeline import (
            PipelineConfig,
            PipelineMode,
            RealtimeMultiPersonPipeline,
        )

        config = PipelineConfig(
            mode=PipelineMode.SIMULATION,
            enable_camera=True,
            enable_lidar=True,
            enable_uwb=True,
            enable_imu=True,
        )
        pipeline = RealtimeMultiPersonPipeline(config=config)
        result = pipeline.initialize()
        assert result is True
        assert pipeline._simulator is not None

    def test_pipeline_fusion_produces_tracking_data(self):
        """测试融合管道输出跟踪数据"""
        from plugins.indoor_fence.standalone.realtime_pipeline import (
            PipelineConfig,
            PipelineMode,
            RealtimeMultiPersonPipeline,
        )

        config = PipelineConfig(
            mode=PipelineMode.SIMULATION,
            enable_camera=True,
            enable_lidar=True,
            enable_uwb=True,
            enable_imu=True,
        )
        pipeline = RealtimeMultiPersonPipeline(config=config)
        pipeline.initialize()
        pipeline.start()

        time.sleep(0.5)  # 等待足够帧

        data = pipeline.get_tracking_data()
        assert "persons" in data
        assert "stats" in data

        pipeline.stop()

    def test_pipeline_reset_clears_fusion(self):
        """测试重置清理融合状态"""
        from plugins.indoor_fence.standalone.realtime_pipeline import (
            PipelineConfig,
            PipelineMode,
            RealtimeMultiPersonPipeline,
        )

        config = PipelineConfig(mode=PipelineMode.SIMULATION)
        pipeline = RealtimeMultiPersonPipeline(config=config)
        pipeline.initialize()
        pipeline.start()

        time.sleep(0.2)

        pipeline.reset()
        assert pipeline.state.frame_count == 0

        pipeline.stop()


class TestSensorFusionV3Hungarian:
    """SensorFusionV3 Hungarian 分配测试"""

    def test_fusion_hungarian_multi_target(self):
        """测试融合引擎的 Hungarian 多目标分配"""
        from plugins.indoor_fence.core.fusion.sensor_fusion_v3 import SensorFusionV3
        from plugins.indoor_fence.protocols import SensorData, SensorType

        fusion = SensorFusionV3()
        ts = time.time()

        # 两个相机检测 + 两个 LiDAR 检测
        camera_data = SensorData(
            sensor_type=SensorType.CAMERA,
            timestamp=ts,
            data={"detections": [
                {"position_2d": (2.0, 1.0)},
                {"position_2d": (6.0, 3.0)},
            ]},
            confidence=0.9,
        )
        lidar_data = SensorData(
            sensor_type=SensorType.LIDAR,
            timestamp=ts,
            data={"clusters": [
                {"position_2d": (2.1, 1.1)},
                {"position_2d": (6.1, 3.1)},
            ]},
            confidence=0.95,
        )

        outputs = fusion.update({
            SensorType.CAMERA: camera_data,
            SensorType.LIDAR: lidar_data,
        })

        # 应产生至少 2 个跟踪输出
        assert len(outputs) >= 2

    def test_fusion_imu_velocity(self):
        """测试 IMU 速度融合"""
        from plugins.indoor_fence.core.fusion.sensor_fusion_v3 import SensorFusionV3
        from plugins.indoor_fence.protocols import SensorData, SensorType

        fusion = SensorFusionV3()

        # 第一帧建立轨迹
        ts = time.time()
        camera = SensorData(
            sensor_type=SensorType.CAMERA,
            timestamp=ts,
            data={"detections": [{"position_2d": (3.0, 2.0)}]},
            confidence=0.9,
        )
        fusion.update({SensorType.CAMERA: camera})

        # 第二帧带 IMU
        imu = SensorData(
            sensor_type=SensorType.IMU,
            timestamp=ts + 0.1,
            data={
                "acceleration": {"x": 1.0, "y": 0.5, "z": 9.81},
                "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            confidence=0.95,
        )
        camera2 = SensorData(
            sensor_type=SensorType.CAMERA,
            timestamp=ts + 0.1,
            data={"detections": [{"position_2d": (3.1, 2.0)}]},
            confidence=0.9,
        )
        outputs = fusion.update({
            SensorType.CAMERA: camera2,
            SensorType.IMU: imu,
        })

        assert len(outputs) >= 1
        # IMU should contribute to sources
        has_imu = any(SensorType.IMU in o.sources for o in outputs)
        assert has_imu

    def test_fusion_3d_uwb_tracking(self):
        """测试 UWB 3D 跟踪"""
        from plugins.indoor_fence.core.fusion.sensor_fusion_v3 import SensorFusionV3
        from plugins.indoor_fence.protocols import SensorData, SensorType

        fusion = SensorFusionV3()

        for i in range(5):
            ts = time.time() + i * 0.1
            uwb = SensorData(
                sensor_type=SensorType.UWB,
                timestamp=ts,
                data={"positions": [
                    {"x": 3.0 + i * 0.1, "y": 2.0, "z": 0.5},
                ]},
                confidence=0.9,
            )
            outputs = fusion.update({SensorType.UWB: uwb})

        assert len(outputs) >= 1
        # z should be non-zero from UWB
        assert any(abs(o.position_3d[2]) > 0.01 for o in outputs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
