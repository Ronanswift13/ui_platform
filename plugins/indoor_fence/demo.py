#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内电子围栏 V2.0 演示脚本
========================

演示多人安全监测系统的核心功能:
1. 核心模块测试
2. 适配器测试
3. 状态机评估
4. 融合跟踪

运行方式:
    python -m plugins.indoor_fence.demo
"""

import logging
import time
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_separator(title=""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("=" * 50)


def demo_geometry():
    """演示几何模块"""
    print_separator("几何模块演示")

    from plugins.indoor_fence.core import Point2D, Polygon, point_in_polygon, point_to_polygon_distance

    # 创建多边形
    vertices = [Point2D(0, 0), Point2D(4, 0), Point2D(4, 3), Point2D(0, 3)]
    polygon = Polygon(vertices)

    # 测试点
    test_points = [
        (Point2D(2, 1.5), "内部点"),
        (Point2D(5, 1), "外部点"),
        (Point2D(0, 0), "顶点"),
        (Point2D(2, 0), "边上点"),
    ]

    for point, desc in test_points:
        inside = point_in_polygon(point, polygon)
        distance, _, _ = point_to_polygon_distance(point, polygon)
        status = "在多边形内" if inside else "在多边形外"
        print(f"  {desc} ({point.x}, {point.y}): {status}, 距离边界: {abs(distance):.2f}m")


def demo_zone_config():
    """演示区域配置"""
    print_separator("区域配置演示")

    from plugins.indoor_fence.core import ZoneConfigLoader

    # 加载默认配置
    loader = ZoneConfigLoader()
    config = loader.load()

    print(f"  配置ID: {config.config_id}")
    print(f"  配置名称: {config.config_name}")
    print(f"  区域数量: {len(config.zones)}")
    print(f"  机柜数量: {len(config.cabinets)}")

    print("\n  机柜列表:")
    for cab_id, cab in config.cabinets.items():
        print(f"    - 机柜{cab_id}: {cab.name} (x={cab.x_start:.2f}~{cab.x_end:.2f})")


def demo_state_machine():
    """演示状态机"""
    print_separator("状态机演示")

    from plugins.indoor_fence.core import (
        Point2D, ZoneConfigLoader, StateMachine, PersonState
    )

    # 初始化
    loader = ZoneConfigLoader()
    zone_config = loader.load()
    state_machine = StateMachine(zone_config)

    # 模拟人员位置
    test_cases = [
        ("T001", Point2D(5.0, 1.0), "主通道中央"),
        ("T002", Point2D(0.8, 2.3), "机柜1黄线附近"),
        ("T003", Point2D(2.5, 4.0), "机柜2内部"),
        ("T004", Point2D(8.5, 5.5), "机柜6深处"),
    ]

    # 设置授权
    state_machine.set_authorization("T001", [1, 2])
    state_machine.set_authorization("T002", [1])

    # 评估状态
    person_positions = {pid: pos for pid, pos, _ in test_cases}
    global_state = state_machine.evaluate_global(person_positions, {})

    print(f"  全局告警级别: {global_state.light_color}")
    print(f"  状态消息: {global_state.status_message}")
    print(f"\n  人员状态:")

    state_emoji = {
        PersonState.NORMAL: '正常',
        PersonState.ON_LINE: '压线',
        PersonState.MISPLACED: '未授权',
        PersonState.CROSS_LINE: '越线',
        PersonState.HIGH_RISK: '高风险',
    }

    for ps in global_state.person_states:
        desc = next((d for p, _, d in test_cases if p == ps.person_id), "")
        status = state_emoji.get(ps.state, "未知")
        auth = "已授权" if ps.is_authorized else "未授权"
        print(f"    - {ps.person_id} [{desc}]: {status} | 机柜{ps.current_cabinet or '-'} | {auth}")


def demo_tracking():
    """演示多目标跟踪"""
    print_separator("多目标跟踪演示")

    from plugins.indoor_fence.core import Point2D, Detection, MultiTargetTracker

    # 创建跟踪器
    tracker = MultiTargetTracker(
        max_age=10,
        min_hits=2,
        distance_threshold=1.0,
        use_kalman=True
    )

    # 模拟5帧检测
    frames = [
        # 帧1: 2个检测
        [Detection(detection_id='d1', position=Point2D(2.0, 1.0), confidence=0.9),
         Detection(detection_id='d2', position=Point2D(5.0, 1.5), confidence=0.85)],
        # 帧2: 3个检测 (新增1个)
        [Detection(detection_id='d3', position=Point2D(2.1, 1.1), confidence=0.88),
         Detection(detection_id='d4', position=Point2D(5.1, 1.4), confidence=0.87),
         Detection(detection_id='d5', position=Point2D(8.0, 1.0), confidence=0.8)],
        # 帧3: 3个检测 (继续移动)
        [Detection(detection_id='d6', position=Point2D(2.2, 1.2), confidence=0.9),
         Detection(detection_id='d7', position=Point2D(5.2, 1.3), confidence=0.86),
         Detection(detection_id='d8', position=Point2D(8.1, 1.1), confidence=0.82)],
        # 帧4: 2个检测 (第一个消失)
        [Detection(detection_id='d9', position=Point2D(5.3, 1.2), confidence=0.88),
         Detection(detection_id='d10', position=Point2D(8.2, 1.2), confidence=0.85)],
        # 帧5: 2个检测
        [Detection(detection_id='d11', position=Point2D(5.4, 1.1), confidence=0.87),
         Detection(detection_id='d12', position=Point2D(8.3, 1.3), confidence=0.83)],
    ]

    for i, detections in enumerate(frames):
        tracks = tracker.update(detections)
        all_tracks = tracker.get_all_tracks()
        confirmed_tracks = tracker.get_confirmed_tracks()

        print(f"  帧{i+1}: 检测数={len(detections)}, "
              f"活跃轨迹={len(all_tracks)}, "
              f"确认轨迹={len(confirmed_tracks)}")

        for track in confirmed_tracks:
            print(f"    - {track.track_id}: ({track.position.x:.2f}, {track.position.y:.2f}) "
                  f"置信度={track.confidence:.2f}")


def demo_adapters():
    """演示适配器"""
    print_separator("适配器演示")

    from plugins.indoor_fence.adapters import (
        CameraAdapter, CameraConfig,
        LidarAdapter, LidarConfig,
        LightAdapter, LightConfig, LightColor
    )

    # 相机适配器
    print("  [相机适配器]")
    cam_config = CameraConfig(
        source="0",
        resolution=(640, 480),
        fps=30,
        confidence_threshold=0.5,
    )
    cam_config.device_id = "camera_test"
    camera = CameraAdapter(cam_config)
    camera.connect()
    print(f"    状态: {camera.status.value}")
    print(f"    模拟模式: {camera.is_simulated}")

    # 模拟检测
    detections = camera.get_person_detections(None)
    print(f"    检测人数: {len(detections) if detections else 0}")
    camera.disconnect()

    # 雷达适配器
    print("\n  [雷达适配器]")
    lidar_config = LidarConfig(
        device_ip="192.168.1.100",
        device_port=2112,
    )
    lidar_config.device_id = "lidar_test"
    lidar = LidarAdapter(lidar_config)
    lidar.connect()
    print(f"    状态: {lidar.status.value}")

    scan = lidar.get_scan()
    if scan:
        clusters = lidar.get_clusters(scan)
        print(f"    扫描点数: {scan.num_points}")
        print(f"    聚类数量: {len(clusters)}")
    lidar.disconnect()

    # 灯光适配器
    print("\n  [灯光适配器]")
    light_config = LightConfig(output_type="simulate")
    light = LightAdapter(light_config)
    light.connect()
    print(f"    状态: {light.status.value}")

    # 测试灯光切换
    for color in [LightColor.GREEN, LightColor.YELLOW, LightColor.RED]:
        light.set_color(color)
        state = light.get_state()
        print(f"    设置{color.value}灯 -> 当前: {state.color.value}")

    light.all_off()
    light.disconnect()


def demo_fusion():
    """演示传感器融合"""
    print_separator("传感器融合演示")

    from plugins.indoor_fence.core import (
        VisualDetection, LidarDetection, FusedTracker
    )

    # 创建融合跟踪器
    tracker = FusedTracker(
        fusion_config={"max_time_diff_ms": 100, "distance_match_threshold_m": 0.5},
        tracker_config={"max_age": 10, "min_hits": 2, "use_kalman": True}
    )

    # 模拟视觉和雷达检测
    import time
    ts = time.time()

    visual_dets = [
        VisualDetection(
            detection_id="v1",
            bbox=(100, 200, 50, 120),
            foot_pixel=(125, 320),
            confidence=0.9,
            timestamp=ts,
        ),
        VisualDetection(
            detection_id="v2",
            bbox=(300, 180, 55, 130),
            foot_pixel=(327, 310),
            confidence=0.85,
            timestamp=ts,
        ),
    ]

    lidar_dets = [
        LidarDetection(
            cluster_id="l1",
            center_distance=3.0,
            center_angle=-10,
            point_count=15,
            confidence=0.88,
        ),
        LidarDetection(
            cluster_id="l2",
            center_distance=4.5,
            center_angle=15,
            point_count=12,
            confidence=0.82,
        ),
    ]

    print(f"  视觉检测数: {len(visual_dets)}")
    print(f"  雷达检测数: {len(lidar_dets)}")

    # 融合更新
    for i in range(5):
        tracks = tracker.update(visual_dets, lidar_dets)
        stats = tracker.get_statistics()
        print(f"\n  帧{i+1}: 融合轨迹数={len(tracks)}, 总检测={stats.get('total_detections', 0)}")

        for track in tracks:
            print(f"    - {track.track_id}: ({track.position.x:.2f}, {track.position.y:.2f})")


def demo_multi_person_violation():
    """演示多人违规检测"""
    print_separator("多人违规检测演示")

    from plugins.indoor_fence.core import (
        Point2D, ZoneConfigLoader, StateMachine
    )

    # 初始化
    loader = ZoneConfigLoader()
    zone_config = loader.load()
    state_machine = StateMachine(zone_config)

    # 模拟同一机柜前多人
    positions = {
        "P001": Point2D(0.8, 3.0),  # 机柜1内
        "P002": Point2D(0.9, 3.5),  # 机柜1内 - 违规
        "P003": Point2D(2.5, 3.0),  # 机柜2内
    }

    # 设置授权
    state_machine.set_authorization("P001", [1])
    state_machine.set_authorization("P002", [1])
    state_machine.set_authorization("P003", [2])

    # 评估
    global_state = state_machine.evaluate_global(positions, {})
    violations = state_machine.check_multi_person_violation(global_state.person_states)

    print(f"  检测人数: {global_state.total_persons}")
    print(f"  告警级别: {global_state.light_color}")

    if violations:
        print(f"\n  检测到违规: {len(violations)}个")
        for v in violations:
            print(f"    - 机柜{v.get('cabinet_id', '?')}: {v.get('message', '')}")
    else:
        print("\n  无违规情况")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("       室内电子围栏 V2.0 - 多人安全监测系统演示")
    print("=" * 60)

    try:
        # 几何模块
        demo_geometry()

        # 区域配置
        demo_zone_config()

        # 状态机
        demo_state_machine()

        # 多目标跟踪
        demo_tracking()

        # 适配器
        demo_adapters()

        # 传感器融合
        demo_fusion()

        # 多人违规检测
        demo_multi_person_violation()

        print("\n" + "=" * 60)
        print("                    所有演示完成!")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        logger.exception(f"演示出错: {e}")


if __name__ == "__main__":
    main()
