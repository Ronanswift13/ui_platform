"""End-to-end integration test for V3 indoor fence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
from plugins.indoor_fence.plugin import IndoorFencePlugin


def test_full_pipeline_simulation():
    """Run the full V3 pipeline with simulated sensors."""
    plugin = IndoorFencePlugin.create_standalone()
    assert plugin is not None
    assert plugin.id == "indoor_fence"

    # Health check
    health = plugin.healthcheck()
    assert health is not None

    # Cleanup
    plugin.cleanup()


def test_simulator_driven_pipeline():
    """Run V3 pipeline driven by simulator data."""
    from plugins.indoor_fence.adapters.simulator import Simulator, SimulatorConfig
    from plugins.indoor_fence.protocols import SensorType

    config = SimulatorConfig(
        num_persons=2,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA, SensorType.LIDAR],
        path_type="cross_line",
        yellow_line_y=2.5,
    )
    sim = Simulator(config)

    # Verify simulator produces data for 20 steps
    all_data = []
    for _ in range(20):
        sim_data = sim.step()
        assert isinstance(sim_data, dict)
        all_data.append(sim_data)

    assert len(all_data) == 20
    # Each step should have camera data
    for d in all_data:
        assert SensorType.CAMERA in d


def test_v3_imports():
    """Verify all V3 modules can be imported correctly."""
    from plugins.indoor_fence.protocols import (
        SensorData, SensorType, PersonStateV3, RiskAssessment,
    )
    from plugins.indoor_fence.adapters.uwb_adapter import UWBAdapter
    from plugins.indoor_fence.adapters.imu_adapter import IMUAdapter
    from plugins.indoor_fence.adapters.simulator import Simulator
    from plugins.indoor_fence.detection.object_detector import ObjectDetector
    from plugins.indoor_fence.detection.pose_estimator import PoseEstimatorV3
    from plugins.indoor_fence.detection.behavior_recognizer import BehaviorRecognizerV3
    from plugins.indoor_fence.core.fusion.ekf_fusion import EKF6DOF
    from plugins.indoor_fence.core.fusion.sensor_fusion_v3 import SensorFusionV3
    from plugins.indoor_fence.core.rules.rule_engine import RuleEngine
    from plugins.indoor_fence.core.rules.risk_scorer import RiskScorer
    from plugins.indoor_fence.standalone.data_recorder import DataRecorder
    from plugins.indoor_fence.standalone.data_replayer import DataReplayer
    from plugins.indoor_fence.standalone.training_pipeline import TrainingPipeline


def test_ekf_multi_sensor_end_to_end():
    """Test full fusion pipeline: simulator -> EKF -> tracker."""
    from plugins.indoor_fence.adapters.simulator import Simulator, SimulatorConfig
    from plugins.indoor_fence.protocols import SensorType
    from plugins.indoor_fence.core.fusion.sensor_fusion_v3 import SensorFusionV3
    from plugins.indoor_fence.core.tracking.multi_target_tracker_v3 import (
        MultiTargetTrackerV3, TrackState,
    )

    config = SimulatorConfig(
        num_persons=1,
        scene_bounds=(0.0, 0.0, 10.0, 6.0),
        sensor_types=[SensorType.CAMERA, SensorType.LIDAR],
        path_type="fixed",
        yellow_line_y=2.5,
    )
    sim = Simulator(config)
    fusion = SensorFusionV3()
    tracker = MultiTargetTrackerV3(max_age=10, min_hits=2)

    for _ in range(15):
        sensor_data = sim.step()
        fusion_outputs = fusion.update(sensor_data)
        tracks = tracker.update(fusion_outputs)

    active = tracker.get_active_tracks()
    assert len(active) >= 1
    assert active[0].state == TrackState.ACTIVE


def test_detection_behavior_pipeline():
    """Test detection -> pose -> behavior pipeline."""
    from plugins.indoor_fence.detection.yolo_detector import YOLODetector
    from plugins.indoor_fence.detection.pose_estimator import PoseEstimatorV3
    from plugins.indoor_fence.detection.behavior_recognizer import BehaviorRecognizerV3

    detector = YOLODetector(model_path=None, device="cpu")
    pose_est = PoseEstimatorV3(model_path=None)
    behavior = BehaviorRecognizerV3(window_size=5)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(5):
        detections = detector.detect(frame)
        assert len(detections) >= 1

        det = detections[0]
        bbox = (det.x1, det.y1, det.x2, det.y2)
        pose = pose_est.estimate(frame, bbox)
        assert pose is not None

        result = behavior.update(
            f"track_{det.class_id}",
            pose,
            position=(float(det.center[0]) / 100.0, float(det.center[1]) / 100.0),
        )
        assert result is not None


def test_rule_engine_with_state_machine():
    """Test state machine + rule engine integration."""
    from plugins.indoor_fence.core.state_machine import StateMachine, PersonState
    from plugins.indoor_fence.core.geometry import Point2D
    from plugins.indoor_fence.core.zone_config import ZoneConfigLoader
    from plugins.indoor_fence.core.rules.rule_engine import RuleEngine, Rule, RuleAction
    from plugins.indoor_fence.core.rules.risk_scorer import RiskScorer
    from plugins.indoor_fence.protocols import PersonStateV3

    loader = ZoneConfigLoader()
    zone_config = loader._create_default_config()
    sm = StateMachine(zone_config)

    engine = RuleEngine()
    engine.add_rule(Rule(
        rule_id="fallen",
        condition="person_state == 'fallen'",
        action=RuleAction.ALARM_CRITICAL,
        priority=10,
        cooldown_seconds=0,
    ))

    scorer = RiskScorer()

    # Simulate fallen person
    result = sm.evaluate_person(
        "p1", Point2D(5.0, 3.0),
        metadata={"behavior": "fallen"}
    )
    assert result.state == PersonState.FALLEN

    # Evaluate rule
    actions = engine.evaluate({"person_state": result.state.value})
    assert RuleAction.ALARM_CRITICAL in actions

    # Score risk
    score = scorer.score(
        person_state=PersonStateV3.FALLEN,
        distance_to_danger=0.5,
        is_authorized=False,
        behavior="fallen",
    )
    assert score > 0.5
