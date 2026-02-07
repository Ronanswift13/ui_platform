"""
室内电子围栏 - 集成测试
========================

测试范围:
1. 插件加载与初始化
2. API端点功能
3. WebSocket连接与消息
4. 事件契约验证
5. 坐标转换正确性
6. 端到端流程

版本: 3.0.0
作者: 变电站AI监控平台架构组

运行方式:
    pytest tests/integration/test_indoor_fence_v3.py -v --tb=short
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, List, Any
import pytest
from unittest.mock import Mock, patch, AsyncMock

import numpy as np

# 添加项目根目录到导入路径（与现有测试约定保持一致）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# =============================================================================
# 测试配置
# =============================================================================

class TestConfig:
    """测试配置"""
    API_BASE = "/api/v3/indoor/fence"
    TIMEOUT = 10.0
    PLUGIN_ID = "indoor_fence"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def event_schema():
    """统一事件契约schema"""
    return {
        "required": [
            "event_id", "timestamp", "type", "location", "value",
            "confidence", "evidence", "suggestion", "trace_id"
        ],
        "properties": {
            "event_id": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"},
            "type": {"type": "string"},
            "location": {
                "type": "object",
                "required": ["ground_position"],
                "properties": {
                    "ground_position": {
                        "type": "object",
                        "required": ["x", "y"],
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"}
                        }
                    }
                }
            },
            "value": {"type": "object"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "object"},
            "suggestion": {"type": "object"},
            "trace_id": {"type": "string"}
        }
    }


@pytest.fixture
def sample_track():
    """示例跟踪数据"""
    return {
        "track_id": "T0001",
        "person_state": "safe",
        "ground_position": {"x": 2.5, "y": 1.0},
        "velocity": {"x": 0.1, "y": 0.0},
        "confidence": 0.95,
        "cabinet_id": None,
        "is_authorized": True,
        "last_seen": datetime.now().isoformat(),
        "hits": 10,
        "age": 15
    }


@pytest.fixture
def sample_event():
    """示例事件数据"""
    return {
        "event_id": f"evt_{uuid.uuid4().hex[:12]}",
        "timestamp": datetime.now().isoformat(),
        "type": "person_detected",
        "location": {
            "ground_position": {"x": 2.5, "y": 1.0}
        },
        "value": {"person_count": 1},
        "confidence": 0.95,
        "evidence": {
            "frame_raw": "/evidence/frame_001.jpg"
        },
        "suggestion": {
            "suggestion_type": "monitor",
            "message": "持续监控"
        },
        "trace_id": f"trace_{uuid.uuid4().hex[:16]}_{int(time.time() * 1000)}",
        "alert_level": "info",
        "person_id": "T0001",
        "source_plugin": "indoor_fence"
    }


@pytest.fixture
def mock_plugin_manager():
    """模拟插件管理器"""
    with patch('platform_core.plugin_manager.PluginManager') as MockPM:
        pm = Mock()

        class _DummyAdapter:
            def __init__(self, data_rate_hz: float):
                self._health = SimpleNamespace(
                    status=SimpleNamespace(value="running"),
                    data_rate_hz=data_rate_hz,
                )

            def get_health(self):
                return self._health

        class _DummyTracker:
            def __init__(self, track_obj):
                self._track_obj = track_obj

            def get_confirmed_tracks(self):
                return [self._track_obj]

        class _DummyPlugin:
            def __init__(self, track_obj):
                self.id = "indoor_fence"
                self.version = "3.0.0"
                self.status = SimpleNamespace(value="ready")
                self._tracker = _DummyTracker(track_obj)
                self._camera_adapter = _DummyAdapter(data_rate_hz=12.5)
                self._lidar_adapter = _DummyAdapter(data_rate_hz=10.0)

            def healthcheck(self):
                return SimpleNamespace(
                    healthy=True,
                    message="正常",
                    details={"frame_count": 100},
                )

            def update_allow_list(self, _cabinet_ids):
                return None

        # 模拟跟踪目标
        track = SimpleNamespace(
            track_id="T0001",
            position=SimpleNamespace(x=2.5, y=1.0),
            velocity=SimpleNamespace(x=0.1, y=0.0),
            confidence=0.95,
            last_seen=time.time(),
            hits=10,
            age=15,
            is_confirmed=True,
            is_authorized=True,
            cabinet_id=None,
        )

        plugin = _DummyPlugin(track)

        pm.get_plugin.return_value = plugin
        pm.load_plugin.return_value = plugin
        MockPM.return_value = pm

        yield pm


# =============================================================================
# 事件契约测试
# =============================================================================

class TestEventSchema:
    """事件契约测试"""
    
    def test_event_has_required_fields(self, sample_event, event_schema):
        """测试事件包含所有必填字段"""
        for field in event_schema["required"]:
            assert field in sample_event, f"缺少必填字段: {field}"
    
    def test_confidence_range(self, sample_event):
        """测试置信度范围 (0-1)"""
        assert 0 <= sample_event["confidence"] <= 1
    
    def test_trace_id_format(self, sample_event):
        """测试trace_id格式"""
        trace_id = sample_event["trace_id"]
        assert trace_id.startswith("trace_")
        assert len(trace_id) > 20
    
    def test_location_has_ground_position(self, sample_event):
        """测试位置信息包含地面坐标"""
        assert "ground_position" in sample_event["location"]
        assert "x" in sample_event["location"]["ground_position"]
        assert "y" in sample_event["location"]["ground_position"]
    
    def test_event_builder_creates_valid_event(self):
        """测试事件构建器创建有效事件"""
        from platform_core.schema.indoor_fence_events import (
            IndoorFenceEventBuilder, Point2D, BoundingBox, EventValidator
        )
        
        builder = IndoorFenceEventBuilder()
        event = builder.build_person_detected_event(
            person_id="T0001",
            ground_position=Point2D(2.5, 1.0),
            bbox=BoundingBox(0.1, 0.2, 0.3, 0.4),
            confidence=0.95
        )
        
        # 验证事件
        is_valid, errors = EventValidator.validate(event)
        assert is_valid, f"事件验证失败: {errors}"
    
    def test_alarm_event_requires_evidence(self):
        """测试告警事件必须有证据"""
        from platform_core.schema.indoor_fence_events import (
            UnifiedEvent, EventType, AlertLevel, Location, Point2D,
            Evidence, Suggestion, SuggestionType, generate_event_id,
            generate_trace_id, EventValidator
        )
        
        # 创建告警级别事件但没有帧证据
        event = UnifiedEvent(
            event_id=generate_event_id(),
            timestamp=datetime.now().isoformat(),
            type=EventType.YELLOW_LINE_ALARM,
            location=Location(ground_position=Point2D(2.5, 1.0)),
            value={"distance": 0.1},
            confidence=0.95,
            evidence=Evidence(),  # 空证据
            suggestion=Suggestion(
                suggestion_type=SuggestionType.SOUND_ALARM,
                message="越线"
            ),
            trace_id=generate_trace_id(),
            alert_level=AlertLevel.ALARM
        )
        
        is_valid, errors = EventValidator.validate(event)
        assert not is_valid, "告警事件无证据应验证失败"
        assert any("帧图像证据" in e for e in errors)


# =============================================================================
# 坐标转换测试
# =============================================================================

class TestCoordinateTransform:
    """坐标转换测试"""
    
    def test_pixel_to_ground_basic(self):
        """测试基本像素到地面坐标转换"""
        from plugins.indoor_fence.core.geometry import pixel_to_ground
        
        # 图像中心点应该对应坐标原点附近
        result = pixel_to_ground(320, 240, 640, 480)
        assert isinstance(result.x, float)
        assert isinstance(result.y, float)
    
    def test_lidar_to_ground_basic(self):
        """测试激光雷达到地面坐标转换"""
        from plugins.indoor_fence.core.geometry import lidar_to_ground
        
        # 正前方3米
        result = lidar_to_ground(3.0, 0, lidar_height=2.5, lidar_tilt_deg=20)
        assert result.y > 0  # 应该在前方
        assert abs(result.x) < 0.1  # 应该在中心线附近
    
    def test_fuse_coordinates_both_present(self):
        """测试视觉+雷达融合坐标"""
        from plugins.indoor_fence.core.geometry import fuse_coordinates, Point2D
        
        visual = Point2D(2.5, 1.0)
        lidar = Point2D(2.6, 1.2)
        
        # 默认横向信任视觉,纵向信任雷达
        result = fuse_coordinates(visual, lidar)
        
        assert result.x == visual.x  # X来自视觉
        assert result.y == lidar.y   # Y来自雷达
    
    def test_fuse_coordinates_visual_only(self):
        """测试仅视觉坐标"""
        from plugins.indoor_fence.core.geometry import fuse_coordinates, Point2D
        
        visual = Point2D(2.5, 1.0)
        result = fuse_coordinates(visual, None)
        
        assert result.x == visual.x
        assert result.y == visual.y
    
    def test_fuse_coordinates_lidar_only(self):
        """测试仅雷达坐标"""
        from plugins.indoor_fence.core.geometry import fuse_coordinates, Point2D
        
        lidar = Point2D(2.6, 1.2)
        result = fuse_coordinates(None, lidar)
        
        assert result.x == lidar.x
        assert result.y == lidar.y


# =============================================================================
# 插件集成测试
# =============================================================================

class TestPluginIntegration:
    """插件集成测试"""
    
    def test_plugin_load(self, mock_plugin_manager):
        """测试插件加载"""
        plugin = mock_plugin_manager.get_plugin("indoor_fence")
        assert plugin is not None
        assert plugin.id == "indoor_fence"
    
    def test_plugin_healthcheck(self, mock_plugin_manager):
        """测试插件健康检查"""
        plugin = mock_plugin_manager.get_plugin("indoor_fence")
        health = plugin.healthcheck()
        
        assert health.healthy is True
        assert "frame_count" in health.details
    
    def test_tracker_get_tracks(self, mock_plugin_manager):
        """测试获取跟踪轨迹"""
        plugin = mock_plugin_manager.get_plugin("indoor_fence")
        tracks = plugin._tracker.get_confirmed_tracks()
        
        assert len(tracks) > 0
        assert tracks[0].track_id == "T0001"


# =============================================================================
# API测试 (需要FastAPI TestClient)
# =============================================================================

class TestAPIEndpoints:
    """API端点测试"""
    
    @pytest.fixture
    def client(self, mock_plugin_manager):
        """创建测试客户端"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from apps.indoor_fence_api_v3 import integrate_indoor_fence_api_v3
        
        app = FastAPI()
        integrate_indoor_fence_api_v3(app)
        
        return TestClient(app)
    
    def test_get_status(self, client):
        """测试获取状态接口"""
        response = client.get(f"{TestConfig.API_BASE}/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "plugin_id" in data
        assert "status" in data
        assert "tracked_persons" in data
    
    def test_get_tracks(self, client):
        """测试获取轨迹接口"""
        response = client.get(f"{TestConfig.API_BASE}/tracks")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_events(self, client):
        """测试获取事件接口"""
        response = client.get(f"{TestConfig.API_BASE}/events")
        assert response.status_code == 200
        
        data = response.json()
        assert "total" in data
        assert "events" in data
    
    def test_get_zones(self, client):
        """测试获取区域配置接口"""
        response = client.get(f"{TestConfig.API_BASE}/zones")
        assert response.status_code == 200
    
    def test_health_check(self, client):
        """测试健康检查接口"""
        response = client.get(f"{TestConfig.API_BASE}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "healthy" in data
        assert "components" in data
    
    def test_update_authorization(self, client):
        """测试更新授权接口"""
        response = client.post(
            f"{TestConfig.API_BASE}/authorize",
            json={
                "person_id": "T0001",
                "cabinet_ids": [1, 2, 3]
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"


# =============================================================================
# WebSocket测试
# =============================================================================

class TestWebSocket:
    """WebSocket测试"""
    
    @pytest.fixture
    def ws_client(self, mock_plugin_manager):
        """创建WebSocket测试客户端"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from apps.indoor_fence_api_v3 import integrate_indoor_fence_api_v3
        
        app = FastAPI()
        integrate_indoor_fence_api_v3(app)
        
        return TestClient(app)
    
    def test_websocket_connect(self, ws_client):
        """测试WebSocket连接"""
        with ws_client.websocket_connect(f"{TestConfig.API_BASE}/ws") as ws:
            # 应该收到连接确认
            data = ws.receive_json()
            assert data["type"] == "connection_ack"
            assert "client_id" in data
    
    def test_websocket_ping_pong(self, ws_client):
        """测试WebSocket心跳"""
        with ws_client.websocket_connect(f"{TestConfig.API_BASE}/ws") as ws:
            # 跳过连接确认
            ws.receive_json()
            
            # 发送ping
            ws.send_json({"type": "ping"})
            
            # 应该收到pong
            data = ws.receive_json()
            assert data["type"] == "pong"
    
    def test_websocket_get_status(self, ws_client):
        """测试通过WebSocket获取状态"""
        with ws_client.websocket_connect(f"{TestConfig.API_BASE}/ws") as ws:
            # 跳过连接确认
            ws.receive_json()
            
            # 请求状态
            ws.send_json({"type": "get_status"})
            
            # 应该收到状态数据
            data = ws.receive_json()
            assert data["type"] == "status"
    
    def test_websocket_reconnect_with_client_id(self, ws_client):
        """测试带client_id的WebSocket重连"""
        # 第一次连接
        with ws_client.websocket_connect(f"{TestConfig.API_BASE}/ws") as ws:
            data = ws.receive_json()
            client_id = data["client_id"]
        
        # 使用相同client_id重连
        with ws_client.websocket_connect(
            f"{TestConfig.API_BASE}/ws?client_id={client_id}"
        ) as ws:
            data = ws.receive_json()
            assert data["type"] == "connection_ack"


# =============================================================================
# 端到端测试
# =============================================================================

class TestEndToEnd:
    """端到端测试"""
    
    def test_full_detection_flow(self, mock_plugin_manager):
        """测试完整检测流程"""
        from platform_core.schema.indoor_fence_events import (
            IndoorFenceEventBuilder, Point2D, BoundingBox
        )
        
        # 1. 初始化事件构建器
        builder = IndoorFenceEventBuilder()
        
        # 2. 模拟检测结果
        detection = {
            "person_id": "T0001",
            "position": Point2D(2.5, 1.0),
            "bbox": BoundingBox(0.1, 0.2, 0.3, 0.4),
            "confidence": 0.95
        }
        
        # 3. 构建检测事件
        event = builder.build_person_detected_event(
            person_id=detection["person_id"],
            ground_position=detection["position"],
            bbox=detection["bbox"],
            confidence=detection["confidence"]
        )
        
        # 4. 验证事件
        assert event.type.value == "person_detected"
        assert event.person_id == "T0001"
        assert event.confidence == 0.95
        
        # 5. 验证trace_id贯穿
        assert builder.trace_id == event.trace_id
    
    def test_alarm_chain(self, mock_plugin_manager):
        """测试告警链条"""
        from platform_core.schema.indoor_fence_events import (
            IndoorFenceEventBuilder, Point2D
        )
        
        builder = IndoorFenceEventBuilder()
        
        # 1. 压线预警
        warning_event = builder.build_yellow_line_alarm_event(
            person_id="T0001",
            ground_position=Point2D(2.5, 0.4),
            distance_to_line=0.35,
            confidence=0.92,
            is_cross=False
        )
        
        # 2. 越线告警
        alarm_event = builder.build_yellow_line_alarm_event(
            person_id="T0001",
            ground_position=Point2D(2.5, 0.2),
            distance_to_line=0.15,
            confidence=0.95,
            is_cross=True
        )
        
        # 验证事件链使用相同trace_id
        assert warning_event.trace_id == alarm_event.trace_id
        
        # 验证告警级别升级
        assert warning_event.alert_level.value == "warning"
        assert alarm_event.alert_level.value == "alarm"


# =============================================================================
# 性能测试
# =============================================================================

class TestPerformance:
    """性能测试"""
    
    def test_event_creation_performance(self):
        """测试事件创建性能"""
        from platform_core.schema.indoor_fence_events import (
            IndoorFenceEventBuilder, Point2D, BoundingBox
        )
        
        builder = IndoorFenceEventBuilder()
        
        start = time.time()
        for i in range(1000):
            builder.build_person_detected_event(
                person_id=f"T{i:04d}",
                ground_position=Point2D(float(i % 10), float(i % 6)),
                bbox=BoundingBox(0.1, 0.2, 0.3, 0.4),
                confidence=0.95
            )
        elapsed = time.time() - start
        
        # 1000个事件应该在1秒内创建完成
        assert elapsed < 1.0, f"事件创建太慢: {elapsed:.2f}s"
    
    def test_coordinate_transform_performance(self):
        """测试坐标转换性能"""
        from plugins.indoor_fence.core.geometry import pixel_to_ground
        
        start = time.time()
        for i in range(10000):
            pixel_to_ground(
                float(i % 640), 
                float(i % 480), 
                640, 480
            )
        elapsed = time.time() - start
        
        # 10000次转换应该在0.5秒内完成
        assert elapsed < 0.5, f"坐标转换太慢: {elapsed:.2f}s"


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
