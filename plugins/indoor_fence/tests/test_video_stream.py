"""
视频流服务测试

覆盖:
- VideoStreamService 生命周期 (start/stop)
- 模拟帧非纯黑验证
- 帧标注功能
- MJPEG 生成器格式
- 统计信息
"""

import asyncio
import time
import numpy as np
import pytest

from plugins.indoor_fence.standalone.video_stream import VideoStreamService


# =============================================================================
# Mock Plugin
# =============================================================================


class MockCameraAdapter:
    """模拟相机适配器"""

    def __init__(self, simulated=True):
        self._simulated = simulated
        self.is_connected = True
        self._frame_count = 0

    class config:
        resolution = (640, 480)

    def get_frame(self):
        self._frame_count += 1
        # 返回非纯黑帧
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        return frame

    def get_person_detections(self, frame=None):
        """返回模拟检测结果"""

        class MockDetection:
            def __init__(self):
                self.detection_id = "D00001"
                self.bbox = {'x': 0.2, 'y': 0.3, 'width': 0.1, 'height': 0.3}
                self.confidence = 0.85
                self.track_id = "T001"
                self.foot_pixel = (160, 288)

        return [MockDetection()]


class MockPlugin:
    """模拟插件"""

    def __init__(self):
        self._camera_adapter = MockCameraAdapter(simulated=True)
        self._zone_config = None


# =============================================================================
# 测试用例
# =============================================================================


class TestVideoStreamService:
    """VideoStreamService 测试"""

    def test_service_creation(self):
        """测试服务创建"""
        plugin = MockPlugin()
        service = VideoStreamService(plugin, fps=10)

        assert service is not None
        assert not service.is_running
        assert service._fps == 10

    def test_service_start_stop(self):
        """测试启动/停止生命周期"""
        plugin = MockPlugin()
        service = VideoStreamService(plugin, fps=10)

        # 启动
        service.start()
        assert service.is_running
        assert service._thread is not None
        assert service._thread.is_alive()

        # 等待一些帧被处理
        time.sleep(0.5)

        # 检查帧计数已增加
        assert service._frame_count > 0

        # 停止
        service.stop()
        assert not service.is_running

    def test_service_double_start(self):
        """测试重复启动不会创建多个线程"""
        plugin = MockPlugin()
        service = VideoStreamService(plugin, fps=10)

        service.start()
        thread1 = service._thread

        service.start()  # 应该被忽略
        thread2 = service._thread

        assert thread1 is thread2

        service.stop()

    def test_get_latest_frame(self):
        """测试获取最新帧"""
        plugin = MockPlugin()
        service = VideoStreamService(plugin, fps=15)

        # 未启动时返回 None
        assert service.get_latest_frame() is None

        # 启动后有帧
        service.start()
        time.sleep(0.5)

        frame = service.get_latest_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3  # BGR
        assert frame.shape[0] > 0 and frame.shape[1] > 0

        service.stop()

    def test_simulation_frame_not_black(self):
        """测试模拟帧非纯黑"""
        plugin = MockPlugin()
        service = VideoStreamService(plugin, fps=15)

        # 直接调用渲染方法
        frame = service._render_simulation_frame()

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

        # 验证非纯黑 (背景色为26, 加上网格线应更高)
        assert frame.mean() > 5  # 不是纯黑

    def test_annotate_frame_with_detections(self):
        """测试帧标注包含检测框"""
        plugin = MockPlugin()
        service = VideoStreamService(plugin, fps=15)
        service._frame_count = 10

        frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        detections = [{
            'detection_id': 'D001',
            'bbox': {'x': 0.2, 'y': 0.3, 'width': 0.1, 'height': 0.3},
            'confidence': 0.9,
            'track_id': 'T001',
        }]

        annotated = service._annotate_frame(frame, detections)
        assert annotated is not None
        assert annotated.shape == frame.shape

        # 标注后帧应该与原帧不同 (有绘制内容)
        assert not np.array_equal(annotated, frame)

    def test_annotate_frame_empty_detections(self):
        """测试无检测结果时帧标注"""
        plugin = MockPlugin()
        service = VideoStreamService(plugin, fps=15)

        frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        annotated = service._annotate_frame(frame, [])

        # 即使无检测，信息栏应该被绘制
        assert annotated is not None
        assert not np.array_equal(annotated, frame)

    def test_stats_property(self):
        """测试统计信息"""
        plugin = MockPlugin()
        service = VideoStreamService(plugin, fps=10)

        stats = service.stats
        assert 'running' in stats
        assert 'frame_count' in stats
        assert 'detection_count' in stats
        assert 'fps' in stats
        assert 'target_fps' in stats
        assert stats['running'] is False

        service.start()
        time.sleep(0.5)

        stats = service.stats
        assert stats['running'] is True
        assert stats['frame_count'] > 0

        service.stop()

    def test_snapshot_jpeg(self):
        """测试快照 JPEG 生成"""
        plugin = MockPlugin()
        service = VideoStreamService(plugin, fps=15)

        # 未启动时应返回占位帧的 JPEG
        jpeg = service.get_snapshot_jpeg()
        if jpeg is not None:
            # 验证 JPEG 魔数
            assert jpeg[:2] == b'\xff\xd8'

        # 启动后获取快照
        service.start()
        time.sleep(0.5)

        jpeg = service.get_snapshot_jpeg()
        assert jpeg is not None
        assert len(jpeg) > 100  # JPEG 应该有一定大小
        assert jpeg[:2] == b'\xff\xd8'  # JPEG 魔数

        service.stop()

    def test_mjpeg_generator_format(self):
        """测试 MJPEG 生成器输出格式"""
        plugin = MockPlugin()
        service = VideoStreamService(plugin, fps=10)
        service.start()
        time.sleep(0.3)

        async def collect_frames():
            frames = []
            count = 0
            async for chunk in service.generate_mjpeg():
                frames.append(chunk)
                count += 1
                if count >= 2:
                    break
            return frames

        loop = asyncio.new_event_loop()
        try:
            frames = loop.run_until_complete(collect_frames())

            assert len(frames) >= 1
            # 验证 MJPEG boundary 格式
            first_frame = frames[0]
            assert b'--frame' in first_frame
            assert b'Content-Type: image/jpeg' in first_frame
        finally:
            loop.close()
            service.stop()

    def test_no_camera_adapter_fallback(self):
        """测试无相机适配器时的降级"""
        plugin = MockPlugin()
        plugin._camera_adapter = None  # 无相机
        service = VideoStreamService(plugin, fps=10)

        service.start()
        time.sleep(0.3)

        # 应该生成降级帧而非崩溃
        frame = service.get_latest_frame()
        assert frame is not None

        service.stop()


class TestCameraSimulationFrame:
    """相机模拟帧测试"""

    def test_camera_get_frame_simulated_not_black(self):
        """测试模拟模式下 get_frame() 不返回纯黑帧"""
        from plugins.indoor_fence.adapters.camera_adapter import CameraAdapter, CameraConfig

        config = CameraConfig(
            source="0",
            resolution=(640, 480),
        )
        adapter = CameraAdapter(config)
        # 手动设为模拟模式
        adapter._simulated = True

        frame = adapter.get_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        # 不能是纯黑帧
        assert frame.mean() > 5
