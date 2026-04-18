"""Tests for lidar adapter fallback behavior."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import logging
import time
from plugins.indoor_fence.adapters.lidar_adapter import LidarAdapter, LidarConfig


def test_lidar_serial_unavailable_fallback(caplog):
    """Test lidar adapter fallback when serial port is unavailable."""
    caplog.set_level(logging.ERROR)

    config = LidarConfig(
        protocol="rplidar",
        serial_port="/dev/ttyUSB999",  # Non-existent port
        simulate_if_unavailable=True
    )

    adapter = LidarAdapter(config)
    result = adapter.connect()

    # Should succeed with simulation mode
    assert result is True
    assert adapter.is_simulated

    # Should have logged fallback
    assert any("FALLBACK" in record.message for record in caplog.records)


def test_lidar_network_unavailable_fallback(caplog):
    """Test lidar adapter fallback when network device is unavailable."""
    caplog.set_level(logging.WARNING)

    config = LidarConfig(
        protocol="sick_tim",
        device_ip="192.168.99.99",  # Non-existent IP
        device_port=2368,
        simulate_if_unavailable=True,
        connection_timeout=1.0
    )

    adapter = LidarAdapter(config)
    result = adapter.connect()

    # Should succeed with simulation mode
    assert result is True
    assert adapter.is_simulated

    # Should have logged fallback (WARNING level for this case)
    assert any("FALLBACK" in record.message for record in caplog.records)


def test_lidar_no_fallback_when_disabled(caplog):
    """Test lidar adapter fails when fallback is disabled."""
    caplog.set_level(logging.ERROR)

    config = LidarConfig(
        protocol="rplidar",
        serial_port="/dev/ttyUSB999",
        simulate_if_unavailable=False
    )

    adapter = LidarAdapter(config)
    result = adapter.connect()

    # Should fail
    assert result is False
    assert not adapter.is_connected


def test_lidar_simulation_mode_works(caplog):
    """Test lidar adapter works in simulation mode."""
    config = LidarConfig(
        protocol="simulation",
        simulate_if_unavailable=True
    )

    adapter = LidarAdapter(config)
    result = adapter.connect()

    assert result is True
    assert adapter.is_simulated

    # Wait a bit for simulation to generate data
    time.sleep(0.2)

    # Should be able to get scan data
    scan = adapter.get_scan()
    assert scan is not None
    assert scan.num_points > 0

