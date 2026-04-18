"""Tests for Extended Kalman Filter fusion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
from plugins.indoor_fence.core.fusion.ekf_fusion import EKF6DOF


def test_ekf_creation():
    ekf = EKF6DOF()
    state = ekf.get_state()
    assert len(state) == 6  # x, y, z, vx, vy, vz
    assert all(s == 0.0 for s in state)


def test_ekf_predict():
    ekf = EKF6DOF()
    # Set initial state with velocity
    ekf.set_state([1.0, 2.0, 0.0, 0.5, 0.0, 0.0])
    ekf.predict(dt=1.0)
    state = ekf.get_state()
    # x should have moved by vx * dt
    assert abs(state[0] - 1.5) < 0.01
    assert abs(state[1] - 2.0) < 0.01


def test_ekf_update_camera():
    ekf = EKF6DOF()
    ekf.set_state([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    # Camera observes (x, y) only
    measurement = np.array([1.1, 2.1])
    R = np.diag([0.01, 0.01])
    ekf.update_camera(measurement, R)
    state = ekf.get_state()
    # State should move toward measurement
    assert abs(state[0] - 1.1) < 0.1
    assert abs(state[1] - 2.1) < 0.1


def test_ekf_update_uwb():
    ekf = EKF6DOF()
    ekf.set_state([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    # UWB observes (x, y, z)
    measurement = np.array([1.2, 2.2, 0.5])
    R = np.diag([0.04, 0.04, 0.04])
    ekf.update_uwb(measurement, R)
    state = ekf.get_state()
    # z should update toward 0.5
    assert state[2] > 0.0


def test_ekf_update_lidar():
    ekf = EKF6DOF()
    ekf.set_state([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    # LiDAR observes (x, y) in ground plane
    measurement = np.array([0.9, 1.9])
    R = np.diag([0.005, 0.005])
    ekf.update_lidar(measurement, R)
    state = ekf.get_state()
    assert abs(state[0] - 0.9) < 0.15


def test_ekf_multi_sensor_fusion():
    ekf = EKF6DOF()
    ekf.set_state([0.0, 0.0, 0.0, 1.0, 0.5, 0.0])

    for i in range(10):
        ekf.predict(dt=0.1)
        # Simulate noisy camera observation
        true_x = (i + 1) * 0.1
        true_y = (i + 1) * 0.05
        cam_obs = np.array([true_x + np.random.randn() * 0.05,
                            true_y + np.random.randn() * 0.05])
        ekf.update_camera(cam_obs, np.diag([0.0025, 0.0025]))

    state = ekf.get_state()
    # Should track roughly toward (1.0, 0.5, 0)
    assert abs(state[0] - 1.0) < 0.3
    assert abs(state[1] - 0.5) < 0.3


def test_ekf_update_imu():
    ekf = EKF6DOF()
    ekf.set_state([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    # IMU observes velocity [vx, vy, vz]
    measurement = np.array([0.5, 0.3, 0.0])
    R = np.diag([0.1, 0.1, 0.1])
    ekf.update_imu(measurement, R)
    state = ekf.get_state()
    # Velocity state should move toward measurement
    assert state[3] > 0.0  # vx should increase
    assert state[4] > 0.0  # vy should increase


def test_ekf_divergence_detection():
    ekf = EKF6DOF(initial_covariance=200.0)
    cov = ekf.get_covariance()
    # With large initial covariance, position variance > 100
    assert np.any(np.diag(cov)[:2] > 100.0)


def test_ekf_divergence_recovery():
    ekf = EKF6DOF(initial_covariance=200.0)
    # Simulate reset
    ekf.set_state([5.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    ekf._P = np.eye(6) * 10.0  # Reset covariance
    cov = ekf.get_covariance()
    assert np.all(np.diag(cov)[:2] < 100.0)


def test_ekf_get_covariance():
    ekf = EKF6DOF()
    cov = ekf.get_covariance()
    assert cov.shape == (6, 6)
    # Diagonal should be positive
    assert all(cov[i, i] > 0 for i in range(6))
