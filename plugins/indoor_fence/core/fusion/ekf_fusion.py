"""
Indoor Fence V3.0 - Extended Kalman Filter (6-DOF)
====================================================

Six-state EKF [x, y, z, vx, vy, vz] for multi-sensor position fusion.
Supports camera (2D), LiDAR (2D), UWB (3D), IMU (velocity) updates.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class EKF6DOF:
    """Extended Kalman Filter with 6-DOF state: [x, y, z, vx, vy, vz]."""

    def __init__(
        self,
        process_noise: float = 0.01,
        initial_covariance: float = 1.0,
    ):
        self._n = 6
        self._x = np.zeros(self._n)  # State vector
        self._P = np.eye(self._n) * initial_covariance  # Covariance
        self._q = process_noise  # Process noise scalar

    def get_state(self) -> List[float]:
        """Get current state as list [x, y, z, vx, vy, vz]."""
        return self._x.tolist()

    def set_state(self, state: List[float]) -> None:
        """Set state directly."""
        assert len(state) == self._n
        self._x = np.array(state, dtype=float)

    def get_covariance(self) -> np.ndarray:
        """Get current covariance matrix (6x6)."""
        return self._P.copy()

    def predict(self, dt: float) -> None:
        """Predict step using constant-velocity model.

        State transition: x_new = x + vx*dt, etc.
        """
        # State transition matrix F
        F = np.eye(self._n)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt

        # Process noise Q (discrete white noise acceleration model)
        q = self._q
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        # Block diagonal noise for position-velocity pairs
        Q = np.zeros((self._n, self._n))
        for i in range(3):
            Q[i, i] = q * dt3 / 3.0
            Q[i, i + 3] = q * dt2 / 2.0
            Q[i + 3, i] = q * dt2 / 2.0
            Q[i + 3, i + 3] = q * dt

        # Predict
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Generic Kalman update step."""
        y = z - H @ self._x  # Innovation
        S = H @ self._P @ H.T + R  # Innovation covariance
        K = self._P @ H.T @ np.linalg.inv(S)  # Kalman gain

        self._x = self._x + K @ y
        I = np.eye(self._n)
        # Joseph form for numerical stability
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T

    def update_camera(self, measurement: np.ndarray, R: np.ndarray) -> None:
        """Update with camera observation [x, y]."""
        H = np.zeros((2, self._n))
        H[0, 0] = 1.0  # Observes x
        H[1, 1] = 1.0  # Observes y
        self._update(measurement, H, R)

    def update_lidar(self, measurement: np.ndarray, R: np.ndarray) -> None:
        """Update with LiDAR observation [x, y]."""
        H = np.zeros((2, self._n))
        H[0, 0] = 1.0  # Observes x
        H[1, 1] = 1.0  # Observes y
        self._update(measurement, H, R)

    def update_uwb(self, measurement: np.ndarray, R: np.ndarray) -> None:
        """Update with UWB observation [x, y, z]."""
        H = np.zeros((3, self._n))
        H[0, 0] = 1.0  # Observes x
        H[1, 1] = 1.0  # Observes y
        H[2, 2] = 1.0  # Observes z
        self._update(measurement, H, R)

    def update_imu(self, measurement: np.ndarray, R: np.ndarray) -> None:
        """Update with IMU velocity observation [vx, vy, vz]."""
        H = np.zeros((3, self._n))
        H[0, 3] = 1.0  # Observes vx
        H[1, 4] = 1.0  # Observes vy
        H[2, 5] = 1.0  # Observes vz
        self._update(measurement, H, R)
