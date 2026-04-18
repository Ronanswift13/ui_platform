"""
Indoor Fence V3.0 - Multi-Sensor Fusion Manager
==================================================

Manages per-track EKF instances and routes sensor data
to appropriate update methods. Handles data association.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...protocols import FusionOutput, SensorData, SensorType
from ..tracking.hungarian import linear_assignment
from .ekf_fusion import EKF6DOF
from .weight_manager import WeightManager


@dataclass
class _TrackEKF:
    """Internal track with its own EKF instance."""
    track_id: int
    ekf: EKF6DOF
    last_update_time: float = 0.0
    hit_count: int = 0
    miss_count: int = 0
    sources: List[SensorType] = field(default_factory=list)


class SensorFusionV3:
    """V3 multi-sensor fusion manager using per-track EKF."""

    def __init__(
        self,
        max_association_distance: float = 2.0,
        process_noise: float = 0.05,
    ):
        self._tracks: Dict[int, _TrackEKF] = {}
        self._next_id: int = 0
        self._max_dist = max_association_distance
        self._process_noise = process_noise
        self._weight_manager = WeightManager()
        self._last_time: Optional[float] = None

    @property
    def weight_manager(self) -> WeightManager:
        """Access weight manager for configuration."""
        return self._weight_manager

    def update(
        self, sensor_data: Dict[SensorType, SensorData]
    ) -> List[FusionOutput]:
        """Process one frame of sensor data from multiple sensors.

        Returns list of fused outputs per active track.
        """
        now = time.time()
        dt = 0.1  # Default dt
        if self._last_time is not None:
            dt = max(0.001, now - self._last_time)
        self._last_time = now

        # Predict all existing tracks
        for te in self._tracks.values():
            te.ekf.predict(dt=dt)

        # Extract observations from each sensor
        observations = self._extract_observations(sensor_data)

        # Associate observations with existing tracks
        matched, unmatched_obs = self._associate(observations)

        # Extract IMU velocity for all tracks
        imu_accel = self._extract_imu_velocity(sensor_data)

        # Update matched tracks
        sources_per_track: Dict[int, List[SensorType]] = {}
        for track_id, obs_list in matched.items():
            te = self._tracks[track_id]
            te.hit_count += 1
            te.miss_count = 0
            te.last_update_time = now
            sources_per_track[track_id] = []
            for sensor_type, position in obs_list:
                self._apply_update(te, sensor_type, position)
                sources_per_track[track_id].append(sensor_type)

            # Apply IMU velocity update to matched tracks
            if imu_accel is not None:
                vel_estimate = imu_accel * dt
                R_imu = np.diag([0.5, 0.5, 0.5])
                te.ekf.update_imu(vel_estimate, R_imu)
                if SensorType.IMU not in sources_per_track[track_id]:
                    sources_per_track[track_id].append(SensorType.IMU)

        # Create new tracks for unmatched observations
        for sensor_type, position in unmatched_obs:
            tid = self._next_id
            self._next_id += 1
            ekf = EKF6DOF(process_noise=self._process_noise)
            state = [position[0], position[1], 0.0, 0.0, 0.0, 0.0]
            if len(position) >= 3:
                state[2] = position[2]
            ekf.set_state(state)
            self._tracks[tid] = _TrackEKF(
                track_id=tid,
                ekf=ekf,
                last_update_time=now,
                hit_count=1,
                sources=[sensor_type],
            )
            sources_per_track[tid] = [sensor_type]

        # Mark missed tracks
        for tid, te in self._tracks.items():
            if tid not in matched and tid not in sources_per_track:
                te.miss_count += 1

        # Build outputs
        outputs: List[FusionOutput] = []
        for tid, te in self._tracks.items():
            state = te.ekf.get_state()
            cov = te.ekf.get_covariance()
            sources = sources_per_track.get(tid, te.sources)
            outputs.append(FusionOutput(
                track_id=tid,
                position_3d=(state[0], state[1], state[2]),
                velocity_3d=(state[3], state[4], state[5]),
                confidence=min(0.99, 0.5 + te.hit_count * 0.05),
                sources=sources,
                covariance_diag=(cov[0, 0], cov[1, 1], cov[2, 2]),
            ))

        return outputs

    def _extract_observations(
        self, sensor_data: Dict[SensorType, SensorData]
    ) -> List[Tuple[SensorType, Tuple[float, ...]]]:
        """Extract (sensor_type, position) from sensor data."""
        obs: List[Tuple[SensorType, Tuple[float, ...]]] = []

        for sensor_type, data in sensor_data.items():
            if sensor_type == SensorType.CAMERA:
                detections = data.data.get("detections", [])
                for det in detections:
                    pos = det.get("position_2d")
                    if pos is not None:
                        obs.append((SensorType.CAMERA, tuple(pos)))

            elif sensor_type == SensorType.LIDAR:
                clusters = data.data.get("clusters", [])
                for c in clusters:
                    pos = c.get("position_2d")
                    if pos is not None:
                        obs.append((SensorType.LIDAR, tuple(pos)))

            elif sensor_type == SensorType.UWB:
                positions = data.data.get("positions", [])
                for p in positions:
                    x = p.get("x", 0.0)
                    y = p.get("y", 0.0)
                    z = p.get("z", 0.0)
                    obs.append((SensorType.UWB, (x, y, z)))

            elif sensor_type == SensorType.IMU:
                # IMU provides velocity data, stored for per-track update
                # IMU is not a position source so not added to position obs
                pass

        return obs

    def _extract_imu_velocity(
        self, sensor_data: Dict[SensorType, SensorData]
    ) -> Optional[np.ndarray]:
        """Extract integrated velocity estimate from IMU acceleration data."""
        imu_data = sensor_data.get(SensorType.IMU)
        if imu_data is None:
            return None

        accel = imu_data.data.get("acceleration")
        if accel is None:
            return None

        # Simple velocity from acceleration: remove gravity, scale by dt
        ax = accel.get("x", 0.0)
        ay = accel.get("y", 0.0)
        az = accel.get("z", 9.81) - 9.81  # Remove gravity component

        # Only return meaningful velocity when acceleration exceeds noise floor
        accel_magnitude = (ax * ax + ay * ay + az * az) ** 0.5
        if accel_magnitude < 0.1:
            return None

        return np.array([ax, ay, az])

    def _associate(
        self,
        observations: List[Tuple[SensorType, Tuple[float, ...]]],
    ) -> Tuple[Dict[int, List[Tuple[SensorType, Tuple[float, ...]]]], List[Tuple[SensorType, Tuple[float, ...]]]]:
        """Hungarian optimal assignment with 3D distance."""
        matched: Dict[int, List[Tuple[SensorType, Tuple[float, ...]]]] = {}
        unmatched: List[Tuple[SensorType, Tuple[float, ...]]] = []

        if not observations or not self._tracks:
            return matched, list(observations)

        track_ids = list(self._tracks.keys())
        n_obs = len(observations)
        n_tracks = len(track_ids)

        # Build cost matrix: observations x tracks
        cost_matrix = np.zeros((n_obs, n_tracks))
        for i, (_, pos) in enumerate(observations):
            for j, tid in enumerate(track_ids):
                state = self._tracks[tid].ekf.get_state()
                dx = state[0] - pos[0]
                dy = state[1] - pos[1]
                dz = state[2] - pos[2] if len(pos) >= 3 else 0.0
                cost_matrix[i, j] = (dx * dx + dy * dy + dz * dz) ** 0.5

        # Hungarian assignment
        assignments = linear_assignment(cost_matrix)

        matched_obs: set = set()
        for obs_idx, track_idx in assignments:
            if cost_matrix[obs_idx, track_idx] <= self._max_dist:
                tid = track_ids[track_idx]
                matched.setdefault(tid, []).append(observations[obs_idx])
                matched_obs.add(obs_idx)

        for i in range(n_obs):
            if i not in matched_obs:
                unmatched.append(observations[i])

        return matched, unmatched

    def _apply_update(
        self,
        te: _TrackEKF,
        sensor_type: SensorType,
        position: Tuple[float, ...],
    ) -> None:
        """Apply sensor-specific EKF update."""
        weights = self._weight_manager.get_weights()
        w = weights.get(sensor_type)
        weight = w.weight if w else 0.5

        # Scale observation noise inversely with weight
        noise_scale = max(0.01, 1.0 / (weight + 0.001))

        if sensor_type == SensorType.CAMERA:
            z = np.array([position[0], position[1]])
            R = np.diag([0.01 * noise_scale, 0.01 * noise_scale])
            te.ekf.update_camera(z, R)

        elif sensor_type == SensorType.LIDAR:
            z = np.array([position[0], position[1]])
            R = np.diag([0.005 * noise_scale, 0.005 * noise_scale])
            te.ekf.update_lidar(z, R)

        elif sensor_type == SensorType.UWB:
            z = np.array([position[0], position[1], position[2] if len(position) >= 3 else 0.0])
            R = np.diag([0.04 * noise_scale, 0.04 * noise_scale, 0.04 * noise_scale])
            te.ekf.update_uwb(z, R)
