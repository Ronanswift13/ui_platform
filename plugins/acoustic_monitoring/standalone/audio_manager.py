"""Audio Session Manager - handles simulation and monitoring loops independently.

This is the core backend for the acoustic monitoring plugin's standalone dashboard.
It manages two independent async loops (simulation and monitoring), processes audio
through the plugin pipeline, and broadcasts results via WebSocket.

Can also be used headless (no WebSocket) by calling process_audio() directly.

DarkBreaker Substation Monitoring Platform
"""
import asyncio
import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SessionMode(str, Enum):
    IDLE = "idle"
    SIMULATION = "simulation"
    MONITORING = "monitoring"


class AudioSessionManager:
    """Manages independent simulation and monitoring sessions.

    Two fully independent async loops:
      - Simulation: generates synthetic audio with selectable anomaly types
      - Monitoring: processes real audio (or plugin's mock fallback)

    Both can run simultaneously without interference.
    """

    def __init__(self, plugin, ws_clients: Optional[List] = None):
        self.plugin = plugin
        self._ws_clients = ws_clients or []

        # Session states (independent)
        self._sim_running = False
        self._mon_running = False
        self._sim_task: Optional[asyncio.Task] = None
        self._mon_task: Optional[asyncio.Task] = None

        # Simulation config
        self._sim_config: Dict[str, Any] = {
            "anomaly_type": "normal",
            "anomaly_intensity": 0.7,
            "interval_sec": 2.0,
            "voltage_level": "220kV",
        }

        # Monitoring config
        self._mon_config: Dict[str, Any] = {
            "voltage_level": "220kV",
            "device_id": "mic_001",
        }

        # Stats
        self._sim_count = 0
        self._mon_count = 0
        self._last_result: Optional[Dict] = None

    # ── Simulation ───────────────────────────────────────────────

    async def start_simulation(self, config: Optional[Dict] = None) -> Dict:
        """Start the simulation loop."""
        if self._sim_running:
            return {"status": "already_running", "mode": "simulation"}

        if config:
            self._sim_config.update(config)

        self._sim_running = True
        self._sim_count = 0
        self._sim_task = asyncio.create_task(self._simulation_loop())
        logger.info(f"Simulation started: {self._sim_config}")
        return {"status": "started", "mode": "simulation", "config": self._sim_config}

    async def stop_simulation(self) -> Dict:
        """Stop the simulation loop."""
        self._sim_running = False
        if self._sim_task and not self._sim_task.done():
            self._sim_task.cancel()
            try:
                await self._sim_task
            except asyncio.CancelledError:
                pass
        self._sim_task = None
        logger.info("Simulation stopped")
        return {"status": "stopped", "mode": "simulation", "cycles": self._sim_count}

    async def _simulation_loop(self):
        """Background loop: generate synthetic audio → process → broadcast."""
        try:
            while self._sim_running:
                audio = self._generate_simulation_audio()
                result = self.plugin.process({
                    "audio": audio,
                    "sample_rate": self.plugin.config.sample_rate,
                    "device_id": f"sim_{self._sim_config['voltage_level']}",
                })

                # Add visualization data
                result["waveform"] = self._downsample_for_viz(audio, max_points=500)
                result["mode"] = "simulation"
                result["cycle"] = self._sim_count

                self._last_result = result
                self._sim_count += 1

                await self._broadcast_result(result)
                await asyncio.sleep(self._sim_config["interval_sec"])
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Simulation loop error: {e}")
            self._sim_running = False

    def _generate_simulation_audio(self) -> np.ndarray:
        """委托给插件的多模态模拟器, 避免代码重复。"""
        return self.plugin._generate_mock_audio(
            self.plugin.config.sample_rate,
            self._sim_config["anomaly_type"],
            float(self._sim_config.get("anomaly_intensity", 0.7))
        )

    # ── Monitoring ───────────────────────────────────────────────

    async def start_monitoring(self, config: Optional[Dict] = None) -> Dict:
        """Start the monitoring loop (processes real audio or plugin's mock fallback)."""
        if self._mon_running:
            return {"status": "already_running", "mode": "monitoring"}

        if config:
            self._mon_config.update(config)

        self._mon_running = True
        self._mon_count = 0
        self._mon_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Monitoring started: {self._mon_config}")
        return {"status": "started", "mode": "monitoring", "config": self._mon_config}

    async def stop_monitoring(self) -> Dict:
        """Stop the monitoring loop."""
        self._mon_running = False
        if self._mon_task and not self._mon_task.done():
            self._mon_task.cancel()
            try:
                await self._mon_task
            except asyncio.CancelledError:
                pass
        self._mon_task = None
        logger.info("Monitoring stopped")
        return {"status": "stopped", "mode": "monitoring", "cycles": self._mon_count}

    async def _monitoring_loop(self):
        """Background loop: process audio from real sources or mock fallback."""
        try:
            while self._mon_running:
                # process() with no audio triggers mock audio generation inside plugin
                result = self.plugin.process({
                    "sample_rate": self.plugin.config.sample_rate,
                    "device_id": self._mon_config.get("device_id", "monitor"),
                })

                result["mode"] = "monitoring"
                result["cycle"] = self._mon_count

                self._last_result = result
                self._mon_count += 1

                await self._broadcast_result(result)
                await asyncio.sleep(self.plugin.config.audio_duration)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            self._mon_running = False

    # ── One-shot processing (headless / API) ─────────────────────

    def process_audio(self, audio: np.ndarray, sample_rate: int = 16000,
                      device_id: str = "api") -> Dict:
        """Process a single audio buffer. Works headless without WebSocket."""
        result = self.plugin.process({
            "audio": audio,
            "sample_rate": sample_rate,
            "device_id": device_id,
        })
        result["waveform"] = self._downsample_for_viz(audio, max_points=500)
        self._last_result = result
        return result

    # ── Status ───────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return current status of both sessions."""
        return {
            "simulation": {
                "running": self._sim_running,
                "cycles": self._sim_count,
                "config": self._sim_config,
            },
            "monitoring": {
                "running": self._mon_running,
                "cycles": self._mon_count,
                "config": self._mon_config,
            },
            "last_result": self._last_result,
        }

    # ── Broadcast ────────────────────────────────────────────────

    async def _broadcast_result(self, result: Dict):
        """Send result to all connected WebSocket clients.

        Uses custom message type 'acoustic_result' (not runner's hardcoded 'detection').
        """
        if not self._ws_clients:
            return
        msg = json.dumps({"type": "acoustic_result", "data": result}, default=str)
        disconnected = []
        for ws in self._ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            if ws in self._ws_clients:
                self._ws_clients.remove(ws)

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _downsample_for_viz(audio: np.ndarray, max_points: int = 500) -> List[float]:
        """Downsample audio array for efficient WebSocket transfer."""
        if len(audio) <= max_points:
            return audio.tolist()
        step = len(audio) // max_points
        return audio[::step][:max_points].tolist()
