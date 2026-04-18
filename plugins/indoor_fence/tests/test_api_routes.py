"""Tests for simulator and training API routes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from fastapi.testclient import TestClient
from plugins.indoor_fence.standalone.simulator_routes import create_simulator_router
from plugins.indoor_fence.standalone.training_routes import create_training_router
from plugins.indoor_fence.adapters.simulator import Simulator
from plugins.indoor_fence.standalone.training_pipeline import TrainingPipeline
from fastapi import FastAPI


@pytest.fixture
def sim_client():
    app = FastAPI()
    sim = Simulator.default()
    app.include_router(create_simulator_router(sim), prefix="/api/simulator")
    return TestClient(app)


@pytest.fixture
def training_client(tmp_path):
    app = FastAPI()
    pipeline = TrainingPipeline(data_dir=str(tmp_path))
    app.include_router(create_training_router(pipeline), prefix="/api/training")
    return TestClient(app)


def test_simulator_start(sim_client):
    resp = sim_client.post("/api/simulator/start?num_persons=2&sensor_types=camera&sensor_types=lidar")
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_simulator_step(sim_client):
    sim_client.post("/api/simulator/start?num_persons=1&sensor_types=camera")
    resp = sim_client.post("/api/simulator/step")
    assert resp.status_code == 200
    assert "data" in resp.json()


def test_simulator_scenarios(sim_client):
    resp = sim_client.get("/api/simulator/scenarios")
    assert resp.status_code == 200
    assert "scenarios" in resp.json()


def test_training_status(training_client):
    resp = training_client.get("/api/training/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "idle"
