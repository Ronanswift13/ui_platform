import json
import time
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "field_like_device_replay.json"


def _materialize(sample):
    sample = json.loads(json.dumps(sample))
    metrics = sample["metrics"]
    offset = metrics.pop("last_heartbeat_offset_sec")
    metrics["last_heartbeat"] = time.time() - offset
    return sample


def test_field_like_device_replay_reaches_ticket_and_alarm_path():
    from plugins.device_monitoring.plugin import Plugin

    plugin = Plugin.create_standalone()
    samples = json.loads(FIXTURE.read_text(encoding="utf-8"))
    outputs = []

    for index, sample in enumerate(samples):
        outputs.append(
            plugin.process(
                {
                    "device_id": sample["device_id"],
                    "timestamp": 1700000000 + index * 30,
                    "readings": [_materialize(sample)],
                    "context": {
                        "task_id": "field-like-device-replay",
                        "site_id": "lab-replay",
                        "source": "field_like_fixture",
                    },
                }
            )
        )

    first, last = outputs[0], outputs[-1]

    assert first["success"] is True
    assert first["status"] == "normal"
    assert first["devices"][0]["health_index"] == 100.0
    assert last["success"] is True
    assert last["status"] == "alarm"
    assert last["label"] == "abnormal"
    assert last["devices"][0]["health_index"] == 0
    assert last["alarms"]
    assert last["maintenance_tickets"]
    assert last["maintenance_tickets"][0]["priority"] == "P1"
    assert last["metadata"]["modality"] == "device"
    assert last["metadata"]["runtime_mode"] == "standalone"
    assert last["abnormal_intervals"]
    assert "health_index" in last["abnormal_intervals"][0]["metrics"]
