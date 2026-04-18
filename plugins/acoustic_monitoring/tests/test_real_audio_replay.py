import wave

import numpy as np
import pytest

from plugins.acoustic_monitoring.plugin import Plugin


def _write_wav_fixture(path, sample_rate=16000):
    duration = 1.0
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    audio = (
        0.18 * np.sin(2 * np.pi * 250 * t)
        + 0.04 * np.sin(2 * np.pi * 1000 * t)
    ).astype(np.float32)
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _read_wav_fixture(path):
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return sample_rate, audio


def test_wav_container_replay_produces_stable_standard_output(tmp_path):
    wav_path = tmp_path / "substation_audio_replay.wav"
    _write_wav_fixture(wav_path)
    sample_rate, audio = _read_wav_fixture(wav_path)
    plugin = Plugin.create_standalone({
        "sample_rate": sample_rate,
        "audio_duration": 1.0,
        "n_fft": 1024,
        "hop_length": 256,
    })
    payload = {
        "device_id": "acoustic_replay_channel",
        "timestamp": "2026-01-01T00:00:00Z",
        "audio_buffer": audio.tolist(),
        "sample_rate": sample_rate,
        "data_source": "wav_fixture_replay",
        "context": {"task_id": "wav-replay", "site_id": "lab-fixture"},
    }

    first = plugin.process(payload)
    second = plugin.process(payload)

    assert first["success"] is True
    assert first["metadata"]["modality"] == "acoustic"
    assert first["input_protocol"]["observed_input_type"] == "audio_buffer"
    assert first["input_protocol"]["data_quality"]["status"] == "ok"
    assert first["frequency_analysis"]["dominant_frequency"] == pytest.approx(250.0, abs=16.0)
    assert second["status"] == first["status"]
    assert second["anomaly_type"] == first["anomaly_type"]
    assert second["frequency_analysis"]["dominant_frequency"] == pytest.approx(
        first["frequency_analysis"]["dominant_frequency"],
        abs=1e-6,
    )
