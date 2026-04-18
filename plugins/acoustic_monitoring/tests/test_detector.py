import numpy as np
import pytest

from plugins.acoustic_monitoring.detector import (
    AcousticDetector,
    AcousticDetectorEnhanced,
    AudioFeatureExtractor,
)
from plugins.acoustic_monitoring.plugin import Plugin


def _plugin(config=None):
    return Plugin.create_standalone(config or {})


def _sine(sample_rate=8000, duration=0.5, frequency=250.0, amplitude=0.2):
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_feature_extractor_spectrogram_shape_and_finiteness():
    extractor = AudioFeatureExtractor(sample_rate=8000, n_mels=16, n_fft=256, hop_length=64)
    audio = _sine()

    spectrogram = extractor.compute_spectrogram(audio)
    mel = extractor.compute_mel_spectrogram(audio)

    expected_frames = 1 + (len(audio) - extractor.n_fft) // extractor.hop_length
    assert spectrogram.shape == (extractor.n_fft // 2 + 1, expected_frames)
    assert mel.shape == (extractor.n_mels, expected_frames)
    assert np.isfinite(spectrogram).all()
    assert np.isfinite(mel).all()


def test_detector_rule_scores_stay_inside_quality_bounds():
    plugin = _plugin({
        "sample_rate": 8000,
        "audio_duration": 0.5,
        "n_fft": 256,
        "hop_length": 64,
    })
    detector = AcousticDetector(plugin.config)

    result = detector.detect(_sine(), sample_rate=8000)

    assert 0.0 <= result["anomaly_score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["anomaly_type"] in {"normal", *result["all_scores"].keys()}
    assert all(0.0 <= score <= 1.0 for score in result["all_scores"].values())
    assert np.isfinite(result["spectrogram"]).all()


def test_transformer_hum_threshold_comes_from_config():
    sample_rate = 8000
    t = np.arange(int(sample_rate * 0.5), dtype=np.float32) / sample_rate
    audio = (
        0.10 * np.sin(2 * np.pi * 100 * t)
        + 0.90 * np.sin(2 * np.pi * 700 * t)
    ).astype(np.float32)

    permissive = _plugin({
        "sample_rate": sample_rate,
        "detection_params": {"transformer_hum": {"ratio_threshold": 0.01}},
    })
    strict = _plugin({
        "sample_rate": sample_rate,
        "detection_params": {"transformer_hum": {"ratio_threshold": 0.80}},
    })

    permissive_score = AcousticDetector(permissive.config)._detect_transformer_hum(audio)
    strict_score = AcousticDetector(strict.config)._detect_transformer_hum(audio)

    assert permissive_score > strict_score + 0.5
    assert permissive_score == pytest.approx(1.0)
    assert 0.0 <= strict_score < 0.1


def test_enhanced_detector_uses_model_path_when_registry_is_available():
    class FakeRegistry:
        def __init__(self):
            self.calls = 0

        def is_model_loaded(self, model_id):
            return model_id == "audio_anomaly_transformer"

        def infer(self, model_id, inputs):
            self.calls += 1
            assert model_id == "audio_anomaly_transformer"
            assert inputs["input"].ndim == 4
            return {
                "success": True,
                "outputs": {
                    "anomaly_score": [[0.9]],
                    "anomaly_logits": [[0.02, 0.95, 0.01, 0.01, 0.01]],
                },
            }

    plugin = _plugin({
        "sample_rate": 8000,
        "audio_duration": 0.5,
        "n_fft": 256,
        "hop_length": 64,
        "detection_params": {"anomaly_threshold": 0.1},
    })
    detector = AcousticDetectorEnhanced(plugin.config)
    registry = FakeRegistry()
    detector.set_model_registry(registry)

    result = detector.detect(_sine(), sample_rate=8000)

    assert registry.calls == 1
    assert result["anomaly_type"] == "partial_discharge"
    assert 0.0 <= result["anomaly_score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["traditional_result"]["anomaly_type"] in {
        "normal",
        *result["traditional_result"]["all_scores"].keys(),
    }
    assert result["dl_result"]["success"] is True
