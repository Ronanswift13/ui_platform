import numpy as np

from plugins.acoustic_monitoring.analyzer import AcousticAnalyzer
from plugins.acoustic_monitoring.plugin import Plugin


def _configured_analyzer(sample_rate=8000):
    plugin = Plugin.create_standalone({
        "sample_rate": sample_rate,
        "audio_duration": 0.5,
        "n_fft": 256,
        "hop_length": 64,
    })
    return AcousticAnalyzer(plugin.config)


def _audio(sample_rate=8000, duration=0.5, frequency=250.0):
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    return (0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_analyzer_returns_expected_diagnostic_sections():
    analyzer = _configured_analyzer()

    result = analyzer.analyze(_audio(), sample_rate=8000)

    assert {
        "frequency_analysis",
        "time_analysis",
        "harmonic_analysis",
        "band_analysis",
        "summary",
        "diagnosis_suggestions",
    }.issubset(result)
    assert result["summary"]
    assert result["diagnosis_suggestions"]
    assert result["frequency_analysis"]["dominant_frequencies"]
    assert result["time_analysis"]["duration_seconds"] > 0
    assert np.isfinite(result["time_analysis"]["rms_level"])


def test_analyzer_low_sample_rate_marks_only_available_bands():
    analyzer = _configured_analyzer(sample_rate=8000)

    result = analyzer.analyze(_audio(sample_rate=8000), sample_rate=8000)
    bands = result["band_analysis"]

    assert "mechanical_band" in bands
    assert "transformer_hum_band" in bands
    assert "corona_band" not in bands
    assert "partial_discharge_band" not in bands
    assert all(0.0 <= band["percentage"] <= 100.0 for band in bands.values())
