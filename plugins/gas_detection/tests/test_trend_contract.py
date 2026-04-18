def create_plugin(config=None):
    from plugins.gas_detection.plugin import Plugin

    return Plugin.create_standalone(config)


def _sample(index, *, device_id="gas_trend_sensor"):
    return {
        "device_id": device_id,
        "timestamp": 1700000000 + index * 3600,
        "readings": {
            "SF6": 100.0 + index,
            "H2": 20.0 + 0.2 * index,
            "CO": 80.0,
            "CH4": 30.0,
            "C2H4": 10.0,
            "C2H6": 8.0,
            "C2H2": 0.5,
        },
    }


def test_24_samples_enter_predictor_and_analyzer_contract():
    plugin = create_plugin()
    if plugin._predictor is not None:
        plugin._predictor._dl_initialized = False
        plugin._predictor._gl_translstm = None

    for index in range(24):
        result = plugin.process(_sample(index))

    assert result["success"] is True
    assert result["predictions"]["available"] is True
    assert result["predictions"]["method"] == "traditional"
    assert result["predictions"]["prediction_horizon"] == 24
    assert "SF6" in result["predictions"]["gas_predictions"]
    assert result["trend_analysis"]["available"] is True
    assert result["trend_analysis"]["source"] == "GasDataAnalyzer.analyze_trends"
    assert result["trend_diagnosis"]["available"] is True
    assert result["trend_diagnosis"]["prediction"] == result["predictions"]
    assert result["trend_diagnosis"]["analysis"] == result["trend_analysis"]
    assert result["metadata"]["trend_prediction_available"] is True
    assert result["metadata"]["algorithm_stage"] == "threshold_rules_with_predictor_and_analyzer"


def test_plugin_main_chain_calls_predictor_and_analyzer_objects():
    class SpyPredictor:
        def __init__(self):
            self.called = False

        def predict(self, history):
            self.called = True
            assert len(history["timestamps"]) == 24
            return {
                "available": True,
                "success": True,
                "method": "spy_predictor",
                "prediction_horizon": 24,
                "gas_predictions": {
                    "SF6": {
                        "values": [140.0],
                        "time_steps": [1],
                        "unit": "ppm",
                        "trend_slope": 1.0,
                    }
                },
                "predicted_alarms": [],
            }

    class SpyAnalyzer:
        def __init__(self):
            self.called = False

        def analyze_trends(self, history, current_readings):
            self.called = True
            assert len(history["timestamps"]) == 24
            assert current_readings["SF6"] == 123.0
            return {
                "source": "spy_analyzer",
                "gas_trends": {
                    "SF6": {
                        "pattern": "rapid_increase",
                        "severity": "warning",
                        "is_abnormal": True,
                    }
                },
                "abnormal_trends": [{"gas": "SF6", "severity": "warning"}],
                "rate_of_change": {"SF6": {"hourly": 1.0}},
                "statistical_summary": {"SF6": {"current": 123.0}},
            }

    plugin = create_plugin()
    predictor = SpyPredictor()
    analyzer = SpyAnalyzer()
    plugin._predictor = predictor
    plugin._analyzer = analyzer

    for index in range(24):
        result = plugin.process({
            "device_id": "gas_spy_sensor",
            "timestamp": 1700000000 + index * 3600,
            "readings": {"SF6": 100.0 + index},
        })

    assert predictor.called is True
    assert analyzer.called is True
    assert result["success"] is True
    assert result["predictions"]["method"] == "spy_predictor"
    assert result["trend_analysis"]["source"] == "spy_analyzer"
    assert result["trend_diagnosis"]["available"] is True
