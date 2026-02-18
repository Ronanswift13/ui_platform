"""
Tests for darkbreaker_sdk.utils

Validates logging, config, and model_loader utilities.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

import pytest

from darkbreaker_sdk.utils.logging import setup_plugin_logger
from darkbreaker_sdk.utils.config import load_plugin_config
from darkbreaker_sdk.utils.model_loader import load_onnx_model


class TestSetupPluginLogger:
    """Tests for setup_plugin_logger."""

    def test_returns_logger(self):
        logger = setup_plugin_logger("test-plugin-1")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "darkbreaker.plugin.test-plugin-1"

    def test_logger_level(self):
        logger = setup_plugin_logger("test-plugin-2", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_logger_has_handler(self):
        logger = setup_plugin_logger("test-plugin-3")
        assert len(logger.handlers) >= 1

    def test_duplicate_call_no_extra_handlers(self):
        logger1 = setup_plugin_logger("test-plugin-4")
        count1 = len(logger1.handlers)
        logger2 = setup_plugin_logger("test-plugin-4")
        count2 = len(logger2.handlers)
        assert count1 == count2
        assert logger1 is logger2

    def test_log_to_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            path = f.name

        try:
            logger = setup_plugin_logger("test-plugin-file", log_file=path)
            logger.info("Hello from test")
            # Flush handlers
            for h in logger.handlers:
                h.flush()
            with open(path, "r") as f:
                content = f.read()
            assert "Hello from test" in content
        finally:
            # Clean up handlers to avoid issues with other tests
            logger.handlers.clear()
            os.unlink(path)


class TestLoadPluginConfig:
    """Tests for load_plugin_config."""

    def test_missing_file_returns_empty(self):
        result = load_plugin_config("/nonexistent/path/config.json")
        assert result == {}

    def test_load_json(self):
        data = {"threshold": 0.8, "debug": True}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            result = load_plugin_config(path)
            assert result["threshold"] == 0.8
            assert result["debug"] is True
        finally:
            os.unlink(path)

    def test_load_yaml(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        data = {"model_path": "/models/test.onnx", "batch_size": 4}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            f.flush()
            path = f.name

        try:
            result = load_plugin_config(path)
            assert result["model_path"] == "/models/test.onnx"
            assert result["batch_size"] == 4
        finally:
            os.unlink(path)

    def test_non_dict_json_returns_empty(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([1, 2, 3], f)
            f.flush()
            path = f.name

        try:
            result = load_plugin_config(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_path_as_string(self):
        result = load_plugin_config("/tmp/nonexistent_config.json")
        assert result == {}


class TestLoadOnnxModel:
    """Tests for load_onnx_model."""

    def test_missing_file_returns_none(self):
        result = load_onnx_model("/nonexistent/model.onnx")
        assert result is None

    def test_missing_file_path_object(self):
        result = load_onnx_model(Path("/nonexistent/model.onnx"))
        assert result is None

    def test_default_providers(self):
        # Just verify the function handles missing files gracefully
        result = load_onnx_model("/tmp/no_model.onnx", providers=["CPUExecutionProvider"])
        assert result is None
