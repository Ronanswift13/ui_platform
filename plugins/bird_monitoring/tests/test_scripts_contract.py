"""scripts/ 目录契约测试 — 入口存在、可执行、基本退出码符合预期"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PLUGIN_DIR / "scripts"


SHELL_SCRIPTS = [
    "run_targeted_tests.sh",
    "run_quality_gate.sh",
    "run_regression_tests.sh",
]

PYTHON_SCRIPTS = [
    "check_real_model.py",
    "validate_fixture.py",
]


class TestScriptsLayout:
    def test_scripts_dir_exists(self):
        assert SCRIPTS_DIR.is_dir(), "scripts/ 目录缺失"

    def test_coveragerc_exists_with_fail_under(self):
        rc = PLUGIN_DIR / ".coveragerc"
        assert rc.exists(), ".coveragerc 缺失"
        text = rc.read_text(encoding="utf-8")
        assert "fail_under" in text
        # 与 08_task_routing.md 的覆盖率门槛保持一致
        assert "fail_under = 60" in text

    @pytest.mark.parametrize("name", SHELL_SCRIPTS)
    def test_shell_script_executable_and_shebang(self, name):
        path = SCRIPTS_DIR / name
        assert path.exists(), f"{name} 缺失"
        mode = path.stat().st_mode
        assert mode & stat.S_IXUSR, f"{name} 缺少 +x"
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#!"), f"{name} 缺少 shebang"
        assert "bash" in first_line, f"{name} 必须是 bash 脚本"

    @pytest.mark.parametrize("name", PYTHON_SCRIPTS)
    def test_python_script_executable_and_shebang(self, name):
        path = SCRIPTS_DIR / name
        assert path.exists(), f"{name} 缺失"
        mode = path.stat().st_mode
        assert mode & stat.S_IXUSR, f"{name} 缺少 +x"
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#!"), f"{name} 缺少 shebang"
        assert "python" in first_line, f"{name} 必须是 python 脚本"


class TestRunTargetedTestsHelp:
    def test_unknown_layer_exits_2(self):
        result = subprocess.run(
            [str(SCRIPTS_DIR / "run_targeted_tests.sh"), "bogus"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "unknown layer" in result.stderr


class TestCheckRealModelGating:
    """`check_real_model.py` 不得伪造闭环；面对缺失模型只能 exit 2。"""

    def test_missing_model_path_exits_2(self, tmp_path):
        bogus = tmp_path / "nonexistent_bird.onnx"
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "check_real_model.py"), str(bogus)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, result.stderr
        assert "模型文件不存在" in result.stderr

    def test_missing_arg_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "check_real_model.py")],
            capture_output=True,
            text=True,
        )
        # argparse 缺参 → 退出码 2
        assert result.returncode != 0


class TestValidateFixtureGating:
    """`validate_fixture.py` 不得把 placeholder 槽位伪装成 collected。"""

    def test_unknown_sample_id_exits_2(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "validate_fixture.py"),
                "--sample-id",
                "nonexistent_sample_xyz",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2

    def test_planned_slot_reports_pending_not_pass(self):
        """槽位仍 planned/blocked 时，必须报 slot_pending 且 exit 1。"""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "validate_fixture.py"),
                "--sample-id",
                "bird_no_bird_001",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        # exit 1 因为该槽位仍 planned，文件不存在 → slot_pending issue
        assert result.returncode == 1, result.stdout + result.stderr
        assert "slot_pending" in result.stdout


class TestRoutingDocumentsScripts:
    """脚本必须在 08_task_routing.md 注册，避免成为孤儿文件。"""

    def test_routing_lists_all_scripts(self):
        routing = PLUGIN_DIR / ".agent_skills" / "08_task_routing.md"
        text = routing.read_text(encoding="utf-8")
        for name in SHELL_SCRIPTS + PYTHON_SCRIPTS:
            assert name in text, f"{name} 未在 08_task_routing.md 注册"
