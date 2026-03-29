"""Tests for ClawBench version checks in the validator."""

from pathlib import Path
from unittest.mock import patch

import pytest

from trajectoryrl.base.validator import TrajectoryValidator


@pytest.fixture(autouse=True)
def _capture_logs(caplog):
    with caplog.at_level("INFO"):
        yield


class TestSubmoduleStatus:

    @patch("subprocess.check_output", return_value=" abc123 clawbench (v1.0)\n def456 openclaw (v2.0)\n")
    def test_all_in_sync(self, mock_sub, caplog):
        TrajectoryValidator._check_submodule_status(None)
        assert "out of sync" not in caplog.text
        assert "not initialized" not in caplog.text

    @patch("subprocess.check_output", return_value="+abc123 clawbench (v1.0)\n def456 openclaw (v2.0)\n")
    def test_out_of_sync(self, mock_sub, caplog):
        TrajectoryValidator._check_submodule_status(None)
        assert "Submodule 'clawbench' is out of sync" in caplog.text
        assert "openclaw" not in caplog.text

    @patch("subprocess.check_output", return_value="-abc123 clawbench\n")
    def test_not_initialized(self, mock_sub, caplog):
        TrajectoryValidator._check_submodule_status(None)
        assert "Submodule 'clawbench' is not initialized" in caplog.text

    @patch("subprocess.check_output", return_value="+aaa clawbench\n+bbb openclaw\n")
    def test_multiple_out_of_sync(self, mock_sub, caplog):
        TrajectoryValidator._check_submodule_status(None)
        assert "clawbench" in caplog.text
        assert "openclaw" in caplog.text

    @patch("subprocess.check_output", side_effect=FileNotFoundError("git not found"))
    def test_git_unavailable(self, mock_sub, caplog):
        TrajectoryValidator._check_submodule_status(None)
        assert "out of sync" not in caplog.text  # silently skips


class TestGetClawbenchSha:

    @patch("subprocess.check_output", return_value="abc123def456\n")
    def test_returns_sha(self, mock_sub):
        result = TrajectoryValidator._get_clawbench_sha(Path("/some/path"))
        assert result == "abc123def456"

    @patch("subprocess.check_output", side_effect=FileNotFoundError)
    def test_returns_none_on_error(self, mock_sub):
        result = TrajectoryValidator._get_clawbench_sha(Path("/bad/path"))
        assert result is None
