"""Tests for TrajectoryRL miner: pack building, validation, commitment formatting, daemon."""

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import bittensor as bt
import pytest

from trajectoryrl.base.miner import TrajectoryMiner


# ===================================================================
# Pack Building
# ===================================================================


class TestBuildPack:
    """Tests for TrajectoryMiner.build_pack()."""

    def test_minimal_pack(self):
        """Build a minimal valid pack from inline AGENTS.md content."""
        pack = TrajectoryMiner.build_pack(agents_md="# My Policy\nBe safe.")
        assert pack["schema_version"] == 1
        assert "AGENTS.md" in pack["files"]
        assert pack["files"]["AGENTS.md"] == "# My Policy\nBe safe."
        assert pack["metadata"]["pack_name"] == "my-pack"
        assert pack["metadata"]["pack_version"] == "1.0.0"
        assert pack["metadata"]["target_suite"] == "clawbench_v1"
        assert "exec" in pack["tool_policy"]["allow"]
        assert "admin_*" in pack["tool_policy"]["deny"]

    def test_with_soul_md(self):
        """Pack includes SOUL.md when provided."""
        pack = TrajectoryMiner.build_pack(
            agents_md="# Policy", soul_md="Be friendly."
        )
        assert "SOUL.md" in pack["files"]
        assert pack["files"]["SOUL.md"] == "Be friendly."

    def test_with_extra_files(self):
        """Extra files are included in the pack."""
        pack = TrajectoryMiner.build_pack(
            agents_md="# Policy",
            extra_files={"RULES.md": "Some rules"},
        )
        assert "RULES.md" in pack["files"]

    def test_custom_metadata(self):
        """Custom pack name and version."""
        pack = TrajectoryMiner.build_pack(
            agents_md="# Policy",
            pack_name="super-pack",
            pack_version="2.1.0",
        )
        assert pack["metadata"]["pack_name"] == "super-pack"
        assert pack["metadata"]["pack_version"] == "2.1.0"

    def test_custom_tool_policy(self):
        """Custom allow/deny lists."""
        pack = TrajectoryMiner.build_pack(
            agents_md="# Policy",
            tool_allow=["exec", "read"],
            tool_deny=["shell", "admin_*"],
        )
        assert pack["tool_policy"]["allow"] == ["exec", "read"]
        assert pack["tool_policy"]["deny"] == ["shell", "admin_*"]

    def test_stop_rules(self):
        """Stop rules are included when provided."""
        pack = TrajectoryMiner.build_pack(
            agents_md="# Policy",
            stop_rules=["max_tool_calls: 20"],
        )
        assert pack["stop_rules"] == ["max_tool_calls: 20"]

    def test_no_stop_rules_by_default(self):
        """No stop_rules key when not provided."""
        pack = TrajectoryMiner.build_pack(agents_md="# Policy")
        assert "stop_rules" not in pack

    def test_read_from_file(self, tmp_path):
        """Build pack from a file path."""
        agents_file = tmp_path / "AGENTS.md"
        agents_file.write_text("# File-based policy\nDo good things.")

        pack = TrajectoryMiner.build_pack(agents_md=str(agents_file))
        assert "File-based policy" in pack["files"]["AGENTS.md"]
        assert "Do good things." in pack["files"]["AGENTS.md"]


# ===================================================================
# Pack Hashing
# ===================================================================


class TestPackHash:
    """Tests for content-addressed pack hashing."""

    def test_deterministic(self):
        """Same pack always produces same hash."""
        pack = TrajectoryMiner.build_pack(agents_md="# Test")
        h1 = TrajectoryMiner.compute_pack_hash(pack)
        h2 = TrajectoryMiner.compute_pack_hash(pack)
        assert h1 == h2

    def test_correct_format(self):
        """Hash is 64 lowercase hex chars."""
        pack = TrajectoryMiner.build_pack(agents_md="# Test")
        h = TrajectoryMiner.compute_pack_hash(pack)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_content_different_hash(self):
        """Different AGENTS.md content produces different hash."""
        h1 = TrajectoryMiner.compute_pack_hash(
            TrajectoryMiner.build_pack(agents_md="# Policy A")
        )
        h2 = TrajectoryMiner.compute_pack_hash(
            TrajectoryMiner.build_pack(agents_md="# Policy B")
        )
        assert h1 != h2

    def test_matches_manual_sha256(self):
        """Hash matches manual SHA256 of canonical JSON."""
        pack = TrajectoryMiner.build_pack(agents_md="# Test")
        canonical = json.dumps(pack, sort_keys=True)
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        assert TrajectoryMiner.compute_pack_hash(pack) == expected


# ===================================================================
# Pack Save / Load
# ===================================================================


class TestPackIO:
    """Tests for save_pack and load_pack."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save then load produces identical pack."""
        pack = TrajectoryMiner.build_pack(agents_md="# My Policy")
        path = str(tmp_path / "pack.json")

        TrajectoryMiner.save_pack(pack, path)
        loaded = TrajectoryMiner.load_pack(path)

        assert loaded == pack

    def test_save_returns_hash(self, tmp_path):
        """save_pack returns the correct SHA256 hash."""
        pack = TrajectoryMiner.build_pack(agents_md="# Test")
        path = str(tmp_path / "pack.json")

        returned_hash = TrajectoryMiner.save_pack(pack, path)
        expected_hash = TrajectoryMiner.compute_pack_hash(pack)
        assert returned_hash == expected_hash

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_pack creates parent directories if needed."""
        pack = TrajectoryMiner.build_pack(agents_md="# Test")
        path = str(tmp_path / "nested" / "deep" / "pack.json")

        TrajectoryMiner.save_pack(pack, path)
        assert Path(path).exists()

    def test_load_nonexistent_raises(self):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TrajectoryMiner.load_pack("/tmp/nonexistent_pack_9999.json")


# ===================================================================
# Schema Validation
# ===================================================================


class TestMinerValidate:
    """Tests for miner-side schema validation."""

    def test_valid_pack_passes(self):
        """Standard pack passes validation."""
        pack = TrajectoryMiner.build_pack(agents_md="# Policy\nBe safe.")
        result = TrajectoryMiner.validate(pack)
        assert result.passed
        assert len(result.issues) == 0

    def test_missing_agents_md_fails(self):
        """Pack without AGENTS.md fails."""
        pack = TrajectoryMiner.build_pack(agents_md="# Policy")
        del pack["files"]["AGENTS.md"]
        result = TrajectoryMiner.validate(pack)
        assert not result.passed

    def test_too_large_fails(self):
        """Pack exceeding 32KB fails."""
        pack = TrajectoryMiner.build_pack(agents_md="x" * 40000)
        result = TrajectoryMiner.validate(pack)
        assert not result.passed
        assert any("too large" in issue.lower() or "32" in issue for issue in result.issues)

    def test_bad_semver_fails(self):
        """Invalid pack_version fails."""
        pack = TrajectoryMiner.build_pack(
            agents_md="# Policy", pack_version="not.a.version"
        )
        result = TrajectoryMiner.validate(pack)
        assert not result.passed


# ===================================================================
# Commitment Formatting
# ===================================================================


class TestCommitmentFormat:
    """Tests for format_commitment."""

    def test_basic_format(self):
        """Standard commitment formatting with HTTP URL."""
        c = TrajectoryMiner.format_commitment(
            pack_hash="a" * 64,
            pack_url="https://trajrl.com/samples/pack.json",
        )
        assert c == "a" * 64 + "|https://trajrl.com/samples/pack.json"

    def test_https_url(self):
        """HTTPS URL with path works."""
        c = TrajectoryMiner.format_commitment(
            pack_hash="a" * 64,
            pack_url="https://cdn.example.com/packs/v1/pack.json",
        )
        assert "|https://cdn.example.com/packs/v1/pack.json" in c

    def test_http_url(self):
        """Plain HTTP URL works."""
        c = TrajectoryMiner.format_commitment(
            pack_hash="a" * 64,
            pack_url="http://example.com/pack.json",
        )
        assert "|http://example.com/pack.json" in c

    def test_roundtrip_with_parser(self):
        """Commitment round-trips through parse_commitment."""
        from trajectoryrl.utils.commitments import parse_commitment

        url = "https://trajrl.com/samples/pack.json"
        c = TrajectoryMiner.format_commitment(
            pack_hash="a" * 64,
            pack_url=url,
        )
        parsed = parse_commitment(c)
        assert parsed is not None
        pack_hash, pack_url = parsed
        assert pack_hash == "a" * 64
        assert pack_url == url

    def test_invalid_pack_hash_length(self):
        """Short pack hash raises ValueError."""
        with pytest.raises(ValueError, match="64 hex chars"):
            TrajectoryMiner.format_commitment(
                pack_hash="a" * 63,
                pack_url="https://trajrl.com/samples/pack.json",
            )

    def test_invalid_url_scheme(self):
        """Non-HTTP URL raises ValueError."""
        with pytest.raises(ValueError, match="HTTP"):
            TrajectoryMiner.format_commitment(
                pack_hash="a" * 64,
                pack_url="ftp://example.com/pack.json",
            )

    def test_too_long_url_raises(self):
        """URL that exceeds 256-byte commitment limit raises ValueError."""
        long_url = "https://example.com/" + "a" * 300
        with pytest.raises(ValueError, match="too long"):
            TrajectoryMiner.format_commitment(
                pack_hash="a" * 64,
                pack_url=long_url,
            )


# ===================================================================
# Submit Workflow (mocked Bittensor)
# ===================================================================


class TestSubmitWorkflow:
    """Tests for the full submit() workflow with mocked chain."""

    def _make_miner(self):
        """Create a miner with mocked Bittensor components."""
        with patch("trajectoryrl.base.miner.bt") as mock_bt:
            mock_wallet = MagicMock()
            mock_bt.Wallet.return_value = mock_wallet

            mock_subtensor = MagicMock()
            mock_subtensor.set_commitment.return_value = True
            mock_bt.Subtensor.return_value = mock_subtensor

            miner = TrajectoryMiner(
                wallet_name="test",
                wallet_hotkey="default",
                netuid=11,
                network="test",
            )
            # Force lazy init
            miner._wallet = mock_wallet
            miner._subtensor = mock_subtensor
            return miner

    def test_submit_valid_pack(self):
        """Submit succeeds with valid pack and URL."""
        miner = self._make_miner()
        pack = TrajectoryMiner.build_pack(agents_md="# Policy\nBe safe.")
        pack_hash = TrajectoryMiner.compute_pack_hash(pack)
        pack_url = "https://trajrl.com/samples/pack.json"

        # set_commitment returns True, get_current_block returns a block number,
        # and get_current_commitment returns a commitment string that matches
        # the expected hash (for the on-chain verification step).
        miner._subtensor.set_commitment.return_value = True
        miner._subtensor.get_current_block.return_value = 12345
        commitment_str = f"{pack_hash}|{pack_url}"
        miner.get_current_commitment = MagicMock(return_value=commitment_str)

        success = miner.submit(
            pack=pack,
            pack_url=pack_url,
        )
        assert success
        miner._subtensor.set_commitment.assert_called_once()

    def test_submit_invalid_pack_fails(self):
        """Submit fails if pack doesn't pass schema validation."""
        miner = self._make_miner()
        pack = {"schema_version": 1}  # Missing required fields

        success = miner.submit(
            pack=pack,
            pack_url="https://trajrl.com/samples/pack.json",
        )
        assert not success
        miner._subtensor.set_commitment.assert_not_called()


# ===================================================================
# Status Check
# ===================================================================


class TestStatus:
    """Tests for get_current_commitment."""

    def test_no_commitment(self):
        """Returns None when no commitment exists."""
        with patch("trajectoryrl.base.miner.bt") as mock_bt:
            mock_wallet = MagicMock()
            mock_wallet.hotkey.ss58_address = "5FakeHotkey"
            mock_bt.Wallet.return_value = mock_wallet

            mock_subtensor = MagicMock()
            mock_subtensor.get_all_commitments.return_value = {}
            mock_bt.Subtensor.return_value = mock_subtensor

            miner = TrajectoryMiner()
            miner._wallet = mock_wallet
            miner._subtensor = mock_subtensor

            result = miner.get_current_commitment()
            assert result is None

    def test_existing_commitment(self):
        """Returns raw commitment string when set."""
        raw = "a" * 64 + "|https://trajrl.com/samples/pack.json"

        with patch("trajectoryrl.base.miner.bt") as mock_bt:
            mock_wallet = MagicMock()
            mock_wallet.hotkey.ss58_address = "5FakeHotkey"
            mock_bt.Wallet.return_value = mock_wallet

            mock_subtensor = MagicMock()
            mock_subtensor.get_all_commitments.return_value = {
                "5FakeHotkey": raw,
            }
            mock_bt.Subtensor.return_value = mock_subtensor

            miner = TrajectoryMiner()
            miner._wallet = mock_wallet
            miner._subtensor = mock_subtensor

            result = miner.get_current_commitment()
            assert result == raw


# ===================================================================
# CLI Entry Point
# ===================================================================


class TestMinerCLI:
    """Tests for neurons/miner.py CLI parsing."""

    def test_build_help(self):
        """Build subcommand parses without error."""
        import neurons.miner as cli
        import sys
        with patch.object(sys, "argv", ["miner.py"]):
            result = cli.main()
            assert result == 0

    def test_validate_valid_pack(self, tmp_path):
        """Validate subcommand with a valid pack."""
        import neurons.miner as cli

        pack = TrajectoryMiner.build_pack(agents_md="# Policy\nBe safe.")
        pack_path = str(tmp_path / "pack.json")
        TrajectoryMiner.save_pack(pack, pack_path)

        args = MagicMock()
        args.pack_path = pack_path

        result = cli.cmd_validate(args)
        assert result == 0

    def test_validate_invalid_pack(self, tmp_path):
        """Validate subcommand catches invalid pack."""
        import neurons.miner as cli

        bad_pack = {
            "schema_version": 1,
            "files": {},
            "tool_policy": {"allow": [], "deny": []},
            "metadata": {
                "pack_name": "bad",
                "pack_version": "1.0.0",
                "target_suite": "clawbench_v1",
            },
        }
        pack_path = str(tmp_path / "bad.json")
        with open(pack_path, "w") as f:
            json.dump(bad_pack, f)

        args = MagicMock()
        args.pack_path = pack_path

        result = cli.cmd_validate(args)
        assert result == 1


# ===================================================================
# CLI Help Output (subcommand --help)
# ===================================================================


class TestCLIHelp:
    """Verify that subcommand --help shows the correct arguments.

    Regression test: bittensor's import side-effects can hijack argparse,
    causing --help to show bittensor's logging flags instead of the
    subcommand's own arguments.
    """

    @staticmethod
    def _get_help(subcommand: str) -> str:
        """Capture --help output for a subcommand."""
        import subprocess
        cmd = [sys.executable, "-m", "neurons.miner"]
        if subcommand:
            cmd.append(subcommand)
        cmd.append("--help")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
        )
        return result.stdout + result.stderr

    def test_submit_help_shows_pack_url(self):
        """'miner.py submit --help' must show the pack_url argument."""
        output = self._get_help("submit")
        assert "pack_url" in output, (
            f"submit --help should show 'pack_url' argument, got:\n{output}"
        )

    def test_build_help_shows_agents_md(self):
        """'miner.py build --help' must show --agents-md."""
        output = self._get_help("build")
        assert "--agents-md" in output, (
            f"build --help should show '--agents-md', got:\n{output}"
        )

    def test_run_help_shows_mode(self):
        """'miner.py run --help' must show --mode."""
        output = self._get_help("run")
        assert "--mode" in output, (
            f"run --help should show '--mode', got:\n{output}"
        )

    def test_validate_help_shows_pack_path(self):
        """'miner.py validate --help' must show pack_path."""
        output = self._get_help("validate")
        assert "pack_path" in output, (
            f"validate --help should show 'pack_path', got:\n{output}"
        )

    def test_top_level_help_shows_subcommands(self):
        """'miner.py --help' must list all subcommands."""
        output = self._get_help("")
        for cmd in ("build", "validate", "submit", "run", "status"):
            assert cmd in output, (
                f"top-level --help should list '{cmd}', got:\n{output}"
            )


# ===================================================================
# Tiered Exception Handling
# ===================================================================


class TestSubmitExceptions:
    """Tests for tiered exception handling in submit_commitment/get_current_commitment."""

    def _make_miner(self):
        miner = TrajectoryMiner(
            wallet_name="test", wallet_hotkey="default",
            netuid=11, network="test",
        )
        miner._wallet = MagicMock()
        miner._subtensor = MagicMock()
        return miner

    def test_not_registered_error(self):
        """NotRegisteredError produces actionable btcli guidance."""
        miner = self._make_miner()
        miner._subtensor.set_commitment.side_effect = bt.NotRegisteredError()

        with patch("trajectoryrl.base.miner.logger") as mock_logger:
            result = miner.submit_commitment(
                "a" * 64, "https://trajrl.com/samples/pack.json"
            )
            assert result is False
            log_msg = mock_logger.error.call_args[0][0]
            assert "not registered" in log_msg.lower()
            assert "btcli subnet register" in log_msg

    def test_chain_connection_error(self):
        """ChainConnectionError produces connectivity guidance."""
        miner = self._make_miner()
        miner._subtensor.set_commitment.side_effect = bt.ChainConnectionError()

        with patch("trajectoryrl.base.miner.logger") as mock_logger:
            result = miner.submit_commitment(
                "a" * 64, "https://trajrl.com/samples/pack.json"
            )
            assert result is False
            log_msg = mock_logger.error.call_args[0][0]
            assert "connect" in log_msg.lower()

    def test_get_commitment_chain_error(self):
        """get_current_commitment handles ChainConnectionError gracefully."""
        miner = self._make_miner()
        miner._wallet.hotkey.ss58_address = "5FakeKey"
        miner._subtensor.get_all_commitments.side_effect = bt.ChainConnectionError()

        with patch("trajectoryrl.base.miner.logger") as mock_logger:
            result = miner.get_current_commitment()
            assert result is None
            log_msg = mock_logger.error.call_args[0][0]
            assert "connect" in log_msg.lower()


# ===================================================================
# Daemon Mode
# ===================================================================


class TestDaemonEntryPoint:
    """Tests for CLI entry point behavior with no subcommand."""

    def test_no_subcommand_shows_help(self, capsys):
        """No subcommand → prints help and returns 0."""
        import sys
        import neurons.miner as cli

        with patch.object(sys, "argv", ["miner.py"]):
            result = cli.main()
            assert result == 0
