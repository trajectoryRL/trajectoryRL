"""Tests for TrajectoryRL miner: S1 pack building, validation, commitment formatting."""

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
# Pack Building (Season 1)
# ===================================================================


class TestBuildS1Pack:
    """Tests for TrajectoryMiner.build_s1_pack()."""

    def test_minimal_pack(self):
        """Build a minimal valid pack from SKILL.md content."""
        pack = TrajectoryMiner.build_s1_pack("# My Skill\nBe effective.")
        assert pack["schema_version"] == 1
        assert "SKILL.md" in pack["files"]
        assert pack["files"]["SKILL.md"] == "# My Skill\nBe effective."

    def test_only_schema_version_and_files(self):
        """S1 pack contains only schema_version and files."""
        pack = TrajectoryMiner.build_s1_pack("# Skill")
        assert set(pack.keys()) == {"schema_version", "files"}
        assert set(pack["files"].keys()) == {"SKILL.md"}


# ===================================================================
# Pack Hashing
# ===================================================================


class TestPackHash:
    """Tests for content-addressed pack hashing."""

    def test_deterministic(self):
        """Same pack always produces same hash."""
        pack = TrajectoryMiner.build_s1_pack("# Test")
        h1 = TrajectoryMiner.compute_pack_hash(pack)
        h2 = TrajectoryMiner.compute_pack_hash(pack)
        assert h1 == h2

    def test_correct_format(self):
        """Hash is 64 lowercase hex chars."""
        pack = TrajectoryMiner.build_s1_pack("# Test")
        h = TrajectoryMiner.compute_pack_hash(pack)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_content_different_hash(self):
        """Different SKILL.md content produces different hash."""
        h1 = TrajectoryMiner.compute_pack_hash(
            TrajectoryMiner.build_s1_pack("# Skill A")
        )
        h2 = TrajectoryMiner.compute_pack_hash(
            TrajectoryMiner.build_s1_pack("# Skill B")
        )
        assert h1 != h2

    def test_matches_manual_sha256(self):
        """Hash matches manual SHA256 of canonical JSON."""
        pack = TrajectoryMiner.build_s1_pack("# Test")
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
        pack = TrajectoryMiner.build_s1_pack("# My Skill")
        path = str(tmp_path / "pack.json")

        TrajectoryMiner.save_pack(pack, path)
        loaded = TrajectoryMiner.load_pack(path)

        assert loaded == pack

    def test_save_returns_hash(self, tmp_path):
        """save_pack returns the correct SHA256 hash."""
        pack = TrajectoryMiner.build_s1_pack("# Test")
        path = str(tmp_path / "pack.json")

        returned_hash = TrajectoryMiner.save_pack(pack, path)
        expected_hash = TrajectoryMiner.compute_pack_hash(pack)
        assert returned_hash == expected_hash

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_pack creates parent directories if needed."""
        pack = TrajectoryMiner.build_s1_pack("# Test")
        path = str(tmp_path / "nested" / "deep" / "pack.json")

        TrajectoryMiner.save_pack(pack, path)
        assert Path(path).exists()

    def test_load_nonexistent_raises(self):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TrajectoryMiner.load_pack("/tmp/nonexistent_pack_9999.json")


# ===================================================================
# Schema Validation (Season 1)
# ===================================================================


class TestValidateS1:
    """Tests for S1 schema validation."""

    def test_valid_pack_passes(self):
        """Standard S1 pack passes validation."""
        pack = TrajectoryMiner.build_s1_pack("# Skill\nBe effective.")
        issues = TrajectoryMiner.validate_s1(pack)
        assert issues == []

    def test_missing_skill_md_fails(self):
        """Pack without SKILL.md fails."""
        pack = {"schema_version": 1, "files": {}}
        issues = TrajectoryMiner.validate_s1(pack)
        assert any("SKILL.md" in i for i in issues)

    def test_empty_skill_md_fails(self):
        """Pack with empty SKILL.md fails."""
        pack = {"schema_version": 1, "files": {"SKILL.md": "   "}}
        issues = TrajectoryMiner.validate_s1(pack)
        assert any("empty" in i for i in issues)

    def test_too_large_fails(self):
        """Pack exceeding 32KB fails."""
        pack = TrajectoryMiner.build_s1_pack("x" * 40000)
        issues = TrajectoryMiner.validate_s1(pack)
        assert any("size" in i or "limit" in i for i in issues)

    def test_wrong_schema_version_fails(self):
        """Wrong schema_version fails."""
        pack = {"schema_version": 2, "files": {"SKILL.md": "# Skill"}}
        issues = TrajectoryMiner.validate_s1(pack)
        assert any("schema_version" in i for i in issues)

    def test_missing_files_dict_fails(self):
        """Missing files dict fails."""
        pack = {"schema_version": 1}
        issues = TrajectoryMiner.validate_s1(pack)
        assert any("files" in i for i in issues)


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
        """URL that exceeds 128-byte commitment limit raises ValueError."""
        long_url = "https://example.com/" + "a" * 300
        with pytest.raises(ValueError, match="too long"):
            TrajectoryMiner.format_commitment(
                pack_hash="a" * 64,
                pack_url=long_url,
            )


# ===================================================================
# Submit Commitment (mocked Bittensor)
# ===================================================================


class TestSubmitCommitment:
    """Tests for submit_commitment with mocked chain."""

    def _make_miner(self):
        """Create a miner with mocked Bittensor components."""
        miner = TrajectoryMiner(
            wallet_name="test",
            wallet_hotkey="default",
            netuid=11,
            network="test",
        )
        miner._wallet = MagicMock()
        miner._subtensor = MagicMock()
        return miner

    def test_submit_success(self):
        """submit_commitment succeeds when chain accepts."""
        miner = self._make_miner()
        pack_hash = "a" * 64
        pack_url = "https://trajrl.com/samples/pack.json"

        miner._subtensor.set_commitment.return_value = True
        miner._subtensor.get_current_block.return_value = 12345
        commitment_str = f"{pack_hash}|{pack_url}"
        miner.get_current_commitment = MagicMock(return_value=commitment_str)

        success = miner.submit_commitment(pack_hash, pack_url)
        assert success
        miner._subtensor.set_commitment.assert_called_once()

    def test_not_registered_error(self):
        """NotRegisteredError returns False with actionable message."""
        miner = self._make_miner()
        miner._subtensor.set_commitment.side_effect = bt.NotRegisteredError()

        result = miner.submit_commitment(
            "a" * 64, "https://trajrl.com/samples/pack.json"
        )
        assert result is False

    def test_chain_connection_error(self):
        """ChainConnectionError returns False."""
        miner = self._make_miner()
        miner._subtensor.set_commitment.side_effect = bt.ChainConnectionError()

        result = miner.submit_commitment(
            "a" * 64, "https://trajrl.com/samples/pack.json"
        )
        assert result is False


# ===================================================================
# Status Check
# ===================================================================


class TestStatus:
    """Tests for get_current_commitment."""

    def test_no_commitment(self):
        """Returns None when no commitment exists."""
        miner = TrajectoryMiner()
        miner._wallet = MagicMock()
        miner._wallet.hotkey.ss58_address = "5FakeHotkey"
        miner._subtensor = MagicMock()
        miner._subtensor.get_all_commitments.return_value = {}

        result = miner.get_current_commitment()
        assert result is None

    def test_existing_commitment(self):
        """Returns raw commitment string when set."""
        raw = "a" * 64 + "|https://trajrl.com/samples/pack.json"

        miner = TrajectoryMiner()
        miner._wallet = MagicMock()
        miner._wallet.hotkey.ss58_address = "5FakeHotkey"
        miner._subtensor = MagicMock()
        miner._subtensor.get_all_commitments.return_value = {"5FakeHotkey": raw}

        result = miner.get_current_commitment()
        assert result == raw


# ===================================================================
# CLI Entry Point
# ===================================================================


class TestMinerCLI:
    """Tests for neurons/miner.py CLI."""

    def test_no_subcommand_shows_help(self, capsys):
        """No subcommand prints help and returns 0."""
        import neurons.miner as cli

        with patch.object(sys, "argv", ["miner.py"]):
            result = cli.main()
            assert result == 0

    def test_build_valid_skill(self, tmp_path):
        """Build subcommand with valid SKILL.md."""
        import neurons.miner as cli

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\nBe effective.")

        args = MagicMock()
        args.skill_md = str(skill_file)
        args.output = str(tmp_path / "pack.json")

        result = cli.cmd_build(args)
        assert result == 0
        assert (tmp_path / "pack.json").exists()

    def test_build_empty_skill_fails(self, tmp_path):
        """Build subcommand fails with empty SKILL.md."""
        import neurons.miner as cli

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("")

        args = MagicMock()
        args.skill_md = str(skill_file)
        args.output = str(tmp_path / "pack.json")

        result = cli.cmd_build(args)
        assert result == 1

    def test_validate_valid_pack(self, tmp_path):
        """Validate subcommand with a valid pack."""
        import neurons.miner as cli

        pack = TrajectoryMiner.build_s1_pack("# Skill\nBe effective.")
        pack_path = str(tmp_path / "pack.json")
        TrajectoryMiner.save_pack(pack, pack_path)

        args = MagicMock()
        args.pack_path = pack_path

        result = cli.cmd_validate(args)
        assert result == 0

    def test_validate_invalid_pack(self, tmp_path):
        """Validate subcommand catches invalid pack."""
        import neurons.miner as cli

        bad_pack = {"schema_version": 1, "files": {}}
        pack_path = str(tmp_path / "bad.json")
        with open(pack_path, "w") as f:
            json.dump(bad_pack, f)

        args = MagicMock()
        args.pack_path = pack_path

        result = cli.cmd_validate(args)
        assert result == 1


# ===================================================================
# CLI Help Output
# ===================================================================


class TestCLIHelp:
    """Verify that subcommand --help shows the correct arguments."""

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

    def test_build_help_shows_skill_md(self):
        """'miner.py build --help' must show skill_md argument."""
        output = self._get_help("build")
        assert "skill_md" in output, (
            f"build --help should show 'skill_md' argument, got:\n{output}"
        )

    def test_submit_help_shows_pack_url(self):
        """'miner.py submit --help' must show the pack_url argument."""
        output = self._get_help("submit")
        assert "pack_url" in output, (
            f"submit --help should show 'pack_url' argument, got:\n{output}"
        )

    def test_upload_help_shows_pack_path(self):
        """'miner.py upload --help' must show pack_path argument."""
        output = self._get_help("upload")
        assert "pack_path" in output, (
            f"upload --help should show 'pack_path' argument, got:\n{output}"
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
        for cmd in ("build", "validate", "upload", "submit", "status"):
            assert cmd in output, (
                f"top-level --help should list '{cmd}', got:\n{output}"
            )
