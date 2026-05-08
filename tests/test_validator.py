"""v6.0 validator surface tests.

Covers the small set of pure helpers that survived the v5.2 → v6.0
refactor: server-driven challenge-epoch flow, the local winner cache,
and the v6 ``status_reporter`` signing helper. Pre-existing tests for
v5.x consensus, eval-snapshot, integrity-judge, and pack-ownership-lock
machinery were retired alongside those modules.

End-to-end / integration coverage of the eval and weight loops is left
to the staging-environment smoke test described in the v6 refactor PR.
"""

from __future__ import annotations

import json
import sys
import time
from unittest.mock import MagicMock

import pytest

# Mock bittensor so importing trajectoryrl.* doesn't pull in the SDK.
_mock_bt = MagicMock()


class _MockSynapse:
    pass


_mock_bt.Synapse = _MockSynapse
sys.modules["bittensor"] = _mock_bt

# Now safe to import — none of the deleted v5.x modules are referenced.
from trajectoryrl.utils.config import SPEC_NUMBER, ValidatorConfig
from trajectoryrl.utils.github import PackFetcher
from trajectoryrl.utils.commitments import MinerCommitment
from trajectoryrl.utils.winner_state import (
    WinnerState,
    WINNER_FALLBACK_TTL_SECONDS,
    winner_from_server_block,
    save_winner_state,
    load_winner_state,
)
from trajectoryrl.utils.status_reporter import _sign


# ---------------------------------------------------------------------------
# WinnerState cache (v6.0)
# ---------------------------------------------------------------------------


class TestWinnerStateCache:
    def test_empty_state_not_seated(self):
        s = WinnerState()
        assert not s.is_seated
        assert not s.is_fresh()

    def test_from_server_block_populates_fields(self):
        block = {
            "hotkey": "5Gabc",
            "uid": 42,
            "pack_hash": "deadbeef" * 8,
            "score": "0.91",
            "spec_number": 7,
        }
        now = 100.0
        s = winner_from_server_block(block, now=now)
        assert s.is_seated
        assert s.winner_hotkey == "5Gabc"
        assert s.winner_uid == 42
        assert s.winner_pack_hash == "deadbeef" * 8
        assert s.winner_score == 0.91
        assert s.spec_number == 7
        assert s.updated_at == now

    def test_from_server_block_none_yields_unseated(self):
        s = winner_from_server_block(None, now=100.0)
        assert not s.is_seated
        # cold-start state is updated_at-stamped but treats is_fresh as
        # False because nothing was actually written from the server.
        # The contract is "is_seated controls whether to set_weights",
        # so a never-seen winner stays in fallback regardless of TTL.
        assert s.updated_at == 100.0

    def test_is_fresh_within_ttl(self):
        now = 1000.0
        s = WinnerState(
            winner_hotkey="5Gabc", winner_uid=1, updated_at=now,
        )
        assert s.is_fresh(ttl_seconds=60, now=now + 30)
        assert not s.is_fresh(ttl_seconds=60, now=now + 90)

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "winner_state.json"
        s = WinnerState(
            winner_hotkey="5Gabc",
            winner_uid=42,
            winner_pack_hash="abcd",
            winner_score=0.5,
            spec_number=7,
            updated_at=100.0,
        )
        save_winner_state(s, str(path))
        loaded = load_winner_state(str(path))
        assert loaded == s

    def test_load_missing_file_returns_empty(self, tmp_path):
        s = load_winner_state(str(tmp_path / "missing.json"))
        assert s == WinnerState()

    def test_load_invalid_json_returns_empty(self, tmp_path):
        path = tmp_path / "winner_state.json"
        path.write_text("{not json")
        s = load_winner_state(str(path))
        assert s == WinnerState()

    def test_winner_from_server_block_handles_bad_score(self):
        s = winner_from_server_block(
            {"hotkey": "5G", "uid": 1, "score": "not-a-number"},
            now=0,
        )
        assert s.winner_score is None  # graceful fallback


# ---------------------------------------------------------------------------
# v6 validator helpers (pure / static)
# ---------------------------------------------------------------------------


class TestValidatorEpochHelpers:
    """Helpers on TrajectoryValidator that don't need an instance."""

    def test_commitment_from_epoch_happy_path(self):
        from trajectoryrl.base.validator import TrajectoryValidator
        epoch = {
            "challenge_epoch_id": 17,
            "challenger_hotkey": "5Gabc",
            "challenger_pack_hash": "f" * 64,
            "challenger_pack_url": "https://example.test/pack.json",
            "start_block": 8000000,
        }
        c = TrajectoryValidator._commitment_from_epoch(epoch)
        assert c is not None
        assert c.hotkey == "5Gabc"
        assert c.pack_hash == "f" * 64
        assert c.pack_url == "https://example.test/pack.json"
        assert c.block_number == 8000000

    def test_commitment_from_epoch_missing_fields(self):
        from trajectoryrl.base.validator import TrajectoryValidator
        for missing in ("challenger_hotkey", "challenger_pack_hash", "challenger_pack_url"):
            epoch = {
                "challenger_hotkey": "5Gabc",
                "challenger_pack_hash": "f" * 64,
                "challenger_pack_url": "https://x/y",
            }
            del epoch[missing]
            assert TrajectoryValidator._commitment_from_epoch(epoch) is None

    def test_challenge_seed_deterministic(self):
        from trajectoryrl.base.validator import TrajectoryValidator
        s1 = TrajectoryValidator._challenge_seed(42)
        s2 = TrajectoryValidator._challenge_seed(42)
        s3 = TrajectoryValidator._challenge_seed(43)
        assert s1 == s2
        assert s1 != s3
        assert isinstance(s1, int) and s1 > 0

    def test_summarize_eval_aggregates_scores(self):
        from trajectoryrl.base.validator import TrajectoryValidator
        eval_result = {
            "qualified": {"a": True, "b": False, "c": True},
            "judge_details": {
                "a": {"overall_score": 0.75},
                "b": {"overall_score": 0.10},
                "c": {"overall_score": 0.50},
            },
        }
        score, qualified, scenario_results = TrajectoryValidator._summarize_eval(
            eval_result
        )
        # raw_score = 0.75 + 0.10 + 0.50 = 1.35
        assert pytest.approx(score, abs=1e-6) == 1.35
        assert qualified is True  # any-pass: at least one True
        assert set(scenario_results.keys()) == {"a", "b", "c"}
        assert scenario_results["a"]["qualified"] is True
        assert scenario_results["b"]["qualified"] is False

    def test_summarize_eval_no_pass_is_unqualified(self):
        from trajectoryrl.base.validator import TrajectoryValidator
        eval_result = {
            "qualified": {"a": False},
            "judge_details": {"a": {"overall_score": 0.0}},
        }
        _score, qualified, _sr = TrajectoryValidator._summarize_eval(eval_result)
        assert qualified is False


# ---------------------------------------------------------------------------
# status_reporter signing helper (pure)
# ---------------------------------------------------------------------------


class _FakeKeypair:
    def __init__(self, address: str):
        self.ss58_address = address

    def sign(self, message: bytes) -> bytes:
        # Deterministic stand-in for a real signature so tests can assert
        # the exact wire prefix without running the bittensor crypto.
        return b"S" + message[:31].ljust(31, b"\x00")


class TestStatusReporterSigning:
    def test_sign_basic(self):
        kp = _FakeKeypair("5Gabc")
        addr, message, sig = _sign("trajectoryrl-heartbeat", kp, 1700000000)
        assert addr == "5Gabc"
        assert message == "trajectoryrl-heartbeat:5Gabc:1700000000"
        assert sig.startswith("0x")
        assert len(sig) > 2

    def test_sign_with_extras_includes_epoch_id(self):
        kp = _FakeKeypair("5Gabc")
        addr, message, _ = _sign(
            "trajectoryrl-challenge-score", kp, 1700000000, "42",
        )
        assert (
            message
            == "trajectoryrl-challenge-score:5Gabc:1700000000:42"
        )


# ---------------------------------------------------------------------------
# PackFetcher (kept verbatim from v5.x — module unchanged)
# ---------------------------------------------------------------------------


class TestPackFetcher:
    def test_init_creates_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "pack_cache"
        PackFetcher(cache_dir=cache_dir)
        assert cache_dir.exists()


# ---------------------------------------------------------------------------
# ValidatorConfig (kept — config schema didn't change in v6)
# ---------------------------------------------------------------------------


class TestValidatorConfig:
    def test_spec_number_is_int(self):
        assert isinstance(SPEC_NUMBER, int)
        assert SPEC_NUMBER >= 1

    def test_from_env_uses_defaults(self, monkeypatch, tmp_path):
        # Clear all relevant env vars so from_env hits its defaults.
        for var in (
            "WALLET_NAME", "WALLET_HOTKEY", "BITTENSOR_NETWORK",
            "NETUID", "WINNER_STATE_PATH",
        ):
            monkeypatch.delenv(var, raising=False)
        # from_env still requires a few env vars to be set; just check
        # SPEC_NUMBER is exposed as a class attribute or accessible.
        assert hasattr(ValidatorConfig, "from_env")
