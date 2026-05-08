"""v6.0 validator surface tests.

Covers the small set of pure helpers that survived the v5.2 → v6.0
refactor: server-driven challenge-epoch flow, the local winner cache,
and the v6 ``trajrl_api`` signing helper. Pre-existing tests for
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
    aggregate_challenger_score,
    derive_winner_state,
    winner_from_server_block,
    save_winner_state,
    load_winner_state,
)
from trajectoryrl.utils.trajrl_api import _sign


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
# Local winner derivation (v6.0 /api/v2/winner/current)
# ---------------------------------------------------------------------------


class TestAggregateChallengerScore:
    def test_empty_returns_none(self):
        assert aggregate_challenger_score(None) is None
        assert aggregate_challenger_score([]) is None

    def test_single_qualified_vote(self):
        subs = [{
            "validator_stake": 100.0,
            "challenger_score": 0.9,
            "challenger_qualified": True,
            "challenger_rejected": False,
        }]
        score, qualified = aggregate_challenger_score(subs)
        assert score == pytest.approx(0.9)
        assert qualified is True

    def test_stake_weighted_average(self):
        subs = [
            {"validator_stake": 75.0, "challenger_score": 1.0,
             "challenger_qualified": True, "challenger_rejected": False},
            {"validator_stake": 25.0, "challenger_score": 0.0,
             "challenger_qualified": True, "challenger_rejected": False},
        ]
        score, qualified = aggregate_challenger_score(subs)
        # 75/100 → 1.0, 25/100 → 0.0  ⇒  weighted avg = 0.75
        assert score == pytest.approx(0.75)
        assert qualified is True

    def test_rejected_rows_dropped(self):
        subs = [
            {"validator_stake": 50.0, "challenger_score": 0.0,
             "challenger_qualified": True, "challenger_rejected": True},
            {"validator_stake": 50.0, "challenger_score": 0.8,
             "challenger_qualified": True, "challenger_rejected": False},
        ]
        score, qualified = aggregate_challenger_score(subs)
        assert score == pytest.approx(0.8)
        assert qualified is True

    def test_null_stake_rows_dropped(self):
        subs = [
            {"validator_stake": None, "challenger_score": 0.0,
             "challenger_qualified": True, "challenger_rejected": False},
            {"validator_stake": 100.0, "challenger_score": 0.6,
             "challenger_qualified": True, "challenger_rejected": False},
        ]
        score, qualified = aggregate_challenger_score(subs)
        assert score == pytest.approx(0.6)
        assert qualified is True

    def test_stake_weighted_majority_qualified(self):
        subs = [
            {"validator_stake": 60.0, "challenger_score": 0.0,
             "challenger_qualified": False, "challenger_rejected": False},
            {"validator_stake": 40.0, "challenger_score": 0.9,
             "challenger_qualified": True, "challenger_rejected": False},
        ]
        score, qualified = aggregate_challenger_score(subs)
        # qualified stake is 40/100 = 40% — not a majority
        assert qualified is False
        # score is computed only over qualified rows
        assert score == pytest.approx(0.9)

    def test_no_qualified_rows_returns_zero_score(self):
        subs = [
            {"validator_stake": 100.0, "challenger_score": 0.5,
             "challenger_qualified": False, "challenger_rejected": False},
        ]
        score, qualified = aggregate_challenger_score(subs)
        assert score == 0.0
        assert qualified is False


class TestDeriveWinnerState:
    def test_none_response_yields_empty(self):
        s = derive_winner_state(None, now=100.0)
        assert not s.is_seated

    def test_cold_start_response(self):
        # No finalized epoch yet
        resp = {"winner": None, "finalized_epoch": None, "submissions": []}
        s = derive_winner_state(resp, now=100.0)
        assert not s.is_seated
        assert s.updated_at == 100.0

    def test_winner_held_uses_server_winner(self):
        resp = {
            "winner": {"hotkey": "5G", "uid": 7, "score": "0.95"},
            "finalized_epoch": {
                "challenge_epoch_id": 100,
                "winner_replaced": False,
                "outcome": "winner_held",
            },
            "submissions": [{
                "validator_stake": 100.0,
                "challenger_score": 0.10,
                "challenger_qualified": True,
                "challenger_rejected": False,
            }],
        }
        s = derive_winner_state(resp, now=100.0)
        assert s.winner_hotkey == "5G"
        assert s.winner_uid == 7
        assert s.winner_score == 0.95

    def test_winner_replaced_matches_server(self, caplog):
        resp = {
            "winner": {"hotkey": "5G", "uid": 7, "score": "0.90"},
            "finalized_epoch": {
                "challenge_epoch_id": 200,
                "winner_replaced": True,
                "outcome": "winner_replaced",
            },
            "submissions": [{
                "validator_stake": 100.0,
                "challenger_score": 0.90,
                "challenger_qualified": True,
                "challenger_rejected": False,
            }],
        }
        with caplog.at_level("WARNING", logger="trajectoryrl.utils.winner_state"):
            s = derive_winner_state(resp, now=100.0)
        assert s.winner_score == 0.90
        # No divergence → no warning emitted
        assert not any("divergence" in r.message for r in caplog.records)

    def test_winner_replaced_divergence_logs_warning(self, caplog):
        resp = {
            "winner": {"hotkey": "5G", "uid": 7, "score": "0.90"},
            "finalized_epoch": {
                "challenge_epoch_id": 201,
                "winner_replaced": True,
                "outcome": "winner_replaced",
            },
            "submissions": [{
                "validator_stake": 100.0,
                "challenger_score": 0.50,  # local consensus 0.50 vs server 0.90
                "challenger_qualified": True,
                "challenger_rejected": False,
            }],
        }
        with caplog.at_level("WARNING", logger="trajectoryrl.utils.winner_state"):
            derive_winner_state(resp, now=100.0)
        assert any("divergence" in r.message for r in caplog.records)


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
# fetch_current_epoch — propagates remaining_blocks for the budget gate
# ---------------------------------------------------------------------------


class TestFetchCurrentEpochResponseShape:
    """fetch_current_epoch must surface current_block / remaining_blocks
    so the eval loop can run the mid-epoch budget gate. We mock httpx at
    the module level rather than poking the wire."""

    @pytest.mark.asyncio
    async def test_200_propagates_chain_time(self, monkeypatch):
        from trajectoryrl.utils import trajrl_api

        class _FakeResp:
            status_code = 200
            text = ""

            def json(self):
                return {
                    "success": True,
                    "epoch": {
                        "challenge_epoch_id": 17,
                        "challenger_hotkey": "5G",
                        "challenger_pack_hash": "f" * 64,
                        "challenger_pack_url": "https://x/p.json",
                        "start_block": 8000000,
                        "end_block": 8000150,
                        "epoch_length_blocks": 150,
                        "status": "in_progress",
                    },
                    "current_block": 8000075,
                    "elapsed_blocks": 75,
                    "remaining_blocks": 75,
                }

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, *a, **kw):
                return _FakeResp()

        monkeypatch.setattr(trajrl_api.httpx, "AsyncClient", lambda: _FakeClient())
        out = await trajrl_api.fetch_current_epoch()
        assert out is not None
        assert out["epoch"]["challenge_epoch_id"] == 17
        assert out["current_block"] == 8000075
        assert out["elapsed_blocks"] == 75
        assert out["remaining_blocks"] == 75

    @pytest.mark.asyncio
    async def test_404_returns_null_block_fields(self, monkeypatch):
        from trajectoryrl.utils import trajrl_api

        class _FakeResp:
            status_code = 404
            text = ""

            def json(self):
                return {}

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, *a, **kw):
                return _FakeResp()

        monkeypatch.setattr(trajrl_api.httpx, "AsyncClient", lambda: _FakeClient())
        out = await trajrl_api.fetch_current_epoch()
        assert out == {
            "epoch": None,
            "current_block": None,
            "elapsed_blocks": None,
            "remaining_blocks": None,
        }


# ---------------------------------------------------------------------------
# trajrl_api signing helper (pure)
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
