"""Schedule-driven spec resolution (web /epoch/current → eval scenario set).

Covers the three layers of the cutover-sync feature:
  1. ``SCENARIOS_BY_SPEC`` registry + ``resolve_eval_spec`` clamp/fallback
     in ``sandbox_harness``.
  2. Per-eval scenario threading through ``evaluate_miner_s1`` /
     ``harness.evaluate_miner``.
  3. ``TrajectoryValidator`` eval loop adopting the server-reported
     ``epoch.spec_number`` for scenario selection and outgoing payloads.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock

import pytest

# Mock bittensor so importing trajectoryrl.* doesn't pull in the SDK.
_mock_bt = MagicMock()


class _MockSynapse:
    pass


_mock_bt.Synapse = _MockSynapse
sys.modules["bittensor"] = _mock_bt

from trajectoryrl.utils.config import SPEC_NUMBER
from trajectoryrl.utils import sandbox_harness as sh
from trajectoryrl.utils.commitments import MinerCommitment


SPEC21_ADDITIONS = {"audio-synth-stft-peaks", "puzzle-solver", "query-optimize"}


# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------


class TestScenarioRegistry:
    def test_registry_covers_current_and_previous_spec(self):
        assert SPEC_NUMBER in sh.SCENARIOS_BY_SPEC
        assert SPEC_NUMBER - 1 in sh.SCENARIOS_BY_SPEC

    def test_sandbox_scenarios_is_local_spec_set(self):
        assert sh.SANDBOX_SCENARIOS == sh.SCENARIOS_BY_SPEC[SPEC_NUMBER]

    def test_spec20_is_spec21_minus_additions(self):
        spec21 = set(sh.SCENARIOS_BY_SPEC[21])
        spec20 = set(sh.SCENARIOS_BY_SPEC[20])
        assert spec21 - spec20 == SPEC21_ADDITIONS
        assert spec20 < spec21
        assert len(spec20) == 17 and len(spec21) == 20

    def test_scenario_sets_sorted_and_unique(self):
        for spec, scenarios in sh.SCENARIOS_BY_SPEC.items():
            assert list(scenarios) == sorted(set(scenarios)), spec


# ---------------------------------------------------------------------------
# 2. resolve_eval_spec
# ---------------------------------------------------------------------------


class TestResolveEvalSpec:
    def test_known_spec_exact_match(self):
        spec, scenarios = sh.resolve_eval_spec(20)
        assert spec == 20
        assert scenarios == sh.SCENARIOS_BY_SPEC[20]

    def test_local_spec_exact_match(self):
        spec, scenarios = sh.resolve_eval_spec(SPEC_NUMBER)
        assert spec == SPEC_NUMBER
        assert scenarios == sh.SCENARIOS_BY_SPEC[SPEC_NUMBER]

    def test_numeric_string_coerced(self):
        spec, _ = sh.resolve_eval_spec("20")
        assert spec == 20

    def test_none_falls_back_to_local(self):
        spec, scenarios = sh.resolve_eval_spec(None)
        assert spec == SPEC_NUMBER
        assert scenarios == sh.SCENARIOS_BY_SPEC[SPEC_NUMBER]

    def test_unknown_newer_spec_falls_back_to_local(self, caplog):
        with caplog.at_level("WARNING"):
            spec, _ = sh.resolve_eval_spec(99)
        assert spec == SPEC_NUMBER
        assert any("99" in r.message for r in caplog.records)

    def test_unknown_older_spec_falls_back_to_local(self):
        spec, _ = sh.resolve_eval_spec(3)
        assert spec == SPEC_NUMBER

    def test_garbage_falls_back_to_local(self):
        spec, _ = sh.resolve_eval_spec("not-a-spec")
        assert spec == SPEC_NUMBER


# ---------------------------------------------------------------------------
# 3. Validator threading
# ---------------------------------------------------------------------------


def _bare_validator():
    """A TrajectoryValidator shell with just the attrs the tick path uses."""
    from trajectoryrl.base.validator import TrajectoryValidator

    v = TrajectoryValidator.__new__(TrajectoryValidator)
    v.config = MagicMock()
    v.wallet = MagicMock()
    v.subtensor = MagicMock()
    v.subtensor.get_current_block.return_value = 123
    v.pack_fetcher = MagicMock()
    v._sandbox_harness = MagicMock()
    v._sandbox_harness.harness_name = "trajrl-bench"
    v._sandbox_harness.harness_version = "0.0.0"
    v._sandbox_harness.bench_image_hash = "unknown"
    v._sandbox_harness.scenario_image_hash = "unknown"
    v._sandbox_harness.sandbox_version = "unknown"
    v._last_scored_challenge_epoch_id = None
    v._last_set_weights_at = None
    v._last_eval_at = None
    v._last_set_weights_block = 0
    v._save_eval_state = lambda: None
    return v


def _epoch_block(spec_number, epoch_id=777):
    return {
        "challenge_epoch_id": epoch_id,
        "challenger_hotkey": "hk-challenger",
        "challenger_pack_hash": "ab" * 32,
        "challenger_pack_url": "https://example.com/pack.json",
        "start_block": 100,
        "end_block": 200,
        "epoch_length_blocks": 100,
        "status": "in_progress",
        "spec_number": spec_number,
    }


class TestEvalLoopSpecThreading:
    def test_tick_passes_api_spec_to_score_challenger(self, monkeypatch):
        from trajectoryrl.base import validator as vmod

        v = _bare_validator()
        seen = {}

        async def fake_refresh():
            pass

        async def fake_fetch(*a, **k):
            return {"epoch": _epoch_block(20), "elapsed_blocks": 0}

        async def fake_score(epoch_id, commitment, eval_spec, eval_scenarios):
            seen["spec"] = eval_spec
            seen["scenarios"] = eval_scenarios

        v._refresh_winner_cache = fake_refresh
        v._score_challenger = fake_score
        monkeypatch.setattr(vmod, "fetch_current_epoch", fake_fetch)

        asyncio.run(v._eval_loop_tick())
        assert seen["spec"] == 20
        assert seen["scenarios"] == sh.SCENARIOS_BY_SPEC[20]

    def test_tick_unknown_api_spec_falls_back_to_local(self, monkeypatch):
        from trajectoryrl.base import validator as vmod

        v = _bare_validator()
        seen = {}

        async def fake_refresh():
            pass

        async def fake_fetch(*a, **k):
            return {"epoch": _epoch_block(99), "elapsed_blocks": 0}

        async def fake_score(epoch_id, commitment, eval_spec, eval_scenarios):
            seen["spec"] = eval_spec

        v._refresh_winner_cache = fake_refresh
        v._score_challenger = fake_score
        monkeypatch.setattr(vmod, "fetch_current_epoch", fake_fetch)

        asyncio.run(v._eval_loop_tick())
        assert seen["spec"] == SPEC_NUMBER


class TestScoreChallengerSpecThreading:
    def _run_score(self, monkeypatch, eval_spec, eval_scenarios):
        from trajectoryrl.base import validator as vmod

        v = _bare_validator()
        submitted = {}

        v._get_validator_log_offset = lambda: 0
        v._prepare_eval_log_capture = lambda *a, **k: (MagicMock(), 0, 0)

        async def fake_eval(commitment, epoch_id, spec, scenarios):
            submitted["eval_spec"] = spec
            submitted["eval_scenarios"] = scenarios
            return {
                "success": True,
                "qualified": {"regex-chess": True},
                "judge_details": {"regex-chess": {"overall_score": 1.0}},
            }

        async def fake_submit(wallet, **kwargs):
            submitted["payload_spec"] = kwargs.get("spec_number")
            return True

        async def fake_upload(*a, **k):
            pass

        v._evaluate_challenger = fake_eval
        v._fire_upload_eval_logs = fake_upload
        v._fire_upload_cycle_logs = fake_upload
        monkeypatch.setattr(vmod, "submit_challenge_score", fake_submit)

        commitment = MinerCommitment(
            uid=1, hotkey="hk", pack_hash="ab" * 32,
            pack_url="https://example.com/p.json", block_number=100,
            raw="r",
        )
        asyncio.run(
            v._score_challenger(777, commitment, eval_spec, eval_scenarios)
        )
        return submitted

    def test_submits_resolved_spec_not_local_constant(self, monkeypatch):
        scenarios = sh.SCENARIOS_BY_SPEC[20]
        submitted = self._run_score(monkeypatch, 20, scenarios)
        assert submitted["payload_spec"] == 20
        assert submitted["eval_spec"] == 20
        assert submitted["eval_scenarios"] == scenarios


# ---------------------------------------------------------------------------
# 4. Harness / miner_eval scenario threading
# ---------------------------------------------------------------------------


class TestHarnessScenarioParam:
    def test_run_eval_sync_accepts_scenario_subset(self, monkeypatch):
        """_run_eval_sync must iterate the passed-in scenario set, not the
        module global."""
        harness = sh.TrajectorySandboxHarness.__new__(sh.TrajectorySandboxHarness)
        loaded = []

        def fake_load(name):
            loaded.append(name)
            raise RuntimeError("stop after recording")

        harness._pull_sync = lambda: None
        harness._load_scenario_info = fake_load
        harness._scenario_info = None

        with pytest.raises(RuntimeError):
            harness._run_eval_sync(
                "skill", 1, "salt", "ph",
                scenarios=("db-wal-recovery",),
            )
        assert loaded == ["db-wal-recovery"]

    def test_evaluate_miner_s1_threads_scenarios(self, monkeypatch):
        from trajectoryrl.utils import miner_eval as me

        seen = {}

        class _FakeResult:
            error = None
            aborted_mid_session = False
            scenarios = ["db-wal-recovery"]
            scenario_qualities = {"db-wal-recovery": 1.0}
            scenario_costs_usd = {}
            score = 1.0
            mean_quality = 1.0

            class session_result:
                episodes = []

        class _FakeHarness:
            sandbox_scenarios = ["db-wal-recovery"]
            sandbox_version = "test"

            async def evaluate_miner(self, **kwargs):
                seen["scenarios"] = kwargs.get("scenarios")
                return _FakeResult()

        class _FakeVerification:
            valid = True
            error = None
            pack_content = {"files": {"SKILL.md": "# s"}}

        class _FakeFetcher:
            async def verify_submission(self, pack_url, pack_hash):
                return _FakeVerification()

        commitment = MinerCommitment(
            uid=1, hotkey="hk", pack_hash="ab" * 32,
            pack_url="https://example.com/p.json", block_number=100,
            raw="r",
        )
        outcome = asyncio.run(
            me.evaluate_miner_s1(
                harness=_FakeHarness(),
                pack_fetcher=_FakeFetcher(),
                commitment=commitment,
                epoch_seed=1,
                validator_salt="salt",
                scenarios=("db-wal-recovery",),
            )
        )
        assert seen["scenarios"] == ("db-wal-recovery",)
        assert outcome.success
