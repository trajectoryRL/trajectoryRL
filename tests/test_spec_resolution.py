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
SPEC22_ADDITIONS = {"crack-7z-hash", "parallel-particle-simulator", "regex-engine-from-scratch"}
SPEC23_ADDITIONS = {"attention-mil", "llm-inference-batching-scheduler", "torch-tensor-parallelism"}
# SPEC 24 is the first *replacement* bump: it removes AND adds (net size unchanged),
# so it is asserted as a swap, not a subset like the additive specs above.
SPEC24_ADDITIONS = {"fix-code-vulnerability", "large-scale-text-editing", "postgres-csv-clean"}
SPEC24_REMOVALS = {"schemelike-metacircular-eval", "3d-model-format-legacy", "pcap-to-netflow"}
# SPEC 26 is the first *shrinking* bump: it only removes (26 -> 20), so maxScore
# drops 26 -> 20 on the web side (an intentional discontinuity, like SPEC 16).
SPEC26_REMOVALS = {
    "regex-chess", "race-condition-fix", "custom-memory-heap-crash",
    "attention-mil", "git-leak-recovery", "tree-directory-parser",
}


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


class TestSpec22Set:
    def test_spec22_is_spec21_plus_additions(self):
        spec21 = set(sh.SCENARIOS_BY_SPEC[21])
        spec22 = set(sh.SCENARIOS_BY_SPEC[22])
        assert spec22 - spec21 == SPEC22_ADDITIONS
        assert spec21 < spec22
        assert len(spec21) == 20 and len(spec22) == 23


class TestSpec23Set:
    def test_spec23_is_spec22_plus_additions(self):
        spec22 = set(sh.SCENARIOS_BY_SPEC[22])
        spec23 = set(sh.SCENARIOS_BY_SPEC[23])
        assert spec23 - spec22 == SPEC23_ADDITIONS
        assert spec22 < spec23
        assert len(spec22) == 23 and len(spec23) == 26


class TestSpec24Set:
    def test_spec24_swaps_three_scenarios(self):
        spec23 = set(sh.SCENARIOS_BY_SPEC[23])
        spec24 = set(sh.SCENARIOS_BY_SPEC[24])
        assert spec24 - spec23 == SPEC24_ADDITIONS
        assert spec23 - spec24 == SPEC24_REMOVALS
        # Replacement, not a superset: net size stays 26.
        assert not (spec23 < spec24)
        assert len(spec23) == len(spec24) == 26

    def test_spec23_still_resolves_during_transition(self):
        # resolve_eval_spec must still serve SPEC 23 while validators roll forward.
        assert sh.SCENARIOS_BY_SPEC[23]
        assert sh.resolve_eval_spec(23) == (23, sh.SCENARIOS_BY_SPEC[23])


class TestSpec26Set:
    def test_spec26_drops_six_low_signal_scenarios(self):
        spec25 = set(sh.SCENARIOS_BY_SPEC[25])
        spec26 = set(sh.SCENARIOS_BY_SPEC[26])
        # Pure removal: no additions, six dropped, net 26 -> 20.
        assert spec25 - spec26 == SPEC26_REMOVALS
        assert spec26 - spec25 == set()
        assert spec26 < spec25
        assert len(spec25) == 26 and len(spec26) == 20

    def test_spec25_still_resolves_during_transition(self):
        # resolve_eval_spec must still serve SPEC 25 while validators roll forward
        # and the server has not yet flipped the active spec to 26.
        assert sh.SCENARIOS_BY_SPEC[25]
        assert sh.resolve_eval_spec(25) == (25, sh.SCENARIOS_BY_SPEC[25])


class TestSpec27Set:
    def test_spec27_keeps_the_spec26_scenarios(self):
        # SPEC 27 marks the episode cap change ($1.00 -> $0.30), not a scenario change.
        assert sh.SCENARIOS_BY_SPEC[27] == sh.SCENARIOS_BY_SPEC[26]
        assert len(sh.SCENARIOS_BY_SPEC[27]) == 20

    def test_spec27_is_the_local_default(self):
        from trajectoryrl.utils.config import SPEC_NUMBER
        assert SPEC_NUMBER == 27
        assert sh.SANDBOX_SCENARIOS == sh.SCENARIOS_BY_SPEC[27]

    def test_spec26_still_resolves_during_transition(self):
        # resolve_eval_spec must still serve SPEC 26 while validators roll forward
        # and the server has not yet flipped the active spec to 27.
        assert sh.resolve_eval_spec(26) == (26, sh.SCENARIOS_BY_SPEC[26])


class TestSpecConfig:
    def test_every_spec_has_a_config(self):
        assert set(sh.SPEC_CONFIG_BY_SPEC) == set(sh.SCENARIOS_BY_SPEC)

    def test_spec27_lowers_the_episode_cap(self):
        assert sh.SPEC_CONFIG_BY_SPEC[27].episode_cap_usd == 0.30
        # SPEC 25 and 26 keep the launch cap, so a SPEC 26 epoch means the
        # same on every validator while the fleet rolls forward.
        assert sh.SPEC_CONFIG_BY_SPEC[25].episode_cap_usd == 1.00
        assert sh.SPEC_CONFIG_BY_SPEC[26].episode_cap_usd == 1.00

    def test_spec_config_resolves_like_resolve_eval_spec(self):
        from trajectoryrl.utils.config import SPEC_NUMBER
        local = sh.SPEC_CONFIG_BY_SPEC[SPEC_NUMBER]
        assert sh.spec_config(26) is sh.SPEC_CONFIG_BY_SPEC[26]
        assert sh.spec_config() is local
        assert sh.spec_config(None) is local
        assert sh.spec_config(999) is local

    def test_every_cap_leaves_an_opening_turn(self):
        # ~20k prompt tokens (system prompt, tools, SKILL.md) with the default
        # completion length must fit, unclamped, on every allowlisted model.
        from trajectoryrl.policy import MODEL_PRICES
        from trajectoryrl.policy.meter import (
            DEFAULT_MAX_TOKENS, estimate_prompt_tokens, reserve_for,
        )
        body = {"messages": [{"role": "user", "content": "x" * 48000}]}
        assert 19000 <= estimate_prompt_tokens(body) <= 21000
        for spec, cfg in sh.SPEC_CONFIG_BY_SPEC.items():
            for model in MODEL_PRICES:
                _, mt, clamped = reserve_for(model, body, room_usd=cfg.episode_cap_usd)
                assert mt == DEFAULT_MAX_TOKENS and not clamped, (spec, model)


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
                seen["spec_number"] = kwargs.get("spec_number")
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
                spec_number=26,
            )
        )
        assert seen["scenarios"] == ("db-wal-recovery",)
        assert seen["spec_number"] == 26
        assert outcome.success

    @pytest.mark.parametrize("spec_number,cap", [(26, 1.0), (27, 0.3), (None, 0.3)])
    def test_evaluate_miner_runs_under_the_cap_of_its_spec(self, spec_number, cap):
        """The episode cap is a per-spec value: a SPEC 26 epoch keeps the
        launch cap on a binary whose local spec is 27."""
        harness = sh.TrajectorySandboxHarness.__new__(sh.TrajectorySandboxHarness)
        seen = {}

        def fake_run_eval_sync(*args, **kwargs):
            seen["cap_usd"] = kwargs.get("cap_usd")
            return sh._SessionResult()

        harness._run_eval_sync = fake_run_eval_sync
        asyncio.run(harness.evaluate_miner(
            "skill", 1, validator_salt="salt",
            scenarios=("db-wal-recovery",), spec_number=spec_number,
        ))
        assert seen["cap_usd"] == cap
