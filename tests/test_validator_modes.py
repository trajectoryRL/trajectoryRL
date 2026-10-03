"""Weight-only (default) vs. eval-enabled validator modes.

By default the daemon only mirrors the server-canonical winner into
on-chain weights. The eval loop (challenger polling, sandbox evaluation,
score submission) runs only when ``EVAL_ENABLED`` is switched on.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock bittensor so importing trajectoryrl.* doesn't pull in the SDK.
_mock_bt = MagicMock()


class _MockSynapse:
    pass


_mock_bt.Synapse = _MockSynapse
sys.modules["bittensor"] = _mock_bt

from trajectoryrl.utils.config import ValidatorConfig


def _config(tmp_path, **overrides) -> ValidatorConfig:
    return ValidatorConfig(
        pack_cache_dir=tmp_path / "packs",
        log_dir=tmp_path / "logs",
        eval_state_path=tmp_path / "eval_state.json",
        winner_state_path=tmp_path / "winner_state.json",
        **overrides,
    )


def _make_validator(tmp_path, monkeypatch, **overrides):
    """Construct a TrajectoryValidator with chain + harness mocked out."""
    from trajectoryrl.base import validator as vmod

    harness_cls = MagicMock(name="TrajectorySandboxHarness")
    fetcher_cls = MagicMock(name="PackFetcher")
    monkeypatch.setattr(vmod, "bt", MagicMock())
    monkeypatch.setattr(vmod, "TrajectorySandboxHarness", harness_cls)
    monkeypatch.setattr(vmod, "PackFetcher", fetcher_cls)

    v = vmod.TrajectoryValidator(_config(tmp_path, **overrides))
    return v, harness_cls, fetcher_cls


# ---------------------------------------------------------------------------
# Config switch
# ---------------------------------------------------------------------------


class TestEvalEnabledConfig:
    def _from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PACK_CACHE_DIR", str(tmp_path / "packs"))
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        # Point at a missing dotenv so the operator's real .env.validator
        # on this box can't leak into the test.
        return ValidatorConfig.from_env(dotenv_path=tmp_path / "missing.env")

    def test_dataclass_default_is_weight_only(self, tmp_path):
        assert _config(tmp_path).eval_enabled is False

    def test_from_env_defaults_to_weight_only(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EVAL_ENABLED", raising=False)
        assert self._from_env(tmp_path, monkeypatch).eval_enabled is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
    def test_from_env_truthy_values_enable_eval(
        self, tmp_path, monkeypatch, value,
    ):
        monkeypatch.setenv("EVAL_ENABLED", value)
        assert self._from_env(tmp_path, monkeypatch).eval_enabled is True

    @pytest.mark.parametrize("value", ["0", "", "false", "no", "off", "2"])
    def test_from_env_other_values_stay_weight_only(
        self, tmp_path, monkeypatch, value,
    ):
        monkeypatch.setenv("EVAL_ENABLED", value)
        assert self._from_env(tmp_path, monkeypatch).eval_enabled is False


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestValidatorInit:
    def test_weight_only_skips_eval_components(self, tmp_path, monkeypatch):
        v, harness_cls, fetcher_cls = _make_validator(tmp_path, monkeypatch)

        harness_cls.assert_not_called()
        fetcher_cls.assert_not_called()
        assert v._sandbox_harness is None
        assert v.pack_fetcher is None

    def test_eval_enabled_builds_eval_components(self, tmp_path, monkeypatch):
        v, harness_cls, fetcher_cls = _make_validator(
            tmp_path, monkeypatch, eval_enabled=True,
        )

        harness_cls.assert_called_once_with(v.config)
        fetcher_cls.assert_called_once()
        assert v._sandbox_harness is harness_cls.return_value
        assert v.pack_fetcher is fetcher_cls.return_value


# ---------------------------------------------------------------------------
# run(): which loops start
# ---------------------------------------------------------------------------


def _stub_loops(v):
    for name in (
        "_eval_loop", "_winner_loop", "_weight_loop", "_heartbeat_loop",
        "_replay_pending_uploads", "_refresh_winner_cache",
    ):
        setattr(v, name, AsyncMock(name=name))


class TestRunLoops:
    def test_weight_only_never_starts_eval_loop(self, tmp_path, monkeypatch):
        v, _, _ = _make_validator(tmp_path, monkeypatch)
        _stub_loops(v)

        asyncio.run(v.run())

        v._eval_loop.assert_not_called()
        v._replay_pending_uploads.assert_not_called()
        v._winner_loop.assert_awaited_once()
        v._weight_loop.assert_awaited_once()
        v._heartbeat_loop.assert_awaited_once()

    def test_eval_enabled_starts_eval_loop(self, tmp_path, monkeypatch):
        v, _, _ = _make_validator(tmp_path, monkeypatch, eval_enabled=True)
        _stub_loops(v)

        asyncio.run(v.run())

        v._eval_loop.assert_awaited_once()
        v._replay_pending_uploads.assert_awaited_once()
        v._winner_loop.assert_awaited_once()
        v._weight_loop.assert_awaited_once()
        v._heartbeat_loop.assert_awaited_once()

    def test_winner_cache_is_primed_before_loops_start(
        self, tmp_path, monkeypatch,
    ):
        """A fresh install must not burn its first tempo on an empty cache."""
        v, _, _ = _make_validator(tmp_path, monkeypatch)
        _stub_loops(v)
        order = []
        v._refresh_winner_cache.side_effect = lambda: order.append("refresh")
        v._weight_loop.side_effect = lambda: order.append("weight")

        asyncio.run(v.run())

        assert order == ["refresh", "weight"]

    def test_failed_priming_does_not_block_startup(self, tmp_path, monkeypatch):
        v, _, _ = _make_validator(tmp_path, monkeypatch)
        _stub_loops(v)
        v._refresh_winner_cache.side_effect = RuntimeError("server down")

        asyncio.run(v.run())

        v._weight_loop.assert_awaited_once()


# ---------------------------------------------------------------------------
# Winner loop (decoupled from the eval loop)
# ---------------------------------------------------------------------------


class _StopLoop(Exception):
    pass


class TestWinnerLoop:
    def _run_until_second_sleep(self, v, monkeypatch):
        from trajectoryrl.base import validator as vmod

        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) >= 2:
                raise _StopLoop

        monkeypatch.setattr(vmod.asyncio, "sleep", fake_sleep)
        with pytest.raises(_StopLoop):
            asyncio.run(v._winner_loop())
        return sleeps

    def test_refreshes_every_interval(self, tmp_path, monkeypatch):
        from trajectoryrl.base import validator as vmod

        v, _, _ = _make_validator(tmp_path, monkeypatch)
        v._refresh_winner_cache = AsyncMock()

        sleeps = self._run_until_second_sleep(v, monkeypatch)

        assert v._refresh_winner_cache.await_count == 2
        assert sleeps == [vmod._WINNER_POLL_INTERVAL] * 2

    def test_survives_refresh_errors(self, tmp_path, monkeypatch):
        v, _, _ = _make_validator(tmp_path, monkeypatch)
        v._refresh_winner_cache = AsyncMock(side_effect=RuntimeError("boom"))

        self._run_until_second_sleep(v, monkeypatch)

        assert v._refresh_winner_cache.await_count == 2

    def test_eval_tick_no_longer_refreshes_winner(self, tmp_path, monkeypatch):
        """The winner cache is the winner loop's job: an eval that runs
        for an hour must not be the thing that keeps it fresh."""
        from trajectoryrl.base import validator as vmod

        v, _, _ = _make_validator(tmp_path, monkeypatch, eval_enabled=True)
        v._refresh_winner_cache = AsyncMock()
        monkeypatch.setattr(
            vmod, "fetch_current_epoch", AsyncMock(return_value=None),
        )

        asyncio.run(v._eval_loop_tick())

        v._refresh_winner_cache.assert_not_called()


# ---------------------------------------------------------------------------
# Heartbeat payload
# ---------------------------------------------------------------------------


class TestHeartbeatFields:
    def test_weight_only_omits_eval_telemetry(self, tmp_path, monkeypatch):
        v, _, _ = _make_validator(tmp_path, monkeypatch)
        v._last_set_weights_at = 1234
        v._last_eval_at = 99  # stale value persisted from an eval-mode run
        v._health_issue = "left over"

        assert v._heartbeat_fields() == {"last_set_weights_at": 1234}

    def test_eval_enabled_reports_harness_and_model(self, tmp_path, monkeypatch):
        v, harness_cls, _ = _make_validator(
            tmp_path, monkeypatch, eval_enabled=True,
        )
        harness = harness_cls.return_value
        harness.bench_image_hash = "sha256:bench"
        harness.scenario_image_hash = "sha256:scenario"
        harness.sandbox_version = "4.0.23"
        v._last_set_weights_at = 1234
        v._last_eval_at = 5678
        v._health_issue = "infra: broken"

        assert v._heartbeat_fields() == {
            "last_set_weights_at": 1234,
            "last_eval_at": 5678,
            "bench_image_hash": "sha256:bench",
            "harness_image_hash": "sha256:scenario",
            "bench_version": "4.0.23",
            "llm_model": f"policy:auto (default {v.config.llm_model})",
            "llm_base_url": v.config.llm_base_url,
            "health_issue": "infra: broken",
        }
