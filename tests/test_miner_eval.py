"""Tests for the validator-side miner evaluation pipeline.

Specifically covers the skip-reason taxonomy that the validator uses to
decide whether to POST a score for an epoch or discard the evaluation.

The pack-verify / SKILL.md-structure skip paths are not retested here —
those are covered indirectly via the pack-validation tests in
``test_miner.py``. This file focuses on the post-eval skip decision
that depends on the harness's session output (provider-failure), which
needs a mocked harness to exercise.
"""

from __future__ import annotations

import hashlib
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock bittensor so importing trajectoryrl.* doesn't pull in the SDK.
_mock_bt = MagicMock()
_mock_bt.Synapse = type("Synapse", (), {})
sys.modules["bittensor"] = _mock_bt

from trajectoryrl.utils.miner_eval import (
    SKIP_PROVIDER_FAILURE,
    evaluate_miner_s1,
)
from trajectoryrl.utils.sandbox_harness import (
    SandboxEvaluationResult,
    _EpisodeResult,
    _SessionResult,
)


def _commitment(pack_bytes: bytes) -> MagicMock:
    """Stand-in for ``MinerCommitment`` — only the fields the pipeline reads."""
    c = MagicMock()
    c.hotkey = "5HotkeyAbcdef" + "0" * 35
    c.uid = 42
    c.pack_url = "https://example.com/pack.json"
    c.pack_hash = hashlib.sha256(pack_bytes).hexdigest()
    return c


def _verified_pack_fetcher(pack: dict) -> MagicMock:
    """Pack fetcher that always returns a verified pack matching the
    commitment hash — bypasses the network + hash check so we can drive
    the post-eval pipeline directly."""
    pack_bytes = json.dumps(pack).encode()
    fetcher = MagicMock()
    fetcher.verify_submission = AsyncMock(return_value=MagicMock(
        valid=True,
        error=None,
        pack_content=pack,
    ))
    fetcher._pack_bytes = pack_bytes  # convenience for the test
    return fetcher


def _harness_returning(result: SandboxEvaluationResult) -> MagicMock:
    """Mock harness that bypasses Docker — returns a canned eval result."""
    h = MagicMock()
    h.sandbox_scenarios = ["a", "b", "c"]
    h.sandbox_version = "test"
    h.evaluate_miner = AsyncMock(return_value=result)
    return h


def _make_eval_result(
    episodes: list[_EpisodeResult],
) -> SandboxEvaluationResult:
    sr = _SessionResult(episodes=episodes)
    sr.compute_scores()
    result = SandboxEvaluationResult(sr)
    # Mirror what the harness populates on the result for downstream
    # consumers (miner_eval reads these via attribute access).
    result.scenarios = [ep.scenario for ep in episodes]
    result.scenario_qualities = {ep.scenario: ep.quality for ep in episodes}
    result.scenario_costs_usd = {ep.scenario: ep.cost_usd for ep in episodes}
    result.mean_quality = sr.mean_quality
    result.session_result = sr
    return result


@pytest.mark.asyncio
async def test_provider_failure_skips_with_dedicated_reason():
    """Every scenario's hermes process exited non-zero (LLM round-trip
    never produced output) and no cost was billed → outcome must be
    ``SKIP_PROVIDER_FAILURE``, not a poisoned score submission."""
    eps = [
        _EpisodeResult(
            episode_index=i, scenario=name,
            quality=0.0, cost_usd=None,
            chat_exit=1, agent_output_missing=True,
            transcript="",
        )
        for i, name in enumerate(("a", "b", "c"))
    ]
    pack = {"schema_version": 1, "files": {"SKILL.md": "# Real skill\n"}}
    outcome = await evaluate_miner_s1(
        harness=_harness_returning(_make_eval_result(eps)),
        pack_fetcher=_verified_pack_fetcher(pack),
        commitment=_commitment(json.dumps(pack).encode()),
        epoch_seed=123,
    )
    assert outcome.success is False
    assert outcome.skip_reason == SKIP_PROVIDER_FAILURE


@pytest.mark.asyncio
async def test_real_low_score_still_submits():
    """A pack that genuinely failed (low quality) but DID call the LLM
    must NOT be treated as infra failure — cost is recorded, evaluation
    is real, the network deserves the signal."""
    eps = [
        _EpisodeResult(
            episode_index=0, scenario="a",
            quality=0.0, cost_usd=0.21, chat_exit=0,
            transcript="agent output: <wrong answer>",
        ),
        _EpisodeResult(
            episode_index=1, scenario="b",
            quality=0.125, cost_usd=0.04, chat_exit=0,
            transcript="agent output: <partial>",
        ),
    ]
    pack = {"schema_version": 1, "files": {"SKILL.md": "# Real skill\n"}}
    outcome = await evaluate_miner_s1(
        harness=_harness_returning(_make_eval_result(eps)),
        pack_fetcher=_verified_pack_fetcher(pack),
        commitment=_commitment(json.dumps(pack).encode()),
        epoch_seed=123,
    )
    assert outcome.success is True
    assert outcome.skip_reason is None


