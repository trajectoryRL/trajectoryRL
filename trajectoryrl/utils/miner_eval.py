"""Shared Season 1 single-miner evaluation function.

This helper is the single source of truth for evaluating one miner's
submitted pack via the trajrl-bench sandbox. It is consumed by both:

- ``TrajectoryValidator._evaluate_miner`` — the production validator
  loop, which wraps the result back into the existing pipeline dict
  shape (``qualified``, ``judge_details``, ``_s1_sandbox_result``, ...)
  and updates ``self._disqualified_miners``.
- ``scripts/eval_miners.py`` — a CLI tool that calls the same function
  and renders human-readable summaries.

Keeping a single implementation guarantees that CLI debugging mirrors
the actual validator's evaluation pipeline (pack verify -> SKILL.md
extraction -> trajrl-bench harness call -> per-episode transcript
logging -> judge_details build).

The function takes ``harness`` and ``pack_fetcher`` as parameters
(rather than constructing them) so callers control their lifecycle and
configuration. It does not touch wallet, subtensor, metagraph, or
consensus subsystems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .commitments import MinerCommitment
from .github import PackFetcher
from .sandbox_harness import (
    SandboxEvaluationResult,
    TrajectorySandboxHarness,
    _EpisodeResult,
    _looks_like_provider_failure,
)


# Skip-reason taxonomy — kept stable for callers that branch on these values
# (e.g. validator's ``_disqualified_miners``).
SKIP_PACK_VERIFY = "pack_verify"          # silent skip in validator (pre-eval reject)
SKIP_INVALID_PACK = "invalid_pack_structure"  # pack JSON parsed but root / files not an object
SKIP_MISSING_SKILL_MD = "missing_skill_md"
SKIP_EMPTY_SKILL_MD = "empty_skill_md"
SKIP_S1_EVAL_ERROR = "s1_eval_error"
# Harness gave up on remaining scenarios because the active challenge
# epoch had moved on. The validator discards the partial session rather
# than POSTing a score whose missing scenarios would be counted as 0.
SKIP_EPOCH_CHANGED = "epoch_changed_mid_eval"
# Hermes itself failed on every scenario this session (non-zero exit
# code or deadline kill, with no billed LLM call anywhere). Catches
# OpenRouter HTTP 4xx, OpenAI 5xx, local-server connection refused,
# hermes OOM, etc. — any failure that surfaces as "hermes did not
# produce real output for any scenario". The verifier still ran and
# scored baseline-credit on empty agent outputs; that's not a miner
# signal. Discard rather than POST — silently shipping these scores
# would punish miners for OUR infra failure and pollute consensus.
SKIP_PROVIDER_FAILURE = "llm_provider_failure"


@dataclass
class MinerEvalOutcome:
    """Outcome of a single miner's S1 sandbox evaluation."""

    success: bool
    sandbox_result: Optional[SandboxEvaluationResult] = None
    skip_reason: Optional[str] = None  # one of SKIP_* constants when success=False
    skip_detail: Optional[str] = None  # extra context (e.g. verification error msg)
    judge_details: Optional[Dict[str, Dict]] = None  # validator-pipeline shape


async def evaluate_miner_s1(
    *,
    harness: TrajectorySandboxHarness,
    pack_fetcher: PackFetcher,
    commitment: MinerCommitment,
    epoch_seed: int,
    validator_salt: str = "",
    mlog: Optional[logging.Logger] = None,
    on_episode_start: Optional[Callable[[str, int, int], None]] = None,
    on_episode_verifying: Optional[Callable[[str, int, int], None]] = None,
    on_episode_done: Optional[Callable[[_EpisodeResult, int, int], None]] = None,
    is_epoch_still_current: Optional[Callable[[str], bool]] = None,
) -> MinerEvalOutcome:
    """Evaluate a single miner's pack via trajrl-bench (Season 1).

    Flow (matches the original ``TrajectoryValidator._evaluate_miner``
    + ``_evaluate_miner_s1`` byte-for-byte):

    1. Fetch + verify pack from the commitment URL (hash check).
    2. Extract and validate ``SKILL.md`` from the pack.
    3. Run ``harness.evaluate_miner(...)`` for N episodes.
    4. Build the validator-pipeline ``judge_details`` dict.

    Args:
        harness: Pre-initialized sandbox harness (caller pulls images).
        pack_fetcher: Pre-initialized fetcher (caller chooses cache dir).
        commitment: Miner's on-chain commitment.
        epoch_seed: Epoch-derived RNG seed for fixture generation.
        validator_salt: Extra entropy for fixture generation. Empty
            string lets the harness fall back to its built-in default.
        mlog: Optional per-miner logger for transcript tails. Falls back
            to module logger when omitted.

    Returns:
        ``MinerEvalOutcome`` describing success or skip reason. On
        success, ``sandbox_result`` and ``judge_details`` are populated.
    """
    log = mlog or logging.getLogger(__name__)

    # Step 1: Fetch and verify submission from HTTP URL
    verification = await pack_fetcher.verify_submission(
        pack_url=commitment.pack_url,
        pack_hash=commitment.pack_hash,
    )

    if not verification.valid:
        log.warning(
            f"Pack verification failed: {verification.error}"
        )
        return MinerEvalOutcome(
            success=False,
            skip_reason=SKIP_PACK_VERIFY,
            skip_detail=verification.error,
        )

    pack = verification.pack_content
    if not isinstance(pack, dict):
        # Hash check passes for any well-formed JSON, including arrays /
        # scalars / null. Without this guard, ``pack.get(...)`` below would
        # raise AttributeError and propagate out of the whole eval cycle.
        log.warning(
            "Pack root is %s, expected JSON object", type(pack).__name__
        )
        return MinerEvalOutcome(
            success=False,
            skip_reason=SKIP_INVALID_PACK,
            skip_detail=f"pack root is {type(pack).__name__}, expected object",
        )

    files = pack.get("files")
    if not isinstance(files, dict):
        log.warning(
            "Pack 'files' is %s, expected JSON object",
            type(files).__name__,
        )
        return MinerEvalOutcome(
            success=False,
            skip_reason=SKIP_INVALID_PACK,
            skip_detail=f"'files' is {type(files).__name__}, expected object",
        )

    if "SKILL.md" not in files:
        log.warning("Pack missing SKILL.md in files")
        return MinerEvalOutcome(
            success=False,
            skip_reason=SKIP_MISSING_SKILL_MD,
        )

    extra_files = [f for f in files if f != "SKILL.md"]
    if extra_files:
        log.warning("S1 pack contains unexpected files: %s", extra_files)

    skill_md = files.get("SKILL.md")
    if not isinstance(skill_md, str) or not skill_md.strip():
        log.warning(
            "SKILL.md is %s or empty",
            type(skill_md).__name__,
        )
        return MinerEvalOutcome(
            success=False,
            skip_reason=SKIP_EMPTY_SKILL_MD,
        )

    log.info(
        "S1 evaluation: %d scenarios, skill_md=%d chars",
        len(harness.sandbox_scenarios), len(skill_md),
    )

    # Step 2: Run trajrl-bench sandbox evaluation
    try:
        result = await harness.evaluate_miner(
            skill_md=skill_md,
            epoch_seed=epoch_seed,
            pack_hash=commitment.pack_hash,
            validator_salt=validator_salt,
            on_episode_start=on_episode_start,
            on_episode_verifying=on_episode_verifying,
            on_episode_done=on_episode_done,
            is_epoch_still_current=is_epoch_still_current,
        )
    except Exception as e:
        log.error("S1 evaluation failed: %s", e, exc_info=True)
        return MinerEvalOutcome(
            success=False,
            skip_reason=SKIP_S1_EVAL_ERROR,
            skip_detail=str(e),
        )

    if result.error:
        log.warning("S1 evaluation error: %s", result.error)
        return MinerEvalOutcome(
            success=False,
            sandbox_result=result,
            skip_reason=SKIP_S1_EVAL_ERROR,
            skip_detail=result.error,
        )

    if result.aborted_mid_session:
        log.info(
            "S1 evaluation aborted mid-session: epoch moved on after "
            "%d/%d scenarios; discarding partial result",
            len(result.scenarios), len(harness.sandbox_scenarios),
        )
        return MinerEvalOutcome(
            success=False,
            sandbox_result=result,
            skip_reason=SKIP_EPOCH_CHANGED,
        )

    # Infra-failure gate: when hermes itself failed on every scenario
    # (non-zero exit or deadline kill, with no billed LLM call anywhere
    # in the session), the verifier still ran but on empty agent
    # outputs — the resulting baseline-credit scores look like a
    # legitimately-low miner score but are infra-poisoned. POSTing them
    # would punish honest miners for our key/credits/network failure.
    # The 2026-05-16 incident: OpenRouter HTTP 402 across every call.
    if _looks_like_provider_failure(result.session_result.episodes):
        log.error(
            "S1 evaluation: hermes failed on every scenario (no LLM "
            "round-trip produced output and nothing was billed). "
            "Session result is infra-poisoned, not a real miner score. "
            "Discarding; check the validator's LLM key/credits/network."
        )
        return MinerEvalOutcome(
            success=False,
            sandbox_result=result,
            skip_reason=SKIP_PROVIDER_FAILURE,
        )

    # Step 3: Per-episode transcript tail logging (file-only via mlog)
    for ep in result.session_result.episodes:
        if ep.transcript:
            log.info(
                "[%s] testee transcript tail (%d chars):\n%s",
                ep.scenario, len(ep.transcript), ep.transcript[-3000:],
            )

    # Step 4: Map S1 result to validator pipeline judge_details.
    # One entry per scenario, keyed by scenario name. ``overall_score``
    # is the per-scenario quality (passed/total ∈ [0, 1]); the headline
    # session score = Σ qualities is computed downstream by
    # _fire_submit_eval.
    log.info(
        "S1 result: final_score=%.3f, mean_quality=%.3f, scenarios=%s",
        result.score, result.mean_quality, result.scenario_qualities,
    )

    judge_details: Dict[str, Dict] = {}
    for scenario in result.scenarios:
        quality = float(result.scenario_qualities.get(scenario, 0.0))
        cost = result.scenario_costs_usd.get(scenario)
        judge_details[scenario] = {
            "overall_score": round(quality, 4),
            "qualification_gate": quality > 0.0,
            "cost_usd": round(cost, 6) if cost is not None else None,
            "harness": "trajrl-bench",
            "sandbox_version": harness.sandbox_version,
        }

    return MinerEvalOutcome(
        success=True,
        sandbox_result=result,
        judge_details=judge_details,
    )
