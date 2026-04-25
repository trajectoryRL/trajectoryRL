"""Submission filter pipeline for consensus aggregation.

5-layer pipeline that filters incoming consensus submissions before
stake-weighted aggregation.  Each layer logs skip counts for diagnosing
low participation.

Pipeline order:
  1. Protocol version  — discard mismatched protocol versions
  2. Window number     — discard submissions from wrong window
  3. Trust threshold   — discard validators below min stake
  4. spec_number       — discard submissions whose payload spec_number
                          differs from the chain-derived target spec
  5. Zero-signal       — discard all-zero score submissions (free-riders)

The target spec_number used by layer 4 is not read from a static local
constant. Instead, ``select_target_spec_number`` derives it from the
on-chain stake distribution of the surviving submissions: the
stake-weighted dominant spec_number wins if it holds more than 50% of
participating stake; otherwise the validator falls back to its locally
configured ``SPEC_NUMBER``. This keeps SPEC_NUMBER bumps self-coordinating
across validators (the previous winner keeps receiving emissions until
stake-weighted majority migrates to the new spec).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import SPEC_NUMBER
from .consensus import (
    ConsensusPayload, ConsensusPointer,
    CONSENSUS_PROTOCOL_VERSION,
)

logger = logging.getLogger(__name__)


# Stake-share threshold for adopting an on-chain dominant spec_number as
# this round's target. Reuses the disqualify_stake_threshold semantic from
# scoring (>50% stake-weighted majority).
SPEC_MAJORITY_THRESHOLD = 0.5


@dataclass
class FilterStats:
    """Accumulates per-layer skip counts for diagnostics."""
    total_input: int = 0
    skipped_protocol: int = 0
    skipped_window: int = 0
    skipped_stake: int = 0
    skipped_spec_number: int = 0
    skipped_zero_signal: int = 0
    target_spec_number: int = 0
    target_spec_source: str = ""  # "chain_majority" | "local_fallback"
    passed: int = 0

    def summary(self) -> str:
        return (
            f"Filter: {self.total_input} in → {self.passed} passed | "
            f"protocol={self.skipped_protocol} window={self.skipped_window} "
            f"stake={self.skipped_stake} "
            f"spec={self.skipped_spec_number} (target={self.target_spec_number} "
            f"via {self.target_spec_source}) "
            f"zero={self.skipped_zero_signal}"
        )


@dataclass
class ValidatedSubmission:
    """A submission that passed all filter layers."""
    pointer: ConsensusPointer
    payload: ConsensusPayload
    validator_stake: float


def filter_protocol_version(
    submissions: List[Tuple[ConsensusPointer, ConsensusPayload]],
    expected_version: int = CONSENSUS_PROTOCOL_VERSION,
) -> Tuple[List[Tuple[ConsensusPointer, ConsensusPayload]], int]:
    """Layer 1: discard submissions with mismatched protocol version."""
    passed = []
    skipped = 0
    for ptr, payload in submissions:
        if payload.protocol_version != expected_version:
            logger.debug(
                "Filter[protocol]: skip %s (v%d != v%d)",
                ptr.validator_hotkey[:8], payload.protocol_version, expected_version,
            )
            skipped += 1
        else:
            passed.append((ptr, payload))
    return passed, skipped


def filter_window_number(
    submissions: List[Tuple[ConsensusPointer, ConsensusPayload]],
    expected_window: int,
    accept_newer: bool = False,
) -> Tuple[List[Tuple[ConsensusPointer, ConsensusPayload]], int]:
    """Layer 2: discard submissions from a different evaluation window.

    Default (``accept_newer=False``) requires strict equality
    (``payload.window_number == expected_window``).

    When ``accept_newer=True``, payloads with
    ``payload.window_number >= expected_window`` are kept. This is used by
    startup aggregation: validators that have already moved past the
    "ripe" window have their newer commitment treated as a stand-in vote,
    because Bittensor's ``get_all_commitments`` only returns each
    validator's *latest* commitment (older ones are overwritten and
    cannot be retrieved). Older-than-expected payloads are still
    rejected in both modes.
    """
    passed = []
    skipped = 0
    for ptr, payload in submissions:
        if accept_newer:
            keep = payload.window_number >= expected_window
            cmp = ">="
        else:
            keep = payload.window_number == expected_window
            cmp = "=="
        if not keep:
            logger.debug(
                "Filter[window]: skip %s (window %d not %s %d)",
                ptr.validator_hotkey[:8], payload.window_number,
                cmp, expected_window,
            )
            skipped += 1
        else:
            passed.append((ptr, payload))
    return passed, skipped


def filter_trust_threshold(
    submissions: List[Tuple[ConsensusPointer, ConsensusPayload]],
    validator_stakes: Dict[str, float],
    min_stake: float,
) -> Tuple[List[Tuple[ConsensusPointer, ConsensusPayload]], int]:
    """Layer 3: discard submissions from validators below minimum stake."""
    passed = []
    skipped = 0
    for ptr, payload in submissions:
        stake = validator_stakes.get(ptr.validator_hotkey, 0.0)
        if stake < min_stake:
            logger.debug(
                "Filter[stake]: skip %s (stake %.2f < min %.2f)",
                ptr.validator_hotkey[:8], stake, min_stake,
            )
            skipped += 1
        else:
            passed.append((ptr, payload))
    return passed, skipped


def select_target_spec_number(
    submissions: List[Tuple[ConsensusPointer, ConsensusPayload]],
    validator_stakes: Dict[str, float],
    local_spec: int,
    threshold: float = SPEC_MAJORITY_THRESHOLD,
) -> Tuple[int, str]:
    """Choose this round's target spec_number from on-chain data.

    Algorithm:
      1. Group submissions by ``payload.spec_number``.
      2. Sum validator stake per group.
      3. dominant = group with the largest total stake.
      4. If ``dominant.stake / total_stake > threshold`` → use dominant.spec_number.
         Otherwise fall back to ``local_spec``.

    A tied-for-max set of groups always fails the strict ``> threshold``
    check (since two groups summing > 100% is impossible), and the function
    falls back to ``local_spec`` — no explicit tiebreak is needed.

    Args:
        submissions: Surviving submissions after basic (protocol / window /
            stake) filters. The ``payload.spec_number`` field is the spec
            label this validator self-attested to.
        validator_stakes: ``{validator_hotkey: stake}`` snapshot from the
            metagraph. Missing hotkeys are treated as zero stake.
        local_spec: Validator's locally configured ``SPEC_NUMBER``, used as
            the fallback target when no group reaches majority.
        threshold: Stake-share required for adoption (default 0.5).

    Returns:
        ``(target_spec_number, source)`` where ``source`` is
        ``"chain_majority"`` or ``"local_fallback"``.
    """
    if not submissions:
        return local_spec, "local_fallback"

    stake_by_spec: Dict[int, float] = {}
    for ptr, payload in submissions:
        stake = validator_stakes.get(ptr.validator_hotkey, 0.0)
        if stake <= 0:
            continue
        stake_by_spec[payload.spec_number] = (
            stake_by_spec.get(payload.spec_number, 0.0) + stake
        )

    total_stake = sum(stake_by_spec.values())
    if total_stake <= 0:
        return local_spec, "local_fallback"

    dominant_spec = max(stake_by_spec, key=lambda s: stake_by_spec[s])
    dominant_share = stake_by_spec[dominant_spec] / total_stake

    if dominant_share > threshold:
        return dominant_spec, "chain_majority"
    return local_spec, "local_fallback"


def filter_spec_number(
    submissions: List[Tuple[ConsensusPointer, ConsensusPayload]],
    target_spec_number: int,
) -> Tuple[List[Tuple[ConsensusPointer, ConsensusPayload]], int]:
    """Layer 4: discard submissions whose spec_number != target."""
    passed = []
    skipped = 0
    for ptr, payload in submissions:
        if payload.spec_number != target_spec_number:
            logger.debug(
                "Filter[spec_number]: skip %s (spec=%d != target=%d)",
                ptr.validator_hotkey[:8], payload.spec_number, target_spec_number,
            )
            skipped += 1
        else:
            passed.append((ptr, payload))
    return passed, skipped


def filter_zero_signal(
    submissions: List[Tuple[ConsensusPointer, ConsensusPayload]],
    zero_threshold: float = 1.0,
) -> Tuple[List[Tuple[ConsensusPointer, ConsensusPayload]], int]:
    """Layer 5: discard near-zero-signal submissions when real signal exists.

    A submission is dropped when the fraction of zero scores meets or exceeds
    ``zero_threshold``.  Default 1.0 keeps the legacy behaviour (drop only
    strictly all-zero payloads).  Lower values (e.g. 0.95) treat free-rider
    payloads that sprinkle one or two nonzero scores as all-zero.

    If no submission has any nonzero scores, all pass through (bootstrap
    scenario — nothing to compare against).
    """
    has_nonzero = any(
        any(s != 0.0 for s in payload.scores.values())
        for _, payload in submissions
    )

    if not has_nonzero:
        return submissions, 0

    passed = []
    skipped = 0
    for ptr, payload in submissions:
        scores = payload.scores
        if not scores:
            logger.debug(
                "Filter[zero-signal]: skip %s (empty scores)",
                ptr.validator_hotkey[:8],
            )
            skipped += 1
            continue
        zero_count = sum(1 for s in scores.values() if s == 0.0)
        zero_ratio = zero_count / len(scores)
        if zero_ratio >= zero_threshold:
            logger.debug(
                "Filter[zero-signal]: skip %s (%d/%d = %.1f%% zero >= %.1f%%)",
                ptr.validator_hotkey[:8],
                zero_count, len(scores), 100 * zero_ratio, 100 * zero_threshold,
            )
            skipped += 1
        else:
            passed.append((ptr, payload))
    return passed, skipped


def run_filter_pipeline(
    submissions: List[Tuple[ConsensusPointer, ConsensusPayload]],
    expected_window: int,
    validator_stakes: Dict[str, float],
    min_stake: float,
    local_spec_number: int = SPEC_NUMBER,
    expected_protocol: int = CONSENSUS_PROTOCOL_VERSION,
    zero_signal_threshold: float = 1.0,
    spec_majority_threshold: float = SPEC_MAJORITY_THRESHOLD,
    accept_newer_windows: bool = False,
) -> Tuple[List[ValidatedSubmission], FilterStats]:
    """Run the full 5-layer filter pipeline.

    Layers run in order: protocol → window → stake → spec_number → zero-signal.

    The spec_number target is derived from the surviving submissions after
    the first three layers — see :func:`select_target_spec_number`. Callers
    provide their ``local_spec_number`` only as the fallback target when no
    on-chain group reaches majority; it is not used to pre-filter.

    ``accept_newer_windows`` relaxes layer 2 to ``payload.window_number >=
    expected_window`` (see :func:`filter_window_number`). Used by startup
    aggregation only — main-loop aggregation must keep the strict default.

    Returns:
        - List of ValidatedSubmission that passed all layers
        - FilterStats with per-layer skip counts and the target spec_number
          used (plus its source).
    """
    stats = FilterStats(total_input=len(submissions))

    current = submissions

    current, n = filter_protocol_version(current, expected_protocol)
    stats.skipped_protocol = n

    current, n = filter_window_number(
        current, expected_window, accept_newer=accept_newer_windows,
    )
    stats.skipped_window = n

    current, n = filter_trust_threshold(current, validator_stakes, min_stake)
    stats.skipped_stake = n

    target_spec, source = select_target_spec_number(
        current, validator_stakes, local_spec_number,
        threshold=spec_majority_threshold,
    )
    stats.target_spec_number = target_spec
    stats.target_spec_source = source

    current, n = filter_spec_number(current, target_spec)
    stats.skipped_spec_number = n

    current, n = filter_zero_signal(current, zero_threshold=zero_signal_threshold)
    stats.skipped_zero_signal = n

    validated = []
    for ptr, payload in current:
        stake = validator_stakes.get(ptr.validator_hotkey, 0.0)
        validated.append(ValidatedSubmission(
            pointer=ptr,
            payload=payload,
            validator_stake=stake,
        ))

    stats.passed = len(validated)
    logger.info(stats.summary())

    return validated, stats
