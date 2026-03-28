"""Submission filter pipeline for consensus aggregation.

6-layer pipeline that filters incoming consensus submissions before
stake-weighted aggregation.  Each layer logs skip counts for diagnosing
low participation.

Pipeline order:
  1. Protocol version  — discard mismatched protocol versions
  2. Window number     — discard submissions from wrong window
  3. Trust threshold   — discard validators below min stake
  4. Data integrity    — discard payloads that fail hash verification
  5. ClawBench version — discard incompatible major versions
  6. Zero-signal       — discard all-zero cost submissions (free-riders)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .consensus import (
    ConsensusPayload, ConsensusPointer,
    CONSENSUS_PROTOCOL_VERSION,
)

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Accumulates per-layer skip counts for diagnostics."""
    total_input: int = 0
    skipped_protocol: int = 0
    skipped_window: int = 0
    skipped_stake: int = 0
    skipped_integrity: int = 0
    skipped_version: int = 0
    skipped_zero_signal: int = 0
    passed: int = 0

    def summary(self) -> str:
        return (
            f"Filter: {self.total_input} in → {self.passed} passed | "
            f"protocol={self.skipped_protocol} window={self.skipped_window} "
            f"stake={self.skipped_stake} integrity={self.skipped_integrity} "
            f"version={self.skipped_version} zero={self.skipped_zero_signal}"
        )


@dataclass
class ValidatedSubmission:
    """A submission that passed all filter layers."""
    pointer: ConsensusPointer
    payload: ConsensusPayload
    validator_stake: float


def _parse_major_version(version_str: str) -> Optional[int]:
    """Extract major version number from semver string."""
    try:
        return int(version_str.split(".")[0])
    except (ValueError, IndexError):
        return None


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
) -> Tuple[List[Tuple[ConsensusPointer, ConsensusPayload]], int]:
    """Layer 2: discard submissions from a different evaluation window."""
    passed = []
    skipped = 0
    for ptr, payload in submissions:
        if payload.window_number != expected_window:
            logger.debug(
                "Filter[window]: skip %s (window %d != %d)",
                ptr.validator_hotkey[:8], payload.window_number, expected_window,
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


def filter_data_integrity(
    submissions: List[Tuple[ConsensusPointer, ConsensusPayload]],
) -> Tuple[List[Tuple[ConsensusPointer, ConsensusPayload]], int]:
    """Layer 4: discard payloads whose content hash doesn't match.

    Re-serializes the payload and checks that the sha256 matches.
    """
    passed = []
    skipped = 0
    for ptr, payload in submissions:
        computed_hash = payload.content_hash()
        # For sha256-addressed pointers, verify hash matches
        if ptr.content_address.startswith("sha256:"):
            if computed_hash != ptr.content_address:
                logger.warning(
                    "Filter[integrity]: skip %s (hash mismatch: pointer=%s, computed=%s)",
                    ptr.validator_hotkey[:8], ptr.content_address[:24], computed_hash[:24],
                )
                skipped += 1
                continue
        passed.append((ptr, payload))
    return passed, skipped


def filter_clawbench_version(
    submissions: List[Tuple[ConsensusPointer, ConsensusPayload]],
    local_version: str,
) -> Tuple[List[Tuple[ConsensusPointer, ConsensusPayload]], int]:
    """Layer 5: discard submissions from incompatible ClawBench major versions.

    Different major versions use different scenarios/criteria/judge logic,
    producing incomparable scores.
    """
    local_major = _parse_major_version(local_version)
    if local_major is None:
        return submissions, 0

    passed = []
    skipped = 0
    for ptr, payload in submissions:
        remote_major = _parse_major_version(payload.clawbench_version)
        if remote_major is None or remote_major != local_major:
            logger.debug(
                "Filter[clawbench_version]: skip %s (v%s, local major=%d)",
                ptr.validator_hotkey[:8], payload.clawbench_version, local_major,
            )
            skipped += 1
        else:
            passed.append((ptr, payload))
    return passed, skipped


def filter_zero_signal(
    submissions: List[Tuple[ConsensusPointer, ConsensusPayload]],
) -> Tuple[List[Tuple[ConsensusPointer, ConsensusPayload]], int]:
    """Layer 6: discard all-zero cost submissions when non-zero signals exist.

    Prevents free-riding validators from diluting legitimate signals.
    If ALL submissions are zero, they all pass (bootstrap scenario).
    """
    has_nonzero = any(
        any(c != 0.0 for c in payload.costs.values())
        for _, payload in submissions
    )

    if not has_nonzero:
        return submissions, 0

    passed = []
    skipped = 0
    for ptr, payload in submissions:
        if all(c == 0.0 for c in payload.costs.values()):
            logger.debug(
                "Filter[zero-signal]: skip %s (all-zero costs)",
                ptr.validator_hotkey[:8],
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
    local_version: str,
    expected_protocol: int = CONSENSUS_PROTOCOL_VERSION,
) -> Tuple[List[ValidatedSubmission], FilterStats]:
    """Run the full 6-layer filter pipeline.

    Returns:
        - List of ValidatedSubmission that passed all layers
        - FilterStats with per-layer skip counts
    """
    stats = FilterStats(total_input=len(submissions))

    current = submissions

    current, n = filter_protocol_version(current, expected_protocol)
    stats.skipped_protocol = n

    current, n = filter_window_number(current, expected_window)
    stats.skipped_window = n

    current, n = filter_trust_threshold(current, validator_stakes, min_stake)
    stats.skipped_stake = n

    current, n = filter_data_integrity(current)
    stats.skipped_integrity = n

    current, n = filter_clawbench_version(current, local_version)
    stats.skipped_version = n

    current, n = filter_zero_signal(current)
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
