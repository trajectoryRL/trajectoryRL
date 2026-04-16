"""Winner Protection for consensus-based winner selection.

Each validator maintains a local WinnerState that tracks the current winner's
hotkey, pack hash, and the consensus score at the time they won.  Challengers
must beat winner_score × (1 + score_delta) to dethrone the winner.  The winner
can also self-update by the same rule.

Higher score wins — the best-performing miner takes emission.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from . import consensus as _consensus_mod

logger = logging.getLogger(__name__)


def _scoring_version() -> int:
    """Read the current SCORING_VERSION dynamically from the consensus module."""
    return _consensus_mod.SCORING_VERSION


@dataclass
class WinnerState:
    """Persisted state for Winner Protection."""

    winner_hotkey: Optional[str] = None
    winner_pack_hash: Optional[str] = None
    winner_score: Optional[float] = None
    winner_uid: Optional[int] = None
    scoring_version: int = 1


def select_winner_with_protection(
    consensus_scores: Dict[str, float],
    state: WinnerState,
    score_delta: float = 0.10,
    pack_hashes: Optional[Dict[str, str]] = None,
    hk_to_uid: Optional[Dict[str, int]] = None,
    disable_winner_protection: bool = False,
) -> Tuple[Optional[str], WinnerState]:
    """Select winner using Winner Protection (highest score wins).

    The current winner defends with their winning score (frozen at time of
    winning).  A new winner is elected only if the highest-scoring miner's
    score exceeds winner_score × (1 + score_delta).

    Args:
        consensus_scores: miner_hotkey -> stake-weighted consensus score
        state: persisted winner state from previous window
        score_delta: fraction by which a challenger must beat the winner's
            score to take over (default 0.10 = 10%)
        pack_hashes: optional miner_hotkey -> pack_hash mapping for
            recording the winner's pack hash
        hk_to_uid: optional miner_hotkey -> on-chain UID mapping,
            stored in WinnerState so set_weights can skip metagraph lookup

    Returns:
        (winner_hotkey, updated_state)
        winner_hotkey is None if no miners have scores.
    """
    pack_hashes = pack_hashes or {}
    hk_to_uid = hk_to_uid or {}

    if not consensus_scores:
        logger.warning("No miners with scores in consensus — no winner")
        return None, state

    sorted_miners = sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)
    best_hk, best_score = sorted_miners[0]

    if (
        disable_winner_protection
        or state.winner_hotkey is None
        or state.winner_hotkey not in consensus_scores
    ):
        reason = "no previous winner" if state.winner_hotkey is None else (
            f"previous winner {state.winner_hotkey[:8]} absent from scores"
        )
        new_state = WinnerState(
            winner_hotkey=best_hk,
            winner_pack_hash=pack_hashes.get(best_hk),
            winner_score=best_score,
            winner_uid=hk_to_uid.get(best_hk),
        )
        logger.info(
            "Winner elected (%s): %s UID %s (consensus score %.4f)",
            reason, best_hk[:8], hk_to_uid.get(best_hk, "?"), best_score,
        )
        return best_hk, new_state

    threshold = state.winner_score * (1 + score_delta)

    if best_score > threshold:
        if best_hk == state.winner_hotkey:
            new_state = WinnerState(
                winner_hotkey=best_hk,
                winner_pack_hash=pack_hashes.get(best_hk, state.winner_pack_hash),
                winner_score=best_score,
                winner_uid=hk_to_uid.get(best_hk, state.winner_uid),
            )
            logger.info(
                "Winner %s self-updated: %.4f → %.4f (cleared δ threshold %.4f)",
                best_hk[:8], state.winner_score, best_score, threshold,
            )
        else:
            new_state = WinnerState(
                winner_hotkey=best_hk,
                winner_pack_hash=pack_hashes.get(best_hk),
                winner_score=best_score,
                winner_uid=hk_to_uid.get(best_hk),
            )
            logger.info(
                "Winner overtake: %s UID %s (%.4f) dethrones %s "
                "(winner_score %.4f, δ threshold %.4f)",
                best_hk[:8], hk_to_uid.get(best_hk, "?"), best_score,
                state.winner_hotkey[:8], state.winner_score, threshold,
            )
        return best_hk, new_state

    if state.winner_hotkey in hk_to_uid:
        state.winner_uid = hk_to_uid[state.winner_hotkey]
    logger.info(
        "Winner %s retains (winner_score %.4f): best challenger %s (%.4f) "
        "does not clear δ threshold (%.4f)",
        state.winner_hotkey[:8], state.winner_score,
        best_hk[:8], best_score, threshold,
    )
    return state.winner_hotkey, state


def save_winner_state(state: WinnerState, path: str):
    """Persist winner state to JSON file."""
    data = {
        "scoring_version": state.scoring_version,
        "winner_hotkey": state.winner_hotkey,
        "winner_pack_hash": state.winner_pack_hash,
        "winner_score": state.winner_score,
        "winner_uid": state.winner_uid,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Winner state saved to %s", path)


def load_winner_state(path: str) -> WinnerState:
    """Load winner state from JSON file, or return fresh state.

    Returns an empty ``WinnerState`` when the persisted
    ``scoring_version`` does not match the running ``SCORING_VERSION``,
    so that an obsolete winner cannot block elections under the new
    evaluation criteria.
    """
    try:
        with open(path) as f:
            data = json.load(f)

        file_sv = data.get("scoring_version", 1)
        current_sv = _scoring_version()
        if file_sv != current_sv:
            logger.warning(
                "Winner state scoring_version mismatch (%d != %d), "
                "resetting winner protection",
                file_sv, current_sv,
            )
            return WinnerState()

        return WinnerState(
            winner_hotkey=data.get("winner_hotkey"),
            winner_pack_hash=data.get("winner_pack_hash"),
            winner_score=data.get("winner_score", data.get("winner_cost")),
            winner_uid=data.get("winner_uid"),
            scoring_version=file_sv,
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return WinnerState()
