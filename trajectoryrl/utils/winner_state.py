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

logger = logging.getLogger(__name__)


@dataclass
class WinnerState:
    """Persisted state for Winner Protection.

    ``spec_number`` records which scoring spec the current winner was
    selected under. It does NOT trigger a state reset on its own, but it
    gates Winner Protection: ``select_winner_with_protection`` bypasses
    the δ threshold whenever ``state.spec_number`` differs from the
    chain-derived ``target_spec_number`` for the round, so cross-spec
    score comparisons never happen. After such a bypass the field is
    overwritten to the new target_spec_number, restoring normal
    protection on subsequent rounds.
    """

    winner_hotkey: Optional[str] = None
    winner_pack_hash: Optional[str] = None
    winner_score: Optional[float] = None
    winner_uid: Optional[int] = None
    spec_number: int = 1


def select_winner_with_protection(
    consensus_scores: Dict[str, float],
    state: WinnerState,
    score_delta: float = 0.10,
    pack_hashes: Optional[Dict[str, str]] = None,
    hk_to_uid: Optional[Dict[str, int]] = None,
    disable_winner_protection: bool = False,
    target_spec_number: Optional[int] = None,
) -> Tuple[Optional[str], WinnerState]:
    """Select winner using Winner Protection (highest score wins).

    The current winner defends with their winning score (frozen at time of
    winning).  A new winner is elected only if the highest-scoring miner's
    score exceeds winner_score × (1 + score_delta).

    Winner Protection only compares scores within a single ``spec_number``.
    When ``target_spec_number`` is provided and differs from
    ``state.spec_number`` (i.e. the chain-derived target spec just flipped
    relative to the persisted winner), the δ threshold is bypassed for
    this round and the highest-scoring eligible miner is elected
    immediately. The returned ``WinnerState`` is then stamped with
    ``spec_number = target_spec_number`` so subsequent rounds resume
    normal protection within the new spec.

    Args:
        consensus_scores: miner_hotkey -> stake-weighted consensus score
        state: persisted winner state from previous window
        score_delta: fraction by which a challenger must beat the winner's
            score to take over (default 0.10 = 10%)
        pack_hashes: optional miner_hotkey -> pack_hash mapping for
            recording the winner's pack hash
        hk_to_uid: optional miner_hotkey -> on-chain UID mapping,
            stored in WinnerState so set_weights can skip metagraph lookup
        disable_winner_protection: if True, bypass the δ threshold
            unconditionally (operator override)
        target_spec_number: chain-derived target spec for this round; when
            set, drives the cross-spec bypass and is stamped onto the
            returned state. If None, behaviour is unchanged from the
            single-spec era and ``state.spec_number`` is preserved.

    Returns:
        (winner_hotkey, updated_state)
        winner_hotkey is None if no miners have scores.
    """
    pack_hashes = pack_hashes or {}
    hk_to_uid = hk_to_uid or {}

    if not consensus_scores:
        logger.warning("No miners with scores in consensus — no winner")
        return None, state

    # Restrict candidates to miners still registered on-chain. A hotkey that
    # was deregistered (UID slot reassigned) can linger in consensus_scores
    # via stale per-validator caches; picking it as winner would route weight
    # to a UID held by a different miner. When hk_to_uid is empty (legacy
    # test path), fall back to consensus_scores as-is.
    if hk_to_uid:
        eligible = {
            hk: s for hk, s in consensus_scores.items() if hk in hk_to_uid
        }
    else:
        eligible = consensus_scores

    if not eligible:
        logger.warning(
            "No eligible miners (all %d scored hotkeys absent from metagraph) "
            "— no winner", len(consensus_scores),
        )
        return None, state

    sorted_miners = sorted(eligible.items(), key=lambda x: x[1], reverse=True)
    best_hk, best_score = sorted_miners[0]

    # spec_number stamped onto any newly constructed WinnerState. Falls back
    # to state.spec_number when the caller did not pass a target (legacy /
    # single-spec test paths) so persistence stays stable.
    new_spec_number = (
        target_spec_number if target_spec_number is not None
        else state.spec_number
    )

    cross_spec_transition = (
        target_spec_number is not None
        and state.winner_hotkey is not None
        and state.spec_number != target_spec_number
    )

    if (
        disable_winner_protection
        or state.winner_hotkey is None
        or state.winner_hotkey not in eligible
        or cross_spec_transition
    ):
        if state.winner_hotkey is None:
            reason = "no previous winner"
        elif cross_spec_transition:
            reason = (
                f"cross-spec transition (state.spec={state.spec_number}, "
                f"target={target_spec_number})"
            )
        elif hk_to_uid and state.winner_hotkey not in hk_to_uid:
            reason = f"previous winner {state.winner_hotkey[:8]} deregistered"
        else:
            reason = (
                f"previous winner {state.winner_hotkey[:8]} absent from scores"
            )
        new_state = WinnerState(
            winner_hotkey=best_hk,
            winner_pack_hash=pack_hashes.get(best_hk),
            winner_score=best_score,
            winner_uid=hk_to_uid.get(best_hk),
            spec_number=new_spec_number,
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
                spec_number=new_spec_number,
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
                spec_number=new_spec_number,
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
    # Retain branch is only reachable when state.spec_number == target (the
    # cross-spec branch above would otherwise have re-elected). Stamping
    # explicitly keeps the field in lock-step with the target even in the
    # legacy single-spec path.
    state.spec_number = new_spec_number
    logger.info(
        "Winner %s retains (winner_score %.4f): best challenger %s (%.4f) "
        "does not clear δ threshold (%.4f)",
        state.winner_hotkey[:8], state.winner_score,
        best_hk[:8], best_score, threshold,
    )
    return state.winner_hotkey, state


def save_winner_state(state: WinnerState, path: str):
    """Persist winner state to JSON file."""
    # Emit both `spec_number` (current) and `scoring_version` (legacy mirror)
    # so older validator binaries still in the rollout pool can read the
    # value back without misclassifying it as "missing" (default 1).
    data = {
        "scoring_version": state.spec_number,
        "spec_number": state.spec_number,
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

    The persisted ``spec_number`` is restored as an audit field; it does NOT
    trigger a reset, even when it differs from the running ``SPEC_NUMBER``.
    The new winner takes over naturally once the chain-derived target spec
    flips (see consensus_filter.select_target_spec_number).

    Accepts either ``spec_number`` (current) or ``scoring_version`` (legacy)
    JSON keys for backward compatibility.
    """
    try:
        with open(path) as f:
            data = json.load(f)

        spec_number = data.get("spec_number", data.get("scoring_version", 1))

        return WinnerState(
            winner_hotkey=data.get("winner_hotkey"),
            winner_pack_hash=data.get("winner_pack_hash"),
            winner_score=data.get("winner_score", data.get("winner_cost")),
            winner_uid=data.get("winner_uid"),
            spec_number=spec_number,
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return WinnerState()
