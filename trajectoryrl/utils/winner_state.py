"""Winner Protection for consensus-based winner selection.

Each validator maintains a local WinnerState that tracks the current winner's
hotkey, pack hash, and the consensus cost at the time they won.  Challengers
must beat winner_cost × (1 - cost_delta) to dethrone the winner.  The winner
can also self-update by the same rule.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WinnerState:
    """Persisted state for Winner Protection."""

    winner_hotkey: Optional[str] = None
    winner_pack_hash: Optional[str] = None
    winner_cost: Optional[float] = None
    winner_uid: Optional[int] = None


def select_winner_with_protection(
    consensus_costs: Dict[str, float],
    consensus_qualified: Dict[str, bool],
    state: WinnerState,
    cost_delta: float = 0.10,
    pack_hashes: Optional[Dict[str, str]] = None,
    hk_to_uid: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[str], WinnerState]:
    """Select winner using Winner Protection.

    The current winner defends with their winning cost (frozen at time of
    winning).  A new winner is elected only if the lowest-cost qualified
    miner's cost is below winner_cost × (1 - cost_delta).

    Args:
        consensus_costs: miner_hotkey -> stake-weighted consensus cost
        consensus_qualified: miner_hotkey -> consensus qualification
        state: persisted winner state from previous window
        cost_delta: fraction by which a miner must beat the winner's
            winning cost to take over (default 0.10 = 10%)
        pack_hashes: optional miner_hotkey -> pack_hash mapping for
            recording the winner's pack hash
        hk_to_uid: optional miner_hotkey -> on-chain UID mapping,
            stored in WinnerState so set_weights can skip metagraph lookup

    Returns:
        (winner_hotkey, updated_state)
        winner_hotkey is None if no qualified miners exist.
    """
    pack_hashes = pack_hashes or {}
    hk_to_uid = hk_to_uid or {}

    qualified_miners = {
        hk: cost for hk, cost in consensus_costs.items()
        if consensus_qualified.get(hk, False)
    }

    if not qualified_miners:
        logger.warning("No qualified miners in consensus — no winner")
        return None, state

    sorted_miners = sorted(qualified_miners.items(), key=lambda x: x[1])
    lowest_hk, lowest_cost = sorted_miners[0]

    # No current winner or winner disqualified → lowest cost takes over
    if (
        state.winner_hotkey is None
        or state.winner_hotkey not in qualified_miners
    ):
        reason = "no previous winner" if state.winner_hotkey is None else (
            f"previous winner {state.winner_hotkey[:8]} disqualified"
        )
        new_state = WinnerState(
            winner_hotkey=lowest_hk,
            winner_pack_hash=pack_hashes.get(lowest_hk),
            winner_cost=lowest_cost,
            winner_uid=hk_to_uid.get(lowest_hk),
        )
        logger.info(
            "Winner elected (%s): %s UID %s (consensus cost $%.4f)",
            reason, lowest_hk[:8], hk_to_uid.get(lowest_hk, "?"), lowest_cost,
        )
        return lowest_hk, new_state

    # Winner is qualified — check if anyone beats the protection threshold
    threshold = state.winner_cost * (1 - cost_delta)

    if lowest_cost < threshold:
        if lowest_hk == state.winner_hotkey:
            # Winner self-update
            new_state = WinnerState(
                winner_hotkey=lowest_hk,
                winner_pack_hash=pack_hashes.get(lowest_hk, state.winner_pack_hash),
                winner_cost=lowest_cost,
                winner_uid=hk_to_uid.get(lowest_hk, state.winner_uid),
            )
            logger.info(
                "Winner %s self-updated: $%.4f → $%.4f (cleared δ threshold $%.4f)",
                lowest_hk[:8], state.winner_cost, lowest_cost, threshold,
            )
        else:
            # Challenger dethrones winner
            new_state = WinnerState(
                winner_hotkey=lowest_hk,
                winner_pack_hash=pack_hashes.get(lowest_hk),
                winner_cost=lowest_cost,
                winner_uid=hk_to_uid.get(lowest_hk),
            )
            logger.info(
                "Winner overtake: %s UID %s ($%.4f) dethrones %s "
                "(winner_cost $%.4f, δ threshold $%.4f)",
                lowest_hk[:8], hk_to_uid.get(lowest_hk, "?"), lowest_cost,
                state.winner_hotkey[:8], state.winner_cost, threshold,
            )
        return lowest_hk, new_state

    # No one beats the threshold → winner retains;
    # refresh UID in case it changed due to re-registration
    if state.winner_hotkey in hk_to_uid:
        state.winner_uid = hk_to_uid[state.winner_hotkey]
    logger.info(
        "Winner %s retains (winner_cost $%.4f): best challenger %s ($%.4f) "
        "does not clear δ threshold ($%.4f)",
        state.winner_hotkey[:8], state.winner_cost,
        lowest_hk[:8], lowest_cost, threshold,
    )
    return state.winner_hotkey, state


def save_winner_state(state: WinnerState, path: str):
    """Persist winner state to JSON file."""
    data = {
        "winner_hotkey": state.winner_hotkey,
        "winner_pack_hash": state.winner_pack_hash,
        "winner_cost": state.winner_cost,
        "winner_uid": state.winner_uid,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Winner state saved to %s", path)


def load_winner_state(path: str) -> WinnerState:
    """Load winner state from JSON file, or return fresh state."""
    try:
        with open(path) as f:
            data = json.load(f)
        return WinnerState(
            winner_hotkey=data.get("winner_hotkey"),
            winner_pack_hash=data.get("winner_pack_hash"),
            winner_cost=data.get("winner_cost"),
            winner_uid=data.get("winner_uid"),
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return WinnerState()
