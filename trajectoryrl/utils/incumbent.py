"""Incumbent advantage and cross-window historical best tracking.

Stabilizes winner selection against LLM variance by:
  1. Tracking each miner's best consensus cost within a season.
  2. Requiring challengers to beat the incumbent's historical best
     by a margin (default 10%) to dethrone them.
  3. Resetting historical bests at season boundaries.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class IncumbentState:
    """Persisted state for incumbent advantage tracking."""

    # miner hotkey -> best consensus cost achieved in current season
    historical_best: Dict[str, float] = field(default_factory=dict)

    # current incumbent miner hotkey (winner from last window)
    incumbent_hotkey: Optional[str] = None

    # current season number (derived from window_number // season_length)
    current_season: int = 0

    # window number of the last update
    last_window: int = -1


def select_winner_with_incumbent(
    consensus_costs: Dict[str, float],
    consensus_qualified: Dict[str, bool],
    state: IncumbentState,
    window_number: int,
    season_length: int,
    cost_delta: float = 0.10,
) -> Tuple[Optional[str], IncumbentState]:
    """Select winner using incumbent advantage and historical best.

    Uses the same cost_delta threshold as first-mover protection:
    a challenger must be at least cost_delta (10%) cheaper than
    the incumbent's historical best to dethrone.

    Args:
        consensus_costs: miner_hotkey -> stake-weighted consensus cost
        consensus_qualified: miner_hotkey -> consensus qualification
        state: persisted incumbent state from previous window
        window_number: current evaluation window number
        season_length: number of windows per season
        cost_delta: fraction by which challenger must beat
            incumbent's historical best (default 0.10 = 10%)

    Returns:
        (winner_hotkey, updated_state)
        winner_hotkey is None if no qualified miners exist.
    """
    updated = IncumbentState(
        historical_best=dict(state.historical_best),
        incumbent_hotkey=state.incumbent_hotkey,
        current_season=state.current_season,
        last_window=window_number,
    )

    current_season = window_number // season_length if season_length > 0 else 0
    if current_season != updated.current_season:
        logger.info(
            "Season boundary: %d → %d, resetting historical bests",
            updated.current_season, current_season,
        )
        updated.historical_best = {}
        updated.incumbent_hotkey = None
        updated.current_season = current_season

    # Update historical bests (only for qualified miners — disqualified miners
    # may have artificially low costs from gaming, which should not be recorded)
    for miner_hk, cost in consensus_costs.items():
        if not consensus_qualified.get(miner_hk, False):
            continue
        if miner_hk not in updated.historical_best:
            updated.historical_best[miner_hk] = cost
        else:
            updated.historical_best[miner_hk] = min(
                updated.historical_best[miner_hk], cost
            )

    # Filter to qualified miners
    qualified_miners = {
        hk: cost for hk, cost in consensus_costs.items()
        if consensus_qualified.get(hk, False)
    }

    if not qualified_miners:
        logger.warning("No qualified miners in consensus — no winner")
        return None, updated

    # Sort by consensus cost ascending
    sorted_miners = sorted(qualified_miners.items(), key=lambda x: x[1])
    lowest_cost_hk, lowest_cost = sorted_miners[0]

    # No incumbent: lowest cost wins directly
    if updated.incumbent_hotkey is None:
        updated.incumbent_hotkey = lowest_cost_hk
        logger.info(
            "No incumbent — %s wins with consensus cost $%.4f",
            lowest_cost_hk[:8], lowest_cost,
        )
        return lowest_cost_hk, updated

    # Incumbent still qualified?
    incumbent_hk = updated.incumbent_hotkey
    if incumbent_hk not in qualified_miners:
        # Incumbent disqualified — lowest cost takes over
        updated.incumbent_hotkey = lowest_cost_hk
        logger.info(
            "Incumbent %s disqualified — %s takes over (cost $%.4f)",
            incumbent_hk[:8], lowest_cost_hk[:8], lowest_cost,
        )
        return lowest_cost_hk, updated

    # Incumbent is the cheapest — retains
    if lowest_cost_hk == incumbent_hk:
        logger.info(
            "Incumbent %s retains — lowest consensus cost $%.4f",
            incumbent_hk[:8], qualified_miners[incumbent_hk],
        )
        return incumbent_hk, updated

    # Challenger must beat incumbent's historical best by margin
    incumbent_best = updated.historical_best.get(
        incumbent_hk, qualified_miners[incumbent_hk]
    )
    threshold = incumbent_best * (1 - cost_delta)

    if lowest_cost < threshold:
        logger.info(
            "Incumbent overtake: %s ($%.4f) beats %s historical best "
            "$%.4f × %.0f%% = $%.4f",
            lowest_cost_hk[:8], lowest_cost,
            incumbent_hk[:8], incumbent_best,
            (1 - cost_delta) * 100, threshold,
        )
        updated.incumbent_hotkey = lowest_cost_hk
        return lowest_cost_hk, updated
    else:
        logger.info(
            "Incumbent %s retains: challenger %s ($%.4f) does not "
            "clear margin (required < $%.4f)",
            incumbent_hk[:8], lowest_cost_hk[:8], lowest_cost, threshold,
        )
        return incumbent_hk, updated


def save_incumbent_state(state: IncumbentState, path: str):
    """Persist incumbent state to JSON file."""
    data = {
        "historical_best": state.historical_best,
        "incumbent_hotkey": state.incumbent_hotkey,
        "current_season": state.current_season,
        "last_window": state.last_window,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Incumbent state saved to %s", path)


def load_incumbent_state(path: str) -> IncumbentState:
    """Load incumbent state from JSON file, or return fresh state."""
    try:
        with open(path) as f:
            data = json.load(f)
        return IncumbentState(
            historical_best=data.get("historical_best", {}),
            incumbent_hotkey=data.get("incumbent_hotkey"),
            current_season=data.get("current_season", 0),
            last_window=data.get("last_window", -1),
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return IncumbentState()
