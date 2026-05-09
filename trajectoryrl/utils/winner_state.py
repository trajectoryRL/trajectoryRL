"""Local winner derivation + cache.

In v6.0 the seated winner is **derived locally** by every validator
from the per-vote inputs published at ``GET /api/v2/winner/current``:
the server returns the latest finalized epoch's submissions with
``validator_stake`` snapshotted at score-POST time, and each daemon
runs the same stake-weighted aggregation against those rows.

This module provides:

* ``WinnerState`` — the cache shape (hotkey, uid, score, freshness).
* ``aggregate_challenger_score`` — pure function over ``submissions[]``
  that mirrors the server's finalize aggregation (stake-weighted
  consensus score + stake-weighted majority qualified).
* ``derive_winner_state`` — wraps the parsed ``/api/v2/winner/current``
  response into a ``WinnerState`` and emits a divergence warning when
  the locally-computed consensus disagrees with the server's claim.

The cache is persisted to disk so cold starts and short server outages
survive; ``WINNER_FALLBACK_TTL_SECONDS`` (24 h) is the staleness alarm
threshold beyond which the daemon refuses to set weights.

There is no Winner Protection (δ) computation here: that gate is
applied server-side at finalize time and reflected in the
``winner_replaced`` / ``winner_held`` outcome on each epoch.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Per INCENTIVE_MECHANISM.md "Canonical state" (line 142): 24 hours of
# server unreachability before the daemon stops setting weights. Keeping
# the constant module-local lets callers `from winner_state import
# WINNER_FALLBACK_TTL` without pulling in env config.
WINNER_FALLBACK_TTL_SECONDS = 24 * 60 * 60


@dataclass
class WinnerState:
    """Local mirror of the server-canonical seated winner.

    All fields are optional because the server returns ``winner: null``
    on cold start (no challenge epoch has finalized yet). ``updated_at``
    is the local Unix timestamp of the last successful refresh — used by
    ``is_fresh()`` to decide whether the cache is still trustworthy
    after a server outage.
    """

    winner_hotkey: Optional[str] = None
    winner_uid: Optional[int] = None
    winner_pack_hash: Optional[str] = None
    winner_score: Optional[float] = None
    spec_number: Optional[int] = None
    updated_at: Optional[float] = None

    @property
    def is_seated(self) -> bool:
        """True when the cache holds a real winner (not cold-start empty)."""
        return self.winner_hotkey is not None and self.winner_uid is not None

    def is_fresh(
        self,
        ttl_seconds: int = WINNER_FALLBACK_TTL_SECONDS,
        now: Optional[float] = None,
    ) -> bool:
        """Whether the cache is still within the fallback TTL.

        Returns False when never updated (cold start) so callers don't
        accidentally drive ``set_weights`` from default values.
        """
        if self.updated_at is None:
            return False
        now = now if now is not None else time.time()
        return (now - self.updated_at) <= ttl_seconds


_DIVERGENCE_TOLERANCE = 0.01


def aggregate_challenger_score(
    submissions: Optional[List[Dict[str, Any]]],
) -> Optional[Tuple[float, bool]]:
    """Stake-weighted aggregate of ``(score, qualified)`` over submissions.

    Mirrors the server-side finalize aggregation:

    * Rows with ``challenger_rejected = true`` are dropped.
    * Rows with null/zero ``validator_stake`` are dropped (pre-migration-070
      rows would need a metagraph fallback; out of scope here).
    * ``consensus_qualified`` = stake-weighted majority of ``challenger_qualified
      = true`` (> 0.5 of valid stake).
    * ``consensus_score`` = stake-weighted average of ``challenger_score`` over
      ``challenger_qualified = true`` rows only.

    Returns ``None`` when no rows survive filtering, otherwise
    ``(consensus_score, consensus_qualified)``.
    """
    if not submissions:
        return None

    valid: List[Dict[str, Any]] = []
    for s in submissions:
        if not isinstance(s, dict):
            continue
        if s.get("challenger_rejected"):
            continue
        stake = s.get("validator_stake")
        try:
            stake_f = float(stake) if stake is not None else 0.0
        except (TypeError, ValueError):
            continue
        if stake_f <= 0:
            continue
        s_copy = dict(s)
        s_copy["validator_stake"] = stake_f
        valid.append(s_copy)

    if not valid:
        return None

    total_stake = sum(s["validator_stake"] for s in valid)
    if total_stake <= 0:
        return None

    qualified_subs = [s for s in valid if s.get("challenger_qualified")]
    qualified_stake = sum(s["validator_stake"] for s in qualified_subs)
    consensus_qualified = (qualified_stake / total_stake) > 0.5

    if not qualified_subs or qualified_stake <= 0:
        return 0.0, consensus_qualified

    weighted_sum = 0.0
    used_stake = 0.0
    for s in qualified_subs:
        score_raw = s.get("challenger_score")
        try:
            score_f = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score_f = None
        if score_f is None:
            continue
        weighted_sum += s["validator_stake"] * score_f
        used_stake += s["validator_stake"]

    if used_stake <= 0:
        return 0.0, consensus_qualified

    return weighted_sum / used_stake, consensus_qualified


def derive_winner_state(
    response: Optional[Dict[str, Any]],
    *,
    now: Optional[float] = None,
) -> WinnerState:
    """Build a ``WinnerState`` from the parsed ``/api/v2/winner/current``
    response.

    The seated-winner identity (hotkey/uid/pack_hash) is taken from
    ``response.winner`` because the server already resolved it through
    the deterministic finalize path. The local aggregation runs in
    parallel as a safety check: when this epoch was a
    ``winner_replaced`` transition and the locally computed consensus
    score diverges from the server's claim by more than
    ``_DIVERGENCE_TOLERANCE``, a warning is logged. Persistent
    divergence across many validators would indicate a server bug
    rather than a network-consensus issue.

    Cold start (no ``finalized_epoch`` or no ``winner``) yields an empty
    ``WinnerState``. ``updated_at`` is stamped on every call so the
    fallback TTL can age the cache forward even when the server returns
    cold-start nulls.
    """
    if now is None:
        now = time.time()

    if not response:
        return WinnerState(updated_at=now)

    finalized = response.get("finalized_epoch")
    server_winner = response.get("winner")

    if finalized and server_winner:
        derived = aggregate_challenger_score(response.get("submissions"))
        if derived is not None and finalized.get("winner_replaced"):
            local_score, _local_qualified = derived
            server_score_raw = server_winner.get("score")
            try:
                server_score = (
                    float(server_score_raw) if server_score_raw is not None else None
                )
            except (TypeError, ValueError):
                server_score = None
            if (
                server_score is not None
                and abs(local_score - server_score) > _DIVERGENCE_TOLERANCE
            ):
                logger.warning(
                    "winner derivation divergence at epoch %s: "
                    "local consensus=%.4f, server winner.score=%.4f",
                    finalized.get("challenge_epoch_id"),
                    local_score,
                    server_score,
                )

    return winner_from_server_block(server_winner, now=now)


def winner_from_server_block(
    winner_block: Optional[Dict[str, Any]],
    *,
    now: Optional[float] = None,
) -> WinnerState:
    """Build a ``WinnerState`` from the ``winner`` block returned by
    ``GET /api/v2/epoch/current``.

    A ``None`` block (cold start) yields an empty, never-seated state
    that ``is_seated`` reports False for. The block is read defensively:
    extra fields are ignored, missing fields fall back to ``None``.
    """
    if not winner_block:
        return WinnerState(updated_at=now if now is not None else time.time())

    score = winner_block.get("score")
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None

    uid = winner_block.get("uid")
    try:
        uid_i = int(uid) if uid is not None else None
    except (TypeError, ValueError):
        uid_i = None

    return WinnerState(
        winner_hotkey=winner_block.get("hotkey"),
        winner_uid=uid_i,
        winner_pack_hash=winner_block.get("pack_hash"),
        winner_score=score_f,
        spec_number=winner_block.get("spec_number"),
        updated_at=now if now is not None else time.time(),
    )


def save_winner_state(state: WinnerState, path: str) -> None:
    """Persist the cache to disk for restart / outage fallback."""
    data: Dict[str, Any] = asdict(state)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Winner state cache saved to %s", path)


def load_winner_state(path: str) -> WinnerState:
    """Load the cache, returning a fresh empty state if absent / unreadable."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return WinnerState()

    if not isinstance(data, dict):
        return WinnerState()

    return WinnerState(
        winner_hotkey=data.get("winner_hotkey"),
        winner_uid=data.get("winner_uid"),
        winner_pack_hash=data.get("winner_pack_hash"),
        winner_score=data.get("winner_score"),
        spec_number=data.get("spec_number"),
        updated_at=data.get("updated_at"),
    )
