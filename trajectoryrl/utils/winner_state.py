"""Server winner adoption + local cross-check + cache.

In v6.0 the seated winner is resolved **server-side** at finalize time
and published at ``GET /api/v2/winner/current``. Each validator adopts
that published winner (identity + score) to drive ``set_weights``; it
does not re-derive the winner authoritatively. Alongside the winner the
server publishes the per-vote inputs (``submissions[]`` with
``validator_stake`` snapshotted at score-POST time) so the daemon can
re-run the aggregation locally as an **advisory cross-check** and alarm
on divergence — the recomputed value does not override the server's
winner.

This module provides:

* ``WinnerState`` — the cache shape (hotkey, uid, score, freshness).
* ``aggregate_challenger_score`` — pure function over ``submissions[]``
  that mirrors the server's finalize aggregation (plain-average
  consensus score + head-count majority qualified). Used only for the
  divergence cross-check; its result does not drive ``set_weights``.
* ``derive_winner_state`` — wraps the parsed ``/api/v2/winner/current``
  response into a ``WinnerState`` (taken from the server's ``winner``
  block) and emits a divergence warning when the locally-recomputed
  consensus disagrees with the server's claim.

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

# Noise-aware seating takes effect from this spec onward: the server switches the
# consensus center to a Winsorized mean and the takeover margin to a
# score-dependent δ (3% decaying to 0 near the score ceiling). Earlier epochs use
# the legacy plain mean + flat 3%. The daemon only re-derives the *consensus*
# (Winsorized) here to cross-check the server's winner score; the takeover-margin
# decision itself is server-canonical and not recomputed locally. Pinned in the
# daemon release to mirror the server's NOISE_AWARE_SEATING_MIN_SPEC.
_NOISE_AWARE_SEATING_MIN_SPEC = 16


def _winsorized_mean(values: List[float]) -> float:
    """Robust consensus center — mirrors the server's aggregator.winsorizedMean.

    Cross-validator score spread is pure agent-rollout noise (grading is
    deterministic), and a single validator's rollout is occasionally a wild
    outlier. Plain averaging gives that outlier full 1/n leverage; Winsorizing
    clips the single lowest and highest score to their nearest neighbour before
    averaging. n-adaptive: only clip when n >= 4 (at n <= 3 clipping collapses
    toward a single validator, so fall back to the plain mean).
    """
    n = len(values)
    xs = sorted(values)
    if n >= 4:
        xs[0] = xs[1]
        xs[-1] = xs[-2]
    return sum(xs) / n


def aggregate_challenger_score(
    submissions: Optional[List[Dict[str, Any]]],
    *,
    winsorize: bool = False,
) -> Optional[Tuple[float, bool]]:
    """Unweighted aggregate of ``(score, qualified)`` over submissions.

    Mirrors the server-side finalize aggregation:

    * Rows with ``challenger_rejected = true`` are dropped.
    * Rows with null/zero ``validator_stake`` are dropped — stake is used only
      to identify eligible voters (a proxy for the server's MIN_VALIDATOR_STAKE
      / active-set filter); it does **not** weight the score or the qualified
      majority. Pre-migration-070 rows would need a metagraph fallback; out of
      scope here.
    * ``consensus_qualified`` = majority by head count of ``challenger_qualified
      = true`` (> 0.5 of eligible voters).
    * ``consensus_score`` = (``winsorize=True``) the Winsorized mean of
      ``challenger_score`` over qualified rows, else the plain mean. The caller
      sets ``winsorize`` from the epoch's spec vs ``_NOISE_AWARE_SEATING_MIN_SPEC``.

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
        valid.append(s)

    if not valid:
        return None

    qualified_subs = [s for s in valid if s.get("challenger_qualified")]
    consensus_qualified = (len(qualified_subs) / len(valid)) > 0.5

    if not qualified_subs:
        return 0.0, consensus_qualified

    scores: List[float] = []
    for s in qualified_subs:
        score_raw = s.get("challenger_score")
        try:
            score_f = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score_f = None
        if score_f is None:
            continue
        scores.append(score_f)

    if not scores:
        return 0.0, consensus_qualified

    consensus = _winsorized_mean(scores) if winsorize else sum(scores) / len(scores)
    return consensus, consensus_qualified


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
        # Match the server's aggregation for this epoch's spec: Winsorized from
        # the noise-aware cutover, plain mean before it.
        spec_number = finalized.get("spec_number")
        winsorize = (
            spec_number is not None
            and spec_number >= _NOISE_AWARE_SEATING_MIN_SPEC
        )
        derived = aggregate_challenger_score(
            response.get("submissions"), winsorize=winsorize
        )
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
