"""Local cache of the server-canonical seated winner.

In v6.0 the platform server is the single source of truth for
``winner_state``. Validators read it via the ``winner`` block on
``GET /api/v2/epoch/current`` and use a local cache only as a fallback
when the server is unreachable.

The cache:
  * is overwritten on every successful poll,
  * is persisted to disk so cold starts and short outages survive,
  * has a freshness window (``WINNER_FALLBACK_TTL``) — beyond that, the
    daemon refuses to set weights and emits an alert.

There is no Winner Protection (δ) computation here in v6.0: the server
applies it before publishing ``winner_state``. Earlier versions of this
module owned that logic; it has been removed along with the per-validator
selection path.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

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
