"""Pack ownership lock for first-mover attribution.

Each validator maintains a per-validator persistent table mapping
``pack_hash`` to ``(first_observed_hotkey, first_observed_block)``.
The first hotkey to register a given pack_hash owns it permanently for
the lifetime of the entry — copies submitted by other hotkeys are
rejected before evaluation and receive weight 0. There is no succession:
if the original owner goes inactive, no other miner inherits ownership.

Eviction is "by-active with a grace window": at the end of each
evaluation cycle, the validator passes the set of pack_hashes
referenced by any active commitment plus the current window number.
Entries seen as inactive get their grace clock started; an entry is
only dropped after its pack_hash has been absent for
``EVICTION_GRACE_WINDOWS`` (7) consecutive wall-clock windows. Any
re-activation in between resets the clock. Once both the original owner
and any copy-cats have been silent for the full grace period, a
brand-new submitter can "resurrect" the pack and claim ownership.

This module is the single source of truth for ownership semantics,
JSON persistence format, and legacy migration. The validator owns the
two in-memory dictionaries (ownership table + last-seen window) and
calls these helpers.
"""

import json
import logging
import os
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

# Bumped whenever the on-disk JSON layout changes. Loaders accept any
# version they understand; unknown versions log a warning and fall back
# to best-effort decoding.
#
# v1: { "version": 1, "pack_first_seen": {...} }
# v2: { "version": 2, "pack_first_seen": {...}, "pack_last_seen_window": {...} }
PACK_OWNERSHIP_FORMAT_VERSION = 2

# Number of consecutive wall-clock windows a pack_hash must be absent
# from active commitments before its ownership entry is evicted. Wall
# clock means "validator may have been offline part of the span" still
# counts — we compare current_window against last_seen_window directly,
# not the number of cycles actually executed.
EVICTION_GRACE_WINDOWS = 7


PackOwnershipTable = Dict[str, Tuple[str, int]]
PackLastSeenWindow = Dict[str, int]


def claim_owner(
    table: PackOwnershipTable,
    pack_hash: str,
    hotkey: str,
    block: int,
) -> Tuple[str, int]:
    """Record ownership of ``pack_hash`` if not already recorded.

    First-writer-wins: subsequent calls with a different hotkey return
    the originally claimed ``(owner_hotkey, owner_block)`` tuple. The
    table is mutated in place via ``setdefault`` so callers can compare
    the returned hotkey with their own to detect copies.

    Returns:
        ``(owner_hotkey, owner_block)`` — the recorded owner after this
        call (either pre-existing or just claimed).
    """
    return table.setdefault(pack_hash, (hotkey, block))


def is_owner(
    table: PackOwnershipTable,
    pack_hash: str,
    hotkey: str,
) -> bool:
    """Return True iff ``hotkey`` is the recorded owner for ``pack_hash``."""
    entry = table.get(pack_hash)
    return entry is not None and entry[0] == hotkey


def evict_orphans(
    table: PackOwnershipTable,
    last_seen: PackLastSeenWindow,
    active_hashes: Iterable[str],
    current_window: int,
    grace_windows: int = EVICTION_GRACE_WINDOWS,
) -> List[str]:
    """Drop entries whose pack_hash has been inactive for the grace span.

    Mutates both ``table`` and ``last_seen`` in place:

    * For every ``pack_hash`` in ``active_hashes`` that is also in
      ``table``: refresh ``last_seen[ph] = current_window``. Any prior
      grace clock is reset.
    * For every ``pack_hash`` in ``table`` that is NOT in
      ``active_hashes``: if no ``last_seen`` entry exists yet (e.g.
      newly-claimed entry that races out before a refresh, or a v1
      legacy entry just migrated), start the clock at
      ``current_window``. Otherwise, evict iff
      ``current_window - last_seen[ph] >= grace_windows``.

    Returns the list of evicted pack_hashes (in iteration order) for
    logging / metrics. ``active_hashes`` may be any iterable; it is
    materialized into a set internally for O(1) lookup.
    """
    active = set(active_hashes)

    for ph in active:
        if ph in table:
            last_seen[ph] = current_window

    evicted: List[str] = []
    for ph in list(table.keys()):
        if ph in active:
            continue
        seen_at = last_seen.get(ph)
        if seen_at is None:
            last_seen[ph] = current_window
            continue
        if current_window - seen_at >= grace_windows:
            table.pop(ph, None)
            last_seen.pop(ph, None)
            evicted.append(ph)
    return evicted


def save_pack_first_seen(
    table: PackOwnershipTable,
    last_seen: PackLastSeenWindow,
    path,
) -> None:
    """Persist ``table`` and ``last_seen`` to ``path`` as JSON (format v2).

    Tuples are serialized as two-element lists for JSON compatibility.
    Parent directory is created if missing. Writes are not atomic —
    matching the existing winner_state.save_winner_state semantics — so
    a crash mid-write may leave a partial file (loader will then return
    empty tables on the next start).
    """
    payload = {
        "version": PACK_OWNERSHIP_FORMAT_VERSION,
        "pack_first_seen": {
            ph: [hk, int(blk)] for ph, (hk, blk) in table.items()
        },
        "pack_last_seen_window": {
            ph: int(w) for ph, w in last_seen.items()
        },
    }
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.debug(
        "pack_first_seen saved (%d entries, %d last-seen) to %s",
        len(table), len(last_seen), path,
    )


def load_pack_first_seen(
    path,
) -> Tuple[PackOwnershipTable, PackLastSeenWindow]:
    """Load the ownership table + last-seen window from ``path``.

    Returns ``({}, {})`` if the file is missing, unreadable, or malformed.
    Individual entries that fail decoding are skipped silently (defensive
    against partial corruption).

    Format compatibility:

    * v1 files (no ``pack_last_seen_window`` key): the side dict is
      returned empty. The first ``evict_orphans`` call after upgrade
      starts the grace clock for every still-orphaned entry, so no
      mass-eviction occurs immediately after the format bump.
    * v2 files: both dicts are returned.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}, {}

    table = _decode_table(data.get("pack_first_seen", {}))
    last_seen = _decode_last_seen(data.get("pack_last_seen_window", {}))
    return table, last_seen


def _decode_table(raw) -> PackOwnershipTable:
    """Decode the JSON ``pack_first_seen`` payload into the in-memory shape.

    Shared by ``load_pack_first_seen`` and the validator's legacy
    migration path (which decodes from inside ``eval_state.json``).
    """
    if not isinstance(raw, dict):
        return {}
    out: PackOwnershipTable = {}
    for ph, entry in raw.items():
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        hk, blk = entry[0], entry[1]
        if not isinstance(hk, str):
            continue
        try:
            out[ph] = (hk, int(blk))
        except (TypeError, ValueError):
            continue
    return out


def _decode_last_seen(raw) -> PackLastSeenWindow:
    """Decode the JSON ``pack_last_seen_window`` payload defensively.

    Skips entries whose value is not coercible to ``int``. Returns ``{}``
    for non-dict input.
    """
    if not isinstance(raw, dict):
        return {}
    out: PackLastSeenWindow = {}
    for ph, w in raw.items():
        if not isinstance(ph, str):
            continue
        try:
            out[ph] = int(w)
        except (TypeError, ValueError):
            continue
    return out
