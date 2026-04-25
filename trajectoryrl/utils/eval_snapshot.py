"""Window-start snapshot of the active miner commitment set.

Each evaluation window is anchored at ``window_start = global_anchor +
window_number * window_length``. At the start of a window, every
validator queries the chain for all miner commitments and freezes the
subset that is deterministic across validators:

    snapshot[uid] = commitment   iff   commitment.block_number < window_start

The ``block_number`` field is sourced from the chain (see
``commitments._get_commitment_block``), so two validators that build
the snapshot at any time strictly after ``window_start`` and before the
next miner overwrite produce byte-identical snapshots. Both the lower
bound (any block in window N) and the upper bound (a brief race when a
miner overwrites mid-poll) are accepted as the per-validator polling
granularity error documented in the redesign plan.

The snapshot is persisted to ``active_set_window_{N}.json`` so that a
restart inside the same window N rehydrates the exact same evaluation
set without re-querying the chain. If the file is absent on restart,
the validator rebuilds best-effort from the current chain state and
the same ``block_number < window_start`` filter; the result is
identical for any commitment that was not overwritten in the meantime.

Stale-age filtering (``inactivity_blocks``) is applied at snapshot
time relative to ``window_start`` rather than ``current_block`` so the
snapshot remains stable for the entire window even if it is taken
late. Per-validator runtime filters (validator_permit, blacklist) are
intentionally NOT baked into the snapshot — those are dynamic and must
be re-evaluated against the live metagraph on every cycle.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .commitments import MinerCommitment

logger = logging.getLogger(__name__)

SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_FILE_PATTERN = re.compile(r"^active_set_window_(\d+)\.json$")

# How many recent window snapshots to keep on disk, including the
# current window snapshot file.
DEFAULT_RETENTION = 4


@dataclass(frozen=True)
class EvalSnapshot:
    """Frozen evaluation set for a single window.

    ``commitments`` is keyed by UID. ``window_start`` is the chain
    block at the beginning of the window. ``snapshot_block`` records
    the block height observed when the snapshot was built (purely
    informational — does not affect determinism).
    """

    window_number: int
    window_start: int
    snapshot_block: int
    commitments: Dict[int, MinerCommitment]

    def hotkeys(self) -> List[str]:
        """Sorted hotkey list — useful for cross-validator diff logs."""
        return sorted(c.hotkey for c in self.commitments.values())


def take_snapshot(
    raw_commitments: Dict[int, MinerCommitment],
    window_number: int,
    window_start: int,
    snapshot_block: int,
    inactivity_blocks: Optional[int] = None,
) -> EvalSnapshot:
    """Filter ``raw_commitments`` to the deterministic window-N set.

    Two filters are applied — both reference ``window_start``, never
    ``snapshot_block``, so the result is invariant under polling-time
    jitter:

    * ``commitment.block_number < window_start`` — only commits that
      landed strictly before window N starts are eligible. A commit
      submitted inside window N is held over for window N+1.
    * If ``inactivity_blocks`` is provided:
      ``window_start - commitment.block_number <= inactivity_blocks``.
    """
    eligible: Dict[int, MinerCommitment] = {}
    dropped_in_window = 0
    dropped_stale = 0
    for uid, commitment in raw_commitments.items():
        if commitment.block_number >= window_start:
            dropped_in_window += 1
            continue
        if inactivity_blocks is not None:
            age = window_start - commitment.block_number
            if age > inactivity_blocks:
                dropped_stale += 1
                continue
        eligible[uid] = commitment

    logger.info(
        "Window %d snapshot: %d eligible (filtered %d in-window, %d stale) "
        "from %d raw commitments at block %d (window_start=%d)",
        window_number, len(eligible), dropped_in_window, dropped_stale,
        len(raw_commitments), snapshot_block, window_start,
    )
    return EvalSnapshot(
        window_number=window_number,
        window_start=window_start,
        snapshot_block=snapshot_block,
        commitments=eligible,
    )


def snapshot_path(directory, window_number: int) -> Path:
    """Return the canonical file path for a window snapshot."""
    return Path(directory) / f"active_set_window_{window_number}.json"


def save_snapshot(
    snapshot: EvalSnapshot,
    directory,
    retention: int = DEFAULT_RETENTION,
) -> Path:
    """Persist ``snapshot`` to ``active_set_window_{N}.json`` and prune.

    The write is two-phase (tmp file + rename) so a crash mid-write
    cannot leave a partially-decoded file that a restart would mistake
    for the canonical snapshot.

    After the new snapshot lands, keeps only the most recent
    ``retention`` windows (including current) and unlinks older files.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": SNAPSHOT_FORMAT_VERSION,
        "window_number": snapshot.window_number,
        "window_start": snapshot.window_start,
        "snapshot_block": snapshot.snapshot_block,
        "commitments": [
            {
                "uid": c.uid,
                "hotkey": c.hotkey,
                "pack_hash": c.pack_hash,
                "pack_url": c.pack_url,
                "block_number": int(c.block_number),
                "raw": c.raw,
            }
            for c in snapshot.commitments.values()
        ],
    }

    target = snapshot_path(directory, snapshot.window_number)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, target)
    logger.info(
        "Saved active-set snapshot for window %d (%d commitments) → %s",
        snapshot.window_number, len(snapshot.commitments), target,
    )

    _prune_old_snapshots(directory, snapshot.window_number, retention)
    return target


def load_snapshot(directory, window_number: int) -> Optional[EvalSnapshot]:
    """Load the snapshot for ``window_number``, or None if missing/invalid.

    Missing file is the common case (first cycle in a fresh window) and
    is logged at debug level. A malformed file logs a warning and
    returns None; the caller is expected to rebuild from the chain.
    """
    path = snapshot_path(directory, window_number)
    if not path.exists():
        logger.debug("No snapshot file at %s", path)
        return None

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read snapshot %s: %s", path, e)
        return None

    if not isinstance(data, dict):
        logger.warning("Snapshot %s has invalid root type %s", path, type(data))
        return None

    file_version = data.get("version")
    if file_version != SNAPSHOT_FORMAT_VERSION:
        logger.warning(
            "Snapshot %s has unsupported format version %r (expected %d)",
            path, file_version, SNAPSHOT_FORMAT_VERSION,
        )
        return None

    try:
        loaded_window = int(data["window_number"])
        if loaded_window != window_number:
            logger.warning(
                "Snapshot %s window_number mismatch: file=%d expected=%d",
                path, loaded_window, window_number,
            )
            return None
        window_start = int(data["window_start"])
        snapshot_block = int(data["snapshot_block"])
        raw_entries = data["commitments"]
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("Snapshot %s missing required field: %s", path, e)
        return None

    if not isinstance(raw_entries, list):
        logger.warning("Snapshot %s commitments field is not a list", path)
        return None

    commitments: Dict[int, MinerCommitment] = {}
    for entry in raw_entries:
        try:
            uid = int(entry["uid"])
            commitments[uid] = MinerCommitment(
                uid=uid,
                hotkey=str(entry["hotkey"]),
                pack_hash=str(entry["pack_hash"]),
                pack_url=str(entry["pack_url"]),
                block_number=int(entry["block_number"]),
                raw=str(entry["raw"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                "Snapshot %s: skipping malformed entry (%s): %r",
                path, e, entry,
            )
            continue

    snapshot = EvalSnapshot(
        window_number=loaded_window,
        window_start=window_start,
        snapshot_block=snapshot_block,
        commitments=commitments,
    )
    logger.info(
        "Loaded active-set snapshot for window %d (%d commitments) ← %s",
        snapshot.window_number, len(snapshot.commitments), path,
    )
    return snapshot


def _prune_old_snapshots(directory: Path, current_window: int, retention: int) -> None:
    """Drop snapshot files older than the retained recent window range.

    A negative ``retention`` is treated as "keep everything"; zero
    keeps only the current window file.
    """
    if retention < 0:
        return
    cutoff = current_window - max(retention - 1, 0)
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        match = SNAPSHOT_FILE_PATTERN.match(entry.name)
        if not match:
            continue
        try:
            file_window = int(match.group(1))
        except ValueError:
            continue
        if file_window < cutoff:
            try:
                entry.unlink()
                logger.debug("Pruned old snapshot %s", entry)
            except OSError as e:
                logger.warning("Failed to prune snapshot %s: %s", entry, e)
