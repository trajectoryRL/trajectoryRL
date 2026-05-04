"""Window-start snapshot of the active miner commitment set.

Each evaluation window is anchored at ``window_start = global_anchor +
window_number * window_length``. At the start of a window, every
validator fetches the deterministic eval target set from the subnet's
``/api/v2/validators/epoch_snapshot`` endpoint (see
``status_reporter.fetch_epoch_snapshot``). The endpoint already does
the on-chain commitment scan and the per-miner pre-eval pipeline, so
validators no longer query the chain or call ``/api/v2/miners/pre-eval``
directly at this stage.

The API response is byte-identical for a given epoch across all
validators (``epoch_summary.eval_snapshot`` is frozen the first time
sync runs after cutoff). We mirror it onto disk as
``active_set_window_{N}.json`` so a restart inside the same window N
rehydrates without a fresh HTTP round-trip; if the file is absent or
malformed, we re-fetch from the endpoint.

Each API entry carries ``pre_eval_status`` (``passed`` / ``failed``).
We split the entries into:

* ``commitments`` — passed entries, exposed as ``MinerCommitment``
  values keyed by UID. ``block_number`` is synthesised from the API
  entry's index in the (refresh_time-sorted) ``entries`` list, so the
  downstream ``pack_first_seen`` tiebreak (which sorts by
  ``(block_number, hotkey)``) stays deterministic across validators.
* ``pre_eval_failed`` — failed entries, exposed as ``EpochSnapshotEntry``
  values keyed by hotkey. The validator submits a rejection row for
  each (no eval is run).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .commitments import MinerCommitment

logger = logging.getLogger(__name__)

SNAPSHOT_FILE_PATTERN = re.compile(r"^active_set_window_(\d+)\.json$")

# How many recent window snapshots to keep on disk, including the
# current window snapshot file.
DEFAULT_RETENTION = 4


@dataclass(frozen=True)
class EpochSnapshotEntry:
    """One row from the ``/api/v2/validators/epoch_snapshot`` response.

    ``pre_eval_status`` is either ``"passed"`` or ``"failed"``.
    ``pre_eval_reason`` is populated only on failures (mirrors the
    server-side ``eval_reason`` column — e.g. ``"hardcoded"``,
    ``"hash_mismatch"``, ``"banned until ..."``).
    """

    uid: int
    hotkey: str
    pack_hash: str
    pack_url: str
    refresh_time: str
    pre_eval_status: str
    pre_eval_reason: Optional[str]


@dataclass(frozen=True)
class EvalSnapshot:
    """Frozen evaluation set for a single window.

    ``commitments`` is keyed by UID and contains only entries whose
    ``pre_eval_status == "passed"``. ``pre_eval_failed`` is keyed by
    hotkey and contains the raw API rows for failed entries — the
    validator submits a rejection row per failed entry without running
    the eval.

    ``window_start`` is the chain block at the beginning of the window.
    ``snapshot_block`` records the API-reported ``snapshot_block``
    (latest synced chain height when the server built the snapshot) —
    informational only.
    """

    window_number: int
    window_start: int
    snapshot_block: int
    commitments: Dict[int, MinerCommitment]
    pre_eval_failed: Dict[str, EpochSnapshotEntry] = field(default_factory=dict)

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

    .. deprecated::
        The chain-side snapshot path is superseded by
        ``take_snapshot_from_api`` (the ``/api/v2/validators/epoch_snapshot``
        endpoint absorbs the on-chain query and the per-miner pre-eval
        pipeline). Retained for tests and as a fallback shim until the
        follow-up cleanup removes ``fetch_all_commitments`` entirely.

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
        "Window %d snapshot (legacy chain path): %d eligible "
        "(filtered %d in-window, %d stale) from %d raw commitments "
        "at block %d (window_start=%d)",
        window_number, len(eligible), dropped_in_window, dropped_stale,
        len(raw_commitments), snapshot_block, window_start,
    )
    return EvalSnapshot(
        window_number=window_number,
        window_start=window_start,
        snapshot_block=snapshot_block,
        commitments=eligible,
    )


def take_snapshot_from_api(
    api_response: Dict[str, Any],
    window_number: int,
) -> EvalSnapshot:
    """Build an ``EvalSnapshot`` from a parsed epoch_snapshot response.

    The API guarantees ``entries`` is sorted ``(refresh_time ASC,
    hotkey ASC)`` and is byte-identical across validators for a given
    epoch. We use the per-entry index as the synthesised
    ``MinerCommitment.block_number`` so the validator's existing
    ``pack_first_seen`` tiebreak (sorted by ``(block_number, hotkey)``)
    remains deterministic across validators.

    Failed-pre-eval entries are split into ``pre_eval_failed`` and not
    placed in ``commitments``. The validator surfaces them as rejection
    submissions (``rejected=True``) without running the eval.
    """
    entries_raw = api_response.get("entries", [])
    if not isinstance(entries_raw, list):
        raise ValueError(
            f"epoch_snapshot response 'entries' is not a list: "
            f"{type(entries_raw).__name__}"
        )

    window_start = int(api_response.get("window_start", 0))
    snapshot_block = int(api_response.get("snapshot_block", 0))

    commitments: Dict[int, MinerCommitment] = {}
    pre_eval_failed: Dict[str, EpochSnapshotEntry] = {}
    passed_count = 0
    failed_count = 0
    skipped_malformed = 0

    for index, raw in enumerate(entries_raw):
        try:
            uid = int(raw["uid"])
            hotkey = str(raw["hotkey"])
            pack_hash = str(raw["pack_hash"])
            pack_url = str(raw["pack_url"])
            refresh_time = str(raw.get("refresh_time", ""))
            status = str(raw.get("pre_eval_status", ""))
            reason = raw.get("pre_eval_reason")
            if reason is not None:
                reason = str(reason)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                "epoch_snapshot entry %d: malformed (%s): %r",
                index, e, raw,
            )
            skipped_malformed += 1
            continue

        entry = EpochSnapshotEntry(
            uid=uid,
            hotkey=hotkey,
            pack_hash=pack_hash,
            pack_url=pack_url,
            refresh_time=refresh_time,
            pre_eval_status=status,
            pre_eval_reason=reason,
        )

        if status == "passed":
            commitments[uid] = MinerCommitment(
                uid=uid,
                hotkey=hotkey,
                pack_hash=pack_hash,
                pack_url=pack_url,
                # Deterministic per-entry ordering for pack_first_seen
                # tiebreak — see module docstring.
                block_number=index,
                raw=f"{pack_hash}|{pack_url}",
            )
            passed_count += 1
        elif status == "failed":
            pre_eval_failed[hotkey] = entry
            failed_count += 1
        else:
            logger.warning(
                "epoch_snapshot entry %d (uid=%d, hk=%s…): "
                "unexpected pre_eval_status=%r — ignoring",
                index, uid, hotkey[:8], status,
            )
            skipped_malformed += 1

    logger.info(
        "Window %d snapshot from API: %d passed, %d failed, "
        "%d malformed (window_start=%d, snapshot_block=%d)",
        window_number, passed_count, failed_count, skipped_malformed,
        window_start, snapshot_block,
    )
    return EvalSnapshot(
        window_number=window_number,
        window_start=window_start,
        snapshot_block=snapshot_block,
        commitments=commitments,
        pre_eval_failed=pre_eval_failed,
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
        "pre_eval_failed": [
            {
                "uid": e.uid,
                "hotkey": e.hotkey,
                "pack_hash": e.pack_hash,
                "pack_url": e.pack_url,
                "refresh_time": e.refresh_time,
                "pre_eval_status": e.pre_eval_status,
                "pre_eval_reason": e.pre_eval_reason,
            }
            for e in snapshot.pre_eval_failed.values()
        ],
    }

    target = snapshot_path(directory, snapshot.window_number)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, target)
    logger.info(
        "Saved active-set snapshot for window %d "
        "(%d passed, %d failed) → %s",
        snapshot.window_number,
        len(snapshot.commitments),
        len(snapshot.pre_eval_failed),
        target,
    )

    _prune_old_snapshots(directory, snapshot.window_number, retention)
    return target


def load_snapshot(directory, window_number: int) -> Optional[EvalSnapshot]:
    """Load the snapshot for ``window_number``, or None if missing/invalid.

    Missing file is the common case (first cycle in a fresh window) and
    is logged at debug level. A malformed file or a file written in a
    previous (incompatible) format logs a warning and returns None;
    the caller is expected to re-fetch from the API.
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

    pre_eval_failed: Dict[str, EpochSnapshotEntry] = {}
    raw_failed = data.get("pre_eval_failed", [])
    if isinstance(raw_failed, list):
        for entry in raw_failed:
            try:
                hotkey = str(entry["hotkey"])
                reason = entry.get("pre_eval_reason")
                if reason is not None:
                    reason = str(reason)
                pre_eval_failed[hotkey] = EpochSnapshotEntry(
                    uid=int(entry["uid"]),
                    hotkey=hotkey,
                    pack_hash=str(entry["pack_hash"]),
                    pack_url=str(entry["pack_url"]),
                    refresh_time=str(entry.get("refresh_time", "")),
                    pre_eval_status=str(entry.get("pre_eval_status", "failed")),
                    pre_eval_reason=reason,
                )
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(
                    "Snapshot %s: skipping malformed pre_eval_failed entry "
                    "(%s): %r", path, e, entry,
                )
                continue

    snapshot = EvalSnapshot(
        window_number=loaded_window,
        window_start=window_start,
        snapshot_block=snapshot_block,
        commitments=commitments,
        pre_eval_failed=pre_eval_failed,
    )
    logger.info(
        "Loaded active-set snapshot for window %d "
        "(%d passed, %d failed) ← %s",
        snapshot.window_number,
        len(snapshot.commitments),
        len(snapshot.pre_eval_failed),
        path,
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
