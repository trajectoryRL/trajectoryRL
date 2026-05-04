"""Unit tests for trajectoryrl.utils.eval_snapshot helpers."""

from trajectoryrl.utils.commitments import MinerCommitment
from trajectoryrl.utils.eval_snapshot import (
    DEFAULT_RETENTION,
    EpochSnapshotEntry,
    EvalSnapshot,
    load_snapshot,
    save_snapshot,
    snapshot_path,
    take_snapshot,
    take_snapshot_from_api,
)


def _mk_commitment(
    uid: int,
    hotkey: str,
    block_number: int,
    pack_hash: str = "a" * 64,
) -> MinerCommitment:
    return MinerCommitment(
        uid=uid,
        hotkey=hotkey,
        pack_hash=pack_hash,
        pack_url=f"https://example.com/{uid}.json",
        block_number=block_number,
        raw=f"{pack_hash}|https://example.com/{uid}.json",
    )


def test_take_snapshot_filters_in_window_and_stale():
    raw = {
        1: _mk_commitment(1, "hk_a", 100, "a" * 64),  # keep
        2: _mk_commitment(2, "hk_b", 190, "b" * 64),  # keep
        3: _mk_commitment(3, "hk_c", 200, "c" * 64),  # in-window drop
        4: _mk_commitment(4, "hk_d", 50, "d" * 64),   # stale drop
    }
    snapshot = take_snapshot(
        raw_commitments=raw,
        window_number=7,
        window_start=200,
        snapshot_block=205,
        inactivity_blocks=120,
    )
    assert snapshot.window_number == 7
    assert snapshot.window_start == 200
    assert snapshot.snapshot_block == 205
    assert sorted(snapshot.commitments.keys()) == [1, 2]


def test_save_and_load_snapshot_round_trip(tmp_path):
    snapshot = EvalSnapshot(
        window_number=3,
        window_start=300,
        snapshot_block=307,
        commitments={
            11: _mk_commitment(11, "hk_11", 250, "1" * 64),
            12: _mk_commitment(12, "hk_12", 260, "2" * 64),
        },
    )
    path = save_snapshot(snapshot, tmp_path)
    assert path == snapshot_path(tmp_path, 3)
    loaded = load_snapshot(tmp_path, 3)
    assert loaded is not None
    assert loaded.window_number == 3
    assert loaded.window_start == 300
    assert loaded.snapshot_block == 307
    assert set(loaded.commitments.keys()) == {11, 12}
    assert loaded.commitments[11].hotkey == "hk_11"
    assert loaded.commitments[12].block_number == 260


def test_save_snapshot_prunes_old_files(tmp_path):
    for w in range(1, 8):
        snap = EvalSnapshot(
            window_number=w,
            window_start=w * 100,
            snapshot_block=w * 100 + 3,
            commitments={w: _mk_commitment(w, f"hk_{w}", w * 100 - 1, f"{w}" * 64)},
        )
        save_snapshot(snap, tmp_path, retention=DEFAULT_RETENTION)

    kept = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert kept == [
        "active_set_window_4.json",
        "active_set_window_5.json",
        "active_set_window_6.json",
        "active_set_window_7.json",
    ]


def test_load_snapshot_missing_file_returns_none(tmp_path):
    assert load_snapshot(tmp_path, 99) is None


def _api_response(entries):
    return {
        "epoch_number": 1234,
        "built_at": "2026-05-04T12:05:00.000Z",
        "window_start": 8092800,
        "cutoff_block": 8092080,
        "cutoff_time": "2026-05-04T12:00:00.000Z",
        "eligible_start_time": "2026-05-02T12:00:00.000Z",
        "inactivity_window_hours": 48,
        "snapshot_block": 8092105,
        "entries": entries,
    }


def _entry(uid, hk, ph, status, reason=None, refresh="2026-05-04T08:00:00.000Z"):
    return {
        "uid": uid,
        "hotkey": hk,
        "pack_hash": ph,
        "pack_url": f"https://example.com/{uid}.json",
        "refresh_time": refresh,
        "pre_eval_status": status,
        "pre_eval_reason": reason,
    }


def test_take_snapshot_from_api_splits_passed_and_failed():
    api = _api_response([
        _entry(10, "hk_a", "a" * 64, "passed"),
        _entry(11, "hk_b", "b" * 64, "failed", reason="hardcoded"),
        _entry(12, "hk_c", "c" * 64, "passed"),
    ])
    snap = take_snapshot_from_api(api, window_number=1234)

    # Passed entries land in commitments keyed by uid.
    assert sorted(snap.commitments.keys()) == [10, 12]
    assert snap.commitments[10].hotkey == "hk_a"
    assert snap.commitments[12].pack_hash == "c" * 64

    # Failed entry surfaces in pre_eval_failed keyed by hotkey,
    # carrying the reason for downstream rejection submission.
    assert set(snap.pre_eval_failed.keys()) == {"hk_b"}
    assert snap.pre_eval_failed["hk_b"].pre_eval_reason == "hardcoded"

    # window_start / snapshot_block come from the API response.
    assert snap.window_start == 8092800
    assert snap.snapshot_block == 8092105


def test_take_snapshot_from_api_assigns_index_as_block_number():
    """``block_number`` is synthesised from each passed entry's index in the
    API-sorted ``entries`` list so the existing ``pack_first_seen``
    tiebreak (sort by ``(block_number, hotkey)``) stays deterministic
    across validators receiving the same byte-identical response."""
    api = _api_response([
        _entry(10, "hk_a", "a" * 64, "passed"),
        _entry(11, "hk_b", "b" * 64, "failed", reason="hash_mismatch"),
        _entry(12, "hk_c", "c" * 64, "passed"),
        _entry(13, "hk_d", "d" * 64, "passed"),
    ])
    snap = take_snapshot_from_api(api, window_number=1234)
    # Failed entry consumes index 1; passed entries get 0, 2, 3.
    assert snap.commitments[10].block_number == 0
    assert snap.commitments[12].block_number == 2
    assert snap.commitments[13].block_number == 3


def test_take_snapshot_from_api_drops_unknown_status():
    api = _api_response([
        _entry(10, "hk_a", "a" * 64, "passed"),
        _entry(11, "hk_b", "b" * 64, "pending"),  # not 'passed' or 'failed'
    ])
    snap = take_snapshot_from_api(api, window_number=1234)
    assert list(snap.commitments.keys()) == [10]
    assert "hk_b" not in snap.pre_eval_failed


def test_save_and_load_round_trip_with_pre_eval_failed(tmp_path):
    api = _api_response([
        _entry(10, "hk_a", "a" * 64, "passed"),
        _entry(11, "hk_b", "b" * 64, "failed", reason="hardcoded"),
    ])
    snap = take_snapshot_from_api(api, window_number=42)

    save_snapshot(snap, tmp_path)
    loaded = load_snapshot(tmp_path, 42)

    assert loaded is not None
    assert sorted(loaded.commitments.keys()) == [10]
    assert "hk_b" in loaded.pre_eval_failed
    assert loaded.pre_eval_failed["hk_b"].pre_eval_reason == "hardcoded"


