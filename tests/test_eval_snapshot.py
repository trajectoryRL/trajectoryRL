"""Unit tests for trajectoryrl.utils.eval_snapshot helpers."""

from trajectoryrl.utils.commitments import MinerCommitment
from trajectoryrl.utils.eval_snapshot import (
    DEFAULT_RETENTION,
    EvalSnapshot,
    load_snapshot,
    save_snapshot,
    snapshot_path,
    take_snapshot,
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
