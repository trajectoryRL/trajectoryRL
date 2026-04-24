"""Unit tests for trajectoryrl.utils.pack_ownership helpers."""

import json

import pytest

from trajectoryrl.utils.pack_ownership import (
    claim_owner,
    is_owner,
    evict_orphans,
    save_pack_first_seen,
    load_pack_first_seen,
    EVICTION_GRACE_WINDOWS,
    PACK_OWNERSHIP_FORMAT_VERSION,
)


# ── claim_owner ───────────────────────────────────────────────────────────

class TestClaimOwner:
    def test_first_writer_wins(self):
        table = {}
        owner, blk = claim_owner(table, "hash_x", "hk_alice", 100)
        assert (owner, blk) == ("hk_alice", 100)
        assert table["hash_x"] == ("hk_alice", 100)

    def test_returns_existing_owner_on_collision(self):
        table = {"hash_x": ("hk_alice", 100)}
        owner, blk = claim_owner(table, "hash_x", "hk_bob", 200)
        assert (owner, blk) == ("hk_alice", 100)
        assert table["hash_x"] == ("hk_alice", 100)

    def test_returns_existing_owner_for_same_hotkey(self):
        # Same hotkey re-submitting must not refresh the block number.
        table = {"hash_x": ("hk_alice", 100)}
        owner, blk = claim_owner(table, "hash_x", "hk_alice", 9999)
        assert (owner, blk) == ("hk_alice", 100)


# ── is_owner ──────────────────────────────────────────────────────────────

class TestIsOwner:
    def test_owner_match(self):
        table = {"hash_x": ("hk_alice", 100)}
        assert is_owner(table, "hash_x", "hk_alice") is True

    def test_owner_mismatch(self):
        table = {"hash_x": ("hk_alice", 100)}
        assert is_owner(table, "hash_x", "hk_bob") is False

    def test_unknown_hash(self):
        table = {"hash_x": ("hk_alice", 100)}
        assert is_owner(table, "hash_unknown", "hk_alice") is False


# ── evict_orphans (grace-window semantics) ────────────────────────────────

class TestEvictOrphansActiveRefresh:
    def test_active_hash_bumps_last_seen(self):
        table = {"hash_a": ("hk_a", 100)}
        last_seen = {"hash_a": 5}
        evicted = evict_orphans(
            table, last_seen, ["hash_a"], current_window=42,
        )
        assert evicted == []
        assert last_seen["hash_a"] == 42
        assert "hash_a" in table

    def test_active_hash_resets_stale_clock(self):
        # Even if the entry is well past the grace window, an active
        # reference refreshes the clock and prevents eviction.
        table = {"hash_a": ("hk_a", 100)}
        last_seen = {"hash_a": 0}
        evicted = evict_orphans(
            table, last_seen, ["hash_a"],
            current_window=999, grace_windows=7,
        )
        assert evicted == []
        assert last_seen["hash_a"] == 999
        assert "hash_a" in table

    def test_supports_iterable_active_hashes(self):
        # A generator is consumed exactly once; helper must materialize it.
        table = {"hash_a": ("hk_a", 100), "hash_b": ("hk_b", 200)}
        last_seen = {}
        evicted = evict_orphans(
            table, last_seen,
            (h for h in ["hash_a"]),
            current_window=10,
        )
        assert evicted == []  # both inside grace
        assert last_seen["hash_a"] == 10
        assert last_seen["hash_b"] == 10  # clock just started


class TestEvictOrphansFirstTimeOrphan:
    def test_no_last_seen_entry_starts_clock(self):
        # A pack that first appears as orphaned (no prior last_seen)
        # gets its clock initialized to current_window and is kept.
        table = {"hash_x": ("hk_x", 100)}
        last_seen = {}
        evicted = evict_orphans(
            table, last_seen, [], current_window=50,
        )
        assert evicted == []
        assert last_seen["hash_x"] == 50
        assert "hash_x" in table

    def test_subsequent_sweep_uses_started_clock(self):
        table = {"hash_x": ("hk_x", 100)}
        last_seen = {}
        evict_orphans(table, last_seen, [], current_window=50)
        # 6 windows later: still inside grace (7 needed).
        evicted = evict_orphans(
            table, last_seen, [], current_window=56,
        )
        assert evicted == []
        assert "hash_x" in table


class TestEvictOrphansGraceBoundary:
    def test_within_grace_keeps_entry(self):
        table = {"hash_x": ("hk_x", 100)}
        last_seen = {"hash_x": 10}
        # 6 windows since last seen: 6 < 7 → keep.
        evicted = evict_orphans(
            table, last_seen, [], current_window=16,
        )
        assert evicted == []
        assert "hash_x" in table

    def test_at_grace_boundary_evicts(self):
        table = {"hash_x": ("hk_x", 100)}
        last_seen = {"hash_x": 10}
        # 7 windows since last seen: 7 >= 7 → evict.
        evicted = evict_orphans(
            table, last_seen, [], current_window=17,
        )
        assert evicted == ["hash_x"]
        assert "hash_x" not in table
        assert "hash_x" not in last_seen

    def test_well_past_grace_evicts(self):
        table = {"hash_x": ("hk_x", 100)}
        last_seen = {"hash_x": 10}
        evicted = evict_orphans(
            table, last_seen, [], current_window=999,
        )
        assert evicted == ["hash_x"]
        assert table == {}
        assert last_seen == {}

    def test_custom_grace_windows_respected(self):
        table = {"hash_x": ("hk_x", 100)}
        last_seen = {"hash_x": 10}
        # grace_windows=3 → eviction at current_window=13.
        evicted = evict_orphans(
            table, last_seen, [],
            current_window=12, grace_windows=3,
        )
        assert evicted == []
        evicted = evict_orphans(
            table, last_seen, [],
            current_window=13, grace_windows=3,
        )
        assert evicted == ["hash_x"]


class TestEvictOrphansClockReset:
    def test_reactivation_resets_clock(self):
        table = {"hash_x": ("hk_x", 100)}
        last_seen = {}

        # Window 10: orphan, clock starts.
        evict_orphans(table, last_seen, [], current_window=10)
        # Window 15: still inside grace.
        evict_orphans(table, last_seen, [], current_window=15)
        # Window 16: re-activates → clock resets to 16.
        evict_orphans(
            table, last_seen, ["hash_x"], current_window=16,
        )
        assert last_seen["hash_x"] == 16
        # Window 22: 6 windows since last_seen=16, still inside grace.
        evicted = evict_orphans(
            table, last_seen, [], current_window=22,
        )
        assert evicted == []
        assert "hash_x" in table
        # Window 23: 7 windows since last_seen=16 → evict.
        evicted = evict_orphans(
            table, last_seen, [], current_window=23,
        )
        assert evicted == ["hash_x"]


class TestEvictOrphansMultipleEntries:
    def test_partition_active_orphan_evict(self):
        # active → kept; orphan within grace → kept; orphan past grace → evicted.
        table = {
            "active": ("hk_a", 1),
            "fresh_orphan": ("hk_b", 2),
            "stale_orphan": ("hk_c", 3),
        }
        last_seen = {
            "active": 0,
            "fresh_orphan": 18,
            "stale_orphan": 10,
        }
        evicted = evict_orphans(
            table, last_seen, ["active"], current_window=20,
        )
        assert evicted == ["stale_orphan"]
        assert set(table.keys()) == {"active", "fresh_orphan"}
        assert last_seen["active"] == 20
        assert last_seen["fresh_orphan"] == 18  # untouched
        assert "stale_orphan" not in last_seen


# ── save / load roundtrip ─────────────────────────────────────────────────

class TestPersistenceRoundtrip:
    def test_roundtrip_preserves_both_dicts(self, tmp_path):
        path = tmp_path / "pack_first_seen.json"
        table = {
            "hash_a": ("hk_alice", 100),
            "hash_b": ("hk_bob", 250),
        }
        last_seen = {"hash_a": 13, "hash_b": 17}
        save_pack_first_seen(table, last_seen, path)
        loaded_table, loaded_last_seen = load_pack_first_seen(path)
        assert loaded_table == table
        assert loaded_last_seen == last_seen

    def test_roundtrip_empty(self, tmp_path):
        path = tmp_path / "pack_first_seen.json"
        save_pack_first_seen({}, {}, path)
        assert load_pack_first_seen(path) == ({}, {})

    def test_save_writes_format_version(self, tmp_path):
        path = tmp_path / "pack_first_seen.json"
        save_pack_first_seen({"h": ("hk", 1)}, {"h": 5}, path)
        data = json.loads(path.read_text())
        assert data["version"] == PACK_OWNERSHIP_FORMAT_VERSION
        assert data["pack_first_seen"] == {"h": ["hk", 1]}
        assert data["pack_last_seen_window"] == {"h": 5}

    def test_save_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "subdir" / "pack_first_seen.json"
        save_pack_first_seen({"h": ("hk", 1)}, {"h": 1}, path)
        assert path.exists()


# ── load defensive paths ──────────────────────────────────────────────────

class TestLoadDefensive:
    def test_missing_file_returns_empty(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        assert load_pack_first_seen(path) == ({}, {})

    def test_malformed_json_returns_empty(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not valid json")
        assert load_pack_first_seen(path) == ({}, {})

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        assert load_pack_first_seen(path) == ({}, {})

    def test_missing_keys_returns_empty(self, tmp_path):
        path = tmp_path / "no_key.json"
        path.write_text(json.dumps({"version": PACK_OWNERSHIP_FORMAT_VERSION}))
        assert load_pack_first_seen(path) == ({}, {})

    def test_v1_file_reads_with_empty_last_seen(self, tmp_path):
        """v1 files (no `pack_last_seen_window` key) load with empty side dict.

        Forward-compat path: after upgrade, the first eviction sweep
        starts the grace clock for every still-orphaned entry.
        """
        path = tmp_path / "v1.json"
        path.write_text(json.dumps({
            "version": 1,
            "pack_first_seen": {"hash_a": ["hk_alice", 100]},
        }))
        table, last_seen = load_pack_first_seen(path)
        assert table == {"hash_a": ("hk_alice", 100)}
        assert last_seen == {}

    def test_v1_file_missing_version_field(self, tmp_path):
        # Even older write paths may have omitted `version`.
        path = tmp_path / "v0.json"
        path.write_text(json.dumps({
            "pack_first_seen": {"hash_a": ["hk_alice", 100]},
        }))
        table, last_seen = load_pack_first_seen(path)
        assert table == {"hash_a": ("hk_alice", 100)}
        assert last_seen == {}

    def test_skips_malformed_ownership_entries(self, tmp_path):
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({
            "version": 2,
            "pack_first_seen": {
                "good": ["hk_alice", 100],
                "wrong_type": "not_a_list",
                "too_short": ["only_one"],
                "bad_block": ["hk_bob", "not_an_int"],
                "bad_hotkey": [42, 100],
                "another_good": ["hk_carol", 999],
            },
            "pack_last_seen_window": {},
        }))
        table, _ = load_pack_first_seen(path)
        assert table == {
            "good": ("hk_alice", 100),
            "another_good": ("hk_carol", 999),
        }

    def test_skips_malformed_last_seen_entries(self, tmp_path):
        path = tmp_path / "partial_last_seen.json"
        path.write_text(json.dumps({
            "version": 2,
            "pack_first_seen": {"good": ["hk_alice", 100]},
            "pack_last_seen_window": {
                "good": 42,
                "bad_value": "not_an_int",
                "stringy_int": "13",
                "null_value": None,
            },
        }))
        _, last_seen = load_pack_first_seen(path)
        assert last_seen == {"good": 42, "stringy_int": 13}


def test_eviction_grace_windows_is_seven():
    """Sanity check: the documented default matches the constant."""
    assert EVICTION_GRACE_WINDOWS == 7
