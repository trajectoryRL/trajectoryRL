"""Tests for NCD pairwise dedup library (trajectoryrl.utils.ncd).

The validator no longer gates evaluation on NCD as of v5.1 — paraphrase
defense is delegated to Winner Protection's δ threshold. These tests
cover the library helpers, which are kept for tooling and possible
future re-introduction.
"""

import sys
from unittest.mock import MagicMock

# Mock bittensor before importing trajectoryrl modules (some imports trigger it)
if "bittensor" not in sys.modules:
    _mock_bt = MagicMock()
    class _MockSynapse:
        pass
    _mock_bt.Synapse = _MockSynapse
    sys.modules["bittensor"] = _mock_bt

from trajectoryrl.utils.ncd import (
    deduplicate_packs,
    normalize_policy,
    pack_similarity,
)


def _pack(content: str) -> dict:
    """Helper: build a minimal pack dict with given SKILL.md content."""
    return {"files": {"SKILL.md": content}}


DISTINCT_A = (
    "# Agent Policy\n\nYou are a helpful assistant. Always greet the user "
    "warmly and ask how you can help today. Use a professional tone."
)
DISTINCT_B = (
    "# Router Config\n\nThis agent handles API routing. Parse incoming "
    "requests, validate headers, and forward to the correct microservice."
)
PARAPHRASE_A = (
    "# Agent Policy\n\nYou are a helpful assistant. Always greet the user "
    "warmly and ask how you can help today. Use a professional tone. Be kind."
)


# ── normalize_policy ──────────────────────────────────────────────────────

class TestNormalizePolicy:
    def test_strips_headings(self):
        assert normalize_policy("## Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_policy("a   b\n\nc") == "a b c"

    def test_lowercase(self):
        assert normalize_policy("ABC") == "abc"


# ── pack_similarity ──────────────────────────────────────────────────────

class TestPackSimilarity:
    def test_identical_packs(self):
        p = _pack(DISTINCT_A)
        # NCD of identical short texts won't reach 1.0 due to zlib
        # header overhead, but should be well above the 0.80 threshold.
        assert pack_similarity(p, p) > 0.90

    def test_distinct_packs(self):
        assert pack_similarity(_pack(DISTINCT_A), _pack(DISTINCT_B)) < 0.5

    def test_paraphrase_is_high_similarity(self):
        sim = pack_similarity(_pack(DISTINCT_A), _pack(PARAPHRASE_A))
        assert sim >= 0.80


# ── deduplicate_packs ────────────────────────────────────────────────────

class TestDeduplicatePacks:
    # -- basic cases --

    def test_single_miner_no_exclusion(self):
        info = {"hk_a": (_pack(DISTINCT_A), 100, "hash_a")}
        assert deduplicate_packs(info) == {}

    def test_two_distinct_miners_no_exclusion(self):
        info = {
            "hk_a": (_pack(DISTINCT_A), 100, "hash_a"),
            "hk_b": (_pack(DISTINCT_B), 200, "hash_b"),
        }
        assert deduplicate_packs(info) == {}

    # -- Layer 1: exact copies --

    def test_layer1_exact_copy_excludes_later(self):
        pack = _pack(DISTINCT_A)
        info = {
            "hk_a": (pack, 100, "hash_same"),
            "hk_b": (pack, 200, "hash_same"),
        }
        result = deduplicate_packs(info)
        assert result == {"hk_b": "hk_a"}

    def test_layer1_three_exact_copies(self):
        pack = _pack(DISTINCT_A)
        info = {
            "hk_c": (pack, 300, "hash_same"),
            "hk_a": (pack, 100, "hash_same"),
            "hk_b": (pack, 200, "hash_same"),
        }
        result = deduplicate_packs(info)
        assert result == {"hk_b": "hk_a", "hk_c": "hk_a"}

    # -- Layer 2: NCD paraphrase detection --

    def test_layer2_paraphrase_excludes_later(self):
        info = {
            "hk_a": (_pack(DISTINCT_A), 100, "hash_a"),
            "hk_b": (_pack(PARAPHRASE_A), 200, "hash_b"),
        }
        result = deduplicate_packs(info)
        assert result == {"hk_b": "hk_a"}

    def test_layer2_paraphrase_keeps_earlier(self):
        """Even if insertion order differs, block_number determines priority."""
        info = {
            "hk_b": (_pack(PARAPHRASE_A), 200, "hash_b"),
            "hk_a": (_pack(DISTINCT_A), 100, "hash_a"),
        }
        result = deduplicate_packs(info)
        assert result == {"hk_b": "hk_a"}

    # -- Fix 1: Layer 2 transitive re-attribution --

    def test_layer2_reattributes_layer1_members(self):
        """When hash_j's rep is flagged by NCD, its Layer 1 members
        should be re-attributed to hash_i's original, not left pointing
        at the now-excluded rep."""
        info = {
            "hk_a": (_pack(DISTINCT_A), 100, "hash_a"),
            # B is rep for hash_b group, C is Layer 1 copy of B
            "hk_b": (_pack(PARAPHRASE_A), 200, "hash_b"),
            "hk_c": (_pack(PARAPHRASE_A), 300, "hash_b"),
        }
        result = deduplicate_packs(info)
        # B excluded by NCD (paraphrase of A)
        assert result["hk_b"] == "hk_a"
        # C was Layer 1 copy of B, but B is now excluded →
        # C should point to A (the true original)
        assert result["hk_c"] == "hk_a"

    # -- Fix 2: empty SKILL.md treated as identical --

    def test_empty_skill_md_treated_as_identical(self):
        """Two packs with empty SKILL.md should be similarity=1.0,
        later one excluded (not skipped)."""
        info = {
            "hk_a": (_pack(""), 100, "hash_a"),
            "hk_b": (_pack(""), 200, "hash_b"),
        }
        result = deduplicate_packs(info)
        assert result == {"hk_b": "hk_a"}

    def test_whitespace_only_skill_md(self):
        """Whitespace-only normalizes to empty → same as empty case."""
        info = {
            "hk_a": (_pack("   \n\n  "), 100, "hash_a"),
            "hk_b": (_pack("\t\n"), 200, "hash_b"),
        }
        result = deduplicate_packs(info)
        assert result == {"hk_b": "hk_a"}

    # -- Edge cases --

    def test_missing_skill_md_skipped(self):
        """Pack without SKILL.md is skipped in Layer 2 (no crash)."""
        info = {
            "hk_a": ({"files": {}}, 100, "hash_a"),
            "hk_b": (_pack(DISTINCT_A), 200, "hash_b"),
        }
        result = deduplicate_packs(info)
        assert result == {}

    def test_mixed_layer1_and_layer2(self):
        """Layer 1 exact + Layer 2 NCD both fire in one call."""
        info = {
            "hk_a": (_pack(DISTINCT_A), 100, "hash_a"),
            "hk_b": (_pack(DISTINCT_A), 200, "hash_a"),  # exact copy of A
            "hk_c": (_pack(PARAPHRASE_A), 300, "hash_c"),  # paraphrase of A
            "hk_d": (_pack(DISTINCT_B), 400, "hash_d"),  # unrelated
        }
        result = deduplicate_packs(info)
        assert result["hk_b"] == "hk_a"  # Layer 1
        assert result["hk_c"] == "hk_a"  # Layer 2
        assert "hk_d" not in result       # distinct, kept
        assert "hk_a" not in result       # original, kept

    def test_all_identical_content_different_hashes(self):
        """Same content but different pack_hash → Layer 2 catches it."""
        info = {
            "hk_a": (_pack(DISTINCT_A), 100, "hash_a"),
            "hk_b": (_pack(DISTINCT_A), 200, "hash_b"),
        }
        result = deduplicate_packs(info)
        assert result == {"hk_b": "hk_a"}


# ── PR #51 test plan scenarios ────────────────────────────────────────────
# These match the exact test plan from the PR description.

class TestPRTestPlan:
    """Exact scenarios from PR #51 test plan."""

    def test_plan_1_transitive_reattribution(self):
        """PR test plan #1: 3 miners where A (block 100) and B (block 200)
        have different pack_hash but similar content, and C (block 300)
        has same pack_hash as B.
        Expected: B and C both excluded, attributed to A."""
        info = {
            "hk_a": (_pack(DISTINCT_A), 100, "hash_a"),
            "hk_b": (_pack(PARAPHRASE_A), 200, "hash_b"),
            "hk_c": (_pack(PARAPHRASE_A), 300, "hash_b"),
        }
        result = deduplicate_packs(info)
        assert "hk_a" not in result, "A is the original, must not be excluded"
        assert result["hk_b"] == "hk_a", "B excluded, attributed to A"
        assert result["hk_c"] == "hk_a", "C excluded, attributed to A (not B)"

    def test_plan_2_empty_skill_md(self):
        """PR test plan #2: two packs with empty SKILL.md →
        later submitter excluded."""
        info = {
            "hk_a": (_pack(""), 100, "hash_a"),
            "hk_b": (_pack(""), 200, "hash_b"),
        }
        result = deduplicate_packs(info)
        assert result == {"hk_b": "hk_a"}
