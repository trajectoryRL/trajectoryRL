"""Tests for NCD pairwise dedup (trajectoryrl.utils.ncd)."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# Mock bittensor before importing trajectoryrl modules (validator needs it)
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
    """Helper: build a minimal pack dict with given AGENTS.md content."""
    return {"files": {"AGENTS.md": content}}


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

    # -- Fix 2: empty AGENTS.md treated as identical --

    def test_empty_agents_md_treated_as_identical(self):
        """Two packs with empty AGENTS.md should be similarity=1.0,
        later one excluded (not skipped)."""
        info = {
            "hk_a": (_pack(""), 100, "hash_a"),
            "hk_b": (_pack(""), 200, "hash_b"),
        }
        result = deduplicate_packs(info)
        assert result == {"hk_b": "hk_a"}

    def test_whitespace_only_agents_md(self):
        """Whitespace-only normalizes to empty → same as empty case."""
        info = {
            "hk_a": (_pack("   \n\n  "), 100, "hash_a"),
            "hk_b": (_pack("\t\n"), 200, "hash_b"),
        }
        result = deduplicate_packs(info)
        assert result == {"hk_b": "hk_a"}

    # -- Edge cases --

    def test_missing_agents_md_skipped(self):
        """Pack without AGENTS.md is skipped in Layer 2 (no crash)."""
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

    def test_plan_2_empty_agents_md(self):
        """PR test plan #2: two packs with empty AGENTS.md →
        later submitter excluded."""
        info = {
            "hk_a": (_pack(""), 100, "hash_a"),
            "hk_b": (_pack(""), 200, "hash_b"),
        }
        result = deduplicate_packs(info)
        assert result == {"hk_b": "hk_a"}

    def test_plan_3_cache_clearing(self):
        """PR test plan #3: deregistered miner from previous cycle
        doesn't appear in NCD comparisons.

        Simulates: cycle 1 populates _hotkey_packs with a miner,
        cycle 2 starts and that miner is no longer in commitments.
        The cache should be cleared so the stale entry doesn't leak
        into the weight-phase NCD dedup."""
        from trajectoryrl.base.validator import TrajectoryValidator
        from trajectoryrl.utils.commitments import MinerCommitment

        config = MagicMock()
        config.netuid = 11
        config.eval_interval_blocks = 1200
        config.similarity_threshold = 0.80
        config.log_level = "WARNING"
        config.scenarios = ["client_escalation"]
        config.scenarios_path = Path("/tmp/test_scenarios")
        config.inactivity_blocks = 14400
        config.weight_interval_blocks = 360
        config.ema_alpha = 0.3
        config.cost_ema_alpha = 0.3
        config.cost_delta = 0.10
        config.required_categories = ["safety", "correctness"]
        config.ema_state_path = Path("/tmp/test_ema_state.json")
        config.pack_cache_dir = Path("/tmp/test_packs")
        config.pack_cache_max_size = 100
        config.delta_threshold = 0.05

        mock_subtensor = MagicMock()
        mock_subtensor.get_current_block.return_value = 100000

        mock_metagraph = MagicMock()
        mock_metagraph.n = 2
        mock_metagraph.hotkeys = ["hk_0", "hk_1"]
        mock_metagraph.validator_permit = [False, False]
        mock_metagraph.S = [100.0, 100.0]
        mock_metagraph.stake = [100.0, 100.0]
        mock_subtensor.metagraph.return_value = mock_metagraph

        v = TrajectoryValidator.__new__(TrajectoryValidator)
        v.config = config
        v.metagraph = mock_metagraph
        v.subtensor = mock_subtensor
        v.ema_scores = {}
        v.raw_costs = {}
        v.scenario_qualified = {}
        v._eval_pack_hash = {}
        v.last_eval_block = {}
        v._hotkey_uid_map = {}
        v.scenarios = {"client_escalation": {"weight": 1.0}}
        v.wallet = MagicMock()
        v.last_weight_block = 0
        v._sandbox_harness = None
        v._disqualified_miners = {}

        # Winner state (used by _set_winner_weights, called in _execute_evaluation_cycle)
        from trajectoryrl.utils.winner_state import WinnerState
        v._winner_state = WinnerState()
        v._winner_state_path = "/tmp/test_winner_state.json"

        # Simulate stale data from a previous cycle: a deregistered miner
        stale_pack = _pack("# Stale miner policy that should be cleared")
        v._hotkey_packs = {"hk_deregistered": stale_pack}
        v._pack_by_hash = {"stale_hash_abc": stale_pack}
        v._cycle_eval_id = None
        v._cycle_log_offset = 0
        v._cycle_log_block = 0

        assert len(v._hotkey_packs) == 1
        assert len(v._pack_by_hash) == 1

        # Run _run_evaluation_cycle — it should clear caches as step 1.
        # We mock everything after the cache clear to return early
        # (no active commitments → falls through to fallback weights).
        with patch.object(
            TrajectoryValidator, "_check_llm_keys", return_value=True
        ), patch.object(
            TrajectoryValidator, "_get_validator_log_offset", return_value=0,
        ), patch.object(
            v, "_set_winner_weights", new_callable=AsyncMock,
        ), patch(
            "trajectoryrl.base.validator.fetch_all_commitments",
            return_value={},
        ), patch.object(
            v, "_filter_active_commitments", return_value={},
        ), patch.object(
            v, "_set_fallback_weights", new_callable=AsyncMock,
        ):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(v._run_evaluation_cycle(100000, window_number=0))
            finally:
                loop.close()

        # After cycle, stale entries must be gone
        assert v._hotkey_packs == {}, (
            "Stale _hotkey_packs should be cleared at cycle start"
        )
        assert v._pack_by_hash == {}, (
            "Stale _pack_by_hash should be cleared at cycle start"
        )
