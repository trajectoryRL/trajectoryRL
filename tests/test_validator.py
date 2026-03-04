"""Tests for TrajectoryRL validator components.

Tests the scoring, ClawBench harness, OPP schema validation,
EMA scoring, and config without requiring a live Bittensor network.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock bittensor before importing any trajectoryrl modules
# ---------------------------------------------------------------------------
_mock_bt = MagicMock()

# bt.Synapse must be a real class so PackRequest/PackResponse can inherit
class _MockSynapse:
    pass

_mock_bt.Synapse = _MockSynapse
sys.modules["bittensor"] = _mock_bt

# Now safe to import
from trajectoryrl.utils.clawbench import ClawBenchHarness, EvaluationResult
from trajectoryrl.utils.opp_schema import validate_opp_schema, ValidationResult
from trajectoryrl.scoring import TrajectoryScorer, AggregatedScore
from trajectoryrl.utils.github import PackFetcher, PackVerificationResult
from trajectoryrl.base.validator import TrajectoryValidator
from trajectoryrl.utils.epoch_context import (
    generate_epoch_context, render_context_preamble,
    EpochContext, NAMES, ROLES, COMPANIES, DEPARTMENTS, TIMEZONES,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent  # /data2/trajectory_rl
CLAWBENCH_PATH = REPO_ROOT / "clawbench"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer():
    return TrajectoryScorer(
        rho_reliability=0.1,
        consensus_epsilon=0.02,
    )


@pytest.fixture
def harness():
    return ClawBenchHarness(clawbench_path=CLAWBENCH_PATH, timeout=120)


@pytest.fixture
def valid_pack():
    return {
        "schema_version": 1,
        "files": {
            "AGENTS.md": "# Agent rules\nBe safe and efficient.",
            "SOUL.md": "# Tone\nProfessional and concise.",
        },
        "tool_policy": {
            "allow": ["exec", "slack", "memory_search"],
            "deny": ["group:runtime"],
        },
        "metadata": {
            "pack_name": "test_pack",
            "pack_version": "1.0.0",
            "target_suite": "clawbench_v1",
        },
    }


@pytest.fixture
def sample_results():
    """Sample EvaluationResults for scoring tests."""
    return [
        EvaluationResult(
            scenario_name="client_escalation",
            score=0.92,
            success=True,
            tool_calls=10,
            response="Escalation summary...",
            rubric={"by_category": {"safety": {"score": 1.0}}},
        ),
        EvaluationResult(
            scenario_name="morning_brief",
            score=0.85,
            success=True,
            tool_calls=8,
            response="Daily brief...",
            rubric={"by_category": {"safety": {"score": 1.0}}},
        ),
        EvaluationResult(
            scenario_name="inbox_to_action",
            score=0.78,
            success=True,
            tool_calls=15,
            response="Action queue...",
            rubric={"by_category": {"safety": {"score": 0.9}}},
        ),
        EvaluationResult(
            scenario_name="team_standup",
            score=0.88,
            success=True,
            tool_calls=7,
            response="Standup prep...",
            rubric={"by_category": {"safety": {"score": 1.0}}},
        ),
    ]


# ===================================================================
# OPP Schema Validation Tests
# ===================================================================

class TestOPPSchemaValidation:

    def test_valid_pack_passes(self, valid_pack):
        result = validate_opp_schema(valid_pack)
        assert result.passed, f"Valid pack should pass, got issues: {result.issues}"

    def test_missing_schema_version(self, valid_pack):
        del valid_pack["schema_version"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("schema_version" in i for i in result.issues)

    def test_wrong_schema_version(self, valid_pack):
        valid_pack["schema_version"] = 2
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("schema_version" in i for i in result.issues)

    def test_missing_files(self, valid_pack):
        del valid_pack["files"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("files" in i for i in result.issues)

    def test_missing_agents_md(self, valid_pack):
        del valid_pack["files"]["AGENTS.md"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("AGENTS.md" in i for i in result.issues)

    def test_missing_tool_policy(self, valid_pack):
        del valid_pack["tool_policy"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("tool_policy" in i for i in result.issues)

    def test_missing_metadata(self, valid_pack):
        del valid_pack["metadata"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("metadata" in i for i in result.issues)

    def test_invalid_file_content_type(self, valid_pack):
        valid_pack["files"]["AGENTS.md"] = 123
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("string" in i for i in result.issues)

    def test_legacy_allowed_denied_rejected(self, valid_pack):
        valid_pack["tool_policy"]["allowed"] = ["exec"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("allow" in i and "deny" in i for i in result.issues)

    def test_invalid_semver(self, valid_pack):
        valid_pack["metadata"]["pack_version"] = "1.0"
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("semver" in i for i in result.issues)

    def test_oversized_pack(self, valid_pack):
        valid_pack["files"]["AGENTS.md"] = "x" * 200_000
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("too large" in i.lower() or "32KB" in i for i in result.issues)

    def test_validation_result_bool(self):
        assert bool(ValidationResult(passed=True, issues=[])) is True
        assert bool(ValidationResult(passed=False, issues=["error"])) is False


# ===================================================================
# TrajectoryScorer Tests
# ===================================================================

class TestTrajectoryScorer:

    def test_aggregate_empty_results(self, scorer):
        agg = scorer.aggregate_scores([])
        assert agg.mean_score == 0.0
        assert agg.variance == 0.0
        assert agg.success_rate == 0.0
        assert agg.total_evaluations == 0

    def test_aggregate_single_result(self, scorer):
        results = [
            EvaluationResult(
                scenario_name="test",
                score=0.85,
                success=True,
                tool_calls=5,
                response="ok",
                rubric={},
            )
        ]
        agg = scorer.aggregate_scores(results)
        assert agg.mean_score == 0.85
        assert agg.variance == 0.0  # single result = zero variance
        assert agg.success_rate == 1.0
        assert agg.total_evaluations == 1
        assert agg.scenario_scores == {"test": 0.85}

    def test_aggregate_multiple_results(self, scorer, sample_results):
        agg = scorer.aggregate_scores(sample_results)

        expected_mean = (0.92 + 0.85 + 0.78 + 0.88) / 4
        assert abs(agg.mean_score - expected_mean) < 1e-6
        assert agg.variance > 0  # different scores = non-zero variance
        assert agg.success_rate == 1.0
        assert agg.total_evaluations == 4
        assert len(agg.scenario_scores) == 4

    def test_aggregate_with_failures(self, scorer):
        results = [
            EvaluationResult("a", score=0.9, success=True, tool_calls=5, response="", rubric={}),
            EvaluationResult("b", score=0.0, success=False, tool_calls=0, response="", rubric={}),
        ]
        agg = scorer.aggregate_scores(results)
        assert agg.success_rate == 0.5
        assert agg.mean_score == 0.45

    def test_compute_final_score_no_variance(self, scorer):
        agg = AggregatedScore(
            mean_score=0.9,
            variance=0.0,
            success_rate=1.0,
            total_evaluations=4,
            scenario_scores={"a": 0.9},
        )
        final = scorer.compute_final_score(agg)
        assert final == 0.9  # No penalty

    def test_compute_final_score_with_variance(self, scorer):
        agg = AggregatedScore(
            mean_score=0.9,
            variance=0.1,
            success_rate=1.0,
            total_evaluations=4,
            scenario_scores={"a": 0.9},
        )
        # final = 0.9 - 0.1 * 0.1 = 0.89
        final = scorer.compute_final_score(agg)
        assert abs(final - 0.89) < 1e-6

    def test_compute_final_score_clamped(self, scorer):
        agg = AggregatedScore(
            mean_score=0.01,
            variance=1.0,
            success_rate=0.0,
            total_evaluations=1,
            scenario_scores={},
        )
        final = scorer.compute_final_score(agg)
        assert final == 0.0  # Clamped to 0

    def test_select_winner_basic(self, scorer):
        scores = {0: 0.85, 1: 0.72, 2: 0.91}
        weights = scorer.select_winner(
            scores, first_mover_data={}, delta=0.05, num_active_miners=20
        )

        assert weights[2] == 1.0  # uid 2 has highest score
        assert weights[0] == 0.0
        assert weights[1] == 0.0

    def test_select_winner_empty(self, scorer):
        assert scorer.select_winner({}, {}, delta=0.05) == {}

    def test_select_winner_single_miner(self, scorer):
        scores = {42: 0.8}
        weights = scorer.select_winner(
            scores, first_mover_data={}, delta=0.05, num_active_miners=20
        )
        assert weights[42] == 1.0

    def test_select_winner_first_mover_protection(self, scorer):
        """Early miner A (score=0.85) should be protected against B (score=0.88)
        because B doesn't beat A + delta (0.85 + 0.05 = 0.90)."""
        scores = {0: 0.85, 1: 0.88}
        uid_to_hotkey = {0: "hk_A", 1: "hk_B"}
        first_mover_data = {
            "hk_A": (0.85, 100.0),  # A submitted first
            "hk_B": (0.88, 200.0),  # B submitted later
        }
        weights = scorer.select_winner(
            scores, first_mover_data, delta=0.05, num_active_miners=20,
            uid_to_hotkey=uid_to_hotkey,
        )

        # A should win due to first-mover protection
        assert weights[0] == 1.0
        assert weights[1] == 0.0

    def test_select_winner_first_mover_beaten(self, scorer):
        """New miner B (score=0.91) beats A + delta (0.85 + 0.05 = 0.90)."""
        scores = {0: 0.85, 1: 0.91}
        uid_to_hotkey = {0: "hk_A", 1: "hk_B"}
        first_mover_data = {
            "hk_A": (0.85, 100.0),
            "hk_B": (0.91, 200.0),
        }
        weights = scorer.select_winner(
            scores, first_mover_data, delta=0.05, num_active_miners=20,
            uid_to_hotkey=uid_to_hotkey,
        )

        # B wins because 0.91 > 0.90
        assert weights[1] == 1.0
        assert weights[0] == 0.0

    def test_select_winner_transitive_first_mover(self, scorer):
        """First-mover protection must be transitive.

        A(block 100, score=0.76), B(block 200, score=0.80), C(block 300, score=0.84).
        - B can't beat A: 0.80 <= 0.76 + 0.05 = 0.81
        - C can beat A: 0.84 > 0.81
        - So C should win, NOT B.

        Previously, the code found B blocks C (0.84 <= 0.85) and stopped,
        awarding B the win even though B never legitimately beat A.
        """
        scores = {0: 0.76, 1: 0.80, 2: 0.84}
        uid_to_hotkey = {0: "hk_A", 1: "hk_B", 2: "hk_C"}
        first_mover_data = {
            "hk_A": (0.76, 100.0),
            "hk_B": (0.80, 200.0),
            "hk_C": (0.84, 300.0),
        }
        weights = scorer.select_winner(
            scores, first_mover_data, delta=0.05, num_active_miners=20,
            uid_to_hotkey=uid_to_hotkey,
        )

        # C beats A's protection (0.84 > 0.81), and B never became champion
        assert weights[2] == 1.0, "C should win (beats A's protection)"
        assert weights[0] == 0.0
        assert weights[1] == 0.0

    def test_full_scoring_pipeline(self, scorer, sample_results):
        """End-to-end: results -> aggregate -> final score."""
        agg = scorer.aggregate_scores(sample_results)
        final = scorer.compute_final_score(agg)

        expected_mean = (0.92 + 0.85 + 0.78 + 0.88) / 4  # 0.8575
        expected_penalty = 0.1 * agg.variance
        expected_final = max(0.0, min(1.0, expected_mean - expected_penalty))

        assert abs(final - expected_final) < 1e-6
        assert 0.0 <= final <= 1.0

    def test_compute_final_score_no_quantization(self):
        """Final score is continuous (no quantization)."""
        s = TrajectoryScorer(rho_reliability=0.1)
        agg = AggregatedScore(
            mean_score=0.87,
            variance=0.0,
            success_rate=1.0,
            total_evaluations=1,
            scenario_scores={"a": 0.87},
        )
        # 0.87 - 0 = 0.87, returned as-is
        assert s.compute_final_score(agg) == 0.87

    def test_select_winner_consensus_epsilon_tie(self):
        """Scores within ε → tie broken by earliest timestamp."""
        s = TrajectoryScorer(
            consensus_epsilon=0.02
        )
        # Two miners with nearly identical scores (within ε).
        # Without first_mover_data, miners have no block ordering, so the
        # iterative approach picks the first in iteration order.
        # With first_mover_data: miner 0 submitted first (earliest),
        # miner 1's score (0.91) is within ε of champion (0.90), so it
        # doesn't overtake — earliest miner keeps the crown.
        scores = {0: 0.90, 1: 0.91}
        uid_to_hotkey = {0: "hk_0", 1: "hk_1"}
        first_mover = {"hk_0": (0.90, 100.0), "hk_1": (0.91, 200.0)}
        # Use delta=0 to isolate epsilon behavior from first-mover protection
        weights = s.select_winner(
            scores, first_mover, delta=0.0, num_active_miners=20,
            uid_to_hotkey=uid_to_hotkey,
        )
        # |0.91 - 0.90| = 0.01 ≤ ε=0.02 → tied → earliest (miner 0) wins
        assert weights[0] == 1.0
        assert weights[1] == 0.0

    def test_select_winner_clear_margin_no_tie(self):
        """Scores separated by more than ε → no tie-break."""
        s = TrajectoryScorer(
            consensus_epsilon=0.02
        )
        scores = {0: 0.85, 1: 0.91}
        uid_to_hotkey = {0: "hk_0", 1: "hk_1"}
        first_mover = {"hk_0": (0.85, 100.0), "hk_1": (0.91, 200.0)}
        weights = s.select_winner(
            scores, first_mover, delta=0.05, num_active_miners=20,
            uid_to_hotkey=uid_to_hotkey,
        )

        # |0.91 - 0.85| = 0.06 > ε=0.02 → not tied, miner 1 wins on score
        # Also 0.91 > 0.85 + 0.05 = 0.90 → beats delta threshold
        assert weights[1] == 1.0
        assert weights[0] == 0.0

    def test_select_winner_bootstrap_graduated(self):
        """With few miners, use graduated 70/20/10 curve instead of WTA."""
        s = TrajectoryScorer(bootstrap_threshold=10)
        scores = {0: 0.80, 1: 0.91, 2: 0.85}
        uid_to_hotkey = {0: "hk_0", 1: "hk_1", 2: "hk_2"}
        first_mover = {"hk_0": (0.80, 100.0), "hk_1": (0.91, 200.0), "hk_2": (0.85, 300.0)}
        # 3 miners < threshold 10 → bootstrap mode
        weights = s.select_winner(
            scores, first_mover, delta=0.05, num_active_miners=3,
            uid_to_hotkey=uid_to_hotkey,
        )
        assert weights[1] == 0.70  # 1st (highest score)
        assert weights[2] == 0.20  # 2nd
        assert weights[0] == 0.10  # 3rd

    def test_select_winner_bootstrap_tie_break(self):
        """Bootstrap ties broken by earliest push timestamp."""
        s = TrajectoryScorer(bootstrap_threshold=10)
        scores = {0: 0.90, 1: 0.90}  # same score
        uid_to_hotkey = {0: "hk_0", 1: "hk_1"}
        first_mover = {"hk_0": (0.90, 200.0), "hk_1": (0.90, 100.0)}  # miner 1 pushed first
        weights = s.select_winner(
            scores, first_mover, delta=0.05, num_active_miners=2,
            uid_to_hotkey=uid_to_hotkey,
        )
        # 2 miners: raw 0.70/0.20 normalized to sum to 1.0
        assert weights[1] == pytest.approx(0.70 / 0.90)  # ~0.778
        assert weights[0] == pytest.approx(0.20 / 0.90)  # ~0.222

    def test_select_winner_bootstrap_single_miner(self):
        """Single miner in bootstrap gets weight 1.0 (normalized)."""
        s = TrajectoryScorer(bootstrap_threshold=10)
        scores = {0: 0.75}
        uid_to_hotkey = {0: "hk_0"}
        first_mover = {"hk_0": (0.75, 100.0)}
        weights = s.select_winner(
            scores, first_mover, delta=0.05, num_active_miners=1,
            uid_to_hotkey=uid_to_hotkey,
        )
        assert weights[0] == 1.0

    def test_select_winner_bootstrap_two_miners_sum_to_one(self):
        """Two miners in bootstrap: weights must sum to 1.0."""
        s = TrajectoryScorer(bootstrap_threshold=10)
        scores = {0: 0.80, 1: 0.91}
        uid_to_hotkey = {0: "hk_0", 1: "hk_1"}
        first_mover = {"hk_0": (0.80, 100.0), "hk_1": (0.91, 200.0)}
        weights = s.select_winner(
            scores, first_mover, delta=0.05, num_active_miners=2,
            uid_to_hotkey=uid_to_hotkey,
        )
        assert weights[1] > weights[0]  # higher score wins more
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_select_winner_above_threshold_is_wta(self):
        """Once miner count >= threshold, pure winner-take-all resumes."""
        s = TrajectoryScorer(bootstrap_threshold=3)
        scores = {0: 0.80, 1: 0.91, 2: 0.85}
        weights = s.select_winner(
            scores, first_mover_data={}, delta=0.05, num_active_miners=3
        )
        # 3 >= threshold 3 → WTA
        assert weights[1] == 1.0
        assert weights[0] == 0.0
        assert weights[2] == 0.0


# ===================================================================
# Epoch Seed & Scenario Rotation Tests
# ===================================================================


class TestEpochSeedAndScenarioRotation:
    """Tests for deterministic epoch seed and scenario selection."""

    def test_epoch_seed_deterministic(self):
        """Same epoch + netuid always produces same seed."""
        seed1 = TrajectoryValidator.compute_epoch_seed(42, netuid=11)
        seed2 = TrajectoryValidator.compute_epoch_seed(42, netuid=11)
        assert seed1 == seed2

    def test_epoch_seed_varies_by_epoch(self):
        """Different epochs produce different seeds."""
        seed1 = TrajectoryValidator.compute_epoch_seed(1, netuid=11)
        seed2 = TrajectoryValidator.compute_epoch_seed(2, netuid=11)
        assert seed1 != seed2

    def test_epoch_seed_varies_by_netuid(self):
        """Different netuids produce different seeds."""
        seed1 = TrajectoryValidator.compute_epoch_seed(1, netuid=11)
        seed2 = TrajectoryValidator.compute_epoch_seed(1, netuid=12)
        assert seed1 != seed2

    def test_epoch_seed_is_positive_int(self):
        """Epoch seed is a positive integer."""
        seed = TrajectoryValidator.compute_epoch_seed(1)
        assert isinstance(seed, int)
        assert seed > 0


# ===================================================================
# Epoch Context Tests
# ===================================================================


class TestEpochContext:
    """Tests for epoch context generation and rendering."""

    def test_generate_epoch_context_deterministic(self):
        """Same seed always produces the same context."""
        ctx1 = generate_epoch_context(12345)
        ctx2 = generate_epoch_context(12345)
        assert ctx1 == ctx2

    def test_generate_epoch_context_varies_by_seed(self):
        """Different seeds produce different contexts."""
        ctx1 = generate_epoch_context(1)
        ctx2 = generate_epoch_context(2)
        # At least one field should differ (extremely unlikely all match)
        assert (ctx1.user_name, ctx1.date_str, ctx1.company) != (
            ctx2.user_name, ctx2.date_str, ctx2.company
        )

    def test_generate_epoch_context_fields_from_pools(self):
        """All generated fields come from the defined pools."""
        ctx = generate_epoch_context(42)
        assert ctx.user_name in NAMES
        assert ctx.user_role in ROLES
        assert ctx.company in COMPANIES
        assert ctx.department in DEPARTMENTS
        tz_names = [tz[0] for tz in TIMEZONES]
        assert ctx.timezone in tz_names

    def test_generate_epoch_context_date_in_2026(self):
        """Generated date is in 2026."""
        ctx = generate_epoch_context(42)
        assert "2026" in ctx.date_str

    def test_generate_epoch_context_weekday_matches_date(self):
        """Weekday field matches the generated date."""
        from datetime import datetime
        ctx = generate_epoch_context(42)
        # Parse the date string and check weekday
        parsed = datetime.strptime(ctx.date_str, "%B %d, %Y")
        assert ctx.weekday == parsed.strftime("%A")

    def test_render_context_preamble_contains_fields(self):
        """Rendered preamble includes all context fields."""
        ctx = EpochContext(
            date_str="March 12, 2026",
            weekday="Thursday",
            user_name="Jordan Rivera",
            user_role="Product Manager",
            company="Meridian Technologies",
            department="Engineering",
            timezone="America/Chicago",
            timezone_abbr="CT",
        )
        preamble = render_context_preamble(ctx)
        assert "Jordan Rivera" in preamble
        assert "Product Manager" in preamble
        assert "Meridian Technologies" in preamble
        assert "Engineering" in preamble
        assert "March 12, 2026" in preamble
        assert "Thursday" in preamble
        assert "America/Chicago" in preamble
        assert "CT" in preamble

    def test_render_context_preamble_ends_with_separator(self):
        """Preamble ends with --- separator so miner AGENTS.md follows cleanly."""
        ctx = generate_epoch_context(42)
        preamble = render_context_preamble(ctx)
        assert "---" in preamble
        assert preamble.endswith("\n")

    def test_context_preamble_prepended_to_agents_md(self, harness):
        """Harness prepends context preamble to AGENTS.md in workspace."""
        pack = {
            "files": {"AGENTS.md": "# My Policy\nBe helpful."},
        }
        ctx = generate_epoch_context(42)
        preamble = render_context_preamble(ctx)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            harness._apply_pack_to_workspace(pack, workspace, preamble)

            agents_md = (workspace / "AGENTS.md").read_text()
            # Preamble comes first
            assert agents_md.startswith("<!-- Epoch Evaluation Context")
            # Miner content follows
            assert "# My Policy" in agents_md
            assert "Be helpful." in agents_md
            # Context fields are present
            assert ctx.user_name in agents_md

    def test_no_preamble_when_empty(self, harness):
        """Empty context_preamble leaves AGENTS.md unchanged."""
        pack = {
            "files": {"AGENTS.md": "# Original Content"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            harness._apply_pack_to_workspace(pack, workspace, "")
            agents_md = (workspace / "AGENTS.md").read_text()
            assert agents_md == "# Original Content"

    def test_variation_space_is_large(self):
        """Verify the combinatorial explosion across pools."""
        total = 365 * len(NAMES) * len(ROLES) * len(COMPANIES) * len(DEPARTMENTS) * len(TIMEZONES)
        assert total > 1_000_000, f"Variation space too small: {total}"

    # ---------------------------------------------------------------
    # to_user_context() and template substitution
    # ---------------------------------------------------------------

    def test_to_user_context_keys(self):
        """to_user_context() returns the right template keys."""
        ctx = EpochContext(
            date_str="March 12, 2026", weekday="Thursday",
            user_name="Jordan Rivera", user_role="Engineering Lead",
            company="Vertex Labs", department="Engineering",
            timezone="America/Chicago", timezone_abbr="CT",
        )
        uc = ctx.to_user_context()
        assert uc["USER_NAME"] == "Jordan Rivera"
        assert uc["USER_FIRST_NAME"] == "Jordan"
        assert uc["USER_ROLE"] == "Engineering Lead"
        assert uc["COMPANY"] == "Vertex Labs"

    def test_to_user_context_first_name_derived(self):
        """USER_FIRST_NAME is derived from USER_NAME automatically."""
        ctx = EpochContext(
            date_str="March 12, 2026", weekday="Thursday",
            user_name="Sam Patel", user_role="Product Manager",
            company="Cascade Systems", department="Product",
            timezone="America/Denver", timezone_abbr="MT",
        )
        uc = ctx.to_user_context()
        assert uc["USER_FIRST_NAME"] == "Sam"

    def test_fill_templates_basic(self):
        """ClawBench fill_templates replaces {{PLACEHOLDER}} markers."""
        import sys as _sys
        _sys.path.insert(0, str(CLAWBENCH_PATH / "scripts"))
        from run_episode import fill_templates

        content = "Hello {{USER_NAME}}, welcome to {{COMPANY}}."
        ctx = {"USER_NAME": "Jordan Rivera", "COMPANY": "Vertex Labs"}
        result = fill_templates(content, ctx)
        assert result == "Hello Jordan Rivera, welcome to Vertex Labs."

    def test_fill_templates_auto_first_name(self):
        """fill_templates auto-derives USER_FIRST_NAME from USER_NAME."""
        import sys as _sys
        _sys.path.insert(0, str(CLAWBENCH_PATH / "scripts"))
        from run_episode import fill_templates

        content = 'Sign off: "Best, {{USER_FIRST_NAME}}"'
        ctx = {"USER_NAME": "Jordan Rivera"}
        result = fill_templates(content, ctx)
        assert result == 'Sign off: "Best, Jordan"'

    def test_fill_templates_preserves_unknown(self):
        """Unknown {{PLACEHOLDER}} markers are left as-is."""
        import sys as _sys
        _sys.path.insert(0, str(CLAWBENCH_PATH / "scripts"))
        from run_episode import fill_templates

        content = "{{USER_NAME}} and {{UNKNOWN}}"
        ctx = {"USER_NAME": "Jordan Rivera"}
        result = fill_templates(content, ctx)
        assert "Jordan Rivera" in result
        assert "{{UNKNOWN}}" in result

    def test_fill_templates_on_real_fixture(self):
        """Template substitution works on a real fixture USER.md."""
        fixture_path = CLAWBENCH_PATH / "fixtures" / "client_escalation" / "USER.md"
        if not fixture_path.exists():
            pytest.skip("ClawBench fixtures not available")

        import sys as _sys
        _sys.path.insert(0, str(CLAWBENCH_PATH / "scripts"))
        from run_episode import fill_templates

        content = fixture_path.read_text()
        ctx = {
            "USER_NAME": "Sam Patel",
            "USER_FIRST_NAME": "Sam",
            "USER_ROLE": "Design Lead",
            "COMPANY": "Cascade Systems",
        }
        result = fill_templates(content, ctx)
        # Identity replaced
        assert "Sam Patel" in result
        assert "Cascade Systems" in result
        assert "{{USER_NAME}}" not in result
        assert "{{COMPANY}}" not in result
        # Team members preserved (not templated)
        assert "Sarah Kim" in result
        assert "David Park" in result

    def test_fill_templates_preserves_non_template_content(self):
        """Scenario-specific content is preserved by template substitution."""
        import sys as _sys
        _sys.path.insert(0, str(CLAWBENCH_PATH / "scripts"))
        from run_episode import fill_templates

        content = (
            "**Name:** {{USER_NAME}}\n"
            "Reports to: Sarah Kim (Eng Manager)\n"
            "Sprint 13 ends Friday Feb 7\n"
        )
        ctx = {"USER_NAME": "Morgan Kim", "COMPANY": "Summit Innovations"}
        result = fill_templates(content, ctx)
        assert "Morgan Kim" in result
        assert "Sarah Kim" in result
        assert "Sprint 13" in result

    def test_epoch_context_to_user_context_roundtrip(self):
        """Generate epoch context → to_user_context → fill_templates works."""
        import sys as _sys
        _sys.path.insert(0, str(CLAWBENCH_PATH / "scripts"))
        from run_episode import fill_templates

        ctx = generate_epoch_context(42)
        uc = ctx.to_user_context()
        content = "{{USER_NAME}} works as {{USER_ROLE}} at {{COMPANY}}"
        result = fill_templates(content, uc)
        assert ctx.user_name in result
        assert ctx.user_role in result
        assert ctx.company in result
        assert "{{" not in result


# ===================================================================
# Consensus Evaluation Tests
# ===================================================================


class TestConsensusEvaluation:
    """Tests for majority-vote consensus evaluation."""

    def test_majority_vote_rubric_all_pass(self):
        """All runs agree → all checks pass."""
        rubrics = [
            {"check_a": {"passed": True, "points": 4}, "check_b": {"passed": True, "points": 3}},
            {"check_a": {"passed": True, "points": 4}, "check_b": {"passed": True, "points": 3}},
            {"check_a": {"passed": True, "points": 4}, "check_b": {"passed": True, "points": 3}},
        ]
        voted = ClawBenchHarness._majority_vote_rubric(rubrics, quorum=2)
        assert voted["check_a"]["passed"] is True
        assert voted["check_b"]["passed"] is True

    def test_majority_vote_rubric_2_of_3(self):
        """2/3 runs pass → check passes."""
        rubrics = [
            {"check_a": {"passed": True, "points": 4}},
            {"check_a": {"passed": True, "points": 4}},
            {"check_a": {"passed": False, "points": 4}},
        ]
        voted = ClawBenchHarness._majority_vote_rubric(rubrics, quorum=2)
        assert voted["check_a"]["passed"] is True
        assert voted["check_a"]["_votes"] == "2/3"

    def test_majority_vote_rubric_1_of_3(self):
        """1/3 runs pass → check fails."""
        rubrics = [
            {"check_a": {"passed": True, "points": 4}},
            {"check_a": {"passed": False, "points": 4}},
            {"check_a": {"passed": False, "points": 4}},
        ]
        voted = ClawBenchHarness._majority_vote_rubric(rubrics, quorum=2)
        assert voted["check_a"]["passed"] is False

    def test_majority_vote_rubric_empty(self):
        voted = ClawBenchHarness._majority_vote_rubric([], quorum=2)
        assert voted == {}

    def test_score_from_rubric(self):
        """Score is earned_points / total_points."""
        rubric = {
            "a": {"passed": True, "points": 4},
            "b": {"passed": False, "points": 3},
            "c": {"passed": True, "points": 3},
        }
        score = ClawBenchHarness._score_from_rubric(rubric)
        assert abs(score - 7 / 10) < 1e-6

    def test_score_from_rubric_all_pass(self):
        rubric = {
            "a": {"passed": True, "points": 5},
            "b": {"passed": True, "points": 5},
        }
        assert ClawBenchHarness._score_from_rubric(rubric) == 1.0

    def test_score_from_rubric_empty(self):
        assert ClawBenchHarness._score_from_rubric({}) == 0.0

    def test_score_from_rubric_bool_values(self):
        """Rubric with plain bool values (no dict)."""
        rubric = {"a": True, "b": False, "c": True}
        score = ClawBenchHarness._score_from_rubric(rubric)
        assert abs(score - 2 / 3) < 1e-6

    # --- Tests for real score_episode() output format (bug fix coverage) ---

    def test_majority_vote_rubric_real_format(self):
        """Rubrics in the real score_episode() format with top-level metadata
        and a nested 'checks' list should be handled correctly."""
        rubrics = [
            {
                "score": 0.5, "points_earned": 5, "points_possible": 10,
                "passed": 2, "failed": 2, "total_checks": 4,
                "checks": [
                    {"id": "c1", "passed": True, "points": 3, "max_points": 3},
                    {"id": "c2", "passed": False, "points": 0, "max_points": 3},
                    {"id": "c3", "passed": True, "points": 2, "max_points": 2},
                    {"id": "c4", "passed": False, "points": 0, "max_points": 2},
                ],
                "by_category": {},
            },
            {
                "score": 0.5, "points_earned": 5, "points_possible": 10,
                "passed": 2, "failed": 2, "total_checks": 4,
                "checks": [
                    {"id": "c1", "passed": True, "points": 3, "max_points": 3},
                    {"id": "c2", "passed": True, "points": 3, "max_points": 3},
                    {"id": "c3", "passed": False, "points": 0, "max_points": 2},
                    {"id": "c4", "passed": False, "points": 0, "max_points": 2},
                ],
                "by_category": {},
            },
            {
                "score": 0.3, "points_earned": 3, "points_possible": 10,
                "passed": 1, "failed": 3, "total_checks": 4,
                "checks": [
                    {"id": "c1", "passed": True, "points": 3, "max_points": 3},
                    {"id": "c2", "passed": False, "points": 0, "max_points": 3},
                    {"id": "c3", "passed": False, "points": 0, "max_points": 2},
                    {"id": "c4", "passed": False, "points": 0, "max_points": 2},
                ],
                "by_category": {},
            },
        ]
        voted = ClawBenchHarness._majority_vote_rubric(rubrics, quorum=2)
        # c1: 3/3 pass → True
        assert voted["c1"]["passed"] is True
        # c2: 1/3 pass → False
        assert voted["c2"]["passed"] is False
        # c3: 1/3 pass → False
        assert voted["c3"]["passed"] is False
        # c4: 0/3 pass → False
        assert voted["c4"]["passed"] is False
        # Should only have the 4 check IDs, not top-level keys like "score"
        assert set(voted.keys()) == {"c1", "c2", "c3", "c4"}

    def test_score_from_rubric_max_points(self):
        """Failed checks have points=0 but max_points>0.
        Score must use max_points as the denominator weight so failed checks
        are NOT excluded from the total."""
        rubric = {
            "c1": {"passed": True, "points": 3, "max_points": 3},
            "c2": {"passed": False, "points": 0, "max_points": 3},
            "c3": {"passed": True, "points": 2, "max_points": 2},
            "c4": {"passed": False, "points": 0, "max_points": 2},
        }
        score = ClawBenchHarness._score_from_rubric(rubric)
        # earned = 3 + 2 = 5, total = 3 + 3 + 2 + 2 = 10 → 0.5
        assert abs(score - 0.5) < 1e-6

    def test_score_from_rubric_all_fail_not_100_percent(self):
        """When all checks fail, score should be 0, not 100%.
        This was the core symptom of using points (0 for failures) as weight."""
        rubric = {
            "c1": {"passed": False, "points": 0, "max_points": 5},
            "c2": {"passed": False, "points": 0, "max_points": 5},
        }
        score = ClawBenchHarness._score_from_rubric(rubric)
        assert score == 0.0

    def test_majority_vote_then_score_real_format(self):
        """End-to-end: majority vote on real-format rubrics then score."""
        rubric_template = {
            "score": 0.0, "points_earned": 0, "points_possible": 6,
            "passed": 0, "failed": 2, "total_checks": 2,
            "checks": [
                {"id": "greeting", "passed": True, "points": 3, "max_points": 3},
                {"id": "farewell", "passed": False, "points": 0, "max_points": 3},
            ],
            "by_category": {},
        }
        rubrics = [rubric_template, rubric_template, rubric_template]
        voted = ClawBenchHarness._majority_vote_rubric(rubrics, quorum=2)
        score = ClawBenchHarness._score_from_rubric(voted)
        # greeting passes (3/3), farewell fails (0/3) → 3/6 = 0.5
        assert abs(score - 0.5) < 1e-6


# ===================================================================
# ClawBenchHarness Tests
# ===================================================================

class TestClawBenchHarness:

    def test_init_validates_paths(self):
        with pytest.raises(ValueError, match="scripts not found"):
            ClawBenchHarness(clawbench_path=Path("/nonexistent"))

    def test_init_with_valid_path(self, harness):
        assert harness.clawbench_path == CLAWBENCH_PATH
        assert harness.timeout == 120

    def test_parse_episode_output_clean_json(self, harness):
        output = '{"score": 0.92, "success": true, "tool_calls": 8, "response": "test", "rubric": {}}'
        result = harness._parse_episode_output(output)
        assert result["score"] == 0.92
        assert result["success"] is True
        assert result["tool_calls"] == 8

    def test_parse_episode_output_json_after_logs(self, harness):
        output = (
            "Some log line\n"
            "Another log line\n"
            '{"score": 0.85, "success": false, "tool_calls": 3, "response": "", "rubric": {}}'
        )
        result = harness._parse_episode_output(output)
        assert result["score"] == 0.85
        assert result["success"] is False

    def test_parse_episode_output_no_json(self, harness):
        result = harness._parse_episode_output("no json here at all")
        assert result["score"] == 0.0
        assert "error" in result

    def test_parse_episode_output_invalid_json(self, harness):
        result = harness._parse_episode_output("{invalid json}")
        assert result["score"] == 0.0
        assert "error" in result

    def test_parse_episode_output_empty(self, harness):
        result = harness._parse_episode_output("")
        assert result["score"] == 0.0
        assert "error" in result

    def test_parse_episode_output_full_rubric(self, harness):
        """Test parsing a realistic scored output with full rubric."""
        data = {
            "score": 0.75,
            "success": False,
            "tool_calls": 5,
            "response": "Here is the summary...",
            "rubric": {
                "score": 0.75,
                "points_earned": 30,
                "points_possible": 40,
                "passed": 10,
                "failed": 5,
                "total_checks": 15,
                "by_category": {
                    "safety": {"earned": 12, "possible": 12, "score": 1.0},
                    "correctness": {"earned": 10, "possible": 15, "score": 0.667},
                },
            },
        }
        output = json.dumps(data)
        result = harness._parse_episode_output(output)
        assert result["score"] == 0.75
        assert result["rubric"]["by_category"]["safety"]["score"] == 1.0

    def test_compute_hash(self, harness, valid_pack):
        h = harness._compute_hash(valid_pack)
        expected = hashlib.sha256(
            json.dumps(valid_pack, sort_keys=True).encode()
        ).hexdigest()
        assert h == expected

    def test_compute_hash_deterministic(self, harness, valid_pack):
        h1 = harness._compute_hash(valid_pack)
        h2 = harness._compute_hash(valid_pack)
        assert h1 == h2

    def test_apply_pack_to_workspace(self, harness, valid_pack):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            harness._apply_pack_to_workspace(valid_pack, workspace)

            agents_file = workspace / "AGENTS.md"
            soul_file = workspace / "SOUL.md"

            assert agents_file.exists()
            assert soul_file.exists()
            assert agents_file.read_text() == valid_pack["files"]["AGENTS.md"]
            assert soul_file.read_text() == valid_pack["files"]["SOUL.md"]

    def test_apply_pack_empty_files(self, harness):
        pack = {"files": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            harness._apply_pack_to_workspace(pack, workspace)
            assert workspace.exists()
            # No files written
            assert list(workspace.iterdir()) == []

    def test_evaluate_pack_missing_scenario(self, harness, valid_pack):
        result = asyncio.get_event_loop().run_until_complete(
            harness.evaluate_pack(
                pack=valid_pack,
                scenario_name="nonexistent_scenario",
                seed=0,
            )
        )
        assert result.score == 0.0
        assert result.error is not None
        assert "not found" in result.error.lower()


# ===================================================================
# EvaluationResult Tests
# ===================================================================

class TestEvaluationResult:

    def test_creation(self):
        r = EvaluationResult(
            scenario_name="test",
            score=0.85,
            success=True,
            tool_calls=10,
            response="hello",
            rubric={"key": "value"},
        )
        assert r.scenario_name == "test"
        assert r.score == 0.85
        assert r.error is None

    def test_error_result(self):
        r = EvaluationResult(
            scenario_name="test",
            score=0.0,
            success=False,
            tool_calls=0,
            response="",
            rubric={},
            error="Timeout after 120s",
        )
        assert r.error == "Timeout after 120s"
        assert r.score == 0.0


# ===================================================================
# PackFetcher Tests
# ===================================================================

class TestPackFetcher:

    def test_init_creates_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "pack_cache"
            fetcher = PackFetcher(cache_dir=cache)
            assert cache.exists()

    def test_verify_valid_pack(self):
        """Valid pack URL + matching hash → verification passes."""
        pack = {"schema_version": 1, "files": {"AGENTS.md": "# Test"}}
        pack_json = json.dumps(pack, sort_keys=True)
        pack_hash = hashlib.sha256(pack_json.encode()).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            with patch.object(
                fetcher, "_fetch_pack",
                new_callable=AsyncMock, return_value=pack_json,
            ):
                result = asyncio.get_event_loop().run_until_complete(
                    fetcher.verify_submission(
                        pack_url="https://trajrl.com/samples/pack.json",
                        pack_hash=pack_hash,
                    )
                )

            assert result.valid is True
            assert result.pack_content == pack

    def test_verify_hash_mismatch(self):
        """Pack content doesn't match expected hash → verification fails."""
        pack_json = json.dumps({"schema_version": 1, "files": {"AGENTS.md": "# Test"}}, sort_keys=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            with patch.object(
                fetcher, "_fetch_pack",
                new_callable=AsyncMock, return_value=pack_json,
            ):
                result = asyncio.get_event_loop().run_until_complete(
                    fetcher.verify_submission(
                        pack_url="https://trajrl.com/samples/pack.json",
                        pack_hash="f" * 64,
                    )
                )

            assert result.valid is False
            assert "mismatch" in result.error.lower()

    def test_verify_fetch_failure(self):
        """HTTP fetch fails → verification fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            with patch.object(
                fetcher, "_fetch_pack",
                new_callable=AsyncMock, return_value=None,
            ):
                result = asyncio.get_event_loop().run_until_complete(
                    fetcher.verify_submission(
                        pack_url="https://trajrl.com/samples/pack.json",
                        pack_hash="a" * 64,
                    )
                )

            assert result.valid is False
            assert "fetch" in result.error.lower() or "failed" in result.error.lower()

    def test_verify_invalid_json(self):
        """Non-JSON response → verification fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            with patch.object(
                fetcher, "_fetch_pack",
                new_callable=AsyncMock, return_value="not json {{{",
            ):
                result = asyncio.get_event_loop().run_until_complete(
                    fetcher.verify_submission(
                        pack_url="https://trajrl.com/samples/pack.json",
                        pack_hash="a" * 64,
                    )
                )

            assert result.valid is False
            assert "json" in result.error.lower()

    def test_cache_hit_skips_fetch(self):
        """Cached pack with matching hash → no HTTP fetch needed."""
        pack = {"schema_version": 1, "files": {"AGENTS.md": "# Cached"}}
        pack_json = json.dumps(pack, sort_keys=True)
        pack_hash = hashlib.sha256(pack_json.encode()).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            # Pre-populate cache
            cache_path = Path(tmpdir) / f"{pack_hash}.json"
            cache_path.write_text(json.dumps(pack, sort_keys=True))

            with patch.object(
                fetcher, "_fetch_pack",
                new_callable=AsyncMock,
            ) as mock_fetch:
                result = asyncio.get_event_loop().run_until_complete(
                    fetcher.verify_submission(
                        pack_url="https://trajrl.com/samples/pack.json",
                        pack_hash=pack_hash,
                    )
                )

            assert result.valid is True
            assert result.pack_content == pack
            mock_fetch.assert_not_called()

    def test_fetch_pack_http_error(self):
        """HTTP 404 → _fetch_pack returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            with patch("httpx.AsyncClient") as MockClient:
                mock_resp = MagicMock()
                mock_resp.status_code = 404

                mock_client_instance = AsyncMock()
                mock_client_instance.get.return_value = mock_resp
                mock_client_instance.__aenter__ = AsyncMock(
                    return_value=mock_client_instance
                )
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_client_instance

                result = asyncio.get_event_loop().run_until_complete(
                    fetcher._fetch_pack("https://trajrl.com/samples/pack.json")
                )

            assert result is None

    def test_fetch_pack_success(self):
        """HTTP 200 with valid content → returns text."""
        pack_text = '{"schema_version": 1}'

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            with patch("httpx.AsyncClient") as MockClient:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.content = pack_text.encode()
                mock_resp.text = pack_text

                mock_client_instance = AsyncMock()
                mock_client_instance.get.return_value = mock_resp
                mock_client_instance.__aenter__ = AsyncMock(
                    return_value=mock_client_instance
                )
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_client_instance

                result = asyncio.get_event_loop().run_until_complete(
                    fetcher._fetch_pack("https://trajrl.com/samples/pack.json")
                )

            assert result == pack_text

    # ---------------------------------------------------------------
    # cleanup_cache (LRU eviction)
    # ---------------------------------------------------------------

    def test_cleanup_cache_no_eviction_when_under_limit(self):
        """Cache under limit → no entries evicted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            cached = Path(tmpdir) / "abc123.json"
            cached.write_text('{"test": true}')

            fetcher.cleanup_cache(max_size_mb=100)
            assert cached.exists()

    def test_cleanup_cache_evicts_oldest_first(self):
        """When over limit, oldest entries (by mtime) are evicted first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            old_file = Path(tmpdir) / "old.json"
            old_file.write_bytes(b"x" * 600_000)
            os.utime(old_file, (1000000, 1000000))

            new_file = Path(tmpdir) / "new.json"
            new_file.write_bytes(b"x" * 600_000)

            fetcher.cleanup_cache(max_size_mb=1)
            assert not old_file.exists(), "Old file should be evicted"
            assert new_file.exists(), "New file should be kept"

    def test_cleanup_cache_empty_dir(self):
        """Empty cache dir → no error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))
            fetcher.cleanup_cache(max_size_mb=100)


# ===================================================================
# Config Tests
# ===================================================================

class TestValidatorConfig:

    def test_default_scenarios(self):
        """Test that default scenarios list is correct."""
        from trajectoryrl.utils.config import ValidatorConfig

        # Can't fully instantiate (git check), but can inspect defaults
        defaults = ValidatorConfig.__dataclass_fields__
        assert "scenarios" in defaults
        # Check the default factory produces expected list
        scenarios = defaults["scenarios"].default_factory()
        assert "client_escalation" in scenarios
        assert "morning_brief" in scenarios
        assert "inbox_to_action" in scenarios
        assert "team_standup" in scenarios

    def test_default_scoring_params(self):
        from trajectoryrl.utils.config import ValidatorConfig
        defaults = ValidatorConfig.__dataclass_fields__
        assert defaults["rho_reliability"].default == 0.1
        assert defaults["delta_threshold"].default == 0.05
        assert defaults["seeds_per_task"].default == 1
        assert defaults["eval_interval_blocks"].default == 1200
        assert defaults["ema_alpha"].default == 0.3
        assert defaults["similarity_threshold"].default == 0.80
        assert defaults["inactivity_blocks"].default == 14400
        assert defaults["weight_interval_blocks"].default == 360

    def test_no_github_fields(self):
        """Verify github_token and validator_scores_fork_url are removed."""
        from trajectoryrl.utils.config import ValidatorConfig
        defaults = ValidatorConfig.__dataclass_fields__
        assert "github_token" not in defaults
        assert "validator_scores_fork_url" not in defaults
        assert "validator_scores_local_path" not in defaults


# ===================================================================
# NCD Similarity Tests
# ===================================================================


class TestNCDSimilarity:
    """Tests for Normalized Compression Distance anti-copy check."""

    def test_identical_packs(self):
        from trajectoryrl.utils.ncd import pack_similarity, is_too_similar
        a = {"files": {"AGENTS.md": "Be safe and careful. Follow instructions."}}
        b = {"files": {"AGENTS.md": "Be safe and careful. Follow instructions."}}
        sim = pack_similarity(a, b)
        assert sim > 0.90  # Identical content → near 1.0 (short strings have zlib overhead)
        assert is_too_similar(a, b, threshold=0.80) is True

    def test_completely_different_packs(self):
        from trajectoryrl.utils.ncd import pack_similarity, is_too_similar
        a = {"files": {"AGENTS.md": "Be safe and careful. Follow instructions precisely."}}
        b = {"files": {"AGENTS.md": "Cook pasta for dinner. Add salt to water."}}
        sim = pack_similarity(a, b)
        assert sim < 0.5  # Unrelated content → low similarity
        assert is_too_similar(a, b, threshold=0.80) is False

    def test_whitespace_only_difference(self):
        from trajectoryrl.utils.ncd import pack_similarity
        a = {"files": {"AGENTS.md": "# Rules\n\nBe safe and careful."}}
        b = {"files": {"AGENTS.md": "#  Rules\n\n\nBe  safe  and  careful."}}
        sim = pack_similarity(a, b)
        # After normalization, whitespace is collapsed → should be very similar
        assert sim > 0.90

    def test_no_winner_never_similar(self):
        from trajectoryrl.utils.ncd import is_too_similar
        a = {"files": {"AGENTS.md": "Any content"}}
        assert is_too_similar(a, None, threshold=0.80) is False

    def test_threshold_boundary(self):
        from trajectoryrl.utils.ncd import is_too_similar, pack_similarity
        a = {"files": {"AGENTS.md": "Be safe and careful. Follow all rules."}}
        b = {"files": {"AGENTS.md": "Be safe and careful. Follow all rules."}}
        sim = pack_similarity(a, b)
        # Exact match should be >= threshold
        assert is_too_similar(a, b, threshold=sim) is True
        # Threshold above similarity should pass
        assert is_too_similar(a, b, threshold=sim + 0.01) is False

    def test_normalize_strips_headings(self):
        from trajectoryrl.utils.ncd import normalize_policy
        result = normalize_policy("## My Heading\nSome text")
        assert "#" not in result
        assert "my heading" in result
        assert "some text" in result


# ===================================================================
# Commitment Parsing Tests
# ===================================================================


class TestCommitmentParsing:
    """Tests for on-chain commitment parsing."""

    def test_valid_commitment(self):
        from trajectoryrl.utils.commitments import parse_commitment
        raw = "a" * 64 + "|https://trajrl.com/samples/pack.json"
        result = parse_commitment(raw)
        assert result is not None
        pack_hash, pack_url = result
        assert pack_hash == "a" * 64
        assert pack_url == "https://trajrl.com/samples/pack.json"

    def test_https_url_commitment(self):
        from trajectoryrl.utils.commitments import parse_commitment
        raw = "a" * 64 + "|https://cdn.example.com/packs/v1/pack.json"
        result = parse_commitment(raw)
        assert result is not None
        _, pack_url = result
        assert pack_url == "https://cdn.example.com/packs/v1/pack.json"

    def test_http_url_commitment(self):
        from trajectoryrl.utils.commitments import parse_commitment
        raw = "a" * 64 + "|http://example.com/pack.json"
        result = parse_commitment(raw)
        assert result is not None
        _, pack_url = result
        assert pack_url == "http://example.com/pack.json"

    def test_empty_commitment(self):
        from trajectoryrl.utils.commitments import parse_commitment
        assert parse_commitment("") is None
        assert parse_commitment(None) is None

    def test_no_separator(self):
        from trajectoryrl.utils.commitments import parse_commitment
        assert parse_commitment("a" * 64) is None

    def test_invalid_hex_pack_hash(self):
        from trajectoryrl.utils.commitments import parse_commitment
        raw = "a" * 63 + "|https://trajrl.com/samples/pack.json"
        assert parse_commitment(raw) is None
        raw = "g" * 64 + "|https://trajrl.com/samples/pack.json"
        assert parse_commitment(raw) is None

    def test_invalid_url_scheme(self):
        from trajectoryrl.utils.commitments import parse_commitment
        raw = "a" * 64 + "|ftp://example.com/pack.json"
        assert parse_commitment(raw) is None

    def test_whitespace_handling(self):
        from trajectoryrl.utils.commitments import parse_commitment
        raw = "  " + "a" * 64 + " | https://trajrl.com/samples/pack.json  "
        result = parse_commitment(raw)
        assert result is not None


class TestGetCommitmentBlock:
    """Tests for _get_commitment_block and fetch_all_commitments integration."""

    def test_get_commitment_block_uses_hotkey_ss58_kwarg(self):
        """Verify that get_commitment_metadata is called with hotkey_ss58= (not hotkey=)."""
        from trajectoryrl.utils.commitments import _get_commitment_block

        mock_subtensor = MagicMock()
        mock_subtensor.get_commitment_metadata.return_value = {"block": 42000}

        result = _get_commitment_block(mock_subtensor, netuid=11, hotkey="5FHneW46...")

        mock_subtensor.get_commitment_metadata.assert_called_once_with(
            netuid=11, hotkey_ss58="5FHneW46..."
        )
        assert result == 42000

    def test_get_commitment_block_dict_access(self):
        """Verify dict-style access (meta['block']) works, not attribute access."""
        from trajectoryrl.utils.commitments import _get_commitment_block

        mock_subtensor = MagicMock()
        # Return a plain dict — no .block attribute, only dict key
        mock_subtensor.get_commitment_metadata.return_value = {"block": 99999}

        result = _get_commitment_block(mock_subtensor, netuid=11, hotkey="5Ftest...")
        assert result == 99999

    def test_get_commitment_block_string_block_cast_to_int(self):
        """Block number returned as string should be cast to int."""
        from trajectoryrl.utils.commitments import _get_commitment_block

        mock_subtensor = MagicMock()
        mock_subtensor.get_commitment_metadata.return_value = {"block": "12345"}

        result = _get_commitment_block(mock_subtensor, netuid=11, hotkey="5Ftest...")
        assert result == 12345
        assert isinstance(result, int)

    def test_get_commitment_block_fallback_on_exception(self):
        """Falls back to current block when get_commitment_metadata raises."""
        from trajectoryrl.utils.commitments import _get_commitment_block

        mock_subtensor = MagicMock()
        mock_subtensor.get_commitment_metadata.side_effect = Exception("not found")
        mock_subtensor.get_current_block.return_value = 50000

        result = _get_commitment_block(mock_subtensor, netuid=11, hotkey="5Ftest...")
        assert result == 50000

    def test_get_commitment_block_double_failure_returns_large_sentinel(self):
        """When both get_commitment_metadata and get_current_block fail,
        returns a large sentinel (not 0) to avoid artificially early timestamps."""
        from trajectoryrl.utils.commitments import _get_commitment_block

        mock_subtensor = MagicMock()
        mock_subtensor.get_commitment_metadata.side_effect = Exception("fail")
        mock_subtensor.get_current_block.side_effect = Exception("also fail")

        result = _get_commitment_block(mock_subtensor, netuid=11, hotkey="5Ftest...")
        assert result == 2**63, "Should return large sentinel, not 0"

    def test_get_commitment_block_fallback_on_none_metadata(self):
        """Falls back to current block when metadata is None."""
        from trajectoryrl.utils.commitments import _get_commitment_block

        mock_subtensor = MagicMock()
        mock_subtensor.get_commitment_metadata.return_value = None
        mock_subtensor.get_current_block.return_value = 60000

        result = _get_commitment_block(mock_subtensor, netuid=11, hotkey="5Ftest...")
        assert result == 60000

    def test_get_commitment_block_fallback_on_non_dict_metadata(self):
        """Falls back when metadata is not a dict (e.g. an object or string)."""
        from trajectoryrl.utils.commitments import _get_commitment_block

        mock_subtensor = MagicMock()
        mock_subtensor.get_commitment_metadata.return_value = "unexpected"
        mock_subtensor.get_current_block.return_value = 70000

        result = _get_commitment_block(mock_subtensor, netuid=11, hotkey="5Ftest...")
        assert result == 70000

    def test_fetch_all_commitments_uses_correct_block(self):
        """Integration: fetch_all_commitments passes hotkey to _get_commitment_block correctly."""
        from trajectoryrl.utils.commitments import fetch_all_commitments

        hotkey = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        raw = "a" * 64 + "|https://trajrl.com/samples/pack.json"

        mock_subtensor = MagicMock()
        mock_subtensor.get_all_commitments.return_value = {hotkey: raw}
        mock_subtensor.get_commitment_metadata.return_value = {"block": 42000}

        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = [hotkey]

        result = fetch_all_commitments(mock_subtensor, netuid=11, metagraph=mock_metagraph)

        assert 0 in result
        assert result[0].block_number == 42000
        assert result[0].hotkey == hotkey
        # Verify it was called with hotkey_ss58=, not hotkey=
        mock_subtensor.get_commitment_metadata.assert_called_once_with(
            netuid=11, hotkey_ss58=hotkey
        )


# ===================================================================
# Per-Scenario EMA Tests
# ===================================================================


class TestPerScenarioEMA:
    """Tests for per-scenario EMA scoring (keyed by hotkey)."""

    def _make_validator(self):
        """Create a minimal validator with mocked Bittensor components."""
        with patch("trajectoryrl.base.validator.bt") as mock_bt, \
             patch("trajectoryrl.base.validator.ClawBenchHarness"), \
             patch("trajectoryrl.base.validator.PackFetcher"), \
             patch("trajectoryrl.base.validator.yaml") as mock_yaml, \
             patch("trajectoryrl.base.validator.ValidatorConfig") as MockConfig:

            config = MagicMock()
            config.wallet_name = "test"
            config.wallet_hotkey = "default"
            config.network = "test"
            config.netuid = 11
            config.clawbench_path = Path("/tmp/test_clawbench")
            config.timeout_per_scenario = 120
            config.rho_reliability = 0.1
            config.consensus_epsilon = 0.02
            config.bootstrap_threshold = 10
            config.log_dir = Path("/tmp/test_logs")
            config.log_level = "WARNING"
            config.scenarios = ["client_escalation", "morning_brief"]
            config.scenarios_path = Path("/tmp/test_scenarios")
            config.inactivity_blocks = 14400
            config.eval_interval_blocks = 1200
            config.similarity_threshold = 0.80
            config.weight_interval_blocks = 360
            config.ema_alpha = 0.3
            config.ema_state_path = Path("/tmp/test_ema_state.json")
            config.pack_cache_dir = Path("/tmp/test_packs")
            config.pack_cache_max_size = 100
            config.delta_threshold = 0.05

            mock_subtensor = MagicMock()
            mock_subtensor.get_current_block.return_value = 100000
            mock_bt.Subtensor.return_value = mock_subtensor

            mock_metagraph = MagicMock()
            mock_metagraph.hotkeys = ["hk_0", "hk_1", "hk_2"]
            mock_metagraph.validator_permit = [False, False, False]
            mock_metagraph.S = [100.0, 100.0, 100.0]
            mock_subtensor.metagraph.return_value = mock_metagraph

            validator = TrajectoryValidator.__new__(TrajectoryValidator)
            validator.config = config
            validator.metagraph = mock_metagraph
            validator.subtensor = mock_subtensor
            validator.ema_scores = {}
            validator._ema_pack_hash = {}
            validator.last_eval_block = {}
            validator.first_mover_data = {}
            validator._hotkey_uid_map = {}
            validator._hotkey_packs = {}
            validator._pack_by_hash = {}
            validator.current_winner_pack = None
            validator.current_winner_hotkey = None
            validator.scenarios = {
                "client_escalation": {"weight": 1.5},
                "morning_brief": {"weight": 1.0},
            }
            validator.scorer = TrajectoryScorer(
                rho_reliability=0.1, consensus_epsilon=0.02
            )

            return validator

    def test_ema_first_observation(self):
        """First observation sets EMA directly (no smoothing)."""
        v = self._make_validator()
        v._update_ema("hk_0", "hash_a", {
            "client_escalation": 0.90,
            "morning_brief": 0.80,
        })
        assert v.ema_scores["hk_0"]["client_escalation"] == 0.90
        assert v.ema_scores["hk_0"]["morning_brief"] == 0.80

    def test_ema_smoothing(self):
        """Second observation applies EMA smoothing: α*new + (1-α)*old."""
        v = self._make_validator()
        v._update_ema("hk_0", "hash_a", {
            "client_escalation": 1.0,
            "morning_brief": 0.80,
        })
        v._update_ema("hk_0", "hash_a", {
            "client_escalation": 0.70,
            "morning_brief": 0.90,
        })
        # α=0.3: 0.3*0.70 + 0.7*1.0 = 0.21 + 0.70 = 0.91
        assert abs(v.ema_scores["hk_0"]["client_escalation"] - 0.91) < 1e-6
        # α=0.3: 0.3*0.90 + 0.7*0.80 = 0.27 + 0.56 = 0.83
        assert abs(v.ema_scores["hk_0"]["morning_brief"] - 0.83) < 1e-6

    def test_ema_resets_on_pack_change(self):
        """When pack_hash changes, EMA resets for that hotkey."""
        v = self._make_validator()
        v._update_ema("hk_0", "hash_a", {
            "client_escalation": 0.90,
        })
        assert v.ema_scores["hk_0"]["client_escalation"] == 0.90

        # New pack hash → EMA resets
        v._update_ema("hk_0", "hash_b", {
            "client_escalation": 0.70,
        })
        # Should be 0.70 (fresh start), not smoothed from 0.90
        assert v.ema_scores["hk_0"]["client_escalation"] == 0.70

    def test_ema_independent_per_hotkey(self):
        """EMA state is independent per hotkey."""
        v = self._make_validator()
        v._update_ema("hk_0", "hash_a", {"client_escalation": 0.90})
        v._update_ema("hk_1", "hash_b", {"client_escalation": 0.70})

        assert v.ema_scores["hk_0"]["client_escalation"] == 0.90
        assert v.ema_scores["hk_1"]["client_escalation"] == 0.70

    def test_compute_final_score_from_ema(self):
        """Final score from EMA: weighted_mean - ρ*weighted_variance."""
        v = self._make_validator()
        v.ema_scores["hk_0"] = {
            "client_escalation": 0.90,
            "morning_brief": 0.80,
        }
        # weights: client_escalation=1.5, morning_brief=1.0
        # weighted_mean = (1.5*0.90 + 1.0*0.80) / 2.5 = (1.35 + 0.80) / 2.5 = 0.86
        # weighted_var = (1.5*(0.90-0.86)^2 + 1.0*(0.80-0.86)^2) / 2.5
        #             = (1.5*0.0016 + 1.0*0.0036) / 2.5
        #             = (0.0024 + 0.0036) / 2.5 = 0.0024
        # final = 0.86 - 0.1*0.0024 = 0.85976
        final = v.compute_final_score_from_ema("hk_0")
        assert abs(final - 0.85976) < 1e-4

    def test_compute_final_score_empty_ema(self):
        """Empty EMA returns 0."""
        v = self._make_validator()
        assert v.compute_final_score_from_ema("hk_unknown") == 0.0

    def test_needs_evaluation_new_miner(self):
        """New miner (never evaluated) needs evaluation."""
        v = self._make_validator()
        assert v._needs_evaluation("hk_new", "hash_a", 100000) is True

    def test_needs_evaluation_pack_changed(self):
        """Pack hash change triggers re-evaluation."""
        v = self._make_validator()
        v._ema_pack_hash["hk_0"] = "hash_a"
        v.last_eval_block["hk_0"] = 99999
        assert v._needs_evaluation("hk_0", "hash_b", 100000) is True

    def test_needs_evaluation_within_interval(self):
        """Within eval interval and same pack → no re-evaluation."""
        v = self._make_validator()
        v._ema_pack_hash["hk_0"] = "hash_a"
        v.last_eval_block["hk_0"] = 99500  # 500 blocks ago < 1200
        assert v._needs_evaluation("hk_0", "hash_a", 100000) is False

    def test_needs_evaluation_interval_exceeded(self):
        """Past eval interval → needs re-evaluation."""
        v = self._make_validator()
        v._ema_pack_hash["hk_0"] = "hash_a"
        v.last_eval_block["hk_0"] = 98000  # 2000 blocks ago > 1200
        assert v._needs_evaluation("hk_0", "hash_a", 100000) is True

    def test_ema_persistence_roundtrip(self):
        """EMA state can be saved and loaded."""
        v = self._make_validator()
        v._scenario_config_hash = "test_hash"
        v.ema_scores = {"hk_0": {"client_escalation": 0.85}}
        v._ema_pack_hash = {"hk_0": "hash_a"}
        v.last_eval_block = {"hk_0": 99000}
        v.first_mover_data = {"hk_0": (0.85, 1000.0)}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.ema_state_path = Path(f.name)

        try:
            v._save_ema_state()

            # Create a fresh validator and load
            v2 = self._make_validator()
            v2.config.ema_state_path = v.config.ema_state_path
            v2._scenario_config_hash = "test_hash"
            v2._load_ema_state()

            assert v2.ema_scores == {"hk_0": {"client_escalation": 0.85}}
            assert v2._ema_pack_hash == {"hk_0": "hash_a"}
            assert v2.last_eval_block == {"hk_0": 99000}
            assert v2.first_mover_data == {"hk_0": (0.85, 1000.0)}
        finally:
            v.config.ema_state_path.unlink(missing_ok=True)

    def test_ema_invalidated_on_scenario_pool_change(self):
        """Loading EMA state with different scenario config invalidates all state."""
        v = self._make_validator()
        v._scenario_config_hash = "old_hash"
        v.ema_scores = {"hk_0": {"client_escalation": 0.85}}
        v._ema_pack_hash = {"hk_0": "hash_a"}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.ema_state_path = Path(f.name)

        try:
            v._save_ema_state()

            # Load with different scenario config hash
            v2 = self._make_validator()
            v2.config.ema_state_path = v.config.ema_state_path
            v2._scenario_config_hash = "new_hash"
            v2._load_ema_state()

            assert v2.ema_scores == {}
            assert v2._ema_pack_hash == {}
        finally:
            v.config.ema_state_path.unlink(missing_ok=True)


# ===================================================================
# Inactivity Tests (Block-Based)
# ===================================================================


class TestInactivityBlocks:
    """Tests for block-based miner inactivity tracking."""

    def _make_validator(self):
        """Create a minimal validator with mocked Bittensor components."""
        with patch("trajectoryrl.base.validator.bt") as mock_bt, \
             patch("trajectoryrl.base.validator.ClawBenchHarness"), \
             patch("trajectoryrl.base.validator.PackFetcher"), \
             patch("trajectoryrl.base.validator.yaml") as mock_yaml, \
             patch("trajectoryrl.base.validator.ValidatorConfig") as MockConfig:

            config = MagicMock()
            config.wallet_name = "test"
            config.wallet_hotkey = "default"
            config.network = "test"
            config.netuid = 11
            config.clawbench_path = Path("/tmp/test_clawbench")
            config.timeout_per_scenario = 120
            config.rho_reliability = 0.1
            config.consensus_epsilon = 0.02
            config.bootstrap_threshold = 10
            config.log_dir = Path("/tmp/test_logs")
            config.log_level = "WARNING"
            config.scenarios = ["client_escalation"]
            config.scenarios_path = Path("/tmp/test_scenarios")
            config.inactivity_blocks = 14400
            config.eval_interval_blocks = 1200
            config.similarity_threshold = 0.80
            config.weight_interval_blocks = 360
            config.ema_alpha = 0.3
            config.ema_state_path = Path("/tmp/test_ema_state.json")

            mock_subtensor = MagicMock()
            mock_subtensor.get_current_block.return_value = 100000
            mock_bt.Subtensor.return_value = mock_subtensor

            mock_metagraph = MagicMock()
            mock_metagraph.hotkeys = ["hk_0", "hk_1", "hk_2"]
            mock_metagraph.validator_permit = [False, False, False]
            mock_metagraph.S = [100.0, 100.0, 100.0]
            mock_subtensor.metagraph.return_value = mock_metagraph

            validator = TrajectoryValidator.__new__(TrajectoryValidator)
            validator.config = config
            validator.metagraph = mock_metagraph
            validator.subtensor = mock_subtensor
            validator.ema_scores = {}
            validator._ema_pack_hash = {}
            validator.last_eval_block = {}
            validator.first_mover_data = {}
            validator._hotkey_uid_map = {}
            validator._hotkey_packs = {}
            validator._pack_by_hash = {}
            validator.current_winner_pack = None
            validator.current_winner_hotkey = None

            return validator

    def test_active_miner_within_inactivity_window(self):
        """Miner evaluated recently is still active."""
        v = self._make_validator()
        v.last_eval_block["hk_0"] = 90000  # 10000 blocks ago < 14400

        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            0: MinerCommitment(
                uid=0, hotkey="hk_0", pack_hash="a" * 64,
                pack_url="https://trajrl.com/samples/pack.json",
                block_number=1000, raw="raw",
            ),
        }
        active = v._get_active_miners_from_commitments(commitments, 100000)
        assert 0 in active

    def test_inactive_miner_loses_first_mover(self):
        """Miner inactive > inactivity_blocks loses first-mover protection."""
        v = self._make_validator()
        v.last_eval_block["hk_1"] = 80000  # 20000 blocks ago > 14400
        v.first_mover_data["hk_1"] = (0.85, 5000.0)

        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            1: MinerCommitment(
                uid=1, hotkey="hk_1", pack_hash="a" * 64,
                pack_url="https://trajrl.com/samples/pack.json",
                block_number=1000, raw="raw",
            ),
        }
        active = v._get_active_miners_from_commitments(commitments, 100000)
        assert 1 not in active
        assert "hk_1" not in v.first_mover_data

    def test_never_evaluated_miner_is_active(self):
        """Miner never evaluated (no last_eval_block) is treated as active
        so it can be evaluated this cycle."""
        v = self._make_validator()

        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            0: MinerCommitment(
                uid=0, hotkey="hk_0", pack_hash="a" * 64,
                pack_url="https://trajrl.com/samples/pack.json",
                block_number=1000, raw="raw",
            ),
        }
        active = v._get_active_miners_from_commitments(commitments, 100000)
        assert 0 in active

    def test_reactivation_after_inactivity(self):
        """Miner returning after inactivity (new eval updates last_eval_block)."""
        v = self._make_validator()
        v.last_eval_block["hk_0"] = 80000  # Was inactive

        # After a successful evaluation, last_eval_block is updated
        v.last_eval_block["hk_0"] = 100000

        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            0: MinerCommitment(
                uid=0, hotkey="hk_0", pack_hash="a" * 64,
                pack_url="https://trajrl.com/samples/pack.json",
                block_number=9000, raw="raw",
            ),
        }
        active = v._get_active_miners_from_commitments(commitments, 100000)
        assert 0 in active

    def test_validators_filtered_out(self):
        """UIDs with validator_permit=True are filtered out."""
        v = self._make_validator()
        v.metagraph.validator_permit = [True, False, False]

        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            0: MinerCommitment(
                uid=0, hotkey="hk_0", pack_hash="a" * 64,
                pack_url="https://trajrl.com/samples/pack.json",
                block_number=1000, raw="raw",
            ),
            1: MinerCommitment(
                uid=1, hotkey="hk_1", pack_hash="b" * 64,
                pack_url="https://trajrl.com/samples/pack.json",
                block_number=2000, raw="raw",
            ),
        }
        active = v._get_active_miners_from_commitments(commitments, 100000)
        assert 0 not in active  # Validator
        assert 1 in active      # Miner


# ===================================================================
# Integration: ClawBench Scoring → Validator Scoring Pipeline
# ===================================================================

class TestScoringIntegration:
    """Tests the full data flow from ClawBench output to validator weights."""

    def test_json_output_to_evaluation_result(self, harness):
        """ClawBench --json output → _parse_episode_output → EvaluationResult."""
        json_output = json.dumps({
            "score": 0.88,
            "success": True,
            "tool_calls": 11,
            "response": "Summary of actions taken...",
            "rubric": {
                "score": 0.88,
                "points_earned": 35,
                "points_possible": 40,
                "by_category": {"safety": {"score": 1.0}},
            },
        })

        parsed = harness._parse_episode_output(json_output)
        result = EvaluationResult(
            scenario_name="client_escalation",
            score=parsed["score"],
            success=parsed["success"],
            tool_calls=parsed["tool_calls"],
            response=parsed["response"],
            rubric=parsed["rubric"],
        )

        assert result.score == 0.88
        assert result.success is True
        assert result.tool_calls == 11

    def test_full_pipeline_json_to_weights(self, harness, scorer):
        """Full: 4 scenario JSON outputs → aggregate → final score → winner."""
        scenario_outputs = [
            ("client_escalation", 0.92),
            ("morning_brief", 0.85),
            ("inbox_to_action", 0.78),
            ("team_standup", 0.88),
        ]

        results = []
        for scenario, score in scenario_outputs:
            json_output = json.dumps({
                "score": score,
                "success": score > 0.5,
                "tool_calls": 10,
                "response": f"{scenario} done",
                "rubric": {"score": score},
            })
            parsed = harness._parse_episode_output(json_output)
            results.append(EvaluationResult(
                scenario_name=scenario,
                score=parsed["score"],
                success=parsed["success"],
                tool_calls=parsed["tool_calls"],
                response=parsed["response"],
                rubric=parsed["rubric"],
            ))

        # Aggregate and score
        agg = scorer.aggregate_scores(results)
        final = scorer.compute_final_score(agg)

        # Two miners compete (num_active_miners above threshold for WTA)
        miner_scores = {0: final, 1: final * 0.8}
        weights = scorer.select_winner(
            miner_scores, first_mover_data={}, delta=0.05, num_active_miners=20
        )

        assert weights[0] == 1.0  # Higher score wins
        assert weights[1] == 0.0
        assert 0.0 < final <= 1.0

    def test_clawbench_scoring_roundtrip(self):
        """Test that clawbench scoring.py output matches what harness expects."""
        sys.path.insert(0, str(CLAWBENCH_PATH))
        from clawbench.scoring import score_episode
        import yaml

        with open(CLAWBENCH_PATH / "scenarios" / "client_escalation.yaml") as f:
            scenario = yaml.safe_load(f)

        # Simulate a good result
        tool_calls = [{"tool": "exec"}] * 5 + [{"tool": "slack"}] * 3
        from collections import Counter
        tool_counts = dict(Counter(tc["tool"] for tc in tool_calls))

        scorable = {
            "response": (
                "## P0 Status: Data Export Incident\n\n"
                "Root cause: cursor reset regression in v2.14.5.\n"
                "Fix: PR #356 ready, staging validated. ETA: deploy by 1pm today.\n"
                "Affected: Zenith Financial, GlobalTech, Meridian Health.\n"
                "Calendar conflict: 2pm interview overlaps with Acme call.\n"
                "SOC 2 findings noted — defer until P0 resolved.\n\n"
                "Recommended action plan:\n"
                "1. Approve Marcus's hotfix deploy\n"
                "2. Draft reply to Dana Reeves for your approval\n"
            ),
            "tool_calls_raw": tool_calls,
            "tool_calls_by_type": tool_counts,
            "tool_calls_total": len(tool_calls),
        }

        score_result = score_episode(scorable, scenario["scoring"])

        # Build the JSON output that run_episode.py --json would produce
        output = {
            "score": score_result["score"],
            "success": score_result.get("failed", 1) == 0,
            "tool_calls": len(tool_calls),
            "response": scorable["response"],
            "rubric": score_result,
        }

        # Parse it like the validator harness would
        harness = ClawBenchHarness(clawbench_path=CLAWBENCH_PATH)
        parsed = harness._parse_episode_output(json.dumps(output))

        assert parsed["score"] == score_result["score"]
        assert "by_category" in parsed["rubric"]
        assert parsed["tool_calls"] == 8

        # Feed into EvaluationResult
        eval_result = EvaluationResult(
            scenario_name="client_escalation",
            score=parsed["score"],
            success=parsed["success"],
            tool_calls=parsed["tool_calls"],
            response=parsed["response"],
            rubric=parsed["rubric"],
        )

        # Score should be high for this good result
        assert eval_result.score >= 0.7, f"Expected good score, got {eval_result.score}"
