"""Tests for TrajectoryRL validator components.

Tests the scoring, ClawBench harness, OPP schema validation,
and config without requiring a live Bittensor network.
"""

import asyncio
import hashlib
import json
import os
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
from trajectoryrl.utils.github import GitHubVerifier, GitVerificationResult
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
        # Two miners with nearly identical scores (within ε)
        scores = {0: 0.90, 1: 0.91}
        # No first-mover data → pure epsilon tie-break
        weights = s.select_winner(
            scores, first_mover_data={}, delta=0.05, num_active_miners=20
        )
        # Without first_mover_data, no timestamps to break tie → highest score wins
        assert weights[1] == 1.0

        # Now with timestamps: miner 0 submitted first
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

    # NOTE: select_epoch_scenarios was removed in v0.2.0.
    # All scenarios run every epoch (per INCENTIVE_MECHANISM.md v1.06).


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
# GitHubVerifier Tests
# ===================================================================

class TestGitHubVerifier:

    def test_init_creates_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "git_cache"
            verifier = GitHubVerifier(cache_dir=cache)
            assert cache.exists()

    def test_init_with_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(
                cache_dir=Path(tmpdir), github_token="ghp_test123"
            )
            assert verifier.github_token == "ghp_test123"

    def test_verify_commit_exists_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))
            # Non-git directory
            result = asyncio.get_event_loop().run_until_complete(
                verifier._verify_commit_exists(
                    Path(tmpdir), "abc123" * 7  # fake hash
                )
            )
            assert result is False

    def test_get_commit_timestamp_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))
            result = asyncio.get_event_loop().run_until_complete(
                verifier._get_commit_timestamp(
                    Path(tmpdir), "abc123" * 7
                )
            )
            assert result is None

    # ---------------------------------------------------------------
    # _parse_github_url
    # ---------------------------------------------------------------

    def test_parse_github_url_basic(self):
        owner, repo = GitHubVerifier._parse_github_url(
            "https://github.com/alice/my-pack"
        )
        assert owner == "alice"
        assert repo == "my-pack"

    def test_parse_github_url_trailing_slash(self):
        owner, repo = GitHubVerifier._parse_github_url(
            "https://github.com/alice/my-pack/"
        )
        assert owner == "alice"
        assert repo == "my-pack"

    def test_parse_github_url_dot_git(self):
        owner, repo = GitHubVerifier._parse_github_url(
            "https://github.com/alice/my-pack.git"
        )
        assert owner == "alice"
        assert repo == "my-pack"

    # ---------------------------------------------------------------
    # _get_push_timestamp_events_api (mocked)
    # ---------------------------------------------------------------

    def test_events_api_finds_commit_via_head(self):
        """Events API finds commit as the head of a PushEvent (fast path)."""
        commit_sha = "a" * 40

        mock_events = [
            {
                "type": "CreateEvent",
                "created_at": "2026-02-10T10:00:00Z",
                "payload": {},
            },
            {
                "type": "PushEvent",
                "created_at": "2026-02-15T14:30:00Z",
                "payload": {
                    "repository_id": 123,
                    "push_id": 456,
                    "ref": "refs/heads/main",
                    "head": commit_sha,
                    "before": "b" * 40,
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))

            with patch("httpx.AsyncClient") as MockClient:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = mock_events

                mock_client_instance = AsyncMock()
                mock_client_instance.get.return_value = mock_resp
                mock_client_instance.__aenter__ = AsyncMock(
                    return_value=mock_client_instance
                )
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_client_instance

                ts = asyncio.get_event_loop().run_until_complete(
                    verifier._get_push_timestamp_events_api("alice", "repo", commit_sha)
                )

            assert ts is not None
            from datetime import datetime, timezone
            expected = datetime(2026, 2, 15, 14, 30, 0, tzinfo=timezone.utc).timestamp()
            assert abs(ts - expected) < 1

    def test_events_api_finds_commit_via_compare(self):
        """Events API finds commit via Compare API when head doesn't match."""
        commit_sha = "a" * 40
        head_sha = "c" * 40
        before_sha = "b" * 40

        mock_events = [
            {
                "type": "PushEvent",
                "created_at": "2026-02-15T14:30:00Z",
                "payload": {
                    "ref": "refs/heads/main",
                    "head": head_sha,
                    "before": before_sha,
                },
            },
        ]

        # Compare API response lists commits in the push range
        mock_compare = {
            "commits": [
                {"sha": commit_sha},
                {"sha": head_sha},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))

            with patch("httpx.AsyncClient") as MockClient:
                # Events API response
                mock_events_resp = MagicMock()
                mock_events_resp.status_code = 200
                mock_events_resp.json.return_value = mock_events

                # Compare API response
                mock_compare_resp = MagicMock()
                mock_compare_resp.status_code = 200
                mock_compare_resp.json.return_value = mock_compare

                mock_client_instance = AsyncMock()
                mock_client_instance.get.side_effect = [
                    mock_events_resp,   # events page 1
                    mock_compare_resp,  # compare API call
                ]
                mock_client_instance.__aenter__ = AsyncMock(
                    return_value=mock_client_instance
                )
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_client_instance

                ts = asyncio.get_event_loop().run_until_complete(
                    verifier._get_push_timestamp_events_api("alice", "repo", commit_sha)
                )

            assert ts is not None
            from datetime import datetime, timezone
            expected = datetime(2026, 2, 15, 14, 30, 0, tzinfo=timezone.utc).timestamp()
            assert abs(ts - expected) < 1

    def test_events_api_commit_not_found(self):
        """Events API has no PushEvent with our commit → returns None."""
        mock_events = [
            {
                "type": "PushEvent",
                "created_at": "2026-02-15T14:30:00Z",
                "payload": {
                    "ref": "refs/heads/main",
                    "head": "d" * 40,
                    "before": "e" * 40,
                },
            },
        ]

        # Compare API says commit not in range
        mock_compare = {"commits": [{"sha": "d" * 40}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))

            with patch("httpx.AsyncClient") as MockClient:
                mock_events_resp = MagicMock()
                mock_events_resp.status_code = 200

                mock_compare_resp = MagicMock()
                mock_compare_resp.status_code = 200
                mock_compare_resp.json.return_value = mock_compare

                # Page 1 returns events, page 2 returns empty
                mock_events_resp.json.side_effect = [mock_events, []]

                mock_client_instance = AsyncMock()
                mock_client_instance.get.side_effect = [
                    mock_events_resp,   # events page 1
                    mock_compare_resp,  # compare for the push event
                    mock_events_resp,   # events page 2 (empty)
                ]
                mock_client_instance.__aenter__ = AsyncMock(
                    return_value=mock_client_instance
                )
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_client_instance

                ts = asyncio.get_event_loop().run_until_complete(
                    verifier._get_push_timestamp_events_api("alice", "repo", "a" * 40)
                )

            assert ts is None

    def test_events_api_rate_limited(self):
        """Events API returns 403 rate limit → returns None gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))

            with patch("httpx.AsyncClient") as MockClient:
                mock_resp = MagicMock()
                mock_resp.status_code = 403

                mock_client_instance = AsyncMock()
                mock_client_instance.get.return_value = mock_resp
                mock_client_instance.__aenter__ = AsyncMock(
                    return_value=mock_client_instance
                )
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_client_instance

                ts = asyncio.get_event_loop().run_until_complete(
                    verifier._get_push_timestamp_events_api("alice", "repo", "a" * 40)
                )

            assert ts is None

    # ---------------------------------------------------------------
    # _get_push_timestamp_graphql (mocked)
    # ---------------------------------------------------------------

    def test_graphql_returns_pushed_date(self):
        """GraphQL API returns valid pushedDate."""
        graphql_response = {
            "data": {
                "repository": {
                    "object": {
                        "pushedDate": "2026-02-15T14:30:00Z",
                        "committedDate": "2026-02-15T14:28:00Z",
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(
                cache_dir=Path(tmpdir), github_token="ghp_test"
            )

            with patch("httpx.AsyncClient") as MockClient:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = graphql_response

                mock_client_instance = AsyncMock()
                mock_client_instance.post.return_value = mock_resp
                mock_client_instance.__aenter__ = AsyncMock(
                    return_value=mock_client_instance
                )
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_client_instance

                ts = asyncio.get_event_loop().run_until_complete(
                    verifier._get_push_timestamp_graphql("alice", "repo", "a" * 40)
                )

            assert ts is not None
            from datetime import datetime, timezone
            expected = datetime(2026, 2, 15, 14, 30, 0, tzinfo=timezone.utc).timestamp()
            assert abs(ts - expected) < 1

    def test_graphql_detects_backdating(self):
        """GraphQL logs warning when pushedDate diverges from committedDate."""
        # Committed date is backdated by 1 day
        graphql_response = {
            "data": {
                "repository": {
                    "object": {
                        "pushedDate": "2026-02-15T14:30:00Z",
                        "committedDate": "2026-02-14T10:00:00Z",
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(
                cache_dir=Path(tmpdir), github_token="ghp_test"
            )

            with patch("httpx.AsyncClient") as MockClient:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = graphql_response

                mock_client_instance = AsyncMock()
                mock_client_instance.post.return_value = mock_resp
                mock_client_instance.__aenter__ = AsyncMock(
                    return_value=mock_client_instance
                )
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_client_instance

                with patch("trajectoryrl.utils.github.logger") as mock_logger:
                    ts = asyncio.get_event_loop().run_until_complete(
                        verifier._get_push_timestamp_graphql("alice", "repo", "a" * 40)
                    )

                    # Should still return the push timestamp
                    assert ts is not None
                    # Should have logged a divergence warning
                    warning_calls = [
                        str(c) for c in mock_logger.warning.call_args_list
                    ]
                    assert any("divergence" in w.lower() for w in warning_calls)

    def test_graphql_commit_not_found(self):
        """GraphQL returns null object → None."""
        graphql_response = {
            "data": {"repository": {"object": None}}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(
                cache_dir=Path(tmpdir), github_token="ghp_test"
            )

            with patch("httpx.AsyncClient") as MockClient:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = graphql_response

                mock_client_instance = AsyncMock()
                mock_client_instance.post.return_value = mock_resp
                mock_client_instance.__aenter__ = AsyncMock(
                    return_value=mock_client_instance
                )
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_client_instance

                ts = asyncio.get_event_loop().run_until_complete(
                    verifier._get_push_timestamp_graphql("alice", "repo", "a" * 40)
                )

            assert ts is None

    # ---------------------------------------------------------------
    # _get_server_push_timestamp (orchestration)
    # ---------------------------------------------------------------

    def test_server_push_timestamp_events_api_succeeds(self):
        """Events API succeeds → returns timestamp without trying GraphQL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(
                cache_dir=Path(tmpdir), github_token="ghp_test"
            )

            with patch.object(
                verifier, "_get_push_timestamp_events_api",
                new_callable=AsyncMock, return_value=1700000000.0,
            ) as mock_events, patch.object(
                verifier, "_get_push_timestamp_graphql",
                new_callable=AsyncMock,
            ) as mock_graphql:
                ts = asyncio.get_event_loop().run_until_complete(
                    verifier._get_server_push_timestamp(
                        "https://github.com/alice/repo", "a" * 40
                    )
                )

            assert ts == 1700000000.0
            mock_events.assert_called_once()
            mock_graphql.assert_not_called()  # Should not fall through

    def test_server_push_timestamp_falls_through_to_graphql(self):
        """Events API returns None → falls through to GraphQL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(
                cache_dir=Path(tmpdir), github_token="ghp_test"
            )

            with patch.object(
                verifier, "_get_push_timestamp_events_api",
                new_callable=AsyncMock, return_value=None,
            ), patch.object(
                verifier, "_get_push_timestamp_graphql",
                new_callable=AsyncMock, return_value=1700000000.0,
            ) as mock_graphql:
                ts = asyncio.get_event_loop().run_until_complete(
                    verifier._get_server_push_timestamp(
                        "https://github.com/alice/repo", "a" * 40
                    )
                )

            assert ts == 1700000000.0
            mock_graphql.assert_called_once()

    def test_server_push_timestamp_both_fail_returns_none(self):
        """Both APIs fail → returns None (fail-safe rejection)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(
                cache_dir=Path(tmpdir), github_token="ghp_test"
            )

            with patch.object(
                verifier, "_get_push_timestamp_events_api",
                new_callable=AsyncMock, return_value=None,
            ), patch.object(
                verifier, "_get_push_timestamp_graphql",
                new_callable=AsyncMock, return_value=None,
            ):
                ts = asyncio.get_event_loop().run_until_complete(
                    verifier._get_server_push_timestamp(
                        "https://github.com/alice/repo", "a" * 40
                    )
                )

            assert ts is None

    def test_server_push_timestamp_no_token_skips_graphql(self):
        """No github_token → skips GraphQL entirely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(
                cache_dir=Path(tmpdir), github_token=None
            )

            with patch.object(
                verifier, "_get_push_timestamp_events_api",
                new_callable=AsyncMock, return_value=None,
            ), patch.object(
                verifier, "_get_push_timestamp_graphql",
                new_callable=AsyncMock,
            ) as mock_graphql:
                ts = asyncio.get_event_loop().run_until_complete(
                    verifier._get_server_push_timestamp(
                        "https://github.com/alice/repo", "a" * 40
                    )
                )

            assert ts is None
            mock_graphql.assert_not_called()

    # ---------------------------------------------------------------
    # cleanup_cache (LRU eviction)
    # ---------------------------------------------------------------

    def test_cleanup_cache_no_eviction_when_under_limit(self):
        """Cache under limit → no repos evicted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))

            # Create a small repo dir (well under 100 MB)
            repo = Path(tmpdir) / "alice__pack"
            repo.mkdir()
            (repo / "data.txt").write_text("x" * 1000)

            verifier.cleanup_cache(max_size_mb=100)
            assert repo.exists()

    def test_cleanup_cache_evicts_oldest_first(self):
        """When over limit, oldest repos (by mtime) are evicted first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))

            # Create two repo dirs with different mtimes
            old_repo = Path(tmpdir) / "old__repo"
            old_repo.mkdir()
            (old_repo / "data.txt").write_bytes(b"x" * 600_000)
            # Backdate the old repo
            os.utime(old_repo, (1000000, 1000000))

            new_repo = Path(tmpdir) / "new__repo"
            new_repo.mkdir()
            (new_repo / "data.txt").write_bytes(b"x" * 600_000)

            # Total ~1.2 MB, limit = 1 MB → should evict old_repo
            verifier.cleanup_cache(max_size_mb=1)
            assert not old_repo.exists(), "Old repo should be evicted"
            assert new_repo.exists(), "New repo should be kept"

    def test_cleanup_cache_evicts_multiple(self):
        """Evicts multiple repos until under limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))

            repos = []
            for i in range(5):
                repo = Path(tmpdir) / f"repo{i}__pack"
                repo.mkdir()
                (repo / "data.txt").write_bytes(b"x" * 300_000)
                os.utime(repo, (1000000 + i, 1000000 + i))
                repos.append(repo)

            # Total ~1.5 MB, limit = 1 MB → should evict at least 2 oldest
            verifier.cleanup_cache(max_size_mb=1)
            remaining = [r for r in repos if r.exists()]
            assert len(remaining) < 5
            # The newest repos should survive
            assert repos[4].exists()

    def test_cleanup_cache_empty_dir(self):
        """Empty cache dir → no error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = GitHubVerifier(cache_dir=Path(tmpdir))
            verifier.cleanup_cache(max_size_mb=100)  # Should not raise


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
        assert defaults["epoch_interval"].default == 86400
        assert defaults["similarity_threshold"].default == 0.80
        assert defaults["inactivity_window"].default == 2
        assert defaults["weight_interval_blocks"].default == 360


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
        raw = "a" * 64 + "|" + "b" * 40 + "|alice/my-pack"
        result = parse_commitment(raw)
        assert result is not None
        pack_hash, git_commit, repo_url = result
        assert pack_hash == "a" * 64
        assert git_commit == "b" * 40
        assert repo_url == "https://github.com/alice/my-pack"

    def test_full_url_commitment(self):
        from trajectoryrl.utils.commitments import parse_commitment
        raw = "a" * 64 + "|" + "b" * 40 + "|https://github.com/alice/my-pack"
        result = parse_commitment(raw)
        assert result is not None
        _, _, repo_url = result
        assert repo_url == "https://github.com/alice/my-pack"

    def test_empty_commitment(self):
        from trajectoryrl.utils.commitments import parse_commitment
        assert parse_commitment("") is None
        assert parse_commitment(None) is None

    def test_too_few_parts(self):
        from trajectoryrl.utils.commitments import parse_commitment
        assert parse_commitment("a" * 64 + "|" + "b" * 40) is None

    def test_invalid_hex_pack_hash(self):
        from trajectoryrl.utils.commitments import parse_commitment
        # Too short
        raw = "a" * 63 + "|" + "b" * 40 + "|alice/my-pack"
        assert parse_commitment(raw) is None
        # Non-hex chars
        raw = "g" * 64 + "|" + "b" * 40 + "|alice/my-pack"
        assert parse_commitment(raw) is None

    def test_invalid_hex_git_commit(self):
        from trajectoryrl.utils.commitments import parse_commitment
        raw = "a" * 64 + "|" + "b" * 39 + "|alice/my-pack"
        assert parse_commitment(raw) is None

    def test_invalid_repo_format(self):
        from trajectoryrl.utils.commitments import parse_commitment
        raw = "a" * 64 + "|" + "b" * 40 + "|not-a-repo"
        assert parse_commitment(raw) is None

    def test_whitespace_handling(self):
        from trajectoryrl.utils.commitments import parse_commitment
        raw = "  " + "a" * 64 + " | " + "b" * 40 + " | alice/my-pack  "
        result = parse_commitment(raw)
        assert result is not None


# ===================================================================
# Score Consensus Tests
# ===================================================================


class TestScoreConsensus:
    """Tests for stake-weighted consensus computation."""

    def test_equal_stakes(self):
        from trajectoryrl.utils.score_publisher import (
            ScorePublisher, ValidatorScoreFile,
        )
        files = [
            ValidatorScoreFile(
                validator_hotkey="hk_1", epoch=1, block_height=100,
                scores={"0": {"final_score": 0.8}},
                signature="sig1",
            ),
            ValidatorScoreFile(
                validator_hotkey="hk_2", epoch=1, block_height=100,
                scores={"0": {"final_score": 0.9}},
                signature="sig2",
            ),
        ]
        metagraph = MagicMock()
        metagraph.hotkeys = ["hk_1", "hk_2"]
        metagraph.S = [100.0, 100.0]  # equal stake

        result = ScorePublisher.compute_consensus(files, metagraph)
        assert result.num_validators == 2
        assert abs(result.consensus_scores[0] - 0.85) < 1e-6  # (0.8+0.9)/2

    def test_unequal_stakes(self):
        from trajectoryrl.utils.score_publisher import (
            ScorePublisher, ValidatorScoreFile,
        )
        files = [
            ValidatorScoreFile(
                validator_hotkey="hk_1", epoch=1, block_height=100,
                scores={"0": {"final_score": 0.8}},
                signature="sig1",
            ),
            ValidatorScoreFile(
                validator_hotkey="hk_2", epoch=1, block_height=100,
                scores={"0": {"final_score": 0.9}},
                signature="sig2",
            ),
        ]
        metagraph = MagicMock()
        metagraph.hotkeys = ["hk_1", "hk_2"]
        metagraph.S = [300.0, 100.0]  # hk_1 has 3x stake

        result = ScorePublisher.compute_consensus(files, metagraph)
        # weighted: (300*0.8 + 100*0.9) / 400 = (240+90)/400 = 0.825
        assert abs(result.consensus_scores[0] - 0.825) < 1e-6

    def test_single_validator(self):
        from trajectoryrl.utils.score_publisher import (
            ScorePublisher, ValidatorScoreFile,
        )
        files = [
            ValidatorScoreFile(
                validator_hotkey="hk_1", epoch=1, block_height=100,
                scores={"0": {"final_score": 0.7}, "1": {"final_score": 0.9}},
                signature="sig1",
            ),
        ]
        metagraph = MagicMock()
        metagraph.hotkeys = ["hk_1"]
        metagraph.S = [100.0]

        result = ScorePublisher.compute_consensus(files, metagraph)
        assert result.num_validators == 1
        assert result.consensus_scores[0] == 0.7
        assert result.consensus_scores[1] == 0.9

    def test_zero_stake_excluded(self):
        from trajectoryrl.utils.score_publisher import (
            ScorePublisher, ValidatorScoreFile,
        )
        files = [
            ValidatorScoreFile(
                validator_hotkey="hk_1", epoch=1, block_height=100,
                scores={"0": {"final_score": 0.5}},
                signature="sig1",
            ),
            ValidatorScoreFile(
                validator_hotkey="hk_2", epoch=1, block_height=100,
                scores={"0": {"final_score": 0.9}},
                signature="sig2",
            ),
        ]
        metagraph = MagicMock()
        metagraph.hotkeys = ["hk_1", "hk_2"]
        metagraph.S = [0.0, 100.0]  # hk_1 has zero stake

        result = ScorePublisher.compute_consensus(files, metagraph)
        assert result.num_validators == 1  # only hk_2 counted
        assert result.consensus_scores[0] == 0.9  # only hk_2's score

    def test_empty_input(self):
        from trajectoryrl.utils.score_publisher import ScorePublisher
        metagraph = MagicMock()
        result = ScorePublisher.compute_consensus([], metagraph)
        assert result.num_validators == 0
        assert result.consensus_scores == {}


# ===================================================================
# Inactivity Window Tests
# ===================================================================


class TestInactivityWindow:
    """Tests for miner inactivity tracking and first-mover protection loss."""

    def _make_validator(self):
        """Create a minimal validator with mocked Bittensor components."""
        with patch("trajectoryrl.base.validator.bt") as mock_bt, \
             patch("trajectoryrl.base.validator.ClawBenchHarness"), \
             patch("trajectoryrl.base.validator.GitHubVerifier"), \
             patch("trajectoryrl.base.validator.yaml") as mock_yaml, \
             patch("trajectoryrl.base.validator.ValidatorConfig") as MockConfig:

            # Mock config
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
            config.github_token = None
            config.validator_scores_fork_url = None
            config.scenarios = ["client_escalation"]
            config.scenarios_path = Path("/tmp/test_scenarios")
            config.inactivity_window = 2
            config.epoch_interval = 86400
            config.similarity_threshold = 0.80
            config.weight_interval_blocks = 360

            # Mock subtensor
            mock_subtensor = MagicMock()
            mock_subtensor.get_current_block.return_value = 100000
            mock_bt.Subtensor.return_value = mock_subtensor

            # Mock metagraph
            mock_metagraph = MagicMock()
            mock_metagraph.hotkeys = ["hk_0", "hk_1", "hk_2"]
            mock_metagraph.validator_permit = [False, False, False]
            mock_metagraph.S = [100.0, 100.0, 100.0]
            mock_subtensor.metagraph.return_value = mock_metagraph

            validator = TrajectoryValidator.__new__(TrajectoryValidator)
            validator.config = config
            validator.metagraph = mock_metagraph
            validator.subtensor = mock_subtensor
            validator.first_mover_data = {}
            validator.last_valid_epoch = {}
            validator.current_epoch = 5
            validator.blocks_per_epoch = 7200

            return validator

    def test_active_miner_tracked(self):
        """Miners with valid commitments are marked active."""
        v = self._make_validator()
        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            0: MinerCommitment(
                uid=0, hotkey="hk_0", pack_hash="a" * 64,
                git_commit_hash="b" * 40,
                repo_url="https://github.com/test/pack",
                block_number=1000, raw="raw",
            ),
        }
        active = v._get_active_miners_from_commitments(commitments)
        assert 0 in active
        assert v.last_valid_epoch[0] == 5

    def test_inactive_miner_loses_first_mover(self):
        """Miners inactive > window epochs lose first-mover protection."""
        v = self._make_validator()
        v.current_epoch = 10
        v.last_valid_epoch = {1: 7}  # last seen epoch 7, current=10, gap=3 > window=2
        v.first_mover_data = {"hk_1": (0.85, 5000.0)}

        # No commitments this epoch
        active = v._get_active_miners_from_commitments({})
        assert len(active) == 0
        assert "hk_1" not in v.first_mover_data  # Protection lost

    def test_reactivation_preserves_tracking(self):
        """Miner returning after inactivity gets fresh tracking."""
        v = self._make_validator()
        v.current_epoch = 10
        v.last_valid_epoch = {0: 7}  # Was inactive
        v.first_mover_data = {}  # Already lost protection

        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            0: MinerCommitment(
                uid=0, hotkey="hk_0", pack_hash="a" * 64,
                git_commit_hash="b" * 40,
                repo_url="https://github.com/test/pack",
                block_number=9000, raw="raw",
            ),
        }
        active = v._get_active_miners_from_commitments(commitments)
        assert 0 in active
        assert v.last_valid_epoch[0] == 10  # Updated to current epoch

    def test_within_window_keeps_protection(self):
        """Miner within inactivity window keeps first-mover protection."""
        v = self._make_validator()
        v.current_epoch = 10
        v.last_valid_epoch = {1: 9}  # gap=1, within window=2
        v.first_mover_data = {"hk_1": (0.85, 5000.0)}

        # No commitments this epoch for miner 1
        active = v._get_active_miners_from_commitments({})
        assert "hk_1" in v.first_mover_data  # Protection kept


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
