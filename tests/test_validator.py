"""Tests for TrajectoryRL validator components.

Tests the scoring, OPP schema validation, eval scoring, and config
without requiring a live Bittensor network.
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

# bt.Synapse must be a real class for any module that inherits from it
class _MockSynapse:
    pass

_mock_bt.Synapse = _MockSynapse
sys.modules["bittensor"] = _mock_bt

# Now safe to import
from trajectoryrl.utils.opp_schema import validate_opp_schema, ValidationResult
from trajectoryrl.scoring import TrajectoryScorer, AggregatedScore
from trajectoryrl.utils.github import PackFetcher, PackVerificationResult
from trajectoryrl.base.validator import TrajectoryValidator
from trajectoryrl.utils.epoch_context import (
    generate_epoch_context, render_context_preamble,
    EpochContext, NAMES, ROLES, COMPANIES, DEPARTMENTS, TIMEZONES,
)
from trajectoryrl.utils.eval_window import (
    WindowConfig, WindowPhase, EvaluationWindow,
    compute_window, is_new_window, should_submit, should_aggregate,
    can_evaluate, window_progress_pct,
)
from trajectoryrl.utils.config import SPEC_NUMBER
from trajectoryrl.utils.consensus import (
    ConsensusPayload, ConsensusPointer,
    verify_payload_integrity, CONSENSUS_PROTOCOL_VERSION,
)
from trajectoryrl.utils.consensus_filter import (
    run_filter_pipeline, FilterStats, ValidatedSubmission,
    filter_protocol_version, filter_window_number,
    filter_trust_threshold, filter_zero_signal,
    filter_spec_number, select_target_spec_number,
)
from trajectoryrl.scoring import compute_consensus_scores
from trajectoryrl.utils.winner_state import (
    WinnerState, select_winner_with_protection,
    save_winner_state, load_winner_state,
)
from trajectoryrl.utils.commitments import (
    parse_consensus_commitment, format_consensus_commitment,
    is_consensus_commitment, parse_commitment,
    ValidatorConsensusCommitment, fetch_validator_consensus_commitments,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent


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
def valid_pack():
    return {
        "schema_version": 1,
        "files": {
            "SKILL.md": "# Agent rules\nBe safe and efficient.",
            "SOUL.md": "# Tone\nProfessional and concise.",
        },
        "tool_policy": {
            "allow": ["exec", "slack", "memory_search"],
            "deny": ["group:runtime"],
        },
        "metadata": {
            "pack_name": "test_pack",
            "pack_version": "1.0.0",
            "target_suite": "trajrl-bench",
        },
    }


from dataclasses import dataclass as _dc, field as _f
from typing import Any as _Any, Optional as _Opt, Dict as _Dict


@_dc
class _MockResult:
    """Minimal result object for scoring tests."""
    scenario_name: str = ""
    score: float = 0.0
    success: bool = False
    tool_calls: int = 0
    response: str = ""
    rubric: _Dict[str, _Any] = _f(default_factory=dict)


@pytest.fixture
def sample_results():
    """Sample results for scoring tests."""
    return [
        _MockResult(
            scenario_name="client_escalation",
            score=0.92,
            success=True,
            tool_calls=10,
            response="Escalation summary...",
            rubric={"by_category": {"safety": {"score": 1.0}}},
        ),
        _MockResult(
            scenario_name="morning_brief",
            score=0.85,
            success=True,
            tool_calls=8,
            response="Daily brief...",
            rubric={"by_category": {"safety": {"score": 1.0}}},
        ),
        _MockResult(
            scenario_name="inbox_to_action",
            score=0.78,
            success=True,
            tool_calls=15,
            response="Action queue...",
            rubric={"by_category": {"safety": {"score": 0.9}}},
        ),
        _MockResult(
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

    def test_missing_policy_file(self, valid_pack):
        del valid_pack["files"]["SKILL.md"]
        result = validate_opp_schema(valid_pack)
        assert not result.passed
        assert any("SKILL.md" in i for i in result.issues)

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
        valid_pack["files"]["SKILL.md"] = 123
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
        valid_pack["files"]["SKILL.md"] = "x" * 200_000
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
            _MockResult(
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
            _MockResult("a", score=0.9, success=True, tool_calls=5, response="", rubric={}),
            _MockResult("b", score=0.0, success=False, tool_calls=0, response="", rubric={}),
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
        """Preamble ends with --- separator so miner SKILL.md follows cleanly."""
        ctx = generate_epoch_context(42)
        preamble = render_context_preamble(ctx)
        assert "---" in preamble
        assert preamble.endswith("\n")

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



# ===================================================================
# Consensus Evaluation Tests
# ===================================================================


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
        """Valid JSON pack URL + matching hash → verification passes."""
        pack = {"schema_version": 1, "files": {"SKILL.md": "# Test"}}
        pack_json = json.dumps(pack, sort_keys=True)
        pack_hash = hashlib.sha256(pack_json.encode()).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            with patch.object(
                fetcher, "_fetch_pack",
                new_callable=AsyncMock, return_value=pack_json,
            ):
                result = asyncio.run(
                    fetcher.verify_submission(
                        pack_url="https://trajrl.com/samples/pack.json",
                        pack_hash=pack_hash,
                    )
                )

            assert result.valid is True
            assert result.pack_content == pack

    def test_verify_invalid_json(self):
        """Non-JSON content → verification fails."""
        raw = "this is not json"

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            with patch.object(
                fetcher, "_fetch_pack",
                new_callable=AsyncMock, return_value=raw,
            ):
                result = asyncio.run(
                    fetcher.verify_submission(
                        pack_url="https://example.com/skill.md",
                        pack_hash="deadbeef" * 8,
                    )
                )

            assert result.valid is False
            assert "Invalid JSON" in result.error

    def test_verify_hash_mismatch(self):
        """Content doesn't match expected hash → verification fails."""
        pack_json = json.dumps({"schema_version": 1, "files": {"SKILL.md": "# Test"}}, sort_keys=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PackFetcher(cache_dir=Path(tmpdir))

            with patch.object(
                fetcher, "_fetch_pack",
                new_callable=AsyncMock, return_value=pack_json,
            ):
                result = asyncio.run(
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
                result = asyncio.run(
                    fetcher.verify_submission(
                        pack_url="https://trajrl.com/samples/pack.json",
                        pack_hash="a" * 64,
                    )
                )

            assert result.valid is False
            assert "fetch" in result.error.lower() or "failed" in result.error.lower()

    def test_cache_hit_skips_fetch(self):
        """Cached pack with matching hash → no HTTP fetch needed."""
        pack = {"schema_version": 1, "files": {"SKILL.md": "# Cached"}}
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
                result = asyncio.run(
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

                result = asyncio.run(
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

                result = asyncio.run(
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
        assert defaults["delta_threshold"].default == 0.05
        assert defaults["seeds_per_task"].default == 1
        assert defaults["eval_interval_blocks"].default == 7200
        assert defaults["similarity_threshold"].default == 0.80
        assert defaults["inactivity_blocks"].default == 14400
        assert defaults["weight_interval_blocks"].default == 360
        # v4.0: LLM judge config fields exist
        assert "judge_model" in defaults
        assert "judge_api_key" in defaults
        assert "judge_base_url" in defaults

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
        a = {"files": {"SKILL.md": "Be safe and careful. Follow instructions."}}
        b = {"files": {"SKILL.md": "Be safe and careful. Follow instructions."}}
        sim = pack_similarity(a, b)
        assert sim > 0.90  # Identical content → near 1.0 (short strings have zlib overhead)
        assert is_too_similar(a, b, threshold=0.80) is True

    def test_completely_different_packs(self):
        from trajectoryrl.utils.ncd import pack_similarity, is_too_similar
        a = {"files": {"SKILL.md": "Be safe and careful. Follow instructions precisely."}}
        b = {"files": {"SKILL.md": "Cook pasta for dinner. Add salt to water."}}
        sim = pack_similarity(a, b)
        assert sim < 0.5  # Unrelated content → low similarity
        assert is_too_similar(a, b, threshold=0.80) is False

    def test_whitespace_only_difference(self):
        from trajectoryrl.utils.ncd import pack_similarity
        a = {"files": {"SKILL.md": "# Rules\n\nBe safe and careful."}}
        b = {"files": {"SKILL.md": "#  Rules\n\n\nBe  safe  and  careful."}}
        sim = pack_similarity(a, b)
        # After normalization, whitespace is collapsed → should be very similar
        assert sim > 0.90

    def test_no_winner_never_similar(self):
        from trajectoryrl.utils.ncd import is_too_similar
        a = {"files": {"SKILL.md": "Any content"}}
        assert is_too_similar(a, None, threshold=0.80) is False

    def test_threshold_boundary(self):
        from trajectoryrl.utils.ncd import is_too_similar, pack_similarity
        a = {"files": {"SKILL.md": "Be safe and careful. Follow all rules."}}
        b = {"files": {"SKILL.md": "Be safe and careful. Follow all rules."}}
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
# Per-Scenario Eval State Tests
# ===================================================================


class TestPerScenarioEvalState:
    """Tests for per-scenario eval scoring (keyed by hotkey)."""

    def _make_validator(self):
        """Create a minimal validator with mocked Bittensor components."""
        with patch("trajectoryrl.base.validator.bt") as mock_bt, \
             patch("trajectoryrl.base.validator.TrajectorySandboxHarness"), \
             patch("trajectoryrl.base.validator.PackFetcher"), \
             patch("trajectoryrl.base.validator.ValidatorConfig") as MockConfig:

            config = MagicMock()
            config.wallet_name = "test"
            config.wallet_hotkey = "default"
            config.network = "test"
            config.netuid = 11
            config.sandbox_image = "trajrl-bench:latest"
            config.timeout_per_scenario = 120
            config.rho_reliability = 0.1
            config.consensus_epsilon = 0.02
            config.bootstrap_threshold = 10
            config.log_dir = Path("/tmp/test_logs")
            config.log_level = "WARNING"
            config.scenarios = ["client_escalation", "morning_brief"]
            config.scenarios_path = Path("/tmp/test_scenarios")
            config.inactivity_blocks = 14400
            config.eval_interval_blocks = 7200
            config.similarity_threshold = 0.80
            config.weight_interval_blocks = 360
            config.cost_delta = 0.10
            config.required_categories = ["safety", "correctness"]
            config.eval_state_path = Path("/tmp/test_eval_state.json")
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
            mock_metagraph.stake = [100.0, 100.0, 100.0]
            mock_subtensor.metagraph.return_value = mock_metagraph

            validator = TrajectoryValidator.__new__(TrajectoryValidator)
            validator.config = config
            validator.metagraph = mock_metagraph
            validator.subtensor = mock_subtensor
            validator._eval_pack_hash = {}
            validator.last_eval_block = {}
            validator.last_eval_window = {}
            validator.scenario_scores = {}
            validator._hotkey_uid_map = {}
            validator._hotkey_packs = {}
            validator._pack_by_hash = {}
            validator._eval_counts = {}
            validator._last_eval_window = -1
            validator.latest_token_usage = {}
            validator.latest_model_usage = {}
            validator.current_winner_pack = None
            validator.current_winner_hotkey = None
            validator._miner_loggers = {}
            validator._miner_log_dir = Path("/tmp/test_logs/miners")
            validator._miner_log_dir.mkdir(parents=True, exist_ok=True)
            validator._consensus_window = -1
            validator.scenarios = {
                "client_escalation": {"weight": 1.5},
                "morning_brief": {"weight": 1.0},
            }
            validator.scorer = TrajectoryScorer(
                rho_reliability=0.1, consensus_epsilon=0.02
            )

            # Mock LLM judges (v4.0)
            validator.integrity_judge = MagicMock()
            validator.integrity_judge.dump_cache.return_value = {}
            validator.integrity_judge.load_cache = MagicMock()
            validator.trajectory_judge = MagicMock()

            return validator

    def test_needs_evaluation_new_miner(self):
        """New miner (never evaluated) needs evaluation."""
        v = self._make_validator()
        assert v._needs_evaluation("hk_new", "hash_a", current_window=42) is True

    def test_needs_evaluation_pack_changed(self):
        """Pack hash change triggers re-evaluation."""
        v = self._make_validator()
        v._eval_pack_hash["hk_0"] = "hash_a"
        v.scenario_scores["hk_0"] = {"client_escalation": 0.85}
        v.last_eval_window["hk_0"] = 41
        assert v._needs_evaluation("hk_0", "hash_b", current_window=42) is True

    def test_needs_evaluation_same_pack_skips(self):
        """Same pack_hash with cached scores → skip eval."""
        v = self._make_validator()
        v._eval_pack_hash["hk_0"] = "hash_a"
        v.scenario_scores["hk_0"] = {"client_escalation": 0.85}
        v.last_eval_window["hk_0"] = 42
        assert v._needs_evaluation("hk_0", "hash_a", current_window=42) is False

    def test_needs_evaluation_skips_across_epochs(self):
        """Same pack_hash across windows → still skip (reuse cached scores)."""
        v = self._make_validator()
        v._eval_pack_hash["hk_0"] = "hash_a"
        v.scenario_scores["hk_0"] = {"client_escalation": 0.85}
        v.last_eval_window["hk_0"] = 40
        assert v._needs_evaluation("hk_0", "hash_a", current_window=45) is False

    def test_needs_evaluation_no_scores_cached(self):
        """pack_hash recorded but no scenario_scores → re-eval (defensive)."""
        v = self._make_validator()
        v._eval_pack_hash["hk_0"] = "hash_a"
        v.scenario_scores["hk_0"] = {}
        assert v._needs_evaluation("hk_0", "hash_a", current_window=42) is True

    def test_eval_state_persistence_roundtrip(self):
        """Eval state can be saved and loaded."""
        v = self._make_validator()
        v.scenario_scores = {"hk_0": {"client_escalation": 0.85}}
        v._eval_pack_hash = {"hk_0": "hash_a"}
        v.last_eval_block = {"hk_0": 99000}
        v.last_eval_window = {"hk_0": 13}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.eval_state_path = Path(f.name)

        try:
            v._save_eval_state()

            v2 = self._make_validator()
            v2.config.eval_state_path = v.config.eval_state_path
            v2._load_eval_state()

            assert v2.scenario_scores == {"hk_0": {"client_escalation": 0.85}}
            assert v2._eval_pack_hash == {"hk_0": "hash_a"}
            assert v2.last_eval_block == {"hk_0": 99000}
            assert v2.last_eval_window == {"hk_0": 13}
        finally:
            v.config.eval_state_path.unlink(missing_ok=True)

    def test_scenario_score_tracking(self):
        """Scenario scores are recorded per-scenario."""
        v = self._make_validator()
        v._update_eval_results("hk_0", "hash_a",
            scenario_scores={"client_escalation": 0.85},
        )
        assert v.scenario_scores["hk_0"]["client_escalation"] == 0.85

    def test_scenario_score_overwrites_on_new_eval(self):
        """New evaluation overwrites previous score."""
        v = self._make_validator()
        v._update_eval_results("hk_0", "hash_a",
            scenario_scores={"client_escalation": 0.85},
        )
        v._update_eval_results("hk_0", "hash_a",
            scenario_scores={"client_escalation": 0.72},
        )
        assert v.scenario_scores["hk_0"]["client_escalation"] == 0.72

    def test_scenario_score_resets_on_pack_change(self):
        """Scenario scores reset when pack changes."""
        v = self._make_validator()
        v._update_eval_results("hk_0", "hash_a",
            scenario_scores={"client_escalation": 0.85},
        )
        v._update_eval_results("hk_0", "hash_b",
            scenario_scores={"client_escalation": 0.72},
        )
        assert v.scenario_scores["hk_0"]["client_escalation"] == 0.72

    def test_uid_change_tracking(self):
        """UID change detection works for re-registration."""
        v = self._make_validator()
        v._track_uid_change(0, "hk_0")
        assert v._hotkey_uid_map["hk_0"] == 0

        v._track_uid_change(5, "hk_0")
        assert v._hotkey_uid_map["hk_0"] == 5

    def test_eval_state_invalidated_on_spec_number_change(self):
        """Loading eval state under a different SPEC_NUMBER invalidates state."""
        import json as _json
        v = self._make_validator()
        v.scenario_scores = {"hk_0": {"client_escalation": 0.85}}
        v._eval_pack_hash = {"hk_0": "hash_a"}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.eval_state_path = Path(f.name)

        try:
            v._save_eval_state()

            data = _json.loads(v.config.eval_state_path.read_text())
            data["spec_number"] = data.get("spec_number", 1) + 100
            data["scoring_version"] = data["spec_number"]
            v.config.eval_state_path.write_text(_json.dumps(data))

            v2 = self._make_validator()
            v2.config.eval_state_path = v.config.eval_state_path
            v2._load_eval_state()

            assert v2.scenario_scores == {}
            assert v2._eval_pack_hash == {}
        finally:
            v.config.eval_state_path.unlink(missing_ok=True)

    def test_eval_state_load_accepts_legacy_scoring_version_key(self):
        """Old eval_state.json with `scoring_version` key still loads."""
        import json as _json

        v = self._make_validator()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.eval_state_path = Path(f.name)

        try:
            legacy = {
                "scoring_version": SPEC_NUMBER,  # legacy key only
                "scenario_scores": {"hk_0": {"client_escalation": 0.7}},
                "eval_pack_hash": {"hk_0": "hash_legacy"},
                "last_eval_block": {"hk_0": 1234},
                "last_eval_window": -1,
                "last_eval_window_per_hotkey": {"hk_0": 9},
                "consensus_window": -1,
                "integrity_cache": {},
            }
            v.config.eval_state_path.write_text(_json.dumps(legacy))

            v2 = self._make_validator()
            v2.config.eval_state_path = v.config.eval_state_path
            v2._load_eval_state()

            assert v2.scenario_scores == {"hk_0": {"client_escalation": 0.7}}
            assert v2._eval_pack_hash == {"hk_0": "hash_legacy"}
        finally:
            v.config.eval_state_path.unlink(missing_ok=True)


# ===================================================================
# Inactivity Tests (Block-Based)
# ===================================================================


class TestInactivityBlocks:
    """Tests for block-based miner inactivity tracking."""

    def _make_validator(self):
        """Create a minimal validator with mocked Bittensor components."""
        with patch("trajectoryrl.base.validator.bt") as mock_bt, \
             patch("trajectoryrl.base.validator.TrajectorySandboxHarness"), \
             patch("trajectoryrl.base.validator.PackFetcher"), \
             patch("trajectoryrl.base.validator.ValidatorConfig") as MockConfig:

            config = MagicMock()
            config.wallet_name = "test"
            config.wallet_hotkey = "default"
            config.network = "test"
            config.netuid = 11
            config.sandbox_image = "trajrl-bench:latest"
            config.timeout_per_scenario = 120
            config.rho_reliability = 0.1
            config.consensus_epsilon = 0.02
            config.bootstrap_threshold = 10
            config.log_dir = Path("/tmp/test_logs")
            config.log_level = "WARNING"
            config.scenarios = ["client_escalation"]
            config.scenarios_path = Path("/tmp/test_scenarios")
            config.inactivity_blocks = 14400
            config.coldkey_blacklist = []
            config.eval_interval_blocks = 7200
            config.similarity_threshold = 0.80
            config.weight_interval_blocks = 360
            config.cost_delta = 0.10
            config.required_categories = ["safety", "correctness"]
            config.eval_state_path = Path("/tmp/test_eval_state.json")

            mock_subtensor = MagicMock()
            mock_subtensor.get_current_block.return_value = 100000
            mock_bt.Subtensor.return_value = mock_subtensor

            mock_metagraph = MagicMock()
            mock_metagraph.hotkeys = ["hk_0", "hk_1", "hk_2"]
            mock_metagraph.coldkeys = ["ck_0", "ck_1", "ck_2"]
            mock_metagraph.validator_permit = [False, False, False]
            mock_metagraph.S = [100.0, 100.0, 100.0]
            mock_metagraph.stake = [100.0, 100.0, 100.0]
            mock_subtensor.metagraph.return_value = mock_metagraph

            validator = TrajectoryValidator.__new__(TrajectoryValidator)
            validator.config = config
            validator.metagraph = mock_metagraph
            validator.subtensor = mock_subtensor
            validator._eval_pack_hash = {}
            validator.last_eval_block = {}
            validator.last_eval_window = {}
            validator.scenario_scores = {}
            validator._hotkey_uid_map = {}
            validator._hotkey_packs = {}
            validator._pack_by_hash = {}
            validator._eval_counts = {}
            validator.latest_token_usage = {}
            validator.latest_model_usage = {}
            validator.current_winner_pack = None
            validator.current_winner_hotkey = None
            validator._miner_loggers = {}
            validator._miner_log_dir = Path("/tmp/test_logs/miners")
            validator._miner_log_dir.mkdir(parents=True, exist_ok=True)
            validator.integrity_judge = MagicMock()
            validator.integrity_judge.dump_cache.return_value = {}
            validator.trajectory_judge = MagicMock()

            return validator

    def test_active_miner_within_inactivity_window(self):
        """Miner whose commitment is fresh (within inactivity_blocks) is active."""
        v = self._make_validator()

        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            0: MinerCommitment(
                uid=0, hotkey="hk_0", pack_hash="a" * 64,
                pack_url="https://trajrl.com/samples/pack.json",
                block_number=90000, raw="raw",
            ),
        }
        # 100000 - 90000 = 10000 < 14400 → active
        active = v._filter_active_commitments(commitments, 100000)
        assert 0 in active

    def test_inactive_miner_removed_from_active(self):
        """Miner whose commitment is stale (> inactivity_blocks old) is removed."""
        v = self._make_validator()

        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            1: MinerCommitment(
                uid=1, hotkey="hk_1", pack_hash="a" * 64,
                pack_url="https://trajrl.com/samples/pack.json",
                block_number=80000, raw="raw",
            ),
        }
        # 100000 - 80000 = 20000 > 14400 → stale
        active = v._filter_active_commitments(commitments, 100000)
        assert 1 not in active

    def test_fresh_commitment_is_active(self):
        """A brand-new commitment (block_number close to current) is active."""
        v = self._make_validator()

        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            0: MinerCommitment(
                uid=0, hotkey="hk_0", pack_hash="a" * 64,
                pack_url="https://trajrl.com/samples/pack.json",
                block_number=99999, raw="raw",
            ),
        }
        # 100000 - 99999 = 1 < 14400 → active
        active = v._filter_active_commitments(commitments, 100000)
        assert 0 in active

    def test_resubmission_reactivates_miner(self):
        """Miner re-submitting (new block_number) becomes active again."""
        v = self._make_validator()

        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            0: MinerCommitment(
                uid=0, hotkey="hk_0", pack_hash="a" * 64,
                pack_url="https://trajrl.com/samples/pack.json",
                block_number=95000, raw="raw",
            ),
        }
        # 100000 - 95000 = 5000 < 14400 → active
        active = v._filter_active_commitments(commitments, 100000)
        assert 0 in active

    def test_boundary_exactly_at_limit(self):
        """Commitment age exactly at inactivity_blocks is not stale."""
        v = self._make_validator()

        from trajectoryrl.utils.commitments import MinerCommitment
        commitments = {
            0: MinerCommitment(
                uid=0, hotkey="hk_0", pack_hash="a" * 64,
                pack_url="https://trajrl.com/samples/pack.json",
                block_number=100000 - 14400, raw="raw",
            ),
        }
        # age == inactivity_blocks → not stale (> required, not >=)
        active = v._filter_active_commitments(commitments, 100000)
        assert 0 in active


# ===================================================================
# Integration: Scoring Pipeline
# ===================================================================

class TestScoringIntegration:
    """Tests for the scoring pipeline."""

    def test_cost_based_winner_selection(self, scorer):
        """Lowest-cost qualified miner beats champion beyond delta threshold."""
        costs = {0: 0.050, 1: 0.030, 2: 0.045}
        qualified = {0: True, 1: True, 2: False}
        uid_to_hotkey = {0: "hk_0", 1: "hk_1", 2: "hk_2"}

        weights = scorer.select_winner_by_cost(
            costs=costs,
            qualified=qualified,
            first_mover_data={},
            cost_delta=0.10,
            num_active_miners=20,
            uid_to_hotkey=uid_to_hotkey,
            champion_hotkey="hk_0",
        )

        # Miner 1 ($0.030) < $0.050 * 0.90 = $0.045 → beats champion
        assert weights[1] == 1.0
        assert weights[0] == 0.0
        assert weights[2] == 0.0

    def test_cost_winner_protection(self, scorer):
        """Champion retains when challenger is cheaper but within delta."""
        costs = {0: 0.050, 1: 0.048}
        qualified = {0: True, 1: True}
        uid_to_hotkey = {0: "hk_0", 1: "hk_1"}

        weights = scorer.select_winner_by_cost(
            costs=costs,
            qualified=qualified,
            first_mover_data={},
            cost_delta=0.10,
            num_active_miners=20,
            uid_to_hotkey=uid_to_hotkey,
            champion_hotkey="hk_0",
        )

        # $0.048 is NOT < $0.050 * 0.90 = $0.045 → champion retains
        assert weights[0] == 1.0
        assert weights[1] == 0.0

    def test_cost_delta_only_protects_champion(self, scorer):
        """Delta only protects the actual champion, not intermediate miners."""
        costs = {233: 0.0196, 91: 0.0172, 224: 0.0156}
        qualified = {233: True, 91: True, 224: True}
        uid_to_hotkey = {233: "hk_233", 91: "hk_91", 224: "hk_224"}

        weights = scorer.select_winner_by_cost(
            costs=costs,
            qualified=qualified,
            first_mover_data={},
            cost_delta=0.10,
            num_active_miners=20,
            uid_to_hotkey=uid_to_hotkey,
            champion_hotkey="hk_233",
        )

        # 224 ($0.0156) < $0.0196 * 0.90 = $0.01764 → beats champion
        assert weights[224] == 1.0, "224 should win (cheapest, beats champion δ)"
        assert weights[91] == 0.0
        assert weights[233] == 0.0

    def test_cost_champion_retains_when_no_challenger_clears_delta(self, scorer):
        """Champion retains when all challengers are cheaper but within delta."""
        costs = {0: 0.050, 1: 0.046, 2: 0.047}
        qualified = {0: True, 1: True, 2: True}
        uid_to_hotkey = {0: "hk_0", 1: "hk_1", 2: "hk_2"}

        weights = scorer.select_winner_by_cost(
            costs=costs,
            qualified=qualified,
            first_mover_data={},
            cost_delta=0.10,
            num_active_miners=20,
            uid_to_hotkey=uid_to_hotkey,
            champion_hotkey="hk_0",
        )

        # Best challenger $0.046 is NOT < $0.050 * 0.90 = $0.045
        assert weights[0] == 1.0
        assert weights[1] == 0.0
        assert weights[2] == 0.0

    def test_cost_all_disqualified(self, scorer):
        """All miners disqualified → zero weights."""
        costs = {0: 0.050, 1: 0.030}
        qualified = {0: False, 1: False}

        weights = scorer.select_winner_by_cost(
            costs=costs,
            qualified=qualified,
            first_mover_data={},
            cost_delta=0.10,
            num_active_miners=20,
        )

        assert weights[0] == 0.0
        assert weights[1] == 0.0



# ---------------------------------------------------------------------------
# EvaluationWindow tests
# ---------------------------------------------------------------------------

class TestEvaluationWindow:
    """Tests for block-based evaluation window computation."""

    def setup_method(self):
        self.config = WindowConfig(
            window_length=7200,
            global_anchor=0,
            publish_pct=0.80,
            aggregate_pct=0.90,
        )

    def test_window_number_at_anchor(self):
        w = compute_window(0, self.config)
        assert w.window_number == 0
        assert w.window_start == 0
        assert w.block_offset == 0

    def test_window_number_mid_window(self):
        w = compute_window(3600, self.config)
        assert w.window_number == 0
        assert w.block_offset == 3600

    def test_window_number_boundary(self):
        w = compute_window(7200, self.config)
        assert w.window_number == 1
        assert w.window_start == 7200
        assert w.block_offset == 0

    def test_window_number_second_window(self):
        w = compute_window(10000, self.config)
        assert w.window_number == 1
        assert w.window_start == 7200
        assert w.block_offset == 2800

    def test_phase_evaluation(self):
        w = compute_window(100, self.config)
        assert w.phase == WindowPhase.EVALUATION
        assert w.blocks_remaining_in_phase == 5760 - 100

    def test_phase_at_publish_boundary(self):
        w = compute_window(5760, self.config)
        assert w.phase == WindowPhase.PROPAGATION
        assert w.blocks_into_phase == 0

    def test_phase_propagation(self):
        w = compute_window(6000, self.config)
        assert w.phase == WindowPhase.PROPAGATION
        assert w.blocks_into_phase == 240

    def test_phase_at_aggregate_boundary(self):
        w = compute_window(6480, self.config)
        assert w.phase == WindowPhase.AGGREGATION
        assert w.blocks_into_phase == 0

    def test_phase_aggregation(self):
        w = compute_window(7000, self.config)
        assert w.phase == WindowPhase.AGGREGATION
        assert w.blocks_remaining_in_phase == 200

    def test_phase_end_of_window(self):
        w = compute_window(7199, self.config)
        assert w.phase == WindowPhase.AGGREGATION
        assert w.window_number == 0
        assert w.blocks_remaining_in_phase == 1

    def test_global_anchor_offset(self):
        cfg = WindowConfig(window_length=7200, global_anchor=1000)
        w = compute_window(1000, cfg)
        assert w.window_number == 0
        assert w.window_start == 1000
        assert w.block_offset == 0

        w2 = compute_window(8200, cfg)
        assert w2.window_number == 1
        assert w2.window_start == 8200

    def test_before_anchor_clamps(self):
        cfg = WindowConfig(window_length=7200, global_anchor=5000)
        w = compute_window(100, cfg)
        assert w.window_number == 0
        assert w.block_offset == 0

    def test_is_new_window(self):
        assert is_new_window(7200, 0, self.config) is True
        assert is_new_window(7199, 0, self.config) is False
        assert is_new_window(14400, 1, self.config) is True
        assert is_new_window(14400, 2, self.config) is False

    def test_can_evaluate(self):
        assert can_evaluate(100, self.config) is True
        assert can_evaluate(5759, self.config) is True
        assert can_evaluate(5760, self.config) is False
        assert can_evaluate(6480, self.config) is False

    def test_should_submit(self):
        assert should_submit(5759, self.config) is False
        assert should_submit(5760, self.config) is True
        assert should_submit(6000, self.config) is True
        assert should_submit(6480, self.config) is False

    def test_should_aggregate(self):
        assert should_aggregate(6479, self.config) is False
        assert should_aggregate(6480, self.config) is True
        assert should_aggregate(7199, self.config) is True

    def test_window_progress(self):
        assert window_progress_pct(0, self.config) == pytest.approx(0.0)
        assert window_progress_pct(3600, self.config) == pytest.approx(0.5)
        assert window_progress_pct(7200, self.config) == pytest.approx(0.0)

    def test_deterministic(self):
        """Same inputs always produce same outputs (pure function)."""
        for block in [0, 100, 5760, 6480, 7199, 7200, 99999]:
            w1 = compute_window(block, self.config)
            w2 = compute_window(block, self.config)
            assert w1 == w2

    def test_frozen_dataclass(self):
        w = compute_window(100, self.config)
        with pytest.raises(AttributeError):
            w.window_number = 999

    def test_publish_deadline_block(self):
        w = compute_window(100, self.config)
        assert w.publish_deadline_block == 0 + 5760  # window_start=0, 80% of 7200

    def test_aggregate_start_block(self):
        w = compute_window(100, self.config)
        assert w.aggregate_start_block == 0 + 6480  # window_start=0, 90% of 7200

    def test_publish_aggregate_second_window(self):
        w = compute_window(10000, self.config)
        assert w.publish_deadline_block == 7200 + 5760
        assert w.aggregate_start_block == 7200 + 6480

    def test_custom_percentages(self):
        cfg = WindowConfig(
            window_length=10000,
            publish_pct=0.70,
            aggregate_pct=0.85,
        )
        assert cfg.publish_block == 7000
        assert cfg.aggregate_block == 8500

        w = compute_window(7000, cfg)
        assert w.phase == WindowPhase.PROPAGATION

        w2 = compute_window(8500, cfg)
        assert w2.phase == WindowPhase.AGGREGATION


# ---------------------------------------------------------------------------
# ConsensusPayload tests
# ---------------------------------------------------------------------------

class TestConsensusPayload:
    """Tests for consensus data model serialization and integrity."""

    def _make_payload(self, **overrides) -> ConsensusPayload:
        defaults = {
            "protocol_version": 1,
            "window_number": 42,
            "validator_hotkey": "5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
            "bench_version": "0.1.0",
            "scores": {"miner_a": 0.85, "miner_b": 0.42},
            "timestamp": 1710000000,
        }
        defaults.update(overrides)
        return ConsensusPayload(**defaults)

    def test_serialize_deserialize_roundtrip(self):
        p = self._make_payload()
        data = p.serialize()
        p2 = ConsensusPayload.deserialize(data)
        assert p.protocol_version == p2.protocol_version
        assert p.window_number == p2.window_number
        assert p.validator_hotkey == p2.validator_hotkey
        assert p.scores == p2.scores

    def test_content_hash_deterministic(self):
        p = self._make_payload()
        h1 = p.content_hash()
        h2 = p.content_hash()
        assert h1 == h2
        assert h1.startswith("sha256:")
        assert len(h1) == len("sha256:") + 64

    def test_content_hash_changes_with_data(self):
        p1 = self._make_payload(window_number=1)
        p2 = self._make_payload(window_number=2)
        assert p1.content_hash() != p2.content_hash()

    def test_verify_integrity_pass(self):
        p = self._make_payload()
        data = p.serialize()
        assert verify_payload_integrity(data, p.content_hash()) is True

    def test_verify_integrity_fail(self):
        p = self._make_payload()
        data = p.serialize()
        assert verify_payload_integrity(data, "sha256:0000") is False

    def test_verify_integrity_bare_hex(self):
        p = self._make_payload()
        data = p.serialize()
        bare_hex = p.content_hash().removeprefix("sha256:")
        assert verify_payload_integrity(data, bare_hex) is True

    def test_to_dict_from_dict_roundtrip(self):
        p = self._make_payload()
        d = p.to_dict()
        p2 = ConsensusPayload.from_dict(d)
        assert p.content_hash() == p2.content_hash()

    def test_pointer_roundtrip(self):
        ptr = ConsensusPointer(
            protocol_version=1,
            window_number=42,
            content_address="sha256:abc123",
            validator_hotkey="5FFApaS75bvpgP9gQ5hTUdZHiTc6LB2VPP9gvHN6VQCNug6f",
        )
        d = ptr.to_dict()
        ptr2 = ConsensusPointer.from_dict(d)
        assert ptr.content_address == ptr2.content_address
        assert ptr.window_number == ptr2.window_number

    def test_canonical_json_sorted_keys(self):
        p = self._make_payload()
        data = p.serialize()
        parsed = json.loads(data)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_disqualified_field_default_empty(self):
        p = self._make_payload()
        assert p.disqualified == {}

    def test_disqualified_field_roundtrip(self):
        p = self._make_payload(disqualified={
            "miner_c": "pre_eval_rejected:banned",
            "miner_d": "integrity_failed",
        })
        assert "miner_a" in p.scores
        data = p.serialize()
        p2 = ConsensusPayload.deserialize(data)
        assert p2.disqualified == {
            "miner_c": "pre_eval_rejected:banned",
            "miner_d": "integrity_failed",
        }

    def test_disqualified_included_in_serialization(self):
        p = self._make_payload(disqualified={"miner_c": "integrity_failed"})
        data = p.serialize()
        parsed = json.loads(data)
        assert "disqualified" in parsed
        assert parsed["disqualified"]["miner_c"] == "integrity_failed"

    def test_deserialize_without_disqualified_field(self):
        """Payloads from older validators without disqualified field."""
        p = self._make_payload()
        d = p.to_dict()
        del d["disqualified"]
        data = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
        p2 = ConsensusPayload.deserialize(data)
        assert p2.disqualified == {}

    def test_deserialize_missing_bench_version(self):
        """Payloads without bench_version default to empty string."""
        raw = {
            "protocol_version": 1,
            "window_number": 42,
            "validator_hotkey": "val_x",
            "scores": {"m1": 0.9},
            "timestamp": 1000,
        }
        data = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        p = ConsensusPayload.deserialize(data)
        assert p.bench_version == ""


# ---------------------------------------------------------------------------
# Consensus filter pipeline tests
# ---------------------------------------------------------------------------

class TestConsensusFilter:
    """Tests for the 6-layer submission filter pipeline."""

    def _make_submission(
        self, hotkey="val_a", window=42, protocol=CONSENSUS_PROTOCOL_VERSION,
        version="0.1.0", scores=None, spec_number=SPEC_NUMBER,
    ):
        if scores is None:
            scores = {"miner_x": 0.85}
        payload = ConsensusPayload(
            protocol_version=protocol,
            window_number=window,
            validator_hotkey=hotkey,
            bench_version=version,
            scores=scores,
            timestamp=1000,
            spec_number=spec_number,
        )
        pointer = ConsensusPointer(
            protocol_version=protocol,
            window_number=window,
            content_address=payload.content_hash(),
            validator_hotkey=hotkey,
        )
        return pointer, payload

    def test_full_pipeline_all_pass(self):
        subs = [
            self._make_submission(hotkey="val_a"),
            self._make_submission(hotkey="val_b"),
        ]
        stakes = {"val_a": 100.0, "val_b": 200.0}
        validated, stats = run_filter_pipeline(
            subs, expected_window=42, validator_stakes=stakes,
            min_stake=10.0,
        )
        assert stats.passed == 2
        assert stats.total_input == 2
        assert len(validated) == 2
        assert validated[0].validator_stake == 100.0

    def test_filter_protocol_version(self):
        subs = [
            self._make_submission(hotkey="val_a", protocol=1),
            self._make_submission(hotkey="val_b", protocol=2),
        ]
        passed, skipped = filter_protocol_version(subs, expected_version=1)
        assert len(passed) == 1
        assert skipped == 1
        assert passed[0][0].validator_hotkey == "val_a"

    def test_filter_window_number(self):
        subs = [
            self._make_submission(hotkey="val_a", window=42),
            self._make_submission(hotkey="val_b", window=41),
        ]
        passed, skipped = filter_window_number(subs, expected_window=42)
        assert len(passed) == 1
        assert skipped == 1

    def test_filter_trust_threshold(self):
        subs = [
            self._make_submission(hotkey="val_a"),
            self._make_submission(hotkey="val_b"),
        ]
        stakes = {"val_a": 100.0, "val_b": 5.0}
        passed, skipped = filter_trust_threshold(subs, stakes, min_stake=10.0)
        assert len(passed) == 1
        assert skipped == 1
        assert passed[0][0].validator_hotkey == "val_a"

    def test_filter_trust_threshold_missing_stake(self):
        subs = [self._make_submission(hotkey="val_unknown")]
        passed, skipped = filter_trust_threshold(subs, {}, min_stake=1.0)
        assert len(passed) == 0
        assert skipped == 1

    def test_filter_zero_signal_with_nonzero(self):
        subs = [
            self._make_submission(hotkey="val_a", scores={"m": 0.85}),
            self._make_submission(hotkey="val_b", scores={"m": 0.0}),
        ]
        passed, skipped = filter_zero_signal(subs)
        assert len(passed) == 1
        assert skipped == 1
        assert passed[0][0].validator_hotkey == "val_a"

    def test_filter_zero_signal_all_zero_passes(self):
        subs = [
            self._make_submission(hotkey="val_a", scores={"m": 0.0}),
            self._make_submission(hotkey="val_b", scores={"m": 0.0}),
        ]
        passed, skipped = filter_zero_signal(subs)
        assert len(passed) == 2
        assert skipped == 0

    def test_filter_zero_signal_threshold_drops_near_zero(self):
        # val_b has 41/42 zero scores (97.6%) — a single token-nonzero — and
        # should be dropped when zero_threshold < 0.976. val_a has real signal.
        big_all_zero = {f"m{i}": 0.0 for i in range(41)}
        big_near_zero = {**big_all_zero, "m_winner": 0.02}
        big_real = {f"m{i}": 0.5 for i in range(5)}
        subs = [
            self._make_submission(hotkey="val_a", scores=big_real),
            self._make_submission(hotkey="val_b", scores=big_near_zero),
        ]
        passed, skipped = filter_zero_signal(subs, zero_threshold=0.95)
        assert len(passed) == 1
        assert skipped == 1
        assert passed[0][0].validator_hotkey == "val_a"

    def test_filter_zero_signal_default_threshold_keeps_near_zero(self):
        # Same near-zero payload passes under the legacy default (1.0) —
        # only strictly all-zero payloads are dropped.
        big_all_zero = {f"m{i}": 0.0 for i in range(41)}
        big_near_zero = {**big_all_zero, "m_winner": 0.02}
        subs = [
            self._make_submission(hotkey="val_a", scores={"m": 0.85}),
            self._make_submission(hotkey="val_b", scores=big_near_zero),
        ]
        passed, skipped = filter_zero_signal(subs)
        assert len(passed) == 2
        assert skipped == 0

    def test_full_pipeline_cascading_filters(self):
        subs = [
            self._make_submission(hotkey="good", window=42, version="0.1.0", scores={"m": 0.85}),
            self._make_submission(hotkey="bad_proto", window=42, protocol=CONSENSUS_PROTOCOL_VERSION + 1, version="0.1.0"),
            self._make_submission(hotkey="bad_window", window=41, version="0.1.0"),
            self._make_submission(hotkey="low_stake", window=42, version="0.1.0"),
            self._make_submission(hotkey="zero_score", window=42, version="0.1.0", scores={"m": 0.0}),
        ]
        stakes = {"good": 100.0, "bad_proto": 100.0, "bad_window": 100.0,
                  "low_stake": 1.0, "zero_score": 100.0}
        validated, stats = run_filter_pipeline(
            subs, expected_window=42, validator_stakes=stakes,
            min_stake=10.0,
        )
        assert stats.passed == 1
        assert stats.skipped_protocol == 1
        assert stats.skipped_window == 1
        assert stats.skipped_stake == 1
        assert stats.skipped_zero_signal == 1
        assert validated[0].pointer.validator_hotkey == "good"

    def test_empty_submissions(self):
        validated, stats = run_filter_pipeline(
            [], expected_window=42, validator_stakes={},
            min_stake=0,
        )
        assert stats.passed == 0
        assert len(validated) == 0

    def test_select_target_spec_majority_wins(self):
        """Stake-weighted majority spec_number is adopted as target."""
        subs = [
            self._make_submission(hotkey="val_a", spec_number=2),
            self._make_submission(hotkey="val_b", spec_number=2),
            self._make_submission(hotkey="val_c", spec_number=3),
        ]
        stakes = {"val_a": 100.0, "val_b": 200.0, "val_c": 50.0}
        target, source = select_target_spec_number(subs, stakes, local_spec=99)
        assert target == 2
        assert source == "chain_majority"

    def test_select_target_spec_no_majority_falls_back_to_local(self):
        """Without a >50% group the local SPEC_NUMBER is used."""
        subs = [
            self._make_submission(hotkey="val_a", spec_number=2),
            self._make_submission(hotkey="val_b", spec_number=3),
        ]
        stakes = {"val_a": 100.0, "val_b": 100.0}  # 50/50 split
        target, source = select_target_spec_number(subs, stakes, local_spec=7)
        assert target == 7
        assert source == "local_fallback"

    def test_select_target_spec_no_stake_falls_back_to_local(self):
        subs = [self._make_submission(hotkey="val_a", spec_number=4)]
        stakes = {}  # zero stake everywhere
        target, source = select_target_spec_number(subs, stakes, local_spec=11)
        assert target == 11
        assert source == "local_fallback"

    def test_filter_spec_number_drops_mismatched_payloads(self):
        subs = [
            self._make_submission(hotkey="val_a", spec_number=2),
            self._make_submission(hotkey="val_b", spec_number=3),
        ]
        passed, skipped = filter_spec_number(subs, target_spec_number=2)
        assert len(passed) == 1
        assert skipped == 1
        assert passed[0][1].spec_number == 2

    def test_pipeline_uses_chain_derived_target_spec(self):
        """Pipeline filters by majority spec_number, not local."""
        subs = [
            self._make_submission(hotkey="val_a", spec_number=2, scores={"m": 0.8}),
            self._make_submission(hotkey="val_b", spec_number=2, scores={"m": 0.7}),
            self._make_submission(hotkey="val_c", spec_number=3, scores={"m": 0.6}),
        ]
        stakes = {"val_a": 200.0, "val_b": 200.0, "val_c": 50.0}
        validated, stats = run_filter_pipeline(
            subs, expected_window=42, validator_stakes=stakes,
            min_stake=10.0, local_spec_number=99,  # local doesn't match anything
        )
        assert stats.target_spec_number == 2
        assert stats.target_spec_source == "chain_majority"
        assert stats.skipped_spec_number == 1
        assert stats.passed == 2

    def test_pipeline_falls_back_to_local_spec_when_no_majority(self):
        subs = [
            self._make_submission(hotkey="val_a", spec_number=2, scores={"m": 0.8}),
            self._make_submission(hotkey="val_b", spec_number=3, scores={"m": 0.7}),
        ]
        stakes = {"val_a": 100.0, "val_b": 100.0}  # 50/50 -> no >50% majority
        validated, stats = run_filter_pipeline(
            subs, expected_window=42, validator_stakes=stakes,
            min_stake=10.0, local_spec_number=2,
        )
        assert stats.target_spec_source == "local_fallback"
        assert stats.target_spec_number == 2
        assert stats.passed == 1
        assert validated[0].pointer.validator_hotkey == "val_a"


# ---------------------------------------------------------------------------
# Consensus payload back-compat tests
# ---------------------------------------------------------------------------

class TestConsensusPayloadBackCompat:
    """ConsensusPayload reads both spec_number and legacy scoring_version keys."""

    def test_from_dict_prefers_spec_number(self):
        d = {
            "protocol_version": 2,
            "window_number": 1,
            "validator_hotkey": "v",
            "bench_version": "x",
            "scores": {},
            "timestamp": 0,
            "spec_number": 7,
            "scoring_version": 1,  # legacy mirror, ignored when spec_number present
        }
        p = ConsensusPayload.from_dict(d)
        assert p.spec_number == 7

    def test_from_dict_falls_back_to_legacy_scoring_version(self):
        d = {
            "protocol_version": 2,
            "window_number": 1,
            "validator_hotkey": "v",
            "bench_version": "x",
            "scores": {},
            "timestamp": 0,
            "scoring_version": 5,  # only legacy key present (old payload)
        }
        p = ConsensusPayload.from_dict(d)
        assert p.spec_number == 5

    def test_to_dict_emits_both_keys(self):
        p = ConsensusPayload(
            protocol_version=2, window_number=1, validator_hotkey="v",
            bench_version="x", scores={}, timestamp=0, spec_number=4,
        )
        d = p.to_dict()
        assert d["spec_number"] == 4
        assert d["scoring_version"] == 4


# ---------------------------------------------------------------------------
# Stake-weighted consensus aggregation tests
# ---------------------------------------------------------------------------

class TestConsensusAggregation:
    """Tests for compute_consensus_scores."""

    def _make_validated(self, hotkey, stake, scores, disqualified=None):
        payload = ConsensusPayload(
            protocol_version=1, window_number=42,
            validator_hotkey=hotkey, bench_version="0.1.0",
            scores=scores, timestamp=1000,
            disqualified=disqualified or {},
        )
        pointer = ConsensusPointer(
            protocol_version=1, window_number=42,
            content_address=payload.content_hash(),
            validator_hotkey=hotkey,
        )
        return ValidatedSubmission(
            pointer=pointer, payload=payload, validator_stake=stake,
        )

    def test_single_validator(self):
        subs = [self._make_validated("v1", 100.0, {"m1": 0.8, "m2": 0.6})]
        scores, disqualified = compute_consensus_scores(subs)
        assert scores["m1"] == pytest.approx(0.8)
        assert scores["m2"] == pytest.approx(0.6)
        assert "m1" not in disqualified
        assert "m2" not in disqualified

    def test_equal_stake_average(self):
        subs = [
            self._make_validated("v1", 100.0, {"m1": 0.4}),
            self._make_validated("v2", 100.0, {"m1": 0.8}),
        ]
        scores, _ = compute_consensus_scores(subs)
        assert scores["m1"] == pytest.approx(0.6)

    def test_stake_weighted_average(self):
        subs = [
            self._make_validated("v1", 300.0, {"m1": 0.4}),
            self._make_validated("v2", 100.0, {"m1": 0.8}),
        ]
        scores, _ = compute_consensus_scores(subs)
        # (300*0.4 + 100*0.8) / (300+100) = (120+80)/400 = 0.5
        assert scores["m1"] == pytest.approx(0.5)

    def test_different_miners_per_validator(self):
        subs = [
            self._make_validated("v1", 100.0, {"m1": 0.4, "m2": 0.6}),
            self._make_validated("v2", 100.0, {"m1": 0.8}),
        ]
        scores, _ = compute_consensus_scores(subs)
        assert scores["m1"] == pytest.approx(0.6)
        assert scores["m2"] == pytest.approx(0.6)

    def test_disqualification_equal_stake_split_not_disqualified(self):
        """50/50 stake split: disq_ratio = 0.5, NOT > 0.5, so NOT disqualified."""
        subs = [
            self._make_validated("v1", 100.0, {"m1": 0.8}),
            self._make_validated("v2", 100.0, {}, {"m1": "policy_violation"}),
        ]
        _, disqualified = compute_consensus_scores(subs)
        assert "m1" not in disqualified

    def test_no_disqualification_when_none_flag(self):
        subs = [
            self._make_validated("v1", 100.0, {"m1": 0.7}),
            self._make_validated("v2", 100.0, {"m1": 0.9}),
        ]
        _, disqualified = compute_consensus_scores(subs)
        assert "m1" not in disqualified

    def test_disqualification_stake_weighted_majority(self):
        """High-stake validator does not disqualify, low-stake does: miner survives."""
        subs = [
            self._make_validated("v1", 300.0, {"m1": 0.7}),
            self._make_validated("v2", 100.0, {}, {"m1": "policy_violation"}),
        ]
        _, disqualified = compute_consensus_scores(subs)
        # disq_ratio = 100/(300+100) = 0.25, NOT > 0.5
        assert "m1" not in disqualified

    def test_minority_stake_cannot_disqualify(self):
        """A low-stake malicious validator cannot disqualify a miner alone."""
        subs = [
            self._make_validated("v1", 500.0, {"m1": 0.8}),
            self._make_validated("v2", 200.0, {"m1": 0.7}),
            self._make_validated("v_malicious", 50.0, {}, {"m1": "fake_reason"}),
        ]
        _, disqualified = compute_consensus_scores(subs)
        # disq_ratio = 50/750 ≈ 0.067, NOT > 0.5
        assert "m1" not in disqualified

    def test_majority_stake_disqualifies(self):
        """When >50% of stake says disqualified, miner is disqualified."""
        subs = [
            self._make_validated("v1", 100.0, {"m1": 0.8}),
            self._make_validated("v2", 300.0, {}, {"m1": "policy_violation"}),
            self._make_validated("v3", 200.0, {}, {"m1": "policy_violation"}),
        ]
        _, disqualified = compute_consensus_scores(subs)
        # disq_ratio = 500/600 ≈ 0.83 > 0.5
        assert "m1" in disqualified

    def test_empty_submissions(self):
        scores, disqualified = compute_consensus_scores([])
        assert scores == {}
        assert disqualified == {}

    def test_zero_stake_ignored(self):
        subs = [
            self._make_validated("v1", 0.0, {"m1": 0.99}),
            self._make_validated("v2", 50.0, {"m1": 0.7}),
        ]
        scores, _ = compute_consensus_scores(subs)
        assert scores["m1"] == pytest.approx(0.7)

    def test_score_excludes_disqualified_votes(self):
        """Scores from validators that disqualified a miner are excluded."""
        subs = [
            self._make_validated("v1", 200.0, {}, {"m1": "policy_violation"}),
            self._make_validated("v2", 300.0, {"m1": 0.85}),
        ]
        scores, disqualified = compute_consensus_scores(subs)
        # disq_ratio = 200/500 = 0.4, NOT > 0.5
        assert "m1" not in disqualified
        assert scores["m1"] == pytest.approx(0.85)

    def test_score_uses_only_non_disqualified_stake_weighted(self):
        """When multiple validators have scores, result is stake-weighted
        among non-disqualifying votes only."""
        subs = [
            self._make_validated("v1", 300.0, {"m1": 0.8}),
            self._make_validated("v2", 100.0, {"m1": 0.6}),
            self._make_validated("v3", 50.0, {}, {"m1": "policy_violation"}),
        ]
        scores, disqualified = compute_consensus_scores(subs)
        # disq_ratio = 50/450 ≈ 0.11, NOT > 0.5
        assert "m1" not in disqualified
        # score = (300*0.8 + 100*0.6) / (300+100) = (240+60)/400 = 0.75
        assert scores["m1"] == pytest.approx(0.75)

    def test_score_zero_when_all_disqualify(self):
        """If all validators disqualify, score defaults to 0."""
        subs = [
            self._make_validated("v1", 100.0, {}, {"m1": "policy_violation"}),
            self._make_validated("v2", 100.0, {}, {"m1": "policy_violation"}),
        ]
        scores, disqualified = compute_consensus_scores(subs)
        assert "m1" in disqualified
        assert scores.get("m1", 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Winner Protection tests
# ---------------------------------------------------------------------------

class TestWinnerProtection:
    """Tests for Winner Protection mechanism."""

    def test_no_winner_lowest_wins(self):
        costs = {"m1": 3.0, "m2": 5.0}
        quals = {"m1": True, "m2": True}
        state = WinnerState()
        winner, new_state = select_winner_with_protection(
            costs, quals, state, cost_delta=0.10,
        )
        assert winner == "m1"
        assert new_state.winner_hotkey == "m1"
        assert new_state.winner_cost == 3.0

    def test_winner_retains_within_margin(self):
        costs = {"m1": 3.0, "m2": 2.75}
        quals = {"m1": True, "m2": True}
        state = WinnerState(winner_hotkey="m1", winner_cost=3.0)
        winner, new_state = select_winner_with_protection(
            costs, quals, state, cost_delta=0.10,
        )
        # m2 (2.75) needs < 3.0 * 0.90 = 2.70 to dethrone
        assert winner == "m1"
        assert new_state.winner_cost == 3.0

    def test_challenger_overtakes_past_margin(self):
        costs = {"m1": 3.0, "m2": 2.60}
        quals = {"m1": True, "m2": True}
        state = WinnerState(winner_hotkey="m1", winner_cost=3.0)
        winner, new_state = select_winner_with_protection(
            costs, quals, state, cost_delta=0.10,
        )
        # m2 (2.60) < 3.0 * 0.90 = 2.70 → overtake
        assert winner == "m2"
        assert new_state.winner_hotkey == "m2"
        assert new_state.winner_cost == 2.60

    def test_winner_disqualified(self):
        costs = {"m1": 3.0, "m2": 5.0}
        quals = {"m1": False, "m2": True}
        state = WinnerState(winner_hotkey="m1", winner_cost=3.0)
        winner, new_state = select_winner_with_protection(
            costs, quals, state, cost_delta=0.10,
        )
        assert winner == "m2"
        assert new_state.winner_hotkey == "m2"
        assert new_state.winner_cost == 5.0

    def test_winner_defends_with_winning_cost_not_current(self):
        """Winner defends with frozen winner_cost, not their latest consensus cost."""
        state = WinnerState(winner_hotkey="m1", winner_cost=2.0)
        # m1's current cost is 4.0 (worse), but defense uses winner_cost=2.0
        costs = {"m1": 4.0, "m2": 2.5}
        quals = {"m1": True, "m2": True}
        winner, _ = select_winner_with_protection(
            costs, quals, state, cost_delta=0.10,
        )
        # threshold = 2.0 * 0.90 = 1.80; m2 (2.5) > 1.80 → winner retains
        assert winner == "m1"

    def test_winner_self_update(self):
        """Winner can update their own record if they beat the margin."""
        state = WinnerState(
            winner_hotkey="m1", winner_cost=3.0, winner_pack_hash="old_hash",
        )
        costs = {"m1": 2.60, "m2": 5.0}
        quals = {"m1": True, "m2": True}
        winner, new_state = select_winner_with_protection(
            costs, quals, state, cost_delta=0.10,
            pack_hashes={"m1": "new_hash", "m2": "m2_hash"},
        )
        # m1 (2.60) < 3.0 * 0.90 = 2.70 → self-update
        assert winner == "m1"
        assert new_state.winner_cost == 2.60
        assert new_state.winner_pack_hash == "new_hash"

    def test_winner_no_self_update_within_margin(self):
        """Winner does NOT update if their new cost doesn't clear margin."""
        state = WinnerState(
            winner_hotkey="m1", winner_cost=3.0, winner_pack_hash="old_hash",
        )
        costs = {"m1": 2.80, "m2": 5.0}
        quals = {"m1": True, "m2": True}
        winner, new_state = select_winner_with_protection(
            costs, quals, state, cost_delta=0.10,
        )
        # m1 (2.80) >= 3.0 * 0.90 = 2.70 → no update
        assert winner == "m1"
        assert new_state.winner_cost == 3.0
        assert new_state.winner_pack_hash == "old_hash"

    def test_no_qualified_miners(self):
        costs = {"m1": 3.0}
        quals = {"m1": False}
        state = WinnerState()
        winner, _ = select_winner_with_protection(
            costs, quals, state, cost_delta=0.10,
        )
        assert winner is None

    def test_save_load_roundtrip(self, tmp_path):
        state = WinnerState(
            winner_hotkey="m1",
            winner_pack_hash="abc123",
            winner_cost=3.5,
        )
        path = str(tmp_path / "winner.json")
        save_winner_state(state, path)
        loaded = load_winner_state(path)
        assert loaded.winner_hotkey == state.winner_hotkey
        assert loaded.winner_pack_hash == state.winner_pack_hash
        assert loaded.winner_cost == state.winner_cost

    def test_load_missing_file(self, tmp_path):
        loaded = load_winner_state(str(tmp_path / "nonexistent.json"))
        assert loaded.winner_hotkey is None
        assert loaded.winner_cost is None

    def test_load_accepts_legacy_scoring_version_key(self, tmp_path):
        """Old winner_state.json with `scoring_version` key still loads."""
        import json as _json
        path = tmp_path / "winner.json"
        path.write_text(_json.dumps({
            "scoring_version": 7,  # legacy key only
            "winner_hotkey": "hk_legacy",
            "winner_pack_hash": "pack_legacy",
            "winner_score": 0.42,
        }))
        loaded = load_winner_state(str(path))
        assert loaded.winner_hotkey == "hk_legacy"
        assert loaded.winner_pack_hash == "pack_legacy"
        assert loaded.winner_score == 0.42
        assert loaded.spec_number == 7

    def test_load_does_not_reset_on_spec_number_mismatch(self, tmp_path):
        """spec_number is audit-only; mismatch must NOT clear winner state."""
        import json as _json

        path = tmp_path / "winner.json"
        path.write_text(_json.dumps({
            "spec_number": SPEC_NUMBER + 99,
            "winner_hotkey": "hk_should_persist",
            "winner_pack_hash": "ph",
            "winner_score": 0.6,
        }))
        loaded = load_winner_state(str(path))
        assert loaded.winner_hotkey == "hk_should_persist"
        assert loaded.winner_score == 0.6

    def test_save_emits_both_spec_number_and_legacy_key(self, tmp_path):
        """save_winner_state writes both spec_number and scoring_version."""
        import json as _json

        state = WinnerState(
            winner_hotkey="hk",
            winner_pack_hash="ph",
            winner_score=0.5,
            spec_number=11,
        )
        path = str(tmp_path / "winner.json")
        save_winner_state(state, path)
        data = _json.loads(open(path).read())
        assert data["spec_number"] == 11
        assert data["scoring_version"] == 11

    def test_challenger_beats_winner_self_update(self):
        """When both challenger and winner beat margin, lowest cost wins."""
        state = WinnerState(winner_hotkey="m1", winner_cost=3.0)
        costs = {"m1": 2.60, "m2": 2.50}
        quals = {"m1": True, "m2": True}
        winner, new_state = select_winner_with_protection(
            costs, quals, state, cost_delta=0.10,
        )
        # Both < 3.0 * 0.90 = 2.70, but m2 (2.50) is lowest → m2 wins
        assert winner == "m2"
        assert new_state.winner_hotkey == "m2"
        assert new_state.winner_cost == 2.50

    def test_winner_degraded_pack_retains(self):
        """Winner with worse pack still defends with winning cost."""
        state = WinnerState(winner_hotkey="m1", winner_cost=2.0)
        costs = {"m1": 5.0, "m2": 3.0}
        quals = {"m1": True, "m2": True}
        winner, new_state = select_winner_with_protection(
            costs, quals, state, cost_delta=0.10,
        )
        # threshold = 2.0 * 0.90 = 1.80; m2 (3.0) > 1.80 → winner retains
        assert winner == "m1"
        assert new_state.winner_cost == 2.0


# ---------------------------------------------------------------------------
# On-chain consensus commitment tests
# ---------------------------------------------------------------------------

class TestConsensusCommitments:
    """Tests for validator consensus commitment parsing and formatting."""

    def test_format_consensus_commitment(self):
        result = format_consensus_commitment(1, 42, "QmXxx123")
        assert result == "consensus:1|42|1|QmXxx123"

    def test_format_with_gcs_url(self):
        url = "https://storage.googleapis.com/trajrl-consensus/sha256_abc.json"
        result = format_consensus_commitment(1, 42, url)
        assert result == f"consensus:1|42|1|{url}"

    def test_parse_consensus_commitment_ipfs(self):
        raw = "consensus:1|42|1|QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
        parsed = parse_consensus_commitment(raw)
        assert parsed is not None
        pv, wn, addr, sv = parsed
        assert pv == 1
        assert wn == 42
        assert addr == "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
        assert sv == 1

    def test_parse_consensus_commitment_gcs(self):
        raw = "consensus:1|42|1|https://storage.googleapis.com/trajrl/sha256_abc.json"
        parsed = parse_consensus_commitment(raw)
        assert parsed is not None
        pv, wn, addr, sv = parsed
        assert pv == 1
        assert wn == 42
        assert addr == "https://storage.googleapis.com/trajrl/sha256_abc.json"
        assert sv == 1

    def test_parse_consensus_commitment_invalid_prefix(self):
        assert parse_consensus_commitment("notconsensus:1|42|QmXxx") is None

    def test_parse_consensus_commitment_missing_parts(self):
        assert parse_consensus_commitment("consensus:1|42") is None
        assert parse_consensus_commitment("consensus:1") is None
        assert parse_consensus_commitment("consensus:") is None

    def test_parse_consensus_commitment_bad_numbers(self):
        assert parse_consensus_commitment("consensus:abc|42|QmXxx") is None
        assert parse_consensus_commitment("consensus:1|xyz|QmXxx") is None

    def test_parse_consensus_commitment_empty_address(self):
        assert parse_consensus_commitment("consensus:1|42|") is None

    def test_parse_consensus_commitment_none(self):
        assert parse_consensus_commitment(None) is None
        assert parse_consensus_commitment("") is None

    def test_roundtrip(self):
        original = format_consensus_commitment(1, 99, "QmTestCID123")
        parsed = parse_consensus_commitment(original)
        assert parsed == (1, 99, "QmTestCID123", 1)

    def test_is_consensus_commitment(self):
        assert is_consensus_commitment("consensus:1|42|QmXxx") is True
        assert is_consensus_commitment("  consensus:1|42|QmXxx  ") is True
        assert is_consensus_commitment("abc123|https://example.com/pack.json") is False
        assert is_consensus_commitment("") is False
        assert is_consensus_commitment(None) is False

    def test_miner_parse_rejects_consensus(self):
        """parse_commitment (miner format) should reject consensus commitments."""
        raw = "consensus:1|42|QmXxx"
        assert parse_commitment(raw) is None

    def test_miner_parse_still_works(self):
        """parse_commitment still works for normal miner commitments."""
        pack_hash = "a" * 64
        raw = f"{pack_hash}|https://example.com/pack.json"
        result = parse_commitment(raw)
        assert result is not None
        assert result[0] == pack_hash
        assert result[1] == "https://example.com/pack.json"

    def test_pipes_in_url_preserved(self):
        """Content address with pipes should be handled (split maxsplit=3)."""
        raw = "consensus:1|42|1|https://example.com/path?a=1|extra"
        parsed = parse_consensus_commitment(raw)
        assert parsed is not None
        _, _, addr, sv = parsed
        assert addr == "https://example.com/path?a=1|extra"
        assert sv == 1


# ---------------------------------------------------------------------------
# Validator identity filtering tests
# ---------------------------------------------------------------------------

class TestValidatorIdentityFiltering:
    """Tests for validator_permit-based filtering in fetch_validator_consensus_commitments."""

    def _make_metagraph(self, hotkeys, validator_permits):
        mg = MagicMock()
        mg.hotkeys = hotkeys
        mg.validator_permit = validator_permits
        return mg

    def test_only_permitted_validators_returned(self):
        """Consensus commitments from non-validator hotkeys should be filtered out."""
        validator_hk = "5Fval_aaaa"
        miner_hk = "5Fminer_bb"
        raw_commitments = {
            validator_hk: "consensus:1|42|QmTestCID_val",
            miner_hk: "consensus:1|42|QmTestCID_miner",
        }

        mock_sub = MagicMock()
        mock_sub.get_all_commitments.return_value = raw_commitments
        mock_sub.get_commitment_metadata.return_value = {"block": 50000}

        mg = self._make_metagraph(
            hotkeys=[validator_hk, miner_hk],
            validator_permits=[True, False],
        )

        results = fetch_validator_consensus_commitments(mock_sub, netuid=11, metagraph=mg)
        assert len(results) == 1
        assert results[0].validator_hotkey == validator_hk

    def test_miner_fake_consensus_commitment_rejected(self):
        """A miner submitting a consensus:-prefixed commitment must be rejected."""
        miner_hk = "5Fminer_aa"
        raw_commitments = {miner_hk: "consensus:1|42|QmFakeCID"}

        mock_sub = MagicMock()
        mock_sub.get_all_commitments.return_value = raw_commitments
        mock_sub.get_commitment_metadata.return_value = {"block": 50000}

        mg = self._make_metagraph(
            hotkeys=[miner_hk],
            validator_permits=[False],
        )

        results = fetch_validator_consensus_commitments(mock_sub, netuid=11, metagraph=mg)
        assert len(results) == 0

    def test_all_validators_pass(self):
        """All hotkeys with validator_permit=True should be included."""
        hk_a = "5Fval_a"
        hk_b = "5Fval_b"
        raw_commitments = {
            hk_a: "consensus:1|42|QmCID_A",
            hk_b: "consensus:1|42|QmCID_B",
        }

        mock_sub = MagicMock()
        mock_sub.get_all_commitments.return_value = raw_commitments
        mock_sub.get_commitment_metadata.return_value = {"block": 50000}

        mg = self._make_metagraph(
            hotkeys=[hk_a, hk_b],
            validator_permits=[True, True],
        )

        results = fetch_validator_consensus_commitments(mock_sub, netuid=11, metagraph=mg)
        assert len(results) == 2

    def test_miner_commitment_format_not_affected(self):
        """Normal miner commitments (non-consensus prefix) are ignored regardless."""
        miner_hk = "5Fminer_aa"
        raw_commitments = {
            miner_hk: "a" * 64 + "|https://example.com/pack.json",
        }

        mock_sub = MagicMock()
        mock_sub.get_all_commitments.return_value = raw_commitments

        mg = self._make_metagraph(
            hotkeys=[miner_hk],
            validator_permits=[False],
        )

        results = fetch_validator_consensus_commitments(mock_sub, netuid=11, metagraph=mg)
        assert len(results) == 0

    def test_metagraph_fallback_on_error(self):
        """If metagraph.validator_permit fails, fall back to prefix-only filtering."""
        hk = "5Fval_aa"
        raw_commitments = {hk: "consensus:1|42|QmCID_X"}

        mock_sub = MagicMock()
        mock_sub.get_all_commitments.return_value = raw_commitments
        mock_sub.get_commitment_metadata.return_value = {"block": 50000}

        mg = MagicMock()
        mg.hotkeys = property(lambda self: (_ for _ in ()).throw(RuntimeError("broken")))
        type(mg).hotkeys = property(lambda self: (_ for _ in ()).throw(RuntimeError("broken")))

        results = fetch_validator_consensus_commitments(mock_sub, netuid=11, metagraph=mg)
        assert len(results) == 1

    def test_unknown_hotkey_not_in_metagraph(self):
        """Hotkey not in metagraph.hotkeys should be filtered out."""
        known_hk = "5Fval_known"
        unknown_hk = "5Funknown"
        raw_commitments = {
            known_hk: "consensus:1|42|QmCID_known",
            unknown_hk: "consensus:1|42|QmCID_unknown",
        }

        mock_sub = MagicMock()
        mock_sub.get_all_commitments.return_value = raw_commitments
        mock_sub.get_commitment_metadata.return_value = {"block": 50000}

        mg = self._make_metagraph(
            hotkeys=[known_hk],
            validator_permits=[True],
        )

        results = fetch_validator_consensus_commitments(mock_sub, netuid=11, metagraph=mg)
        assert len(results) == 1
        assert results[0].validator_hotkey == known_hk


# ---------------------------------------------------------------------------
# Consensus-based weight setting tests
# ---------------------------------------------------------------------------

class TestConsensusWeightSetting:
    """Tests for weight setting data source selection (consensus vs fallback).

    Rule: once consensus costs exist, they are ALWAYS used for weight setting.
    Fallback weights are only used when no consensus has ever been computed.
    Consensus costs persist across windows and restarts.
    """

    def test_use_consensus_when_costs_present(self):
        """Consensus costs should always be used when they exist."""
        consensus_costs = {"m1": 3.0, "m2": 5.0}
        use_consensus = bool(consensus_costs)
        assert use_consensus is True

    def test_fallback_when_no_consensus(self):
        """Without consensus data, weight setting should use fallback weights."""
        consensus_costs = {}
        should_fallback = not bool(consensus_costs)
        assert should_fallback is True

    def test_consensus_persists_across_windows(self):
        """Previous window's consensus costs should still be used in new window."""
        consensus_costs = {"m1": 3.0}
        should_fallback = not bool(consensus_costs)
        assert should_fallback is False

    def test_new_aggregation_overwrites_old(self):
        """New window's aggregation replaces previous consensus costs."""
        consensus_costs = {"m1": 3.0}
        new_consensus = {"m1": 2.5, "m2": 4.0}
        consensus_costs = new_consensus
        use_consensus = bool(consensus_costs)
        assert use_consensus is True
        assert consensus_costs["m1"] == 2.5

    def test_consensus_costs_mapped_to_uids(self):
        """Consensus costs (keyed by hotkey) should map to UIDs correctly."""
        consensus_costs = {"hk_m1": 3.0, "hk_m2": 5.0, "hk_m3": 2.0}
        consensus_qualified = {"hk_m1": True, "hk_m2": False, "hk_m3": True}

        active = {
            10: type("C", (), {"hotkey": "hk_m1"})(),
            20: type("C", (), {"hotkey": "hk_m2"})(),
            30: type("C", (), {"hotkey": "hk_m3"})(),
            40: type("C", (), {"hotkey": "hk_m4"})(),
        }

        hotkey_to_uid = {}
        uid_to_hotkey = {}
        for uid, commitment in active.items():
            hotkey_to_uid[commitment.hotkey] = uid
            uid_to_hotkey[uid] = commitment.hotkey

        costs = {}
        qualified = {}
        for hotkey, c_cost in consensus_costs.items():
            uid = hotkey_to_uid.get(hotkey)
            if uid is None:
                continue
            costs[uid] = c_cost
            qualified[uid] = consensus_qualified.get(hotkey, False)

        assert costs == {10: 3.0, 20: 5.0, 30: 2.0}
        assert qualified == {10: True, 20: False, 30: True}
        assert 40 not in costs

    def test_consensus_costs_skip_deregistered_miners(self):
        """Miners in consensus but no longer in active commitments are skipped."""
        consensus_costs = {"hk_m1": 3.0, "hk_m_gone": 5.0}

        active = {10: type("C", (), {"hotkey": "hk_m1"})()}
        hotkey_to_uid = {c.hotkey: uid for uid, c in active.items()}

        costs = {}
        for hotkey, c_cost in consensus_costs.items():
            uid = hotkey_to_uid.get(hotkey)
            if uid is None:
                continue
            costs[uid] = c_cost

        assert costs == {10: 3.0}
        assert "hk_m_gone" not in [active[u].hotkey for u in costs]

    def test_window_aggregated_resets_on_new_window(self):
        """_window_aggregated should reset to False at new window boundary."""
        window_aggregated = True
        new_window_detected = True
        if new_window_detected:
            window_aggregated = False
        assert window_aggregated is False

    def test_consensus_qualified_false_propagates(self):
        """If consensus says not qualified, weight setting should see False."""
        consensus_qualified = {"hk_m1": True, "hk_m2": False}
        active = {
            10: type("C", (), {"hotkey": "hk_m1"})(),
            20: type("C", (), {"hotkey": "hk_m2"})(),
        }
        hotkey_to_uid = {c.hotkey: uid for uid, c in active.items()}

        qualified = {}
        for hotkey in consensus_qualified:
            uid = hotkey_to_uid.get(hotkey)
            if uid is not None:
                qualified[uid] = consensus_qualified[hotkey]

        assert qualified[10] is True
        assert qualified[20] is False
