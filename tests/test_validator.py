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
from trajectoryrl.utils.pack_ownership import (
    claim_owner, evict_orphans, load_pack_first_seen,
    EVICTION_GRACE_WINDOWS,
)
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
                    fetcher._fetch_pack(
                        "https://trajrl.com/samples/pack.json",
                        "a" * 64,
                    )
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
                    fetcher._fetch_pack(
                        "https://trajrl.com/samples/pack.json",
                        "a" * 64,
                    )
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


class TestFetchAllCommitmentsResilience:
    """Tests for chain-failure handling and reconnect retry in fetch_all_commitments.

    Distinguishes "chain query failed" (returns None — caller MUST NOT persist
    a snapshot) from "chain query succeeded with no commitments" (returns {} —
    legitimate empty state).
    """

    def _hotkey(self) -> str:
        return "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

    def _good_raw(self) -> str:
        return "a" * 64 + "|https://trajrl.com/samples/pack.json"

    def _metagraph_with(self, hotkey: str):
        mg = MagicMock()
        mg.hotkeys = [hotkey]
        return mg

    def test_returns_none_when_chain_query_fails_and_no_reconnect_cb(self):
        """Without a reconnect callback, a chain exception → returns None
        (signals failure so caller skips snapshot persistence)."""
        from trajectoryrl.utils.commitments import fetch_all_commitments

        mock_subtensor = MagicMock()
        mock_subtensor.get_all_commitments.side_effect = Exception("ws closed")

        result = fetch_all_commitments(
            mock_subtensor, netuid=11, metagraph=self._metagraph_with(self._hotkey()),
        )
        assert result is None

    def test_returns_empty_dict_when_chain_truly_empty(self):
        """A successful chain query that returns 0 commitments → returns {} (not None).
        Distinguishes legit empty state from query failure."""
        from trajectoryrl.utils.commitments import fetch_all_commitments

        mock_subtensor = MagicMock()
        mock_subtensor.get_all_commitments.return_value = {}

        result = fetch_all_commitments(
            mock_subtensor, netuid=11, metagraph=self._metagraph_with(self._hotkey()),
        )
        assert result == {}

    def test_retries_via_reconnect_callback_when_first_attempt_fails(self):
        """When the first chain query raises, the reconnect callback is invoked
        and the returned subtensor is used for a single retry."""
        from trajectoryrl.utils.commitments import fetch_all_commitments

        hotkey = self._hotkey()

        broken_subtensor = MagicMock()
        broken_subtensor.get_all_commitments.side_effect = Exception("ws closed")

        fresh_subtensor = MagicMock()
        fresh_subtensor.get_all_commitments.return_value = {hotkey: self._good_raw()}
        fresh_subtensor.get_commitment_metadata.return_value = {"block": 42000}

        reconnect_calls = {"count": 0}

        def reconnect():
            reconnect_calls["count"] += 1
            return fresh_subtensor

        result = fetch_all_commitments(
            broken_subtensor,
            netuid=11,
            metagraph=self._metagraph_with(hotkey),
            reconnect=reconnect,
        )

        assert reconnect_calls["count"] == 1
        assert result is not None
        assert 0 in result
        assert result[0].hotkey == hotkey

    def test_returns_none_when_reconnected_fetch_also_fails(self):
        """If the chain query fails after reconnect too, returns None."""
        from trajectoryrl.utils.commitments import fetch_all_commitments

        broken_subtensor = MagicMock()
        broken_subtensor.get_all_commitments.side_effect = Exception("ws closed")

        still_broken = MagicMock()
        still_broken.get_all_commitments.side_effect = Exception("still broken")

        result = fetch_all_commitments(
            broken_subtensor,
            netuid=11,
            metagraph=self._metagraph_with(self._hotkey()),
            reconnect=lambda: still_broken,
        )
        assert result is None

    def test_does_not_invoke_reconnect_when_first_attempt_succeeds(self):
        """No reconnect call when the chain query succeeds on the first try."""
        from trajectoryrl.utils.commitments import fetch_all_commitments

        hotkey = self._hotkey()

        mock_subtensor = MagicMock()
        mock_subtensor.get_all_commitments.return_value = {hotkey: self._good_raw()}
        mock_subtensor.get_commitment_metadata.return_value = {"block": 1000}

        reconnect_calls = {"count": 0}

        def reconnect():
            reconnect_calls["count"] += 1
            return mock_subtensor

        result = fetch_all_commitments(
            mock_subtensor,
            netuid=11,
            metagraph=self._metagraph_with(hotkey),
            reconnect=reconnect,
        )
        assert reconnect_calls["count"] == 0
        assert result is not None and 0 in result


class TestAcquireWindowSnapshotGuard:
    """Tests for _acquire_window_snapshot's behavior when chain query fails.

    A failed chain query (fetch_all_commitments returns None) MUST NOT be
    persisted as an "empty active set" snapshot — that locks the validator
    into a no-eval state for the rest of the ~24h window.
    """

    def test_does_not_save_snapshot_when_fetch_returns_none(self, tmp_path):
        """When fetch_all_commitments returns None, _acquire_window_snapshot
        must return None and must NOT call save_snapshot — so the next loop
        iteration (or restart) can re-attempt the fetch instead of being
        locked by a poisoned cache file."""
        from trajectoryrl.base import validator as validator_module
        from trajectoryrl.utils.eval_snapshot import EvalSnapshot

        # Build a minimal stub validator that exposes the methods/attrs
        # _acquire_window_snapshot needs, without going through the full
        # TrajectoryValidator constructor.
        validator = MagicMock()
        validator.config.active_set_dir = str(tmp_path)
        validator.config.netuid = 11
        validator.config.inactivity_blocks = 120
        validator.subtensor = MagicMock()
        validator.metagraph = MagicMock()

        # Bind the real _acquire_window_snapshot onto our stub.
        validator._acquire_window_snapshot = (
            validator_module.TrajectoryValidator._acquire_window_snapshot.__get__(
                validator
            )
        )

        # Build a minimal EvaluationWindow-like object (the method only reads
        # window_number and window_start from it).
        window = MagicMock()
        window.window_number = 1123
        window.window_start = 8085600

        with patch.object(
            validator_module, "fetch_all_commitments", return_value=None,
        ) as fetch_mock, patch.object(
            validator_module, "save_snapshot",
        ) as save_mock, patch.object(
            validator_module, "load_snapshot", return_value=None,
        ):
            result = validator._acquire_window_snapshot(window, current_block=8085614)

        assert fetch_mock.called
        assert result is None, "should return None to signal caller to fall back"
        assert not save_mock.called, (
            "must NOT persist a snapshot when chain query failed — "
            "would poison the cache and block the next ~24h window"
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
            config.weight_interval_blocks = 360
            config.cost_delta = 0.10
            config.required_categories = ["safety", "correctness"]
            config.eval_state_path = Path("/tmp/test_eval_state.json")
            config.pack_first_seen_path = Path("/tmp/test_pack_first_seen.json")
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
            validator.pack_first_seen = {}
            validator.pack_last_seen_window = {}
            validator.scenario_scores = {}
            validator._hotkey_uid_map = {}
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
            validator._target_window = 0
            validator._target_submit_done = False
            validator._waiting_for_quorum = False
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
        """Eval state (scores, pack_hash, blocks, windows) survives save/load.

        Note: ``pack_first_seen`` lives in its own file; covered by
        ``test_pack_first_seen_persistence_roundtrip``.
        """
        v = self._make_validator()
        v.scenario_scores = {"hk_0": {"client_escalation": 0.85}}
        v._eval_pack_hash = {"hk_0": "hash_a"}
        v.last_eval_block = {"hk_0": 99000}
        v.last_eval_window = {"hk_0": 13}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.eval_state_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.pack_first_seen_path = Path(f.name)

        try:
            v._save_eval_state()

            # eval_state.json must NOT carry the legacy pack_first_seen key.
            saved = json.loads(v.config.eval_state_path.read_text())
            assert "pack_first_seen" not in saved
            assert "quorum_wait_cycles" not in saved
            assert "last_quorum_ratio" not in saved
            assert "last_submitted_stake" not in saved
            assert "last_total_validator_stake" not in saved

            v2 = self._make_validator()
            v2.config.eval_state_path = v.config.eval_state_path
            v2.config.pack_first_seen_path = v.config.pack_first_seen_path
            v2._load_eval_state()

            assert v2.scenario_scores == {"hk_0": {"client_escalation": 0.85}}
            assert v2._eval_pack_hash == {"hk_0": "hash_a"}
            assert v2.last_eval_block == {"hk_0": 99000}
            assert v2.last_eval_window == {"hk_0": 13}
        finally:
            v.config.eval_state_path.unlink(missing_ok=True)
            v.config.pack_first_seen_path.unlink(missing_ok=True)

    def test_load_heals_legacy_waiting_for_quorum_false_positive(self):
        """Pre-PR197 binaries could persist waiting_for_quorum=True with
        target_submit_done=False. Loading such a file must clear the flag
        so the main loop does not keep burning on a phantom quorum wait.
        """
        v = self._make_validator()
        v._waiting_for_quorum = True
        v._target_submit_done = False
        v._target_window = 42
        v._consensus_window = 41

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.eval_state_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.pack_first_seen_path = Path(f.name)

        try:
            v._save_eval_state()

            poisoned = json.loads(v.config.eval_state_path.read_text())
            assert poisoned["waiting_for_quorum"] is True
            assert poisoned["target_submit_done"] is False

            v2 = self._make_validator()
            v2.config.eval_state_path = v.config.eval_state_path
            v2.config.pack_first_seen_path = v.config.pack_first_seen_path
            v2._load_eval_state()

            assert v2._waiting_for_quorum is False
            assert v2._target_submit_done is False
            assert v2._target_window == 42
            assert v2._consensus_window == 41

            healed = json.loads(v2.config.eval_state_path.read_text())
            assert healed["waiting_for_quorum"] is False
        finally:
            v.config.eval_state_path.unlink(missing_ok=True)
            v.config.pack_first_seen_path.unlink(missing_ok=True)

    def test_load_preserves_legitimate_waiting_for_quorum(self):
        """Waiting_for_quorum=True with target_submit_done=True is a real
        post-submission wait and must NOT be cleared by the legacy heal.
        """
        v = self._make_validator()
        v._waiting_for_quorum = True
        v._target_submit_done = True
        v._target_window = 42
        v._consensus_window = 41

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.eval_state_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.pack_first_seen_path = Path(f.name)

        try:
            v._save_eval_state()

            v2 = self._make_validator()
            v2.config.eval_state_path = v.config.eval_state_path
            v2.config.pack_first_seen_path = v.config.pack_first_seen_path
            v2._load_eval_state()

            assert v2._waiting_for_quorum is True
            assert v2._target_submit_done is True
        finally:
            v.config.eval_state_path.unlink(missing_ok=True)
            v.config.pack_first_seen_path.unlink(missing_ok=True)

    def test_pack_first_seen_persistence_roundtrip(self):
        """pack_first_seen + pack_last_seen_window are persisted together."""
        v = self._make_validator()
        v.pack_first_seen = {
            "hash_a": ("hk_0", 98500),
            "hash_b": ("hk_1", 98800),
        }
        v.pack_last_seen_window = {"hash_a": 13, "hash_b": 17}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.eval_state_path = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.pack_first_seen_path = Path(f.name)

        try:
            v._save_eval_state()

            v2 = self._make_validator()
            v2.config.eval_state_path = v.config.eval_state_path
            v2.config.pack_first_seen_path = v.config.pack_first_seen_path
            v2._load_eval_state()

            assert v2.pack_first_seen == {
                "hash_a": ("hk_0", 98500),
                "hash_b": ("hk_1", 98800),
            }
            assert v2.pack_last_seen_window == {"hash_a": 13, "hash_b": 17}
            # Independent verification via the helper.
            loaded_table, loaded_last_seen = load_pack_first_seen(
                v.config.pack_first_seen_path
            )
            assert loaded_table == {
                "hash_a": ("hk_0", 98500),
                "hash_b": ("hk_1", 98800),
            }
            assert loaded_last_seen == {"hash_a": 13, "hash_b": 17}
        finally:
            v.config.eval_state_path.unlink(missing_ok=True)
            v.config.pack_first_seen_path.unlink(missing_ok=True)

    def test_pack_first_seen_load_missing_file_defaults_empty(self):
        """pack_first_seen.json missing AND eval_state.json carries no
        legacy key → load yields an empty table."""
        v = self._make_validator()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.eval_state_path = Path(f.name)
        # Point at a path that does not exist.
        v.config.pack_first_seen_path = Path(
            tempfile.gettempdir()
        ) / "definitely_does_not_exist_pack_first_seen.json"
        v.config.pack_first_seen_path.unlink(missing_ok=True)

        try:
            legacy = {
                "spec_number": SPEC_NUMBER,
                "scenario_scores": {},
                "eval_pack_hash": {},
                "last_eval_block": {},
                "last_eval_window": -1,
                "last_eval_window_per_hotkey": {},
                "consensus_window": -1,
                "integrity_cache": {},
            }
            v.config.eval_state_path.write_text(json.dumps(legacy))

            v2 = self._make_validator()
            v2.config.eval_state_path = v.config.eval_state_path
            v2.config.pack_first_seen_path = v.config.pack_first_seen_path
            v2._load_eval_state()

            assert v2.pack_first_seen == {}
            assert v2.pack_last_seen_window == {}
        finally:
            v.config.eval_state_path.unlink(missing_ok=True)
            v.config.pack_first_seen_path.unlink(missing_ok=True)

    def test_pack_first_seen_legacy_migration_from_eval_state(self):
        """Legacy eval_state.json with embedded `pack_first_seen` key:
        loader migrates entries into the dedicated file, populates the
        in-memory table, and a subsequent save no longer rewrites the
        legacy key.
        """
        v = self._make_validator()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            v.config.eval_state_path = Path(f.name)
        v.config.pack_first_seen_path = Path(
            tempfile.gettempdir()
        ) / "test_legacy_migration_pack_first_seen.json"
        v.config.pack_first_seen_path.unlink(missing_ok=True)

        try:
            legacy = {
                "spec_number": SPEC_NUMBER,
                "scenario_scores": {},
                "eval_pack_hash": {},
                "last_eval_block": {},
                "last_eval_window": -1,
                "last_eval_window_per_hotkey": {},
                "consensus_window": -1,
                "integrity_cache": {},
                "pack_first_seen": {
                    "hash_a": ["hk_alice", 12345],
                    "hash_b": ["hk_bob", 67890],
                },
            }
            v.config.eval_state_path.write_text(json.dumps(legacy))

            v._load_eval_state()

            # (a) in-memory table populated from legacy block
            assert v.pack_first_seen == {
                "hash_a": ("hk_alice", 12345),
                "hash_b": ("hk_bob", 67890),
            }
            # Legacy v1 carries no last-seen tracker; the side dict
            # starts empty so the grace clock begins on the next sweep.
            assert v.pack_last_seen_window == {}
            # (b) dedicated pack_first_seen.json now exists with same data
            assert v.config.pack_first_seen_path.exists()
            loaded_table, loaded_last_seen = load_pack_first_seen(
                v.config.pack_first_seen_path
            )
            assert loaded_table == {
                "hash_a": ("hk_alice", 12345),
                "hash_b": ("hk_bob", 67890),
            }
            assert loaded_last_seen == {}

            # First eviction sweep must not drop migrated entries even
            # if they look orphaned: the clock has to start ticking now.
            evicted = evict_orphans(
                v.pack_first_seen,
                v.pack_last_seen_window,
                set(),
                current_window=100,
            )
            assert evicted == []
            assert v.pack_last_seen_window == {"hash_a": 100, "hash_b": 100}

            # (c) subsequent _save_eval_state must NOT rewrite legacy key
            v._save_eval_state()
            saved = json.loads(v.config.eval_state_path.read_text())
            assert "pack_first_seen" not in saved
        finally:
            v.config.eval_state_path.unlink(missing_ok=True)
            v.config.pack_first_seen_path.unlink(missing_ok=True)

    def test_pack_first_seen_ownership_lock_first_writer_wins(self):
        """First hotkey to claim a pack_hash owns it; subsequent claims by
        other hotkeys return the original owner unchanged. Mirrors the
        per-miner loop in ``_execute_evaluation_cycle`` where copies get
        short-circuited to weight 0.
        """
        v = self._make_validator()

        owner, blk = claim_owner(v.pack_first_seen, "hash_x", "hk_owner", 1000)
        assert (owner, blk) == ("hk_owner", 1000)

        owner2, blk2 = claim_owner(
            v.pack_first_seen, "hash_x", "hk_copy", 1500,
        )
        assert (owner2, blk2) == ("hk_owner", 1000), (
            "Ownership must not transfer to later submitters"
        )
        assert v.pack_first_seen["hash_x"] == ("hk_owner", 1000)

    def test_pack_first_seen_no_succession_when_owner_gone(self):
        """Owner hotkey gone but entry not yet evicted: copies still see
        the dead owner and are treated as copies (no inheritance).
        """
        v = self._make_validator()
        v.pack_first_seen["hash_x"] = ("hk_dead_owner", 500)

        owner, _ = claim_owner(v.pack_first_seen, "hash_x", "hk_new", 2000)
        assert owner == "hk_dead_owner"
        assert owner != "hk_new"

    def test_pack_first_seen_eviction_grace_window_keeps_then_drops(self):
        """``evict_orphans`` honors the grace window: an orphaned entry
        is only dropped after `EVICTION_GRACE_WINDOWS` consecutive
        windows of inactivity. The first sweep starts the clock.
        """
        v = self._make_validator()
        v.pack_first_seen = {
            "hash_active": ("hk_a", 100),
            "hash_orphan": ("hk_b", 200),
        }
        v.pack_last_seen_window = {}

        # Window 50: first sweep. Active hash kept and clocked; orphan
        # has its clock initialized — NOT evicted on first observation.
        evicted = evict_orphans(
            v.pack_first_seen,
            v.pack_last_seen_window,
            {"hash_active"},
            current_window=50,
        )
        assert evicted == []
        assert v.pack_last_seen_window == {"hash_active": 50, "hash_orphan": 50}

        # Window 50 + (grace - 1): still inside grace → kept.
        evicted = evict_orphans(
            v.pack_first_seen,
            v.pack_last_seen_window,
            {"hash_active"},
            current_window=50 + EVICTION_GRACE_WINDOWS - 1,
        )
        assert evicted == []
        assert "hash_orphan" in v.pack_first_seen

        # Window 50 + grace: boundary reached → evict orphan, keep active.
        evicted = evict_orphans(
            v.pack_first_seen,
            v.pack_last_seen_window,
            {"hash_active"},
            current_window=50 + EVICTION_GRACE_WINDOWS,
        )
        assert evicted == ["hash_orphan"]
        assert v.pack_first_seen == {"hash_active": ("hk_a", 100)}
        assert "hash_orphan" not in v.pack_last_seen_window

    def test_pack_first_seen_resurrection_after_eviction(self):
        """After the full grace window with no active reference, a
        brand-new submitter claims the pack_hash.
        """
        v = self._make_validator()
        v.pack_first_seen["hash_x"] = ("hk_old_owner", 500)
        v.pack_last_seen_window["hash_x"] = 10

        # Within grace: no eviction yet.
        evict_orphans(
            v.pack_first_seen,
            v.pack_last_seen_window,
            set(),
            current_window=10 + EVICTION_GRACE_WINDOWS - 1,
        )
        assert "hash_x" in v.pack_first_seen

        # Past grace: evicted.
        evict_orphans(
            v.pack_first_seen,
            v.pack_last_seen_window,
            set(),
            current_window=10 + EVICTION_GRACE_WINDOWS,
        )
        assert "hash_x" not in v.pack_first_seen

        # Resurrection: brand-new submitter claims ownership.
        owner, blk = claim_owner(v.pack_first_seen, "hash_x", "hk_new", 9000)
        assert (owner, blk) == ("hk_new", 9000)

    def test_pack_first_seen_grace_window_clock_reset(self):
        """A re-activation inside the grace window resets the clock,
        forcing a full new grace span before eviction. Drives the
        validator-level dicts directly to mirror cycle progression.
        """
        v = self._make_validator()
        v.pack_first_seen["hash_x"] = ("hk_owner", 100)
        v.pack_last_seen_window["hash_x"] = 10

        # Window 15: still orphaned, well inside grace.
        evict_orphans(
            v.pack_first_seen, v.pack_last_seen_window,
            set(), current_window=15,
        )
        assert "hash_x" in v.pack_first_seen
        assert v.pack_last_seen_window["hash_x"] == 10  # untouched

        # Window 16: owner re-submits → clock resets to 16.
        evict_orphans(
            v.pack_first_seen, v.pack_last_seen_window,
            {"hash_x"}, current_window=16,
        )
        assert v.pack_last_seen_window["hash_x"] == 16

        # Window 16 + grace - 1: still inside fresh grace → keep.
        evict_orphans(
            v.pack_first_seen, v.pack_last_seen_window,
            set(), current_window=16 + EVICTION_GRACE_WINDOWS - 1,
        )
        assert "hash_x" in v.pack_first_seen

        # Window 16 + grace: full new grace elapsed → evict.
        evicted = evict_orphans(
            v.pack_first_seen, v.pack_last_seen_window,
            set(), current_window=16 + EVICTION_GRACE_WINDOWS,
        )
        assert evicted == ["hash_x"]
        assert "hash_x" not in v.pack_first_seen

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
            config.weight_interval_blocks = 360
            config.cost_delta = 0.10
            config.required_categories = ["safety", "correctness"]
            config.eval_state_path = Path("/tmp/test_eval_state.json")
            config.pack_first_seen_path = Path("/tmp/test_pack_first_seen.json")

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

    # ------------------------------------------------------------------
    # Cross-spec winner protection bypass
    # ------------------------------------------------------------------

    def test_winner_protection_bypassed_on_cross_spec_transition(self):
        """When state.spec_number != target_spec_number, the δ threshold is
        skipped and the highest-scoring eligible miner is elected."""
        state = WinnerState(
            winner_hotkey="m1",
            winner_score=0.9,  # high score, would normally defend
            spec_number=3,
        )
        consensus_scores = {"m1": 0.4, "m2": 0.5}  # both far below 0.9*1.10
        winner, new_state = select_winner_with_protection(
            consensus_scores=consensus_scores,
            state=state,
            score_delta=0.10,
            target_spec_number=4,
        )
        assert winner == "m2"
        assert new_state.winner_hotkey == "m2"
        assert new_state.winner_score == 0.5
        assert new_state.spec_number == 4

    def test_winner_protection_active_when_specs_match(self):
        """When state.spec_number == target_spec_number, normal δ rules apply."""
        state = WinnerState(
            winner_hotkey="m1",
            winner_score=0.9,
            spec_number=4,
        )
        consensus_scores = {"m1": 0.85, "m2": 0.5}  # m2 nowhere near 0.9*1.10
        winner, new_state = select_winner_with_protection(
            consensus_scores=consensus_scores,
            state=state,
            score_delta=0.10,
            target_spec_number=4,
        )
        assert winner == "m1"
        assert new_state.winner_hotkey == "m1"
        assert new_state.winner_score == 0.9  # frozen
        assert new_state.spec_number == 4

    def test_new_state_carries_target_spec_number(self):
        """Elect / overtake / self-update branches all stamp target_spec_number."""
        # Elect (no previous winner)
        elect_state = WinnerState(spec_number=2)
        _, new_elect = select_winner_with_protection(
            consensus_scores={"m1": 0.7},
            state=elect_state,
            score_delta=0.10,
            target_spec_number=5,
        )
        assert new_elect.spec_number == 5

        # Overtake (challenger clears δ threshold under same spec)
        overtake_state = WinnerState(
            winner_hotkey="m1", winner_score=0.5, spec_number=5,
        )
        _, new_overtake = select_winner_with_protection(
            consensus_scores={"m1": 0.5, "m2": 0.99},  # 0.99 > 0.5*1.10
            state=overtake_state,
            score_delta=0.10,
            target_spec_number=5,
        )
        assert new_overtake.winner_hotkey == "m2"
        assert new_overtake.spec_number == 5

        # Self-update (winner beats own threshold)
        selfup_state = WinnerState(
            winner_hotkey="m1", winner_score=0.5, spec_number=5,
        )
        _, new_selfup = select_winner_with_protection(
            consensus_scores={"m1": 0.99},  # 0.99 > 0.5*1.10
            state=selfup_state,
            score_delta=0.10,
            target_spec_number=5,
        )
        assert new_selfup.winner_hotkey == "m1"
        assert new_selfup.winner_score == 0.99
        assert new_selfup.spec_number == 5


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


class TestIssue1EpochSkipSemantics:
    """Tests for dual-window orchestration and quorum-gated aggregation."""

    def _make_validator(self):
        v = TrajectoryValidator.__new__(TrajectoryValidator)
        v.config = MagicMock()
        v.config.full_cycle_on_startup = False
        v.config.aggregate_when_start = False
        v.config.weight_interval_blocks = 999999
        v.config.pack_cache_max_size = 100
        v.config.quorum_threshold = 0.5
        v.config.netuid = 11
        v.config.log_level = "WARNING"

        v._window_config = WindowConfig(
            window_length=100,
            global_anchor=0,
            publish_pct=0.8,
            aggregate_pct=0.9,
        )

        v._last_eval_window = -1
        v._consensus_window = -1
        v._target_window = -1
        v._target_submit_done = False
        v._waiting_for_quorum = False
        v._last_eval_at = None
        v._last_eval_date = None
        v.last_weight_block = 0

        v._cycle_eval_id = None
        v._cycle_log_offset = 0
        v._cycle_log_block = 0
        v._cycle_window_number = 0

        v.wallet = MagicMock()
        v.wallet.hotkey.ss58_address = "validator_hotkey"
        v.subtensor = MagicMock()
        v.metagraph = MagicMock()
        v.metagraph.hotkeys = ["v1", "v2", "v3"]
        v.metagraph.validator_permit = [True, True, True]
        v.metagraph.stake = [60.0, 30.0, 10.0]
        v.metagraph.n = 3

        v.pack_fetcher = MagicMock()
        v._sandbox_harness = MagicMock()
        v._sandbox_harness.bench_image_hash = "bench"
        v._sandbox_harness.harness_image_hash = "harness"
        v._sandbox_harness.sandbox_version = "vtest"

        v._replay_pending_uploads = AsyncMock()
        v._run_evaluation_cycle = AsyncMock()
        v._submit_consensus_payload = AsyncMock(return_value=True)
        v._run_consensus_aggregation = AsyncMock()
        v._set_winner_weights = AsyncMock()
        v._set_burn_weights = AsyncMock()
        v._set_fallback_weights = AsyncMock()
        v._save_eval_state = MagicMock()
        v._check_own_commitment_on_chain = MagicMock(return_value=False)
        v._is_metagraph_healthy = MagicMock(return_value=True)
        v._sync_metagraph = MagicMock(return_value=True)
        v._fire_upload_cycle_logs = AsyncMock()
        v._heartbeat_loop = AsyncMock()
        return v

    def test_compute_quorum_status_uses_effective_spec(self):
        v = self._make_validator()
        commitments = [
            ValidatorConsensusCommitment(
                protocol_version=2,
                window_number=9,
                content_address="cid-1",
                validator_hotkey="v1",
                block_number=100,
                raw="r1",
                spec_number=7,
            ),
            ValidatorConsensusCommitment(
                protocol_version=2,
                window_number=9,
                content_address="cid-2",
                validator_hotkey="v2",
                block_number=101,
                raw="r2",
                spec_number=7,
            ),
            ValidatorConsensusCommitment(
                protocol_version=2,
                window_number=9,
                content_address="cid-3",
                validator_hotkey="v3",
                block_number=102,
                raw="r3",
                spec_number=4,
            ),
        ]
        with patch(
            "trajectoryrl.base.validator.fetch_validator_consensus_commitments",
            return_value=commitments,
        ):
            meets, ratio, submitted, total = v._compute_quorum_status(9)

        assert meets is True
        assert ratio == pytest.approx(0.9)
        assert submitted == pytest.approx(90.0)
        assert total == pytest.approx(100.0)

    def test_submit_blocked_in_evaluation_phase(self):
        """Submission must NOT fire while we're still in the evaluation phase.

        The old behaviour was "phase-decoupled": as soon as eval completed
        (set ``_last_eval_window``) the next loop tick committed to chain
        regardless of phase. This is unsafe — if eval terminates
        pathologically fast (e.g. chain query returned 0 commitments due
        to ws timeout) the validator commits stale data immediately and
        locks itself out of the window. Submission is now gated on
        ``window.phase != EVALUATION`` so the cycle has time to either
        produce real data or be retried.
        """
        v = self._make_validator()
        aligned = 7986780 + ((100 - (7986780 % 100)) % 100)
        block = aligned + 10  # evaluation phase (offset 10/100)
        v.subtensor.get_current_block.side_effect = [block, block]
        target = block // 100
        v._target_window = target
        v._last_eval_window = target
        v._consensus_window = target - 1

        def _close_task(coro):
            coro.close()
            return MagicMock()

        with patch("trajectoryrl.base.validator.asyncio.create_task", side_effect=_close_task), \
             patch("trajectoryrl.base.validator.asyncio.sleep", AsyncMock(side_effect=KeyboardInterrupt())):
            asyncio.run(v.run())

        assert v._submit_consensus_payload.await_count == 0
        assert v._target_submit_done is False

    def test_submit_fires_in_propagation_phase(self):
        """Submission fires once we cross into propagation phase (>= 80%)."""
        v = self._make_validator()
        aligned = 7986780 + ((100 - (7986780 % 100)) % 100)
        block = aligned + 85  # propagation phase (offset 85/100, between 80 and 90)
        v.subtensor.get_current_block.side_effect = [block, block]
        target = block // 100
        v._target_window = target
        v._last_eval_window = target
        v._consensus_window = target - 1

        def _close_task(coro):
            coro.close()
            return MagicMock()

        with patch("trajectoryrl.base.validator.asyncio.create_task", side_effect=_close_task), \
             patch("trajectoryrl.base.validator.asyncio.sleep", AsyncMock(side_effect=KeyboardInterrupt())):
            asyncio.run(v.run())

        assert v._submit_consensus_payload.await_count == 1
        assert v._target_submit_done is True

    def test_phase_gate_logic(self):
        """Unit test for the phase-gate decision controlling submit.

        Two bypasses to the ``EVALUATION`` block: ``target_already_passed``
        (eval finished after the physical window advanced) and any non-
        evaluation phase. Otherwise we hold submission until propagation
        phase to give the eval cycle time to either produce real data or
        be retried (defends against ws-timeout fast-fail submitting empty
        payloads).
        """
        from trajectoryrl.utils.eval_window import WindowPhase

        def gate(target, physical, phase):
            target_already_passed = target < physical
            return (
                target_already_passed
                or phase != WindowPhase.EVALUATION
            )

        # Blocked in evaluation phase when target == physical
        assert gate(1123, 1123, WindowPhase.EVALUATION) is False
        # Allowed in propagation phase
        assert gate(1123, 1123, WindowPhase.PROPAGATION) is True
        # Allowed in aggregation phase
        assert gate(1123, 1123, WindowPhase.AGGREGATION) is True
        # Cross-window: target already passed → bypass
        assert gate(1123, 1124, WindowPhase.EVALUATION) is True

    def test_submit_fires_when_target_window_already_passed(self):
        """Phase gate is bypassed when target_window < physical_window.

        Late-finish path: eval may complete after the target window has
        rolled into the next window's evaluation phase. The phase gate
        uses *current* window's phase, so without the bypass we'd be
        stuck unable to publish the prior-window payload. The bypass
        ensures published-late payloads always go through.
        """
        v = self._make_validator()
        aligned = 7986780 + ((100 - (7986780 % 100)) % 100)
        # Physical = target+1, offset 10 (next window's evaluation phase).
        # Without the target_already_passed bypass, the EVALUATION-phase
        # gate would block submission of the prior-window payload.
        block = aligned + 1 * 100 + 10
        v.subtensor.get_current_block.side_effect = [block, block]
        target = block // 100 - 1   # one window in the past
        v._target_window = target
        v._last_eval_window = target
        v._consensus_window = target - 1

        def _close_task(coro):
            coro.close()
            return MagicMock()

        with patch("trajectoryrl.base.validator.asyncio.create_task", side_effect=_close_task), \
             patch("trajectoryrl.base.validator.asyncio.sleep", AsyncMock(side_effect=KeyboardInterrupt())):
            asyncio.run(v.run())

        assert v._submit_consensus_payload.await_count == 1
        assert v._target_submit_done is True

    def test_quorum_miss_keeps_target_and_sets_burn_weights(self):
        v = self._make_validator()
        aligned = 7986780 + ((100 - (7986780 % 100)) % 100)
        block = aligned + 3 * 100 + 95  # physical window +3, aggregation phase
        v.subtensor.get_current_block.side_effect = [block, block]
        v._target_window = 0
        v._last_eval_window = 0
        v._target_submit_done = True
        v._consensus_window = -1
        v._compute_quorum_status = MagicMock(
            return_value=(False, 0.49, 49.0, 100.0)
        )

        def _close_task(coro):
            coro.close()
            return MagicMock()

        with patch("trajectoryrl.base.validator.asyncio.create_task", side_effect=_close_task), \
             patch("trajectoryrl.base.validator.asyncio.sleep", AsyncMock(side_effect=KeyboardInterrupt())):
            asyncio.run(v.run())

        assert v._set_burn_weights.await_count >= 1
        assert v._target_window == 0
        assert v._waiting_for_quorum is True

    def test_quorum_success_jumps_target_to_physical_window(self):
        v = self._make_validator()
        aligned = 7986780 + ((100 - (7986780 % 100)) % 100)
        block = aligned + 3 * 100 + 95  # physical window +3, aggregation phase
        v.subtensor.get_current_block.side_effect = [block, block]
        v._target_window = 0
        v._last_eval_window = 0
        v._target_submit_done = True
        v._consensus_window = -1
        v._compute_quorum_status = MagicMock(
            return_value=(True, 0.8, 80.0, 100.0)
        )

        async def _agg(window):
            v._consensus_window = window.window_number

        v._run_consensus_aggregation = AsyncMock(side_effect=_agg)
        v._check_own_commitment_on_chain.return_value = False

        def _close_task(coro):
            coro.close()
            return MagicMock()

        with patch("trajectoryrl.base.validator.asyncio.create_task", side_effect=_close_task), \
             patch("trajectoryrl.base.validator.asyncio.sleep", AsyncMock(side_effect=KeyboardInterrupt())):
            asyncio.run(v.run())

        assert v._run_consensus_aggregation.await_count == 1
        assert v._target_window == (block // 100)
        assert v._waiting_for_quorum is False

    def test_quorum_gate_skipped_when_target_not_yet_submitted(self):
        # After a successful aggregation in window N's agg phase, target
        # bumps to N+1 on the next iteration but the validator hasn't
        # evaluated/submitted for N+1 yet. The agg gate must NOT fire on
        # that freshly-bumped target — otherwise it trivially misses
        # quorum (zero / tiny stake at N+1) and latches
        # _waiting_for_quorum=True, which then poisons the next eval phase
        # via the main-loop tempo refresh's burn branch.
        v = self._make_validator()
        aligned = 7986780 + ((100 - (7986780 % 100)) % 100)
        block = aligned + 3 * 100 + 95  # physical window +3, aggregation phase
        v.subtensor.get_current_block.side_effect = [block, block]
        v._target_window = (block // 100) + 1   # bumped past physical
        v._last_eval_window = block // 100      # evaluated only the prior window
        v._consensus_window = block // 100      # just succeeded on the prior window
        v._target_submit_done = False           # ← key: not submitted for the new target

        v._compute_quorum_status = MagicMock(
            return_value=(False, 0.0, 0.0, 100.0)
        )

        def _close_task(coro):
            coro.close()
            return MagicMock()

        with patch("trajectoryrl.base.validator.asyncio.create_task", side_effect=_close_task), \
             patch("trajectoryrl.base.validator.asyncio.sleep", AsyncMock(side_effect=KeyboardInterrupt())):
            asyncio.run(v.run())

        v._compute_quorum_status.assert_not_called()
        assert v._set_burn_weights.await_count == 0
        assert v._waiting_for_quorum is False


class TestAggregateOnStartupConsensusWindowPersist:
    """Regression: startup aggregation must persist _consensus_window on success."""

    def _make_minimal_validator(self):
        v = TrajectoryValidator.__new__(TrajectoryValidator)
        v.config = MagicMock(netuid=11)
        v.subtensor = MagicMock()
        v.subtensor.get_current_block.return_value = 999
        v.metagraph = MagicMock(hotkeys=[])
        v._window_config = WindowConfig(
            window_length=7200,
            global_anchor=0,
            publish_pct=0.8,
            aggregate_pct=0.9,
        )
        v._consensus_window = -1
        v.last_weight_block = 0
        v._winner_state = WinnerState(winner_hotkey="5abcd")
        v._sync_metagraph = MagicMock(return_value=True)
        v._save_eval_state = MagicMock()
        v._set_winner_weights = AsyncMock()
        return v

    def test_aggregate_on_startup_success_keeps_consensus_window(self):
        v = self._make_minimal_validator()
        vc = ValidatorConsensusCommitment(
            protocol_version=2,
            window_number=1121,
            content_address="cid",
            validator_hotkey="vk",
            block_number=1,
            raw="raw",
            spec_number=4,
        )

        async def fake_agg(window):
            v._consensus_window = window.window_number

        v._run_consensus_aggregation = AsyncMock(side_effect=fake_agg)

        with patch(
            "trajectoryrl.base.validator.fetch_validator_consensus_commitments",
            return_value=[vc],
        ):
            asyncio.run(v._aggregate_on_startup())

        assert v._consensus_window == 1121
        v._save_eval_state.assert_called_once()
        v._set_winner_weights.assert_awaited_once()

    def test_aggregate_on_startup_failure_restores_prior_consensus_window(self):
        v = self._make_minimal_validator()
        v._consensus_window = 1120
        vc = ValidatorConsensusCommitment(
            protocol_version=2,
            window_number=1121,
            content_address="cid",
            validator_hotkey="vk",
            block_number=1,
            raw="raw",
            spec_number=4,
        )

        async def fake_agg(_window):
            pass

        v._run_consensus_aggregation = AsyncMock(side_effect=fake_agg)

        with patch(
            "trajectoryrl.base.validator.fetch_validator_consensus_commitments",
            return_value=[vc],
        ):
            asyncio.run(v._aggregate_on_startup())

        assert v._consensus_window == 1120
        v._save_eval_state.assert_called_once()
        v._set_winner_weights.assert_not_called()


class TestRunConsensusAggregationQuorumGate:
    """Regression: ``_run_consensus_aggregation`` must respect quorum.

    The main loop checks quorum via ``_compute_quorum_status`` BEFORE invoking
    ``_run_consensus_aggregation`` (validator.py:1503-1548), so that path is
    safe. But ``_aggregate_on_startup`` and ``_full_cycle_on_startup`` invoke
    ``_run_consensus_aggregation`` directly with no upstream gate. Without an
    internal gate, a single stale on-chain submission (e.g. our own pointer
    left behind from a prior failed cycle) is enough to bump
    ``_consensus_window``, which then forces ``target_window`` past the
    actual physical window and locks the validator out of the real eval.

    This was the second of two bugs that caused the 2026-05-01 window-1123
    incident on UID 221. The first bug (ws-failure-poisons-snapshot) was
    fixed in PR #213; this regression locks in the second.
    """

    def _make_validator(self):
        v = TrajectoryValidator.__new__(TrajectoryValidator)
        v.config = MagicMock(
            netuid=11, quorum_threshold=0.5,
        )
        v.subtensor = MagicMock()
        v.metagraph = MagicMock(hotkeys=[], n=10)
        v._consensus_window = 1122
        v._is_metagraph_healthy = MagicMock(return_value=True)
        v._sync_metagraph = MagicMock(return_value=True)
        v._compute_quorum_status = MagicMock()
        v._consensus_store = MagicMock()
        v.wallet = MagicMock()
        v.wallet.hotkey.ss58_address = "5ValidatorOwn"
        return v

    def test_aggregation_skipped_when_quorum_not_met(self):
        """Without quorum, ``_run_consensus_aggregation`` must return early
        and leave ``_consensus_window`` untouched."""
        v = self._make_validator()
        v._compute_quorum_status.return_value = (False, 0.10, 100.0, 1000.0)
        window = MagicMock(window_number=1123)

        with patch(
            "trajectoryrl.base.validator.fetch_validator_consensus_commitments"
        ) as fetch_mock:
            asyncio.run(v._run_consensus_aggregation(window))

        assert v._consensus_window == 1122, (
            "consensus_window must NOT advance when quorum is not met "
            "(otherwise a single stale submission can lock the validator out "
            "of the real eval cycle for that window)"
        )
        assert not fetch_mock.called, (
            "should bail before fetching commitments when quorum gate fails"
        )
        v._compute_quorum_status.assert_called_once_with(1123)

    def test_aggregation_proceeds_when_quorum_met(self):
        """When quorum is met, the function must proceed past the gate
        (we verify by observing the chain fetch is called)."""
        v = self._make_validator()
        v._compute_quorum_status.return_value = (True, 0.60, 600.0, 1000.0)
        window = MagicMock(window_number=1123)

        with patch(
            "trajectoryrl.base.validator.fetch_validator_consensus_commitments",
            return_value=[],  # empty downstream so the function still bails
        ) as fetch_mock:
            asyncio.run(v._run_consensus_aggregation(window))

        assert fetch_mock.called, (
            "should proceed to fetch chain commitments when quorum is met"
        )


class TestRunEvaluationCycleReturnPropagation:
    """Regression: ``_run_evaluation_cycle`` must propagate the inner cycle's
    return value. PR #213 changed ``_execute_evaluation_cycle`` to return
    ``False`` on snapshot-fetch failure so the caller can skip the
    ``_last_eval_window`` bump and avoid locking the validator out of the
    window. The wrapper had ``await self._execute_evaluation_cycle(...)``
    rather than ``return await ...`` — the False was silently swallowed,
    the caller's ``if cycle_result is False:`` branch never fired, and
    every snapshot failure still poisoned the window. This test pins the
    propagation behaviour."""

    def _make_validator(self):
        v = TrajectoryValidator.__new__(TrajectoryValidator)
        v.config = MagicMock()
        v._cycle_eval_id = None
        v._cycle_log_offset = 0
        v._cycle_log_block = 0
        v._cycle_window_number = 0
        v._get_validator_log_offset = MagicMock(return_value=0)
        return v

    def test_returns_false_when_inner_returns_false(self):
        v = self._make_validator()
        v._execute_evaluation_cycle = AsyncMock(return_value=False)
        result = asyncio.run(v._run_evaluation_cycle(123, 1123))
        assert result is False, (
            "wrapper must propagate inner False so caller skips the "
            "_last_eval_window state update"
        )

    def test_returns_none_when_inner_returns_none(self):
        v = self._make_validator()
        v._execute_evaluation_cycle = AsyncMock(return_value=None)
        result = asyncio.run(v._run_evaluation_cycle(123, 1123))
        assert result is None, (
            "wrapper must propagate inner success (None) unchanged"
        )


class TestTopNRecheck:
    """Tests for the anti-cheese top-N cache wipe at eval cycle start.

    Mirrors the lightweight validator factory used by TestPerScenarioEvalState
    so each test is self-contained with the minimum state needed by
    `_wipe_top_n_for_recheck` and `_save_eval_state`.
    """

    def _make_validator(self):
        v = TrajectoryValidator.__new__(TrajectoryValidator)
        v.scenario_scores = {}
        v._eval_pack_hash = {}
        v.last_eval_block = {}
        v.last_eval_window = {}
        v._eval_counts = {}
        v.latest_token_usage = {}
        v.latest_model_usage = {}
        v._save_eval_state = MagicMock()
        return v

    def _seed(self, v, hotkey, scores):
        v.scenario_scores[hotkey] = dict(scores)
        v._eval_pack_hash[hotkey] = f"hash_{hotkey}"
        v.last_eval_block[hotkey] = 100
        v.last_eval_window[hotkey] = 10
        v._eval_counts[hotkey] = 1
        v.latest_token_usage[hotkey] = {"client_escalation": {"input_tokens": 1}}
        v.latest_model_usage[hotkey] = {"client_escalation": [{"name": "x"}]}

    def test_wipes_top_3_by_mean(self):
        v = self._make_validator()
        # Means: hk_a=0.9, hk_b=0.8, hk_c=0.7, hk_d=0.6, hk_e=0.5
        for hk, mean in [
            ("hk_a", 0.9), ("hk_b", 0.8), ("hk_c", 0.7),
            ("hk_d", 0.6), ("hk_e", 0.5),
        ]:
            self._seed(v, hk, {"s1": mean, "s2": mean})

        wiped = v._wipe_top_n_for_recheck(window_number=42)

        assert sorted(wiped) == ["hk_a", "hk_b", "hk_c"]
        assert "hk_a" not in v.scenario_scores
        assert "hk_b" not in v.scenario_scores
        assert "hk_c" not in v.scenario_scores
        assert v.scenario_scores["hk_d"] == {"s1": 0.6, "s2": 0.6}
        assert v.scenario_scores["hk_e"] == {"s1": 0.5, "s2": 0.5}

    def test_no_active_set_filter(self):
        """Selection is from scenario_scores directly, not filtered by
        any external active-set parameter. (Regression check that the
        signature is window_number-only.)"""
        v = self._make_validator()
        for hk, mean in [("hk_a", 0.9), ("hk_b", 0.8), ("hk_c", 0.7)]:
            self._seed(v, hk, {"s1": mean})

        wiped = v._wipe_top_n_for_recheck(window_number=42)

        assert sorted(wiped) == ["hk_a", "hk_b", "hk_c"]
        # Active commitments are not consulted; method takes only window_number.

    def test_tiebreak_by_hotkey_ascending(self):
        v = self._make_validator()
        # All four hotkeys have mean 0.8. Top 3 must be the lex-smallest.
        for hk in ["hk_d", "hk_a", "hk_c", "hk_b"]:
            self._seed(v, hk, {"s1": 0.8})

        wiped = v._wipe_top_n_for_recheck(window_number=42)

        assert wiped == ["hk_a", "hk_b", "hk_c"]
        assert "hk_d" in v.scenario_scores

    def test_fewer_than_n_candidates(self):
        v = self._make_validator()
        self._seed(v, "hk_a", {"s1": 0.9})
        self._seed(v, "hk_b", {"s1": 0.8})

        wiped = v._wipe_top_n_for_recheck(window_number=42)

        assert sorted(wiped) == ["hk_a", "hk_b"]
        assert v.scenario_scores == {}

    def test_empty_cache_noop(self):
        v = self._make_validator()

        wiped = v._wipe_top_n_for_recheck(window_number=42)

        assert wiped == []
        v._save_eval_state.assert_not_called()

    def test_skips_hotkeys_with_empty_scenario_dict(self):
        """A hotkey present in scenario_scores but with an empty inner dict
        must not be selectable (mean is undefined; treat as no cache)."""
        v = self._make_validator()
        v.scenario_scores["hk_empty"] = {}
        self._seed(v, "hk_a", {"s1": 0.5})

        wiped = v._wipe_top_n_for_recheck(window_number=42)

        assert wiped == ["hk_a"]
        assert "hk_empty" in v.scenario_scores

    def test_wipe_clears_all_state_stores(self):
        v = self._make_validator()
        self._seed(v, "hk_a", {"s1": 0.9})

        v._wipe_top_n_for_recheck(window_number=42)

        for store_name in (
            "scenario_scores", "_eval_pack_hash",
            "last_eval_block", "last_eval_window",
            "_eval_counts", "latest_token_usage", "latest_model_usage",
        ):
            assert "hk_a" not in getattr(v, store_name), (
                f"{store_name} still contains hk_a after wipe"
            )

    def test_post_wipe_needs_evaluation_returns_true(self):
        """After wipe, _needs_evaluation should return True for the wiped
        hotkey on its (unchanged) pack_hash — confirming the wipe actually
        re-triggers evaluation."""
        v = self._make_validator()
        # _needs_evaluation calls self._get_miner_logger; stub it.
        v._get_miner_logger = MagicMock(return_value=MagicMock())
        self._seed(v, "hk_a", {"s1": 0.9})
        original_pack_hash = v._eval_pack_hash["hk_a"]

        assert v._needs_evaluation("hk_a", original_pack_hash, current_window=42) is False

        v._wipe_top_n_for_recheck(window_number=42)

        assert v._needs_evaluation("hk_a", original_pack_hash, current_window=42) is True

    def test_save_called_once_per_invocation(self):
        v = self._make_validator()
        for hk, mean in [("hk_a", 0.9), ("hk_b", 0.8), ("hk_c", 0.7)]:
            self._seed(v, hk, {"s1": mean})

        v._wipe_top_n_for_recheck(window_number=42)

        # One save covers all N wipes (vs N saves if we'd called the
        # per-miner _drop_miner_eval_state primitive in a loop).
        assert v._save_eval_state.call_count == 1


