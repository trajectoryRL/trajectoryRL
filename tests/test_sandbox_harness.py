"""Tests for the Season 1 trajectory-sandbox harness adapter.

Tests the harness adapter, config wiring, and result mapping without
requiring Docker or real LLM API calls.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass, field

from trajectoryrl.utils.config import ValidatorConfig
from trajectoryrl.utils.sandbox_harness import (
    TrajectorySandboxHarness,
    SandboxEvaluationResult,
)


# ---------------------------------------------------------------------------
# Helpers: mock trajectory-sandbox objects
# ---------------------------------------------------------------------------

@dataclass
class MockEpisodeResult:
    episode_index: int
    quality: float = 0.0
    tool_calls: int = 5
    transcript: str = ""
    mock_state: dict = field(default_factory=dict)
    error: str | None = None
    timed_out: bool = False
    duration_s: float = 10.0
    harness_stdout: str = ""
    harness_stderr: str = ""
    novel_calls: int = 0


@dataclass
class MockEvalSessionResult:
    episodes: list = field(default_factory=list)
    early_mean: float = 0.0
    late_mean: float = 0.0
    delta: float = 0.0
    mean_quality: float = 0.0
    learning_bonus: float = 0.0
    final_score: float = 0.0
    pack_hash: str = ""
    validator_salt: str = ""
    miner_uid: int | None = None
    scenario: str = ""
    fixture_hash: str = ""

    def compute_scores(self, alpha=0.5, early_floor=0.3, delta_threshold=0.4):
        scores = [ep.quality for ep in self.episodes]
        if len(scores) < 4:
            self.mean_quality = sum(scores) / len(scores) if scores else 0.0
            self.final_score = self.mean_quality
            return
        self.early_mean = (scores[0] + scores[1]) / 2
        self.late_mean = (scores[2] + scores[3]) / 2
        self.delta = self.late_mean - self.early_mean
        if self.early_mean < early_floor and self.delta > delta_threshold:
            self.delta = 0.0
        self.mean_quality = sum(scores) / len(scores)
        self.learning_bonus = alpha * max(0.0, self.delta)
        self.final_score = self.mean_quality * (1 + self.learning_bonus)


def _make_session_result(qualities: list[float]) -> MockEvalSessionResult:
    """Build a MockEvalSessionResult with given episode qualities."""
    result = MockEvalSessionResult(
        episodes=[
            MockEpisodeResult(episode_index=i, quality=q)
            for i, q in enumerate(qualities)
        ],
    )
    result.compute_scores()
    return result


# ---------------------------------------------------------------------------
# Tests: SandboxEvaluationResult mapping
# ---------------------------------------------------------------------------

class TestSandboxEvaluationResult:
    def test_maps_final_score(self):
        sr = _make_session_result([0.6, 0.7, 0.8, 0.9])
        result = SandboxEvaluationResult(sr)
        assert result.score == sr.final_score
        assert result.success is True

    def test_zero_score_not_qualified(self):
        sr = _make_session_result([0.0, 0.0, 0.0, 0.0])
        result = SandboxEvaluationResult(sr)
        assert result.score == 0.0
        assert result.success is False

    def test_episode_qualities_exposed(self):
        sr = _make_session_result([0.5, 0.6, 0.7, 0.8])
        result = SandboxEvaluationResult(sr)
        assert result.episode_qualities == [0.5, 0.6, 0.7, 0.8]

    def test_delta_and_means(self):
        sr = _make_session_result([0.4, 0.4, 0.8, 0.8])
        result = SandboxEvaluationResult(sr)
        assert abs(result.early_mean - 0.4) < 1e-9
        assert abs(result.late_mean - 0.8) < 1e-9
        assert abs(result.delta - 0.4) < 1e-9

    def test_tool_calls_summed(self):
        sr = _make_session_result([0.5, 0.5, 0.5, 0.5])
        result = SandboxEvaluationResult(sr)
        # Each mock episode has 5 tool_calls
        assert result.tool_calls == 20

    def test_error_field_default_none(self):
        sr = _make_session_result([0.5, 0.5, 0.5, 0.5])
        result = SandboxEvaluationResult(sr)
        assert result.error is None

    def test_cost_fields_none_for_s1(self):
        sr = _make_session_result([0.5, 0.5, 0.5, 0.5])
        result = SandboxEvaluationResult(sr)
        assert result.cost_usd is None
        assert result.token_usage is None


# ---------------------------------------------------------------------------
# Tests: Config wiring
# ---------------------------------------------------------------------------

class TestConfigWiring:
    def test_default_harness_is_clawbench(self):
        config = ValidatorConfig.__new__(ValidatorConfig)
        # Check the default without __post_init__
        assert ValidatorConfig.evaluation_harness == "clawbench"

    def test_season1_config_fields_exist(self):
        """Verify all S1 fields are on ValidatorConfig."""
        assert hasattr(ValidatorConfig, "evaluation_harness")
        assert hasattr(ValidatorConfig, "sandbox_image")
        assert hasattr(ValidatorConfig, "harness_image")
        assert hasattr(ValidatorConfig, "sandbox_timeout_per_episode")
        assert hasattr(ValidatorConfig, "sandbox_num_episodes")

    def test_season1_defaults(self):
        """Check default values for S1 config."""
        assert ValidatorConfig.sandbox_num_episodes == 4
        assert ValidatorConfig.sandbox_timeout_per_episode == 600


# ---------------------------------------------------------------------------
# Tests: extract_skill_md
# ---------------------------------------------------------------------------

class TestExtractSkillMd:
    def test_extracts_from_files(self):
        pack = {"files": {"SKILL.md": "# My Skill\nDo stuff."}}
        assert TrajectorySandboxHarness.extract_skill_md(pack) == "# My Skill\nDo stuff."

    def test_lowercase_fallback(self):
        pack = {"files": {"skill.md": "# lowercase"}}
        assert TrajectorySandboxHarness.extract_skill_md(pack) == "# lowercase"

    def test_returns_none_for_v4_pack(self):
        pack = {"files": {"AGENTS.md": "# Agent config"}}
        assert TrajectorySandboxHarness.extract_skill_md(pack) is None

    def test_returns_none_for_empty_files(self):
        pack = {"files": {}}
        assert TrajectorySandboxHarness.extract_skill_md(pack) is None

    def test_returns_none_for_no_files_key(self):
        pack = {"agents": []}
        assert TrajectorySandboxHarness.extract_skill_md(pack) is None


# ---------------------------------------------------------------------------
# Tests: TrajectorySandboxHarness with mocked trajectory-sandbox
# ---------------------------------------------------------------------------

class TestHarnessEvaluateMiner:
    """Test evaluate_miner with mocked EvalSession."""

    @pytest.fixture
    def mock_sandbox_modules(self):
        """Patch the lazy import to return mocks."""
        mock_session_cls = MagicMock()
        mock_config_cls = MagicMock()
        mock_factory_cls = MagicMock()
        mock_result_cls = MockEvalSessionResult
        mock_scorer_cls = MagicMock()
        mock_judge_cls = MagicMock()

        with patch(
            "trajectoryrl.utils.sandbox_harness._import_sandbox",
            return_value=(
                mock_session_cls,
                mock_config_cls,
                mock_factory_cls,
                mock_result_cls,
                mock_scorer_cls,
                mock_judge_cls,
            ),
        ) as mock_import:
            yield {
                "import": mock_import,
                "session_cls": mock_session_cls,
                "config_cls": mock_config_cls,
                "factory_cls": mock_factory_cls,
                "scorer_cls": mock_scorer_cls,
                "judge_cls": mock_judge_cls,
            }

    def _make_config(self, **overrides) -> ValidatorConfig:
        """Build a ValidatorConfig without triggering __post_init__ path checks."""
        config = ValidatorConfig.__new__(ValidatorConfig)
        defaults = {
            "wallet_name": "test",
            "wallet_hotkey": "test",
            "netuid": 11,
            "network": "test",
            "evaluation_harness": "trajectory-sandbox",
            "sandbox_image": "ghcr.io/trajectoryrl/trajectory-sandbox:latest",
            "harness_image": "nousresearch/hermes-agent:latest",
            "sandbox_timeout_per_episode": 600,
            "sandbox_num_episodes": 4,
            "judge_base_url": "https://openrouter.ai/api/v1",
            "judge_api_key": "test-key",
            "judge_model": "test/model",
            "clawbench_base_url": "",
            "clawbench_api_key": "",
            "clawbench_default_model": "",
        }
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(config, k, v)
        return config

    def test_harness_init(self, mock_sandbox_modules):
        config = self._make_config()
        harness = TrajectorySandboxHarness(config)
        assert harness is not None
        assert harness.config is config

    def test_evaluate_miner_runs_session(self, mock_sandbox_modules):
        config = self._make_config()
        harness = TrajectorySandboxHarness(config)

        # Mock the factory
        mock_world = MagicMock()
        mock_episode_fixtures = MagicMock()
        mock_episode_fixtures.instruction_md = "# Task"
        mock_episode_fixtures.to_dict.return_value = {"inbox": []}

        factory_instance = mock_sandbox_modules["factory_cls"].return_value
        factory_instance.generate_world.return_value = mock_world
        factory_instance.generate_episode.return_value = mock_episode_fixtures

        # Mock scorer
        mock_scorer = MagicMock()
        mock_sandbox_modules["scorer_cls"].for_incident_response.return_value = mock_scorer

        # Mock the session to return a good result
        session_result = _make_session_result([0.6, 0.7, 0.8, 0.9])

        mock_session_ctx = MagicMock()
        mock_session_ctx.__enter__ = MagicMock(return_value=mock_session_ctx)
        mock_session_ctx.__exit__ = MagicMock(return_value=False)
        mock_session_ctx.run_all_episodes.return_value = session_result
        mock_sandbox_modules["session_cls"].return_value = mock_session_ctx

        result = asyncio.get_event_loop().run_until_complete(
            harness.evaluate_miner(
                skill_md="# Test Skill",
                epoch_seed=42,
                pack_hash="abc123",
            )
        )

        assert result.score == session_result.final_score
        assert result.success is True
        assert result.error is None

    def test_evaluate_miner_handles_exception(self, mock_sandbox_modules):
        config = self._make_config()
        harness = TrajectorySandboxHarness(config)

        # Mock factory
        factory_instance = mock_sandbox_modules["factory_cls"].return_value
        factory_instance.generate_world.return_value = MagicMock()
        mock_ef = MagicMock()
        mock_ef.instruction_md = "# Task"
        mock_ef.to_dict.return_value = {}
        factory_instance.generate_episode.return_value = mock_ef

        mock_sandbox_modules["scorer_cls"].for_incident_response.return_value = MagicMock()
        mock_sandbox_modules["judge_cls"].return_value = MagicMock()

        # Session raises
        mock_sandbox_modules["session_cls"].side_effect = RuntimeError("Docker not available")

        result = asyncio.get_event_loop().run_until_complete(
            harness.evaluate_miner(
                skill_md="# Test",
                epoch_seed=1,
            )
        )

        assert result.score == 0.0
        assert result.success is False
        assert result.error is not None
        assert "Docker" in result.error
