"""Tests for the Season 1 trajrl-bench harness adapter.

Tests the result mapping, config wiring, and SKILL.md extraction
without requiring Docker or real LLM API calls.
"""

import pytest
from dataclasses import dataclass, field
from unittest.mock import MagicMock

from trajectoryrl.utils.config import ValidatorConfig
from trajectoryrl.utils.sandbox_harness import (
    TrajectorySandboxHarness,
    SandboxEvaluationResult,
    _SessionResult,
    _EpisodeResult,
    _EMPTY_TRANSCRIPT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_result(qualities: list[float]) -> _SessionResult:
    result = _SessionResult(
        episodes=[
            _EpisodeResult(episode_index=i, quality=q)
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
        sr = _SessionResult(episodes=[
            _EpisodeResult(episode_index=0, quality=0.5, tool_calls=5),
            _EpisodeResult(episode_index=1, quality=0.5, tool_calls=10),
        ])
        sr.compute_scores()
        result = SandboxEvaluationResult(sr)
        assert result.tool_calls == 15

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
# Tests: Split-half delta scoring
# ---------------------------------------------------------------------------

class TestSessionResultScoring:
    def test_basic_delta(self):
        sr = _make_session_result([0.4, 0.4, 0.8, 0.8])
        assert abs(sr.early_mean - 0.4) < 1e-9
        assert abs(sr.late_mean - 0.8) < 1e-9
        assert abs(sr.delta - 0.4) < 1e-9
        # final = 0.6 * (1 + 0.5 * 0.4) = 0.6 * 1.2 = 0.72
        assert abs(sr.final_score - 0.72) < 0.01

    def test_negative_delta_no_bonus(self):
        sr = _make_session_result([0.8, 0.8, 0.4, 0.4])
        assert sr.delta < 0
        assert sr.learning_bonus == 0.0
        assert abs(sr.final_score - 0.6) < 1e-9

    def test_anti_sandbagging(self):
        sr = _make_session_result([0.1, 0.1, 0.8, 0.8])
        # early_mean=0.1 < 0.3, delta=0.7 > 0.4 → delta zeroed
        assert sr.delta == 0.0
        assert abs(sr.final_score - sr.mean_quality) < 1e-9

    def test_fewer_than_4_episodes(self):
        sr = _make_session_result([0.5, 0.8])
        assert abs(sr.final_score - 0.65) < 0.01


# ---------------------------------------------------------------------------
# Tests: Config wiring
# ---------------------------------------------------------------------------

class TestConfigWiring:
    def test_default_harness_is_trajrl_bench(self):
        assert ValidatorConfig.evaluation_harness == "trajrl-bench"

    def test_season1_config_fields_exist(self):
        assert hasattr(ValidatorConfig, "evaluation_harness")
        assert hasattr(ValidatorConfig, "sandbox_image")
        assert hasattr(ValidatorConfig, "harness_image")
        assert hasattr(ValidatorConfig, "sandbox_timeout_per_episode")
        assert hasattr(ValidatorConfig, "sandbox_num_episodes")
        assert hasattr(ValidatorConfig, "sandbox_idle_timeout")

    def test_season1_defaults(self):
        assert ValidatorConfig.sandbox_num_episodes == 4
        assert ValidatorConfig.sandbox_timeout_per_episode == 180
        assert ValidatorConfig.sandbox_idle_timeout == 45


# ---------------------------------------------------------------------------
# Tests: Empty-transcript judge skip
# ---------------------------------------------------------------------------

class TestEmptyTranscriptThreshold:
    def test_threshold_is_reasonable(self):
        # Hermes startup boilerplate is ~130 chars; a working agent
        # always produces multi-kB transcripts. The threshold sits
        # between those two regimes.
        assert 130 < _EMPTY_TRANSCRIPT_THRESHOLD < 1000


# ---------------------------------------------------------------------------
# Tests: Orphan container cleanup
# ---------------------------------------------------------------------------

class TestOrphanCleanup:
    def _make_harness(self, tmp_path) -> TrajectorySandboxHarness:
        cfg = ValidatorConfig(
            pack_cache_dir=tmp_path / "packs",
            log_dir=tmp_path / "logs",
            eval_state_path=tmp_path / "eval_state.json",
            winner_state_path=tmp_path / "winner_state.json",
        )
        h = TrajectorySandboxHarness(cfg)
        h._docker_client = MagicMock()
        return h

    def test_cleanup_removes_labeled_containers_and_networks(self, tmp_path):
        h = self._make_harness(tmp_path)
        c1 = MagicMock(); c1.name = "sandbox_abc"
        c2 = MagicMock(); c2.name = "testee_abc_ep0"
        n1 = MagicMock(); n1.name = "eval_abc"
        h.client.containers.list.return_value = [c1, c2]
        h.client.networks.list.return_value = [n1]

        h._cleanup_orphan_containers()

        h.client.containers.list.assert_called_once_with(
            all=True, filters={"label": "trajectoryrl.role"},
        )
        h.client.networks.list.assert_called_once_with(
            filters={"label": "trajectoryrl.role"},
        )
        c1.remove.assert_called_once_with(force=True)
        c2.remove.assert_called_once_with(force=True)
        n1.remove.assert_called_once_with()

    def test_cleanup_swallows_per_item_failures(self, tmp_path):
        h = self._make_harness(tmp_path)
        bad = MagicMock(); bad.name = "bad"; bad.remove.side_effect = RuntimeError("nope")
        good = MagicMock(); good.name = "good"
        h.client.containers.list.return_value = [bad, good]
        h.client.networks.list.return_value = []

        # Must not raise even when one container removal fails.
        h._cleanup_orphan_containers()

        good.remove.assert_called_once_with(force=True)

    def test_cleanup_swallows_list_failure(self, tmp_path):
        h = self._make_harness(tmp_path)
        h.client.containers.list.side_effect = RuntimeError("docker down")
        h.client.networks.list.side_effect = RuntimeError("docker down")

        # Must not raise even if Docker is entirely unreachable.
        h._cleanup_orphan_containers()


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
