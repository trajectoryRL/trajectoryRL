"""Season 1 trajectory-sandbox harness for evaluating SKILL.md packs.

Wraps trajectory-sandbox's EvalSession to produce EvaluationResult objects
compatible with the existing validator pipeline. Replaces ClawBench for S1.

Season 1 differences from v4.0 (ClawBench):
  - Miners submit SKILL.md (static instruction pack) instead of pack.json
  - Agent runs in SSH sandbox with mock services (not OpenClaw tool-call API)
  - 4 episodes of same scenario with different fixtures (not 1 episode × N scenarios)
  - Split-half delta scoring: quality × (1 + 0.5 × max(0, delta))
  - 100% LLM judge scoring (no rule-based checks)
  - Default harness: Hermes Agent (not OpenClaw)

Usage:
    harness = TrajectorySandboxHarness(config)
    result = await harness.evaluate_miner(skill_md, epoch_seed)
    # result.score = final_score (split-half delta)
    # result.success = final_score > 0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from functools import partial
from typing import Any, Dict, List, Optional

from ..utils.config import ValidatorConfig

logger = logging.getLogger(__name__)


def _import_sandbox():
    """Lazy import trajectory-sandbox to keep it optional."""
    try:
        from trajectory_sandbox import (
            EvalSession,
            SandboxConfig,
            FixtureFactory,
            EvalSessionResult,
        )
        from trajectory_sandbox.episode_scorer import EpisodeScorer
        from trajectory_sandbox.judge import EpisodeJudge
        return EvalSession, SandboxConfig, FixtureFactory, EvalSessionResult, EpisodeScorer, EpisodeJudge
    except ImportError as e:
        raise ImportError(
            "trajectory-sandbox is required for Season 1 evaluation. "
            "Install with: pip install trajectoryrl[season1]"
        ) from e


class SandboxEvaluationResult:
    """Result from a Season 1 sandbox evaluation.

    Maps trajectory-sandbox's EvalSessionResult to the fields the validator
    pipeline needs. Intentionally duck-type compatible with
    clawbench.EvaluationResult for the fields the validator reads.
    """

    def __init__(self, session_result, scenario_name: str = "incident_response"):
        self.scenario_name = scenario_name
        self.score = session_result.final_score
        self.success = session_result.final_score > 0.0
        self.tool_calls = sum(ep.tool_calls for ep in session_result.episodes)
        self.response = ""
        self.rubric = {}
        self.error: Optional[str] = None

        # Cost: not directly tracked in S1 (quality-based, not cost-based)
        self.cost_usd: Optional[float] = None
        self.token_usage: Optional[Dict[str, int]] = None
        self.model_usage: Optional[List[Dict[str, Any]]] = None
        self.trajectory: Optional[List[Dict[str, Any]]] = None
        self.input_message: Optional[str] = None
        self.raw_llm_response: Optional[Dict[str, Any]] = None
        self.all_requests: Optional[List[Dict[str, Any]]] = None
        self.session_key: Optional[str] = None
        self.session_file: Optional[str] = None

        # S1-specific fields
        self.session_result = session_result
        self.early_mean = session_result.early_mean
        self.late_mean = session_result.late_mean
        self.delta = session_result.delta
        self.mean_quality = session_result.mean_quality
        self.learning_bonus = session_result.learning_bonus
        self.episode_qualities = [ep.quality for ep in session_result.episodes]


class TrajectorySandboxHarness:
    """Season 1 harness using trajectory-sandbox for SSH-based evaluations."""

    def __init__(self, config: ValidatorConfig):
        self.config = config

        # Lazy import — only fail when actually used
        (
            self._EvalSession,
            self._SandboxConfig,
            self._FixtureFactory,
            self._EvalSessionResult,
            self._EpisodeScorer,
            self._EpisodeJudge,
        ) = _import_sandbox()

        # Build sandbox config from validator config
        self._sandbox_config = self._SandboxConfig(
            sandbox_image=config.sandbox_image,
            harness_image=config.harness_image,
            llm_api_url=config.judge_base_url or config.clawbench_base_url,
            llm_api_key=config.judge_api_key or config.clawbench_api_key,
            llm_model=config.judge_model or config.clawbench_default_model,
            harness_timeout_s=config.sandbox_timeout_per_episode,
        )

        logger.info(
            "TrajectorySandboxHarness initialized "
            "(sandbox=%s, harness=%s, episodes=%d, timeout=%ds)",
            config.sandbox_image,
            config.harness_image,
            config.sandbox_num_episodes,
            config.sandbox_timeout_per_episode,
        )

    async def evaluate_miner(
        self,
        skill_md: str,
        epoch_seed: int,
        pack_hash: str = "",
        validator_salt: str = "",
    ) -> SandboxEvaluationResult:
        """Run a full Season 1 evaluation for one miner.

        Generates fixtures deterministically from epoch_seed + validator_salt,
        runs N episodes in the sandbox with LLM judge scoring, and returns
        the split-half delta final score.

        Args:
            skill_md: Miner's SKILL.md content (extracted from pack)
            epoch_seed: Epoch seed for deterministic fixture generation
            pack_hash: Pack hash for logging/tracing
            validator_salt: Validator-specific salt for fixture variation

        Returns:
            SandboxEvaluationResult with final_score and episode details
        """
        num_episodes = self.config.sandbox_num_episodes

        logger.info(
            "S1 evaluation starting: pack_hash=%s, seed=%d, episodes=%d",
            pack_hash[:12] if pack_hash else "?", epoch_seed, num_episodes,
        )

        # Select scenario: rotate based on epoch_seed across available scenarios
        from trajectory_sandbox.fixture_factory import SCENARIOS
        scenario = SCENARIOS[epoch_seed % len(SCENARIOS)]

        logger.info("S1 scenario selected: %s (seed=%d, pool=%s)",
                     scenario, epoch_seed, SCENARIOS)

        # Generate fixtures deterministically
        factory = self._FixtureFactory(
            epoch_seed=str(epoch_seed),
            validator_salt=validator_salt or self._default_salt(),
            scenario=scenario,
        )
        world = factory.generate_world()
        episodes_fixtures = [factory.generate_episode(i, world) for i in range(num_episodes)]

        # Build instructions and fixture dicts
        instructions = [ef.instruction_md for ef in episodes_fixtures]
        fixtures_per_episode = [ef.to_dict() for ef in episodes_fixtures]

        # Build per-episode scorers (evidence + LLM judge)
        judge = self._EpisodeJudge()  # picks up LLM_API_KEY etc. from env
        scorers = [
            self._EpisodeScorer.for_scenario(scenario, world, ef, judge=judge)
            for ef in episodes_fixtures
        ]

        # Run in executor to avoid blocking the async event loop
        # (EvalSession uses synchronous Docker SDK calls)
        loop = asyncio.get_event_loop()
        try:
            session_result = await loop.run_in_executor(
                None,
                partial(
                    self._run_session_sync,
                    skill_md=skill_md,
                    instructions=instructions,
                    fixtures_per_episode=fixtures_per_episode,
                    scorers=scorers,
                ),
            )
        except Exception as e:
            logger.error("S1 evaluation failed: %s", e, exc_info=True)
            result = SandboxEvaluationResult(self._EvalSessionResult())
            result.error = str(e)
            return result

        result = SandboxEvaluationResult(session_result, scenario_name=scenario)
        result.session_result.pack_hash = pack_hash
        result.session_result.validator_salt = validator_salt

        logger.info(
            "S1 evaluation complete: final_score=%.3f "
            "(mean_q=%.3f, delta=%.3f, episodes=%s)",
            result.score,
            result.mean_quality,
            result.delta,
            result.episode_qualities,
        )
        return result

    def _run_session_sync(
        self,
        skill_md: str,
        instructions: list[str],
        fixtures_per_episode: list[dict],
        scorers: list,
    ):
        """Run the sandbox session synchronously (called from executor)."""
        with self._EvalSession(self._sandbox_config) as session:
            return session.run_all_episodes(
                skill_md=skill_md,
                instructions=instructions,
                fixtures_per_episode=fixtures_per_episode,
                scorer=scorers,
            )

    def _default_salt(self) -> str:
        """Derive a stable salt from the validator's hotkey."""
        # Use a hash of config values as fallback salt
        data = f"{self.config.wallet_hotkey}:{self.config.netuid}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @staticmethod
    def extract_skill_md(pack: dict) -> str | None:
        """Extract SKILL.md content from a miner's pack.

        Season 1 packs include SKILL.md in the files dict.
        Returns None if no SKILL.md found (not an S1 pack).
        """
        files = pack.get("files", {})
        # Try SKILL.md first, then skill.md
        return files.get("SKILL.md") or files.get("skill.md")
