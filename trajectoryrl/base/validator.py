"""TrajectoryRL Validator - Main validator implementation."""

import asyncio
import hashlib
import json
import logging
import random
import sys
import time
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np

from ..protocol.synapse import PackRequest, PackResponse
from ..utils.opp_schema import validate_opp_schema
from ..utils.config import ValidatorConfig
from ..utils.clawbench import ClawBenchHarness, EvaluationResult
from ..scoring import TrajectoryScorer, AggregatedScore
from ..utils.github import GitHubVerifier
from ..utils.epoch_context import generate_epoch_context, render_context_preamble

logger = logging.getLogger(__name__)


class TrajectoryValidator:
    """TrajectoryRL validator that evaluates policy packs using ClawBench.

    The validator:
    1. Queries miners for policy packs (via Bittensor synapses)
    2. Verifies pack hashes and caches them
    3. Runs ClawBench scenarios against each pack
    4. Scores results and sets on-chain weights

    Example:
        >>> config = ValidatorConfig.from_env()
        >>> validator = TrajectoryValidator(config)
        >>> await validator.run()
    """

    def __init__(self, config: ValidatorConfig):
        """Initialize validator.

        Args:
            config: Validator configuration
        """
        self.config = config

        # Setup logging
        self._setup_logging()

        logger.info("=" * 60)
        logger.info("TrajectoryRL Validator v0.1.0")
        logger.info("=" * 60)

        # Initialize Bittensor components
        logger.info("Initializing Bittensor components...")
        self.wallet = bt.Wallet(
            name=config.wallet_name,
            hotkey=config.wallet_hotkey
        )
        self.subtensor = bt.Subtensor(network=config.network)
        self.metagraph = self.subtensor.metagraph(config.netuid)
        self.dendrite = bt.Dendrite(wallet=self.wallet)

        logger.info(f"Wallet: {self.wallet}")
        logger.info(f"Network: {config.network}")
        logger.info(f"Netuid: {config.netuid}")

        # Initialize ClawBench harness
        logger.info("Initializing ClawBench harness...")
        self.harness = ClawBenchHarness(
            clawbench_path=config.clawbench_path,
            timeout=config.timeout_per_scenario
        )

        # Initialize scorer
        self.scorer = TrajectoryScorer(
            lambda_cost=config.lambda_cost,
            mu_safety=config.mu_safety,
            rho_reliability=config.rho_reliability,
            score_quantization=config.score_quantization,
            consensus_epsilon=config.consensus_epsilon,
            bootstrap_threshold=config.bootstrap_threshold,
        )

        # Initialize GitHub verifier
        logger.info("Initializing GitHub verifier...")
        self.github_verifier = GitHubVerifier(
            cache_dir=config.log_dir / "git_cache",
            github_token=config.github_token,
        )

        # Pack cache (content-addressed)
        self.pack_cache: Dict[str, dict] = {}

        # Score history for tracking
        self.score_history: Dict[int, List[float]] = defaultdict(list)

        # First-mover tracking: {miner_uid: (first_score, first_timestamp)}
        self.first_mover_data: Dict[int, Tuple[float, float]] = {}

        # Epoch number — derived from Bittensor block height so all
        # validators agree on the same epoch (and thus the same seed,
        # persona, and scenario subset).
        # Approx 12 s per block → blocks_per_epoch = epoch_interval / 12.
        self.blocks_per_epoch: int = max(1, self.config.epoch_interval // 12)
        self.current_epoch: int = self._block_epoch()

        # Load scenarios
        self.scenarios = self._load_scenarios()
        logger.info(f"Loaded {len(self.scenarios)} scenarios: {list(self.scenarios.keys())}")

        logger.info("Validator initialization complete!")

    def _block_epoch(self) -> int:
        """Derive epoch number from current Bittensor block height.

        All validators see the same block height, so they compute the
        same epoch number → same seed → same persona/scenarios.
        """
        current_block = self.subtensor.get_current_block()
        return current_block // self.blocks_per_epoch

    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.config.log_dir / f"validator_{int(time.time())}.log"
                )
            ]
        )

    def _load_scenarios(self) -> Dict[str, dict]:
        """Load scenario configurations.

        Returns:
            Dict of scenario_name -> scenario config
        """
        scenarios = {}
        for scenario_name in self.config.scenarios:
            scenario_path = self.config.scenarios_path / f"{scenario_name}.yaml"
            if not scenario_path.exists():
                logger.warning(f"Scenario not found: {scenario_path}")
                continue

            with open(scenario_path) as f:
                scenario = yaml.safe_load(f)
                scenarios[scenario_name] = scenario
                logger.debug(f"Loaded scenario: {scenario_name}")

        if not scenarios:
            raise ValueError("No scenarios loaded!")

        return scenarios

    async def run(self):
        """Main validator loop.

        Epoch numbers are derived from Bittensor block height
        (block // blocks_per_epoch) so every validator agrees on the
        same epoch — and therefore the same seed, persona, and
        scenario subset — regardless of when it started.
        """
        logger.info("Starting validator main loop...")
        logger.info(
            f"Epoch interval: {self.config.epoch_interval}s "
            f"(~{self.blocks_per_epoch} blocks)"
        )

        last_completed_epoch: Optional[int] = None

        while True:
            try:
                self.current_epoch = self._block_epoch()

                # Skip if we already evaluated this epoch
                if self.current_epoch == last_completed_epoch:
                    await asyncio.sleep(60)  # Check again in 1 min
                    continue

                logger.info("=" * 60)
                logger.info(f"Epoch {self.current_epoch} starting")
                logger.info("=" * 60)

                await self.run_epoch()

                last_completed_epoch = self.current_epoch
                logger.info(
                    f"Epoch {self.current_epoch} complete. "
                    f"Waiting for next epoch..."
                )

                # Sleep until roughly the next epoch boundary
                await asyncio.sleep(self.config.epoch_interval)

            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Epoch failed: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 min before retry

    @staticmethod
    def compute_epoch_seed(epoch: int, netuid: int = 11) -> int:
        """Deterministic seed for this epoch.

        The epoch number is derived from Bittensor block height
        (current_block // blocks_per_epoch), so all validators agree
        on the same epoch and thus the same seed.

        Args:
            epoch: Epoch number (block_height // blocks_per_epoch)
            netuid: Subnet UID (mixed in to avoid cross-subnet collisions)

        Returns:
            Deterministic integer seed
        """
        raw = f"trajectoryrl-{netuid}-epoch-{epoch}".encode()
        return int(hashlib.sha256(raw).hexdigest()[:8], 16)

    @staticmethod
    def select_epoch_scenarios(
        all_scenarios: List[str],
        epoch_seed: int,
        max_scenarios: int = 4,
    ) -> List[str]:
        """Deterministically select which scenarios to run this epoch.

        When the scenario pool is larger than ``max_scenarios``, a
        different subset is chosen each epoch (seeded).  Because the seed
        is deterministic, all validators select the same subset.

        Rotating the scenario subset across epochs means a stale policy
        that was tuned for scenarios A-D may face scenarios B-E next
        epoch, naturally degrading cached solutions.

        Args:
            all_scenarios: Full scenario pool
            epoch_seed: Deterministic seed for this epoch
            max_scenarios: Max scenarios to evaluate per epoch

        Returns:
            Sorted list of selected scenario names
        """
        if len(all_scenarios) <= max_scenarios:
            return sorted(all_scenarios)
        rng = random.Random(epoch_seed)
        selected = rng.sample(all_scenarios, max_scenarios)
        return sorted(selected)

    async def run_epoch(self):
        """Run one evaluation epoch.

        Each epoch:
        1. Computes a deterministic epoch seed (same across all validators)
        2. Generates an epoch context (persona, date, company, etc.)
        3. Selects a scenario subset from the pool (epoch-seeded)
        4. Evaluates all active miners on the selected scenarios
        5. Sets on-chain weights (winner-take-all)
        """
        # 0. Compute epoch seed and epoch context
        epoch_seed = self.compute_epoch_seed(self.current_epoch, self.config.netuid)
        epoch_scenarios = self.select_epoch_scenarios(
            all_scenarios=list(self.scenarios.keys()),
            epoch_seed=epoch_seed,
            max_scenarios=self.config.scenarios_per_epoch,
        )

        # Generate epoch context — unique persona/date per epoch, prepended
        # to every miner's AGENTS.md so policies must be identity-agnostic.
        epoch_ctx = generate_epoch_context(epoch_seed)
        context_preamble = render_context_preamble(epoch_ctx)
        user_context = epoch_ctx.to_user_context()

        logger.info(
            f"Epoch {self.current_epoch}: seed={epoch_seed}, "
            f"scenarios={epoch_scenarios}, "
            f"context=[{epoch_ctx.user_name}, {epoch_ctx.user_role}, "
            f"{epoch_ctx.date_str}]"
        )

        # 1. Sync metagraph
        logger.info("Syncing metagraph...")
        self.metagraph.sync(subtensor=self.subtensor)

        # 2. Get active miners
        miners = self._get_active_miners()
        logger.info(f"Found {len(miners)} active miners")

        if not miners:
            logger.warning("No active miners found!")
            return

        # 3. Evaluate each miner on this epoch's scenario subset
        scores = {}
        for miner_uid in miners:
            score = await self._evaluate_miner(
                miner_uid, epoch_scenarios, epoch_seed,
                context_preamble, user_context,
            )
            scores[miner_uid] = score

        # 4. Set weights (pass miner count for bootstrap detection)
        if scores:
            await self._set_weights(scores, num_active_miners=len(miners))
        else:
            logger.warning("No scores to set!")

    def _get_active_miners(self) -> List[int]:
        """Get list of active miner UIDs.

        Returns:
            List of miner UIDs
        """
        # Get all UIDs with non-zero stake
        miners = []
        for uid in range(len(self.metagraph.S)):
            # Skip if UID is a validator (high stake)
            if self.metagraph.S[uid] > 1000:  # Arbitrary threshold
                continue

            # Skip if UID is not registered
            if self.metagraph.axons[uid].ip == "0.0.0.0":
                continue

            miners.append(uid)

        return miners

    async def _evaluate_miner(
        self,
        miner_uid: int,
        epoch_scenarios: List[str],
        epoch_seed: int,
        context_preamble: str = "",
        user_context: Optional[Dict] = None,
    ) -> float:
        """Evaluate a single miner on this epoch's scenarios.

        Args:
            miner_uid: Miner UID
            epoch_scenarios: Scenarios selected for this epoch
            epoch_seed: Deterministic seed for this epoch
            context_preamble: Epoch context markdown prepended to AGENTS.md
            user_context: Dict of {{PLACEHOLDER}} overrides for USER.md

        Returns:
            Final score [0, 1]
        """
        logger.info(f"Evaluating miner {miner_uid}...")

        # Step 1: Fetch pack
        pack_response = await self._fetch_pack(miner_uid)
        if pack_response is None:
            logger.warning(f"Miner {miner_uid}: Failed to fetch pack")
            return 0.0

        # Step 2: Verify GitHub submission
        if not pack_response.git_commit_hash or not pack_response.repo_url:
            logger.warning(
                f"Miner {miner_uid}: Missing git_commit_hash or repo_url"
            )
            return 0.0

        verification = await self.github_verifier.verify_submission(
            repo_url=pack_response.repo_url,
            git_commit_hash=pack_response.git_commit_hash,
            pack_hash=pack_response.pack_hash,
            on_chain_submission_time=time.time()  # TODO: Get actual on-chain time
        )

        if not verification.valid:
            logger.warning(
                f"Miner {miner_uid}: GitHub verification failed: {verification.error}"
            )
            return 0.0

        pack = verification.pack_content
        commit_timestamp = verification.commit_timestamp
        pack_hash = pack_response.pack_hash[:8]
        logger.info(
            f"Miner {miner_uid}: Got pack {pack_hash} "
            f"(commit: {pack_response.git_commit_hash[:8]}, "
            f"timestamp: {commit_timestamp})"
        )

        # Step 3: Static lint
        lint_result = validate_opp_schema(pack)
        if not lint_result.passed:
            logger.warning(
                f"Miner {miner_uid}: Pack lint failed: {lint_result.issues}"
            )
            return 0.0

        # Step 4: Run this epoch's scenarios with majority-vote consensus.
        # Each scenario is run seeds_per_task times; individual rubric checks
        # are majority-voted so that independent validators converge on the
        # same binary pass/fail per check despite LLM non-determinism.
        # The epoch_seed is mixed into per-run seeds so that each epoch
        # evaluates under slightly different conditions.
        results = []
        for scenario_name in epoch_scenarios:
            try:
                result = await self.harness.evaluate_pack_consensus(
                    pack=pack,
                    scenario_name=scenario_name,
                    num_runs=self.config.seeds_per_task,
                    base_seed=epoch_seed,
                    context_preamble=context_preamble,
                    user_context=user_context,
                )
                results.append(result)

                logger.info(
                    f"Miner {miner_uid}: {scenario_name} → "
                    f"score={result.score:.3f} (consensus x{self.config.seeds_per_task})"
                )

            except Exception as e:
                logger.error(
                    f"Miner {miner_uid}: {scenario_name} failed: {e}",
                    exc_info=True
                )

        # Step 5: Aggregate scores (weighted by scenario YAML weight field)
        if not results:
            logger.warning(f"Miner {miner_uid}: No results!")
            return 0.0

        scenario_weights = {
            name: cfg.get("weight", 1.0)
            for name, cfg in self.scenarios.items()
        }
        aggregated = self.scorer.aggregate_scores(results, scenario_weights)
        final_score = self.scorer.compute_final_score(aggregated)

        logger.info(
            f"Miner {miner_uid}: Final score = {final_score:.3f} "
            f"(mean={aggregated.mean_score:.3f}, var={aggregated.variance:.3f})"
        )

        # Track history
        self.score_history[miner_uid].append(final_score)

        # Update first-mover tracking
        if miner_uid not in self.first_mover_data:
            self.first_mover_data[miner_uid] = (final_score, commit_timestamp)
            logger.info(
                f"Miner {miner_uid}: First submission recorded "
                f"(score={final_score:.3f}, timestamp={commit_timestamp})"
            )
        elif final_score > self.first_mover_data[miner_uid][0]:
            self.first_mover_data[miner_uid] = (final_score, commit_timestamp)
            logger.info(
                f"Miner {miner_uid}: Score improved "
                f"(new={final_score:.3f}, old={self.first_mover_data[miner_uid][0]:.3f})"
            )

        return final_score

    async def _fetch_pack(self, miner_uid: int) -> Optional[PackResponse]:
        """Fetch pack from miner.

        Args:
            miner_uid: Miner UID

        Returns:
            PackResponse or None if failed
        """
        request = PackRequest(
            suite_id="clawbench_v1",
            schema_version=1,
            max_bytes=65536,
            want_pointer_ok=True
        )

        try:
            # Query miner via dendrite
            response = await self.dendrite.forward(
                axons=[self.metagraph.axons[miner_uid]],
                synapse=request,
                timeout=10.0
            )

            # dendrite.forward returns list of synapses (same object mutated).
            # If the miner didn't fill in response fields, it's still a
            # PackRequest — not a usable PackResponse.
            if not response or len(response) == 0:
                return None

            resp = response[0]
            if not isinstance(resp, PackResponse):
                logger.debug(f"Miner {miner_uid}: got {type(resp).__name__} instead of PackResponse")
                return None

            # Check that the miner actually filled in required fields
            if not resp.pack_hash:
                logger.debug(f"Miner {miner_uid}: empty pack_hash in response")
                return None

            return resp

        except Exception as e:
            logger.warning(f"Failed to fetch pack from miner {miner_uid}: {e}")
            return None

    async def _set_weights(
        self,
        scores: Dict[int, float],
        num_active_miners: Optional[int] = None,
    ):
        """Set on-chain weights based on scores.

        Uses winner-take-all in steady state, or graduated top-3 curve
        during the bootstrap phase (when active miners < bootstrap_threshold).

        Args:
            scores: Dict of miner_uid -> score [0, 1]
            num_active_miners: Total active miners (for bootstrap detection)
        """
        logger.info(f"Setting weights for {len(scores)} miners...")

        # Filter out miners below minimum score threshold.
        # Without this, all-zero scores would give weight=1.0 to an
        # arbitrary miner.  When no miner qualifies, we skip setting
        # weights entirely — Bittensor's Yuma Consensus redirects
        # the miner alpha share to validators.
        eligible = {
            uid: s for uid, s in scores.items()
            if s >= self.config.min_score_threshold
        }
        if not eligible:
            # No miner qualifies, but we must still call set_weights
            # to keep the validator active on-chain (prevents
            # deregistration after activity_cutoff blocks).
            # Set uniform weights so no single miner is favoured.
            logger.warning(
                f"No miners above min_score_threshold "
                f"({self.config.min_score_threshold}). "
                f"Setting uniform weights to stay active."
            )
            uids = list(scores.keys())
            uniform_w = 1.0 / len(uids)
            weights = [uniform_w] * len(uids)
            try:
                self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=uids,
                    weights=weights,
                    wait_for_inclusion=True,
                    wait_for_finalization=False,
                )
                logger.info("✓ Uniform weights set (no eligible miners)")
            except Exception as e:
                logger.error(f"Error setting uniform weights: {e}", exc_info=True)
            return

        # Select winner (or bootstrap curve) with first-mover advantage
        weights_dict = self.scorer.select_winner(
            scores=eligible,
            first_mover_data=self.first_mover_data,
            delta=self.config.delta_threshold,
            num_active_miners=num_active_miners,
        )

        # Prepare for Bittensor API
        uids = list(weights_dict.keys())
        weights = [weights_dict[uid] for uid in uids]

        # Find winner
        winner_uid = max(weights_dict.keys(), key=lambda uid: weights_dict[uid])

        logger.info("=" * 60)
        logger.info("WINNER-TAKE-ALL RESULTS")
        logger.info("=" * 60)
        logger.info(f"🏆 Winner: Miner {winner_uid} (score={eligible[winner_uid]:.3f})")
        logger.info("-" * 60)
        logger.info(f"Eligible miners ({len(eligible)}/{len(scores)}):")
        for uid, weight in sorted(
            weights_dict.items(),
            key=lambda x: eligible[x[0]],
            reverse=True
        ):
            marker = "🏆 WINNER" if weight == 1.0 else ""
            logger.info(f"  Miner {uid}: weight={weight:.4f}, score={eligible[uid]:.3f} {marker}")

        # Set weights on chain
        try:
            success = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=False
            )

            if success:
                logger.info("✓ Weights set successfully!")
            else:
                logger.error("✗ Failed to set weights")

        except Exception as e:
            logger.error(f"Error setting weights: {e}", exc_info=True)


async def main():
    """Entry point for validator."""
    config = ValidatorConfig.from_env()
    validator = TrajectoryValidator(config)
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
