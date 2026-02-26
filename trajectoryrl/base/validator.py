"""TrajectoryRL Validator — Main validator implementation.

Architecture (v1.06):
    1. Read on-chain commitments (subtensor.get_all_commitments)
    2. Fetch packs from miners' public GitHub repos
    3. Validate schema + NCD similarity check
    4. Run ALL ClawBench scenarios with majority-vote consensus
    5. Publish scores to shared validator-scores repo
    6. Compute stake-weighted consensus across validators
    7. Set on-chain weights (winner-take-all / bootstrap)
    8. Re-set weights every tempo (~72 min)
"""

import asyncio
import hashlib
import json
import logging
import time
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np

from ..utils.opp_schema import validate_opp_schema
from ..utils.config import ValidatorConfig
from ..utils.clawbench import ClawBenchHarness, EvaluationResult
from ..scoring import TrajectoryScorer, AggregatedScore
from ..utils.github import GitHubVerifier
from ..utils.epoch_context import generate_epoch_context, render_context_preamble
from ..utils.commitments import MinerCommitment, fetch_all_commitments
from ..utils.ncd import is_too_similar, pack_similarity
from ..utils.score_publisher import ScorePublisher

logger = logging.getLogger(__name__)


class TrajectoryValidator:
    """TrajectoryRL validator that evaluates policy packs using ClawBench.

    The validator:
    1. Reads on-chain commitments from miners
    2. Fetches and verifies packs from miners' public GitHub repos
    3. Checks NCD similarity against current winner (anti-copy)
    4. Runs ALL ClawBench scenarios with majority-vote consensus
    5. Publishes scores to shared repo and computes stake-weighted consensus
    6. Sets on-chain weights (winner-take-all or bootstrap)
    7. Re-sets weights every tempo (~72 min) for convergence

    Example:
        >>> config = ValidatorConfig.from_env()
        >>> validator = TrajectoryValidator(config)
        >>> await validator.run()
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config

        # Setup logging
        self._setup_logging()

        logger.info("=" * 60)
        logger.info("TrajectoryRL Validator v0.2.0")
        logger.info("=" * 60)

        # Initialize Bittensor components
        logger.info("Initializing Bittensor components...")
        self.wallet = bt.Wallet(
            name=config.wallet_name,
            hotkey=config.wallet_hotkey,
        )
        self.subtensor = bt.Subtensor(network=config.network)
        self.metagraph = self.subtensor.metagraph(config.netuid)

        logger.info(f"Wallet: {self.wallet}")
        logger.info(f"Network: {config.network}")
        logger.info(f"Netuid: {config.netuid}")

        # Initialize ClawBench harness
        logger.info("Initializing ClawBench harness...")
        self.harness = ClawBenchHarness(
            clawbench_path=config.clawbench_path,
            timeout=config.timeout_per_scenario,
        )

        # Initialize scorer
        self.scorer = TrajectoryScorer(
            rho_reliability=config.rho_reliability,
            consensus_epsilon=config.consensus_epsilon,
            bootstrap_threshold=config.bootstrap_threshold,
        )

        # Initialize GitHub verifier
        logger.info("Initializing GitHub verifier...")
        self.github_verifier = GitHubVerifier(
            cache_dir=config.pack_cache_dir,
            github_token=config.github_token,
        )

        # Score publisher (optional — solo mode if not configured)
        if config.validator_scores_fork_url:
            self.score_publisher: Optional[ScorePublisher] = ScorePublisher(
                wallet=self.wallet,
                fork_repo_url=config.validator_scores_fork_url,
                local_path=config.validator_scores_local_path,
                github_token=config.github_token,
            )
            logger.info("Score publisher initialized")
        else:
            self.score_publisher = None
            logger.warning(
                "No VALIDATOR_SCORES_FORK_URL set; "
                "running in solo mode (no multi-validator consensus)"
            )

        # Pack score cache: pack_hash -> (final_score, per_scenario_scores)
        self.pack_score_cache: Dict[str, Tuple[float, Dict[str, float]]] = {}

        # Pack content cache: pack_hash -> pack dict.
        # Allows cache-hitting miners to still populate _uid_packs so that
        # current_winner_pack is never left as None after winner selection.
        self._pack_by_hash: Dict[str, dict] = {}

        # Current winner tracking (for NCD comparison).
        # Stays None until first epoch completes and a winner is selected.
        # This is safe: on the first epoch there is no winner to copy, so
        # NCD protection is not needed.  From epoch 2 onward, the winner
        # pack persists in memory across epoch transitions.
        self.current_winner_pack: Optional[dict] = None
        self.current_winner_uid: Optional[int] = None

        # Evaluated packs by UID (populated during epoch, used to set current_winner_pack)
        self._uid_packs: Dict[int, dict] = {}

        # Score history for tracking
        self.score_history: Dict[int, List[float]] = defaultdict(list)

        # First-mover tracking: {hotkey: (best_score, first_block_number)}
        # Keyed by miner hotkey so history is never inherited on UID recycling.
        self.first_mover_data: Dict[str, Tuple[float, float]] = {}

        # Tracks which UID each hotkey was last evaluated at.  Used to detect
        # re-registration (same hotkey at a new UID), which must reset the
        # first-mover block_number to the current commitment block so the miner
        # doesn't inherit an unfair chronological advantage.
        self._hotkey_uid_map: Dict[str, int] = {}

        # Inactivity tracking: {uid: last_epoch_with_valid_commitment}
        self.last_valid_epoch: Dict[int, int] = {}

        # Epoch derived from block height for cross-validator consensus
        self.blocks_per_epoch: int = max(1, self.config.epoch_interval // 12)
        self.current_epoch: int = self._block_epoch()

        # Weight cadence tracking
        self.last_weight_block: int = 0
        self.cached_scores: Dict[int, float] = {}
        self.cached_per_scenario: Dict[int, Dict[str, float]] = {}

        # Load scenarios
        self.scenarios = self._load_scenarios()
        logger.info(
            f"Loaded {len(self.scenarios)} scenarios: "
            f"{list(self.scenarios.keys())}"
        )

        logger.info("Validator initialization complete!")

    # ------------------------------------------------------------------
    # Block / epoch helpers
    # ------------------------------------------------------------------

    def _block_epoch(self) -> int:
        """Derive epoch number from current Bittensor block height."""
        current_block = self.subtensor.get_current_block()
        return current_block // self.blocks_per_epoch

    def _get_block_timestamp(self) -> float:
        """Get timestamp of the current block from the chain.

        Uses the Substrate Timestamp pallet for a deterministic,
        chain-agreed timestamp.  Falls back to time.time() on failure.
        """
        try:
            block = self.subtensor.get_current_block()
            block_hash = self.subtensor.substrate.get_block_hash(block)
            result = self.subtensor.substrate.query(
                module="Timestamp",
                storage_function="Now",
                block_hash=block_hash,
            )
            return result.value / 1000
        except Exception as e:
            logger.warning(f"Failed to get block timestamp: {e}")
            return time.time()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_logging(self):
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.config.log_dir / f"validator_{int(time.time())}.log"
                ),
            ],
        )

    def _load_scenarios(self) -> Dict[str, dict]:
        scenarios = {}
        for scenario_name in self.config.scenarios:
            scenario_path = self.config.scenarios_path / f"{scenario_name}.yaml"
            if not scenario_path.exists():
                logger.warning(f"Scenario not found: {scenario_path}")
                continue
            with open(scenario_path) as f:
                scenarios[scenario_name] = yaml.safe_load(f)
        if not scenarios:
            raise ValueError("No scenarios loaded!")
        return scenarios

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @staticmethod
    def compute_epoch_seed(epoch: int, netuid: int = 11) -> int:
        raw = f"trajectoryrl-{netuid}-epoch-{epoch}".encode()
        return int(hashlib.sha256(raw).hexdigest()[:8], 16)

    async def run(self):
        """Main validator loop with dual cadence.

        - Epoch cadence (~24 h): full evaluation of new/changed packs.
        - Tempo cadence (~72 min): re-pull scores, recompute consensus,
          re-set weights to stay active on-chain.
        """
        logger.info("Starting validator main loop...")
        logger.info(
            f"Epoch interval: {self.config.epoch_interval}s "
            f"(~{self.blocks_per_epoch} blocks)"
        )

        last_completed_epoch: Optional[int] = None

        while True:
            try:
                # Single RPC call for consistency: both epoch derivation and
                # block-based checks use the same snapshot.
                current_block = self.subtensor.get_current_block()
                self.current_epoch = current_block // self.blocks_per_epoch

                # --- Epoch work: evaluate new/changed packs ---
                if self.current_epoch != last_completed_epoch:
                    logger.info("=" * 60)
                    logger.info(f"Epoch {self.current_epoch} starting")
                    logger.info("=" * 60)

                    # Clear caches so packs are re-evaluated under the new
                    # epoch seed / context (even if pack_hash is unchanged).
                    self.pack_score_cache.clear()
                    self._uid_packs.clear()
                    self._pack_by_hash.clear()

                    await self.run_epoch()
                    last_completed_epoch = self.current_epoch
                    # Use a fresh block number so blocks_since is measured from
                    # when the epoch actually finished, not from when the loop
                    # iteration started (epoch eval can take hours).
                    self.last_weight_block = self.subtensor.get_current_block()

                    # Evict stale cloned repos to keep disk usage bounded.
                    self.github_verifier.cleanup_cache(
                        self.config.pack_cache_max_size
                    )

                    logger.info(
                        f"Epoch {self.current_epoch} complete. "
                        f"Next weight refresh in "
                        f"~{self.config.weight_interval_blocks * 12 // 60} min"
                    )

                # --- Tempo work: re-set weights ---
                # Re-read current_block here so that the tempo cadence is based
                # on time since the epoch finished (not time since loop start).
                # Without this, a multi-hour epoch would make blocks_since huge
                # and immediately fire a redundant tempo refresh.
                current_block = self.subtensor.get_current_block()
                blocks_since = current_block - self.last_weight_block
                if blocks_since >= self.config.weight_interval_blocks:
                    logger.info(
                        f"Tempo weight refresh "
                        f"({blocks_since} blocks since last)"
                    )
                    await self._refresh_and_set_weights()
                    self.last_weight_block = current_block

                await asyncio.sleep(60)

            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Epoch evaluation
    # ------------------------------------------------------------------

    async def run_epoch(self):
        """Run one evaluation epoch.

        1. Compute epoch seed + context
        2. Sync metagraph
        3. Read on-chain commitments
        4. Filter to active miners
        5. Evaluate each miner on ALL scenarios
        6. Publish scores to shared repo
        7. Compute consensus and set weights
        """
        epoch_seed = self.compute_epoch_seed(
            self.current_epoch, self.config.netuid
        )

        # ALL scenarios run every epoch (no subset selection)
        epoch_scenarios = sorted(self.scenarios.keys())

        # Epoch context for identity variation
        epoch_ctx = generate_epoch_context(epoch_seed)
        context_preamble = render_context_preamble(epoch_ctx)
        user_context = epoch_ctx.to_user_context()
        epoch_timestamp = self._get_block_timestamp()

        logger.info(
            f"Epoch {self.current_epoch}: seed={epoch_seed}, "
            f"scenarios={epoch_scenarios}, "
            f"context=[{epoch_ctx.user_name}, {epoch_ctx.user_role}, "
            f"{epoch_ctx.date_str}]"
        )

        # 1. Sync metagraph
        logger.info("Syncing metagraph...")
        self.metagraph.sync(subtensor=self.subtensor)

        # 2. Read on-chain commitments
        logger.info("Reading on-chain commitments...")
        commitments = fetch_all_commitments(
            self.subtensor, self.config.netuid, self.metagraph
        )
        logger.info(f"Found {len(commitments)} valid commitments")

        # 3. Filter to active miners
        active = self._get_active_miners_from_commitments(commitments)
        logger.info(f"Active miners: {len(active)}")

        if not active:
            logger.warning("No active miners with valid commitments!")
            # Clear stale scores so tempo refreshes don't keep directing
            # emission to UIDs from a previous epoch (avoids UID-recycling risk).
            self.cached_scores = {}
            self.cached_per_scenario = {}
            await self._set_uniform_weights()
            return

        # 4. Evaluate each miner
        scores: Dict[int, float] = {}
        per_scenario: Dict[int, Dict[str, float]] = {}

        for uid, commitment in active.items():
            score, breakdown = await self._evaluate_miner(
                uid, commitment, epoch_scenarios, epoch_seed,
                context_preamble, user_context, epoch_timestamp,
            )
            scores[uid] = score
            per_scenario[uid] = breakdown

        self.cached_scores = scores
        self.cached_per_scenario = per_scenario

        # 5. Publish scores to shared repo
        if self.score_publisher and scores:
            block_height = self.subtensor.get_current_block()
            payload = {
                str(uid): {
                    "final_score": score,
                    "per_scenario": per_scenario.get(uid, {}),
                }
                for uid, score in scores.items()
            }
            try:
                await self.score_publisher.publish_scores(
                    epoch=self.current_epoch,
                    block_height=block_height,
                    scores=payload,
                )
                logger.info("Scores published to shared repo")
            except Exception as e:
                logger.error(f"Failed to publish scores: {e}")

        # 6. Consensus + set weights
        await self._refresh_and_set_weights()

    # ------------------------------------------------------------------
    # Miner filtering
    # ------------------------------------------------------------------

    def _get_active_miners_from_commitments(
        self,
        commitments: Dict[int, MinerCommitment],
    ) -> Dict[int, MinerCommitment]:
        """Filter commitments to active miners.

        A miner is active if:
        1. Has a parseable on-chain commitment
        2. Is not a validator
        3. Has been valid within the inactivity window
        """
        active: Dict[int, MinerCommitment] = {}
        for uid, commitment in commitments.items():
            # Skip validators
            if uid < len(self.metagraph.validator_permit) and self.metagraph.validator_permit[uid]:
                continue

            # Record that this miner was valid this epoch
            self.last_valid_epoch[uid] = self.current_epoch

            active[uid] = commitment

        # Remove miners that have been inactive too long.
        # (only affects miners from previous epochs not in current commitments)
        for uid in list(self.last_valid_epoch.keys()):
            if uid in active:
                continue
            last = self.last_valid_epoch[uid]
            if (self.current_epoch - last) > self.config.inactivity_window:
                hotkey = (
                    self.metagraph.hotkeys[uid]
                    if uid < len(self.metagraph.hotkeys)
                    else None
                )
                if hotkey and hotkey in self.first_mover_data:
                    logger.info(
                        f"Miner {uid}: inactive for "
                        f">{self.config.inactivity_window} epochs, "
                        f"first-mover protection lost"
                    )
                    del self.first_mover_data[hotkey]
                    self._hotkey_uid_map.pop(hotkey, None)
                # Always remove from last_valid_epoch once the inactivity window
                # is exceeded; otherwise stale UIDs accumulate unboundedly and
                # pollute _set_uniform_weights() with potentially-recycled slots.
                del self.last_valid_epoch[uid]

        return active

    # ------------------------------------------------------------------
    # First-mover tracking
    # ------------------------------------------------------------------

    def _update_first_mover(
        self,
        miner_uid: int,
        hotkey: str,
        score: float,
        block_number: float,
    ) -> None:
        """Detect re-registration and update first-mover data for a miner.

        Called from both the cache-hit and full-eval paths so that the
        re-registration detection + first-mover insert/update logic lives
        in a single place.
        """
        # Detect re-registration: same hotkey at a different UID means the
        # miner deregistered and re-registered.  Reset first-mover data so
        # they don't inherit an unfair chronological advantage.
        prev_uid = self._hotkey_uid_map.get(hotkey)
        if prev_uid is not None and prev_uid != miner_uid:
            logger.info(
                f"Miner {miner_uid}: hotkey {hotkey[:8]} previously at UID "
                f"{prev_uid}; re-registration detected, resetting first-mover data"
            )
            self.first_mover_data.pop(hotkey, None)
        self._hotkey_uid_map[hotkey] = miner_uid

        if hotkey not in self.first_mover_data:
            self.first_mover_data[hotkey] = (score, block_number)
            logger.info(
                f"Miner {miner_uid}: First submission "
                f"(score={score:.3f}, block={block_number:.0f})"
            )
        elif score > self.first_mover_data[hotkey][0]:
            original_block = self.first_mover_data[hotkey][1]
            self.first_mover_data[hotkey] = (score, original_block)
            logger.info(
                f"Miner {miner_uid}: Score improved to {score:.3f}"
            )

    # ------------------------------------------------------------------
    # Miner evaluation
    # ------------------------------------------------------------------

    async def _evaluate_miner(
        self,
        miner_uid: int,
        commitment: MinerCommitment,
        epoch_scenarios: List[str],
        epoch_seed: int,
        context_preamble: str = "",
        user_context: Optional[Dict] = None,
        epoch_timestamp: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate a single miner using their on-chain commitment.

        Returns:
            (final_score, {scenario_name: score})
        """
        logger.info(
            f"Evaluating miner {miner_uid} "
            f"(hash={commitment.pack_hash[:12]}...)"
        )

        # Step 1: Check pack cache — skip LLM evaluation if pack_hash unchanged.
        # NCD and first-mover tracking still run even on a cache hit so that:
        #  (a) copy-cat miners who share the same pack_hash as the winner are
        #      still NCD-rejected rather than inheriting the cached score, and
        #  (b) every miner gets its own first_mover_data entry for tie-breaking.
        if commitment.pack_hash in self.pack_score_cache:
            cached_score, cached_breakdown = self.pack_score_cache[
                commitment.pack_hash
            ]
            cached_pack = self._pack_by_hash.get(commitment.pack_hash)

            # NCD gate: a different miner submitting the same pack_hash as the
            # current winner is a copy attack — reject even without re-evaluating.
            if (
                miner_uid != self.current_winner_uid
                and cached_pack is not None
                and self.current_winner_pack is not None
            ):
                if is_too_similar(
                    cached_pack, self.current_winner_pack,
                    self.config.similarity_threshold
                ):
                    sim = pack_similarity(cached_pack, self.current_winner_pack)
                    logger.warning(
                        f"Miner {miner_uid}: NCD similarity={sim:.3f} "
                        f">= {self.config.similarity_threshold} "
                        f"(cache hit), rejected"
                    )
                    return 0.0, {}

            # Populate _uid_packs so that if this miner wins, current_winner_pack
            # is never left as None.
            if cached_pack is not None:
                self._uid_packs[miner_uid] = cached_pack

            # Update first-mover tracking so the second miner to share a
            # pack_hash still gets its own entry for tie-breaking.
            self._update_first_mover(
                miner_uid, commitment.hotkey,
                cached_score, float(commitment.block_number),
            )

            logger.info(
                f"Miner {miner_uid}: pack_hash unchanged, "
                f"cached score={cached_score:.3f}"
            )
            return cached_score, cached_breakdown

        # Step 2: Fetch and verify from GitHub
        verification = await self.github_verifier.verify_submission(
            repo_url=commitment.repo_url,
            git_commit_hash=commitment.git_commit_hash,
            pack_hash=commitment.pack_hash,
            on_chain_submission_time=epoch_timestamp or time.time(),
        )

        if not verification.valid:
            logger.warning(
                f"Miner {miner_uid}: Git verification failed: "
                f"{verification.error}"
            )
            return 0.0, {}

        pack = verification.pack_content

        # Step 3: Schema validation
        lint_result = validate_opp_schema(pack)
        if not lint_result.passed:
            logger.warning(
                f"Miner {miner_uid}: Schema failed: {lint_result.issues}"
            )
            return 0.0, {}

        # Step 4: NCD similarity check vs current winner.
        # The current winner is exempt — they must be allowed to re-submit the
        # same pack and defend their position.  Without this exception,
        # pack_similarity(X, X) = 1.0 ≥ threshold and the winner would always
        # be rejected in the very next epoch, breaking first-mover protection.
        if miner_uid != self.current_winner_uid and is_too_similar(
            pack, self.current_winner_pack, self.config.similarity_threshold
        ):
            sim = pack_similarity(pack, self.current_winner_pack)
            logger.warning(
                f"Miner {miner_uid}: NCD similarity={sim:.3f} "
                f">= {self.config.similarity_threshold}, rejected"
            )
            return 0.0, {}

        # Only add to _uid_packs after the pack has cleared schema + NCD.
        # This prevents schema-invalid or NCD-rejected packs from being used
        # as the NCD comparison target if a bug ever allows them to appear as
        # winner candidates.
        self._uid_packs[miner_uid] = pack

        # Step 5: Run ALL scenarios with majority-vote consensus
        results: List[EvaluationResult] = []
        scenario_breakdown: Dict[str, float] = {}

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
                scenario_breakdown[scenario_name] = result.score
                logger.info(
                    f"Miner {miner_uid}: {scenario_name} -> "
                    f"score={result.score:.3f}"
                )
            except Exception as e:
                logger.error(
                    f"Miner {miner_uid}: {scenario_name} failed: {e}",
                    exc_info=True,
                )
                scenario_breakdown[scenario_name] = 0.0

        if not results:
            logger.warning(f"Miner {miner_uid}: No scenario results!")
            return 0.0, scenario_breakdown

        # Step 6: Aggregate scores
        scenario_weights = {
            name: cfg.get("weight", 1.0)
            for name, cfg in self.scenarios.items()
        }
        aggregated = self.scorer.aggregate_scores(results, scenario_weights)
        final_score = self.scorer.compute_final_score(aggregated)

        logger.info(
            f"Miner {miner_uid}: Final={final_score:.3f} "
            f"(mean={aggregated.mean_score:.3f}, "
            f"var={aggregated.variance:.3f})"
        )

        # Step 7: Cache result (score + pack content for _uid_packs on cache hits)
        self.pack_score_cache[commitment.pack_hash] = (
            final_score,
            scenario_breakdown,
        )
        self._pack_by_hash[commitment.pack_hash] = pack

        # Step 8: Update first-mover tracking (on-chain block number)
        self._update_first_mover(
            miner_uid, commitment.hotkey,
            final_score, float(commitment.block_number),
        )

        # Track history
        self.score_history[miner_uid].append(final_score)

        return final_score, scenario_breakdown

    # ------------------------------------------------------------------
    # Weight setting
    # ------------------------------------------------------------------

    async def _refresh_and_set_weights(self):
        """Re-pull scores from shared repo, recompute consensus, set weights.

        Called every tempo (~72 min).
        """
        scores = self.cached_scores

        # Always sync metagraph so UID/hotkey mappings are fresh
        try:
            self.metagraph.sync(subtensor=self.subtensor)
        except Exception as e:
            logger.warning(f"Metagraph sync failed, using cached: {e}")

        # If score publisher is configured, use stake-weighted consensus
        if self.score_publisher:
            try:
                score_files = await self.score_publisher.pull_all_scores(
                    self.current_epoch
                )
                if score_files:
                    consensus = ScorePublisher.compute_consensus(
                        score_files, self.metagraph
                    )
                    if consensus.consensus_scores:
                        # Filter to UIDs within the current metagraph bounds.
                        # Score files are published at evaluation time; by the
                        # next tempo refresh some UIDs may have been deregistered
                        # or recycled.  Stale UIDs would cause set_weights() to
                        # fail or (worse) direct emission to the new occupant of
                        # a recycled slot.
                        n_uids = len(self.metagraph.S)
                        scores = {
                            uid: s
                            for uid, s in consensus.consensus_scores.items()
                            if uid < n_uids
                        }
                        logger.info(
                            f"Consensus from {consensus.num_validators} "
                            f"validators (stake={consensus.total_stake:.1f}), "
                            f"{len(scores)} valid UIDs"
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to pull consensus, using local scores: {e}"
                )

        if scores:
            await self._set_weights(scores)
        else:
            await self._set_uniform_weights()

    async def _set_weights(self, scores: Dict[int, float]):
        """Set on-chain weights based on scores.

        Only miners with score > 0 are eligible for emission.  A score of 0
        means the pack failed evaluation (git, schema, NCD, or all scenarios)
        and must not earn any weight regardless of bootstrap phase.
        """
        logger.info(f"Setting weights for {len(scores)} miners...")

        if not scores:
            await self._set_uniform_weights()
            return

        # Only miners with positive scores are eligible for emission.
        # Filtering here prevents bootstrap mode from awarding weight to a
        # miner whose pack completely failed evaluation (score=0).
        scored = {uid: s for uid, s in scores.items() if s > 0}
        if not scored:
            logger.warning(
                "All evaluated miners scored 0 — no emission directed to miners"
            )
            await self._set_uniform_weights()
            return

        # Count active miners for bootstrap detection
        num_active = len(scored)

        # Build UID -> hotkey mapping (only for scored miners)
        uid_to_hotkey = {}
        for uid in scored:
            if uid < len(self.metagraph.hotkeys):
                uid_to_hotkey[uid] = self.metagraph.hotkeys[uid]

        # Select winner (or bootstrap curve) with first-mover advantage.
        # Pass only scored (score > 0) miners so bootstrap thresholding and
        # winner-take-all never consider zero-score entries.
        weights_dict = self.scorer.select_winner(
            scores=scored,
            first_mover_data=self.first_mover_data,
            delta=self.config.delta_threshold,
            num_active_miners=num_active,
            uid_to_hotkey=uid_to_hotkey,
        )

        # Track current winner for NCD comparison.
        # Only commit the new winner when we hold the verified pack locally.
        # During tempo refreshes, consensus may resolve to a UID that this
        # validator didn't evaluate (not in _uid_packs).  Writing None to
        # current_winner_pack would silently disable NCD protection next epoch.
        winner_uid = max(weights_dict, key=weights_dict.get)
        if weights_dict.get(winner_uid, 0) > 0:
            winner_pack = self._uid_packs.get(winner_uid)
            if winner_pack is not None:
                self.current_winner_uid = winner_uid
                self.current_winner_pack = winner_pack
            else:
                logger.debug(
                    f"Winner UID {winner_uid} not in local _uid_packs "
                    f"(consensus-only result); keeping previous winner state "
                    f"for NCD protection"
                )

        # Log results
        logger.info("=" * 60)
        logger.info("WEIGHT RESULTS")
        logger.info("=" * 60)
        for uid, weight in sorted(
            weights_dict.items(),
            key=lambda x: scored.get(x[0], 0),
            reverse=True,
        ):
            marker = " <- WINNER" if uid == winner_uid else ""
            logger.info(
                f"  Miner {uid}: weight={weight:.4f}, "
                f"score={scored.get(uid, 0):.3f}{marker}"
            )

        # Set weights on chain
        uids = list(weights_dict.keys())
        weights = [weights_dict[uid] for uid in uids]
        try:
            self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )
            logger.info("Weights set successfully")
        except Exception as e:
            logger.error(f"Error setting weights: {e}", exc_info=True)

    async def _set_uniform_weights(self):
        """Set uniform weights to stay active on-chain when no miners qualify.

        Distributes only among miners that have been seen in at least one
        previous epoch (i.e. have a valid commitment history).  This prevents
        a freshly-registered miner from earning emission before their pack has
        been evaluated.  Falls back to the full metagraph only when no miners
        have ever been seen (e.g. very first epoch after validator start).
        """
        try:
            known_uids = [
                uid for uid in self.last_valid_epoch
                if uid < len(self.metagraph.S)
            ]
            if known_uids:
                uids = known_uids
                logger.info(
                    f"Uniform weights over {len(uids)} previously-seen miners "
                    f"(no eligible scored miners this round)"
                )
            else:
                # No miners have been evaluated yet — fall back to all UIDs
                # so the validator stays active on-chain.
                n_uids = len(self.metagraph.S)
                if n_uids == 0:
                    return
                uids = list(range(n_uids))
                logger.info(
                    "Uniform weights over full metagraph "
                    "(no miners evaluated yet)"
                )

            weights = [1.0 / len(uids)] * len(uids)
            self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )
            logger.info("Uniform weights set (no eligible miners)")
        except Exception as e:
            logger.error(f"Error setting uniform weights: {e}", exc_info=True)


async def main():
    """Entry point for validator."""
    config = ValidatorConfig.from_env()
    validator = TrajectoryValidator(config)
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
