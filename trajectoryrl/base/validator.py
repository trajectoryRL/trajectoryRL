"""TrajectoryRL Validator — Main validator implementation.

Architecture (v2.0):
    1. Continuous evaluation loop with dual cadence:
       - eval_interval (~4h): re-evaluate all active packs, update per-scenario EMA
       - tempo (~72 min): compute weights from EMA scores, set_weights
    2. Read on-chain commitments (subtensor.get_all_commitments)
    3. Fetch packs from miners' public HTTP URLs
    4. Validate schema + NCD similarity check
    5. Run ALL ClawBench scenarios
    6. Update per-scenario EMA (keyed by miner hotkey)
    7. Set on-chain weights (winner-take-all / bootstrap)

Each validator operates independently — no shared score repo.
YC3 aggregates independent validator weights on-chain.
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

from ..utils.opp_schema import validate_opp_schema
from ..utils.config import ValidatorConfig
from ..utils.clawbench import ClawBenchHarness, EvaluationResult
from ..scoring import TrajectoryScorer
from ..utils.github import PackFetcher
from ..utils.epoch_context import generate_epoch_context, render_context_preamble
from ..utils.commitments import MinerCommitment, fetch_all_commitments
from ..utils.ncd import is_too_similar, pack_similarity

logger = logging.getLogger(__name__)

OWNER_UID = 74
EVAL_START_BLOCK = 0
# TODO: Set SHADOW_MODE = False for official mainnet launch.
# Shadow mode runs real evals and logs results, but always sets weights to owner UID 74.
SHADOW_MODE = True


class TrajectoryValidator:
    """TrajectoryRL validator that evaluates policy packs using ClawBench.

    The validator:
    1. Reads on-chain commitments from miners
    2. Fetches and verifies packs from miners' public HTTP URLs
    3. Checks NCD similarity against current winner (anti-copy)
    4. Runs ALL ClawBench scenarios
    5. Updates per-scenario EMA (keyed by miner hotkey, resets on pack change)
    6. Sets on-chain weights (winner-take-all or bootstrap)
    7. Re-sets weights every tempo (~72 min) for convergence

    Example:
        >>> config = ValidatorConfig.from_env()
        >>> validator = TrajectoryValidator(config)
        >>> await validator.run()
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config

        self._setup_logging()

        logger.info("=" * 60)
        logger.info("TrajectoryRL Validator v2.0.0")
        logger.info("=" * 60)

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

        logger.info("Initializing ClawBench harness...")
        self.harness = ClawBenchHarness(
            clawbench_path=config.clawbench_path,
            timeout=config.timeout_per_scenario,
        )

        self.scorer = TrajectoryScorer(
            rho_reliability=config.rho_reliability,
            consensus_epsilon=config.consensus_epsilon,
            bootstrap_threshold=config.bootstrap_threshold,
        )

        logger.info("Initializing pack fetcher...")
        self.pack_fetcher = PackFetcher(
            cache_dir=config.pack_cache_dir,
        )

        # Per-scenario EMA state: {hotkey: {scenario: ema_value}}
        self.ema_scores: Dict[str, Dict[str, float]] = {}

        # Per-scenario cost EMA: {hotkey: {scenario: ema_cost_usd}}
        self.ema_costs: Dict[str, Dict[str, float]] = {}

        # Per-scenario qualification (latest, not EMA): {hotkey: {scenario: bool}}
        self.scenario_qualified: Dict[str, Dict[str, bool]] = {}

        # Track the pack_hash that each hotkey's EMA is based on.
        # EMA resets when pack_hash changes.
        self._ema_pack_hash: Dict[str, str] = {}

        # Last eval block for each hotkey (for rate-limiting and inactivity)
        self.last_eval_block: Dict[str, int] = {}

        # Current winner tracking (for NCD comparison)
        self.current_winner_pack: Optional[dict] = None
        self.current_winner_hotkey: Optional[str] = None

        # Pack content cache: pack_hash -> pack dict
        self._pack_by_hash: Dict[str, dict] = {}

        # Packs by hotkey (populated during evaluation for winner tracking)
        self._hotkey_packs: Dict[str, dict] = {}

        # First-mover tracking: {hotkey: (best_score, first_block_number)}
        self.first_mover_data: Dict[str, Tuple[float, float]] = {}

        # Tracks which UID each hotkey was last evaluated at for re-registration detection.
        self._hotkey_uid_map: Dict[str, int] = {}

        # Weight cadence tracking
        self.last_weight_block: int = 0

        # Load scenarios
        self.scenarios = self._load_scenarios()
        logger.info(
            f"Loaded {len(self.scenarios)} scenarios: "
            f"{list(self.scenarios.keys())}"
        )

        # Compute scenario config hash for EMA invalidation on pool change
        self._scenario_config_hash = self._compute_scenario_config_hash()

        # Load persisted EMA state
        self._load_ema_state()

        logger.info("Validator initialization complete!")

    # ------------------------------------------------------------------
    # EMA persistence
    # ------------------------------------------------------------------

    def _compute_scenario_config_hash(self) -> str:
        """Hash the scenario configuration for detecting pool changes."""
        config_str = json.dumps(sorted(self.scenarios.keys()))
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _load_ema_state(self):
        """Load persisted EMA state from disk."""
        path = self.config.ema_state_path
        if not path.exists():
            logger.info("No persisted EMA state found, starting fresh")
            return
        try:
            data = json.loads(path.read_text())
            if data.get("scenario_config_hash") != self._scenario_config_hash:
                logger.info(
                    "Scenario pool changed, invalidating all EMA state"
                )
                return

            self.ema_scores = data.get("ema_scores", {})
            self.ema_costs = data.get("ema_costs", {})
            self.scenario_qualified = data.get("scenario_qualified", {})
            self._ema_pack_hash = data.get("ema_pack_hash", {})
            self.last_eval_block = {
                k: int(v) for k, v in data.get("last_eval_block", {}).items()
            }
            self.first_mover_data = {
                k: (v[0], v[1])
                for k, v in data.get("first_mover_data", {}).items()
            }

            logger.info(
                f"Loaded EMA state: {len(self.ema_scores)} hotkeys, "
                f"{len(self.first_mover_data)} first-mover entries"
            )
        except Exception as e:
            logger.warning(f"Failed to load EMA state: {e}, starting fresh")

    def _save_ema_state(self):
        """Persist EMA state to disk for restart recovery."""
        data = {
            "scenario_config_hash": self._scenario_config_hash,
            "ema_scores": self.ema_scores,
            "ema_costs": self.ema_costs,
            "scenario_qualified": self.scenario_qualified,
            "ema_pack_hash": self._ema_pack_hash,
            "last_eval_block": self.last_eval_block,
            "first_mover_data": self.first_mover_data,
        }
        try:
            self.config.ema_state_path.write_text(
                json.dumps(data, indent=2, sort_keys=True)
            )
        except Exception as e:
            logger.warning(f"Failed to save EMA state: {e}")

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def _update_ema(
        self,
        hotkey: str,
        pack_hash: str,
        scenario_scores: Dict[str, float],
        scenario_costs: Optional[Dict[str, float]] = None,
        scenario_qualified: Optional[Dict[str, bool]] = None,
    ):
        """Update per-scenario EMA for a miner hotkey.

        Resets EMA when pack_hash changes (new pack = new observations).
        """
        if self._ema_pack_hash.get(hotkey) != pack_hash:
            logger.info(
                f"Hotkey {hotkey[:8]}: pack changed "
                f"({self._ema_pack_hash.get(hotkey, 'none')[:8]} -> {pack_hash[:8]}), "
                f"resetting EMA"
            )
            self.ema_scores[hotkey] = {}
            self.ema_costs[hotkey] = {}
            self.scenario_qualified[hotkey] = {}
            self._ema_pack_hash[hotkey] = pack_hash

        # Score EMA (informational, kept for logging)
        alpha = self.config.ema_alpha
        if hotkey not in self.ema_scores:
            self.ema_scores[hotkey] = {}

        for scenario, score in scenario_scores.items():
            prev = self.ema_scores[hotkey].get(scenario)
            if prev is None:
                self.ema_scores[hotkey][scenario] = score
            else:
                self.ema_scores[hotkey][scenario] = (
                    alpha * score + (1 - alpha) * prev
                )

        # Cost EMA
        if scenario_costs:
            cost_alpha = self.config.cost_ema_alpha
            if hotkey not in self.ema_costs:
                self.ema_costs[hotkey] = {}
            for scenario, cost in scenario_costs.items():
                if cost is None:
                    continue
                prev = self.ema_costs[hotkey].get(scenario)
                if prev is None:
                    self.ema_costs[hotkey][scenario] = cost
                else:
                    self.ema_costs[hotkey][scenario] = (
                        cost_alpha * cost + (1 - cost_alpha) * prev
                    )

        # Qualification: latest result, not EMA (binary doesn't smooth well)
        if scenario_qualified:
            if hotkey not in self.scenario_qualified:
                self.scenario_qualified[hotkey] = {}
            self.scenario_qualified[hotkey].update(scenario_qualified)

    def compute_final_score_from_ema(self, hotkey: str) -> float:
        """Compute final_score[hotkey] from smoothed per-scenario EMA values.

        final_score = weighted_mean - ρ * weighted_variance
        """
        ema = self.ema_scores.get(hotkey, {})
        if not ema:
            return 0.0

        scenario_weights = {
            name: cfg.get("weight", 1.0)
            for name, cfg in self.scenarios.items()
        }

        total_weight = 0.0
        weighted_sum = 0.0
        for scenario, score in ema.items():
            w = scenario_weights.get(scenario, 1.0)
            weighted_sum += w * score
            total_weight += w

        if total_weight == 0:
            return 0.0

        mean_score = weighted_sum / total_weight

        weighted_var_sum = 0.0
        for scenario, score in ema.items():
            w = scenario_weights.get(scenario, 1.0)
            weighted_var_sum += w * (score - mean_score) ** 2
        variance = weighted_var_sum / total_weight

        final = mean_score - self.config.rho_reliability * variance
        return max(0.0, min(1.0, final))

    def compute_total_cost_from_ema(self, hotkey: str) -> Optional[float]:
        """Compute weighted average cost from per-scenario cost EMA.

        Returns None if no cost data available.
        """
        ema = self.ema_costs.get(hotkey, {})
        if not ema:
            return None

        scenario_weights = {
            name: cfg.get("weight", 1.0)
            for name, cfg in self.scenarios.items()
        }

        total_weight = 0.0
        weighted_sum = 0.0
        for scenario, cost in ema.items():
            w = scenario_weights.get(scenario, 1.0)
            weighted_sum += w * cost
            total_weight += w

        if total_weight == 0:
            return None

        return weighted_sum / total_weight

    def is_fully_qualified(self, hotkey: str) -> bool:
        """Check if a miner passes the qualification gate on all scenarios."""
        qualified = self.scenario_qualified.get(hotkey, {})
        if not qualified:
            return False
        # Must have qualification data for all scenarios
        for scenario_name in self.scenarios:
            if not qualified.get(scenario_name, False):
                return False
        return True

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
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_epoch_seed(epoch: int, netuid: int = 11) -> int:
        raw = f"trajectoryrl-{netuid}-epoch-{epoch}".encode()
        return int(hashlib.sha256(raw).hexdigest()[:8], 16)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        """Main validator loop with dual cadence.

        - eval_interval (~4h / 1200 blocks): evaluate marked packs, update EMA.
        - tempo (~72 min / 360 blocks): compute weights from EMA, set_weights.
        """
        logger.info("Starting validator main loop...")
        logger.info(
            f"Eval interval: {self.config.eval_interval_blocks} blocks "
            f"(~{self.config.eval_interval_blocks * 12 // 3600}h)"
        )
        logger.info(
            f"Weight interval: {self.config.weight_interval_blocks} blocks "
            f"(~{self.config.weight_interval_blocks * 12 // 60}min)"
        )

        last_eval_sync_block: int = 0

        while True:
            try:
                current_block = self.subtensor.get_current_block()

                # --- Pre-launch phase: fallback weights only ---
                if current_block < EVAL_START_BLOCK:
                    blocks_since_weights = current_block - self.last_weight_block
                    if blocks_since_weights >= self.config.weight_interval_blocks:
                        logger.info(
                            f"Pre-launch phase (block {current_block} < "
                            f"{EVAL_START_BLOCK}), setting fallback weights"
                        )
                        await self._set_fallback_weights()
                        self.last_weight_block = current_block
                    await asyncio.sleep(60)
                    continue

                # --- Eval cadence: evaluate new/changed packs ---
                blocks_since_eval = current_block - last_eval_sync_block
                if blocks_since_eval >= self.config.eval_interval_blocks:
                    logger.info("=" * 60)
                    logger.info(
                        f"Evaluation cycle at block {current_block} "
                        f"({blocks_since_eval} blocks since last)"
                    )
                    logger.info("=" * 60)

                    await self._run_evaluation_cycle(current_block)
                    last_eval_sync_block = self.subtensor.get_current_block()
                    self.last_weight_block = last_eval_sync_block

                    self.pack_fetcher.cleanup_cache(
                        self.config.pack_cache_max_size
                    )
                    self._save_ema_state()

                # --- Tempo cadence: re-set weights ---
                current_block = self.subtensor.get_current_block()
                blocks_since_weights = current_block - self.last_weight_block
                if blocks_since_weights >= self.config.weight_interval_blocks:
                    logger.info(
                        f"Tempo weight refresh at block {current_block} "
                        f"({blocks_since_weights} blocks since last)"
                    )
                    await self._compute_and_set_weights(current_block)
                    self.last_weight_block = current_block

                await asyncio.sleep(60)

            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                self._save_ema_state()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Evaluation cycle
    # ------------------------------------------------------------------

    async def _run_evaluation_cycle(self, current_block: int):
        """Run one evaluation cycle.

        1. Sync metagraph, read on-chain commitments
        2. For each miner hotkey with valid commitment:
           - If pack_hash changed since last eval: mark for re-evaluation
           - If time since last eval >= eval_interval: mark for re-evaluation
        3. Evaluate marked packs on the full scenario set
        4. Update per-scenario EMA for evaluated packs
        5. Set weights from EMA scores
        """
        # Epoch seed for context variation
        epoch = current_block // self.config.eval_interval_blocks
        epoch_seed = self.compute_epoch_seed(epoch, self.config.netuid)
        epoch_ctx = generate_epoch_context(epoch_seed)
        context_preamble = render_context_preamble(epoch_ctx)
        user_context = epoch_ctx.to_user_context()

        logger.info(
            f"Eval cycle: block={current_block}, seed={epoch_seed}, "
            f"context=[{epoch_ctx.user_name}, {epoch_ctx.user_role}]"
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

        # 3. Filter to non-validator miners
        active_commitments = self._filter_active_commitments(commitments)
        logger.info(f"Active miners: {len(active_commitments)}")

        if not active_commitments:
            logger.warning("No active miners with valid commitments!")
            await self._set_fallback_weights()
            return

        # 4. Determine which miners need evaluation
        eval_scenarios = sorted(self.scenarios.keys())
        evaluated_count = 0

        for uid, commitment in active_commitments.items():
            hotkey = commitment.hotkey

            needs_eval = self._needs_evaluation(
                hotkey, commitment.pack_hash, current_block
            )
            if not needs_eval:
                logger.debug(
                    f"Miner {uid} ({hotkey[:8]}): skipping, "
                    f"within eval interval"
                )
                continue

            eval_result = await self._evaluate_miner(
                uid, commitment, eval_scenarios, epoch_seed,
                context_preamble, user_context,
            )

            if eval_result is not None:
                self._update_ema(
                    hotkey, commitment.pack_hash,
                    scenario_scores=eval_result["scores"],
                    scenario_costs=eval_result.get("costs"),
                    scenario_qualified=eval_result.get("qualified"),
                )
                self.last_eval_block[hotkey] = current_block
                evaluated_count += 1

                # First-mover tracks cost (lower = better)
                total_cost = self.compute_total_cost_from_ema(hotkey)
                if total_cost is not None:
                    self._update_first_mover(
                        uid, hotkey, total_cost, float(commitment.block_number)
                    )
                else:
                    # Fallback: use score-based first-mover if no cost data
                    final = self.compute_final_score_from_ema(hotkey)
                    if hotkey not in self.first_mover_data:
                        self.first_mover_data[hotkey] = (final, float(commitment.block_number))

        logger.info(f"Evaluated {evaluated_count} miners this cycle")

        # 5. Set weights from EMA scores
        await self._compute_and_set_weights(current_block)

    def _needs_evaluation(
        self, hotkey: str, pack_hash: str, current_block: int
    ) -> bool:
        """Check if a miner needs re-evaluation.

        Returns True if:
        - pack_hash changed since last eval
        - Time since last eval >= eval_interval
        - Never evaluated before
        """
        if self._ema_pack_hash.get(hotkey) != pack_hash:
            logger.info(
                f"Hotkey {hotkey[:8]}: pack_hash changed, marking for eval"
            )
            return True

        last_block = self.last_eval_block.get(hotkey)
        if last_block is None:
            return True

        blocks_since = current_block - last_block
        if blocks_since >= self.config.eval_interval_blocks:
            return True

        return False

    def _filter_active_commitments(
        self,
        commitments: Dict[int, MinerCommitment],
    ) -> Dict[int, MinerCommitment]:
        """Filter commitments to non-validator miners."""
        active: Dict[int, MinerCommitment] = {}
        for uid, commitment in commitments.items():
            if uid < len(self.metagraph.validator_permit) and self.metagraph.validator_permit[uid]:
                continue
            active[uid] = commitment
        return active

    def _get_active_miners_from_commitments(
        self,
        commitments: Dict[int, MinerCommitment],
        current_block: int,
    ) -> Dict[int, MinerCommitment]:
        """Filter commitments to active miners considering inactivity.

        A miner is active if:
        1. Has a parseable on-chain commitment
        2. Is not a validator
        3. Has been evaluated within inactivity_blocks
        """
        active: Dict[int, MinerCommitment] = {}
        for uid, commitment in commitments.items():
            if uid < len(self.metagraph.validator_permit) and self.metagraph.validator_permit[uid]:
                continue

            hotkey = commitment.hotkey
            last_block = self.last_eval_block.get(hotkey)

            # Never evaluated = not yet active (but will be evaluated this cycle)
            if last_block is not None:
                blocks_since = current_block - last_block
                if blocks_since > self.config.inactivity_blocks:
                    logger.info(
                        f"Miner {uid} ({hotkey[:8]}): inactive for "
                        f"{blocks_since} blocks > {self.config.inactivity_blocks}, "
                        f"removing first-mover protection"
                    )
                    self.first_mover_data.pop(hotkey, None)
                    self._hotkey_uid_map.pop(hotkey, None)
                    continue

            active[uid] = commitment

        return active

    # ------------------------------------------------------------------
    # First-mover tracking
    # ------------------------------------------------------------------

    def _update_first_mover(
        self,
        miner_uid: int,
        hotkey: str,
        cost: float,
        block_number: float,
    ) -> None:
        """Detect re-registration and update first-mover data for a miner.

        Tracks best (lowest) cost. Lower cost = better.
        """
        prev_uid = self._hotkey_uid_map.get(hotkey)
        if prev_uid is not None and prev_uid != miner_uid:
            logger.info(
                f"Miner {miner_uid}: hotkey {hotkey[:8]} previously at UID "
                f"{prev_uid}; re-registration detected, resetting first-mover data"
            )
            self.first_mover_data.pop(hotkey, None)
        self._hotkey_uid_map[hotkey] = miner_uid

        if hotkey not in self.first_mover_data:
            self.first_mover_data[hotkey] = (cost, block_number)
            logger.info(
                f"Miner {miner_uid}: First submission "
                f"(cost=${cost:.4f}, block={block_number:.0f})"
            )
        elif cost < self.first_mover_data[hotkey][0]:
            original_block = self.first_mover_data[hotkey][1]
            self.first_mover_data[hotkey] = (cost, original_block)
            logger.info(
                f"Miner {miner_uid}: Cost improved to ${cost:.4f}"
            )

    # ------------------------------------------------------------------
    # Miner evaluation
    # ------------------------------------------------------------------

    async def _evaluate_miner(
        self,
        miner_uid: int,
        commitment: MinerCommitment,
        eval_scenarios: List[str],
        epoch_seed: int,
        context_preamble: str = "",
        user_context: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Evaluate a single miner on all scenarios.

        Returns:
            Dict with keys "scores", "costs", "qualified" mapping
            scenario_name to values, or None if pre-evaluation checks fail.
        """
        logger.info(
            f"Evaluating miner {miner_uid} "
            f"(hotkey={commitment.hotkey[:8]}, hash={commitment.pack_hash[:12]}...)"
        )

        # Step 1: Fetch and verify pack from HTTP URL
        verification = await self.pack_fetcher.verify_submission(
            pack_url=commitment.pack_url,
            pack_hash=commitment.pack_hash,
        )

        if not verification.valid:
            logger.warning(
                f"Miner {miner_uid}: Pack verification failed: "
                f"{verification.error}"
            )
            return None

        pack = verification.pack_content

        # Step 2: Schema validation
        lint_result = validate_opp_schema(pack)
        if not lint_result.passed:
            logger.warning(
                f"Miner {miner_uid}: Schema failed: {lint_result.issues}"
            )
            return None

        # Step 3: NCD similarity check vs current winner.
        # Current winner is exempt (must defend their position).
        if commitment.hotkey != self.current_winner_hotkey and is_too_similar(
            pack, self.current_winner_pack, self.config.similarity_threshold
        ):
            sim = pack_similarity(pack, self.current_winner_pack)
            logger.warning(
                f"Miner {miner_uid}: NCD similarity={sim:.3f} "
                f">= {self.config.similarity_threshold}, rejected"
            )
            return None

        self._hotkey_packs[commitment.hotkey] = pack
        self._pack_by_hash[commitment.pack_hash] = pack

        # Step 4: Run ALL scenarios
        scenario_scores: Dict[str, float] = {}
        scenario_costs: Dict[str, float] = {}
        scenario_qualified: Dict[str, bool] = {}

        for scenario_name in eval_scenarios:
            try:
                result = await self.harness.evaluate_pack_consensus(
                    pack=pack,
                    scenario_name=scenario_name,
                    num_runs=self.config.seeds_per_task,
                    base_seed=epoch_seed,
                    context_preamble=context_preamble,
                    user_context=user_context,
                )
                scenario_scores[scenario_name] = result.score
                scenario_qualified[scenario_name] = result.success
                if result.cost_usd is not None:
                    scenario_costs[scenario_name] = result.cost_usd

                cost_str = f", cost=${result.cost_usd:.4f}" if result.cost_usd is not None else ""
                gate_str = "PASS" if result.success else "FAIL"
                logger.info(
                    f"Miner {miner_uid}: {scenario_name} -> "
                    f"score={result.score:.3f}{cost_str}, gate={gate_str}"
                )
                if result.model_usage:
                    for m in result.model_usage:
                        logger.info(
                            f"Miner {miner_uid}: {scenario_name}   "
                            f"{m.get('model', '?')}: "
                            f"${m.get('cost_usd', 0):.4f} "
                            f"({m.get('count', 0)} calls)"
                        )
            except Exception as e:
                logger.error(
                    f"Miner {miner_uid}: {scenario_name} failed: {e}",
                    exc_info=True,
                )
                scenario_scores[scenario_name] = 0.0
                scenario_qualified[scenario_name] = False

        if not scenario_scores:
            logger.warning(f"Miner {miner_uid}: No scenario results!")
            return None

        return {
            "scores": scenario_scores,
            "costs": scenario_costs,
            "qualified": scenario_qualified,
        }

    # ------------------------------------------------------------------
    # Weight setting
    # ------------------------------------------------------------------

    async def _compute_and_set_weights(self, current_block: int):
        """Compute weights from EMA scores and set on-chain.

        Maps hotkey -> UID via metagraph, applies winner selection,
        and calls set_weights.
        """
        try:
            self.metagraph.sync(subtensor=self.subtensor)
        except Exception as e:
            logger.warning(f"Metagraph sync failed, using cached: {e}")

        # Read commitments to know which miners are currently valid
        commitments = fetch_all_commitments(
            self.subtensor, self.config.netuid, self.metagraph
        )
        active = self._get_active_miners_from_commitments(
            commitments, current_block
        )

        if not active:
            logger.warning("No active miners for weight setting")
            await self._set_fallback_weights()
            return

        # Build scores, costs, and qualification from EMA
        scores: Dict[int, float] = {}
        costs: Dict[int, float] = {}
        qualified: Dict[int, bool] = {}
        uid_to_hotkey: Dict[int, str] = {}

        for uid, commitment in active.items():
            hotkey = commitment.hotkey
            final = self.compute_final_score_from_ema(hotkey)
            if final > 0:
                scores[uid] = final
                uid_to_hotkey[uid] = hotkey

                total_cost = self.compute_total_cost_from_ema(hotkey)
                if total_cost is not None:
                    costs[uid] = total_cost
                qualified[uid] = self.is_fully_qualified(hotkey)

        if not scores:
            logger.warning("All miners have zero EMA score")
            await self._set_fallback_weights()
            return

        num_active = len(scores)

        # Use cost-based winner selection if cost data available
        if costs:
            weights_dict = self.scorer.select_winner_by_cost(
                costs=costs,
                qualified=qualified,
                first_mover_data=self.first_mover_data,
                cost_delta=self.config.cost_delta,
                num_active_miners=num_active,
                uid_to_hotkey=uid_to_hotkey,
            )
        else:
            # Fallback to score-based selection (transition period)
            logger.info("No cost data available, using score-based selection")
            weights_dict = self.scorer.select_winner(
                scores=scores,
                first_mover_data=self.first_mover_data,
                delta=self.config.delta_threshold,
                num_active_miners=num_active,
                uid_to_hotkey=uid_to_hotkey,
            )

        # Track current winner for NCD comparison
        if weights_dict:
            winner_uid = max(weights_dict, key=weights_dict.get)
            if weights_dict.get(winner_uid, 0) > 0:
                winner_hotkey = uid_to_hotkey.get(winner_uid)
                if winner_hotkey:
                    winner_pack = self._hotkey_packs.get(winner_hotkey)
                    if winner_pack is not None:
                        self.current_winner_hotkey = winner_hotkey
                        self.current_winner_pack = winner_pack
                    else:
                        logger.debug(
                            f"Winner hotkey {winner_hotkey[:8]} not in local packs, "
                            f"keeping previous winner state for NCD protection"
                        )

        # Log results
        logger.info("=" * 60)
        logger.info("WEIGHT RESULTS")
        logger.info("=" * 60)
        for uid, weight in sorted(
            weights_dict.items(),
            key=lambda x: costs.get(x[0], scores.get(x[0], 0)),
        ):
            marker = ""
            hk = uid_to_hotkey.get(uid, "?")
            if weight > 0:
                marker = " <- WINNER" if weight == 1.0 else f" <- TOP-{sum(1 for w in weights_dict.values() if w >= weight)}"
            gate = "PASS" if qualified.get(uid, False) else "FAIL"
            cost_str = f"${costs[uid]:.4f}" if uid in costs else "n/a"
            logger.info(
                f"  Miner {uid} ({hk[:8]}): weight={weight:.4f}, "
                f"cost={cost_str}, gate={gate}, "
                f"score={scores.get(uid, 0):.3f}{marker}"
            )

        # Set weights on chain
        if SHADOW_MODE:
            logger.info("SHADOW MODE: eval complete, setting fallback weights to owner UID")
            await self._set_fallback_weights()
        else:
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

    async def _set_fallback_weights(self):
        """Set weights to subnet owner UID when no miners qualify.

        Directs all emission to the subnet owner so it is not wasted on
        inactive or unregistered UIDs.  The validator must always call
        set_weights every tempo to avoid being deregistered by the chain.
        """
        try:
            uids = [OWNER_UID]
            weights = [1.0]

            logger.info(
                f"No eligible miners — setting fallback weight to "
                f"owner UID {OWNER_UID}"
            )
            self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )
            logger.info(f"Fallback weights set (owner UID {OWNER_UID})")
        except Exception as e:
            logger.error(f"Error setting fallback weights: {e}", exc_info=True)


async def main():
    """Entry point for validator."""
    config = ValidatorConfig.from_env()
    validator = TrajectoryValidator(config)
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
