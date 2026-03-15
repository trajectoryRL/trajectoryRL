"""TrajectoryRL Validator — Main validator implementation.

Architecture (v4.0 — LLM-as-Judge):
    1. Continuous evaluation loop with dual cadence:
       - eval_interval (~24h): re-evaluate all active packs
       - tempo (~72 min): compute weights from qualification + cost, set_weights
    2. Read on-chain commitments (subtensor.get_all_commitments)
    3. Fetch packs from miners' public HTTP URLs
    4. Validate schema + NCD similarity check
    5. Phase 1: LLM pack integrity analysis (static, cached by pack_hash)
    6. Run ALL ClawBench scenarios (single episode per scenario)
    7. Phase 2: LLM trajectory judge per scenario (replaces regex scoring)
    8. Update per-scenario cost EMA (keyed by miner hotkey)
    9. Set on-chain weights (winner-take-all / bootstrap by cost)

Score EMA removed in v4.0 — qualification is a binary judge verdict.
Cost EMA retained — cost genuinely varies across runs.
Each validator operates independently — YC3 aggregates on-chain.
"""

import asyncio
import hashlib
import json
import logging
import time
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

from ..utils.opp_schema import validate_opp_schema
from ..utils.config import ValidatorConfig
from ..utils.clawbench import ClawBenchHarness, EvaluationResult
from ..scoring import TrajectoryScorer
from ..utils.github import PackFetcher
from ..utils.epoch_context import generate_epoch_context, render_context_preamble
from ..utils.commitments import MinerCommitment, fetch_all_commitments
from ..utils.ncd import deduplicate_packs
from ..utils.status_reporter import heartbeat, submit_eval
from ..utils.llm_judge import PackIntegrityJudge, TrajectoryJudge
from .. import __version__

logger = logging.getLogger(__name__)

OWNER_UID = 74
BURN_FRACTION = 0.50  # 50% of miner emissions burned via owner UID
EVAL_START_BLOCK = 0
# Shadow mode runs real evals and logs results, but always sets weights to owner UID 74.
SHADOW_MODE = False


class TrajectoryValidator:
    """TrajectoryRL validator that evaluates policy packs using ClawBench.

    The validator (v4.0):
    1. Reads on-chain commitments from miners
    2. Fetches and verifies packs from miners' public HTTP URLs
    3. Phase 1: LLM pack integrity analysis (rejects gaming packs)
    4. Runs ALL ClawBench scenarios
    5. Phase 2: LLM trajectory judge per scenario (replaces regex scoring)
    6. Updates per-scenario cost EMA (keyed by miner hotkey)
    7. Sets on-chain weights (winner-take-all or bootstrap by cost)
    8. Re-sets weights every tempo (~72 min) for convergence

    Example:
        >>> config = ValidatorConfig.from_env()
        >>> validator = TrajectoryValidator(config)
        >>> await validator.run()
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config

        self._setup_logging()

        logger.info("=" * 60)
        logger.info(f"TrajectoryRL Validator v{__version__}")
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
            clawbench_default_model=config.clawbench_default_model,
            clawbench_api_key=config.clawbench_api_key,
            clawbench_base_url=config.clawbench_base_url,
        )

        self.scorer = TrajectoryScorer(
            consensus_epsilon=config.consensus_epsilon,
            bootstrap_threshold=config.bootstrap_threshold,
        )

        # LLM-as-Judge (v4.0): defaults to same LLM as ClawBench if not set
        judge_model = config.judge_model or config.clawbench_default_model
        judge_api_key = config.judge_api_key or config.clawbench_api_key
        judge_base_url = config.judge_base_url or config.clawbench_base_url
        self._judge_model = judge_model
        self._judge_base_url = judge_base_url
        logger.info("Initializing LLM judges (model=%s)...", judge_model)

        self.integrity_judge = PackIntegrityJudge(
            model=judge_model,
            api_key=judge_api_key,
            base_url=judge_base_url,
        )
        self.trajectory_judge = TrajectoryJudge(
            model=judge_model,
            api_key=judge_api_key,
            base_url=judge_base_url,
        )

        logger.info("Initializing pack fetcher...")
        self.pack_fetcher = PackFetcher(
            cache_dir=config.pack_cache_dir,
        )

        # Per-scenario cost EMA: {hotkey: {scenario: ema_cost_usd}}
        self.ema_costs: Dict[str, Dict[str, float]] = {}

        # Per-scenario qualification (latest judge verdict): {hotkey: {scenario: bool}}
        self.scenario_qualified: Dict[str, Dict[str, bool]] = {}

        # Latest token usage per hotkey/scenario: {hotkey: {scenario: {input_tokens, ...}}}
        self.latest_token_usage: Dict[str, Dict[str, Dict[str, int]]] = {}

        # Latest model usage per hotkey/scenario: {hotkey: {scenario: [model_entry, ...]}}
        self.latest_model_usage: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        # Track the pack_hash that each hotkey's EMA is based on.
        # EMA resets when pack_hash changes.
        self._ema_pack_hash: Dict[str, str] = {}

        # Last eval block for each hotkey (for rate-limiting and inactivity)
        self.last_eval_block: Dict[str, int] = {}

        # Pack content cache: pack_hash -> pack dict
        self._pack_by_hash: Dict[str, dict] = {}

        # Packs by hotkey (populated during evaluation for NCD dedup)
        self._hotkey_packs: Dict[str, dict] = {}

        # First-mover tracking: {hotkey: (best_score, first_block_number)}
        self.first_mover_data: Dict[str, Tuple[float, float]] = {}

        # Tracks which UID each hotkey was last evaluated at for re-registration detection.
        self._hotkey_uid_map: Dict[str, int] = {}

        # Weight cadence tracking
        self.last_weight_block: int = 0

        # Eval count per hotkey for the current pack (resets on pack change)
        self._eval_counts: Dict[str, int] = {}

        # Timestamp of the most recent successful set_weights call
        self._last_set_weights_at: Optional[int] = None

        # Last computed weights, cached for mid-eval tempo replays
        self._last_weights_uids: Optional[List[int]] = None
        self._last_weights: Optional[List[float]] = None

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

            # Load integrity judge cache if present
            integrity_cache = data.get("integrity_cache")
            if integrity_cache:
                self.integrity_judge.load_cache(integrity_cache)

            logger.info(
                f"Loaded EMA state: {len(self.ema_costs)} hotkeys, "
                f"{len(self.first_mover_data)} first-mover entries"
            )
        except Exception as e:
            logger.warning(f"Failed to load EMA state: {e}, starting fresh")

    def _save_ema_state(self):
        """Persist EMA state to disk for restart recovery."""
        data = {
            "scenario_config_hash": self._scenario_config_hash,
            "ema_costs": self.ema_costs,
            "scenario_qualified": self.scenario_qualified,
            "ema_pack_hash": self._ema_pack_hash,
            "last_eval_block": self.last_eval_block,
            "first_mover_data": self.first_mover_data,
            "integrity_cache": self.integrity_judge.dump_cache(),
        }
        try:
            self.config.ema_state_path.parent.mkdir(parents=True, exist_ok=True)
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
        scenario_costs: Optional[Dict[str, float]] = None,
        scenario_qualified: Optional[Dict[str, bool]] = None,
    ):
        """Update per-scenario cost EMA and qualification for a miner.

        Resets EMA when pack_hash changes (new pack = new observations).
        Score EMA removed in v4.0 — qualification is a binary judge verdict.
        """
        if self._ema_pack_hash.get(hotkey) != pack_hash:
            logger.info(
                f"Hotkey {hotkey[:8]}: pack changed "
                f"({self._ema_pack_hash.get(hotkey, 'none')[:8]} -> {pack_hash[:8]}), "
                f"resetting EMA"
            )
            self.ema_costs[hotkey] = {}
            self.scenario_qualified[hotkey] = {}
            self.latest_token_usage.pop(hotkey, None)
            self.latest_model_usage.pop(hotkey, None)
            self._ema_pack_hash[hotkey] = pack_hash

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

        # Qualification: latest judge verdict (binary, not smoothed)
        if scenario_qualified:
            if hotkey not in self.scenario_qualified:
                self.scenario_qualified[hotkey] = {}
            self.scenario_qualified[hotkey].update(scenario_qualified)

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

    async def _heartbeat_loop(self):
        """Send validator heartbeat every 10 minutes, independent of eval cycle."""
        while True:
            try:
                await heartbeat(
                    self.wallet,
                    last_set_weights_at=self._last_set_weights_at,
                )
            except Exception as e:
                logger.warning("Heartbeat error: %s", e)
            await asyncio.sleep(600)

    async def run(self):
        """Main validator loop with dual cadence.

        - eval_interval (~24h / 7200 blocks): evaluate marked packs, update EMA.
        - tempo (~72 min / 360 blocks): compute weights from EMA, set_weights.
        """
        self._start_time = time.time()

        asyncio.create_task(self._heartbeat_loop())

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

                    # Sync ClawBench to latest before evaluation
                    await self._sync_clawbench()

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
    # ClawBench auto-sync
    # ------------------------------------------------------------------

    async def _sync_clawbench(self) -> None:
        """Pull latest ClawBench from GitHub before evaluation.

        If the clawbench directory is a git repo (set up by entrypoint),
        does a fast-forward pull from origin/main.  On any failure, logs
        a warning and continues with the current version.
        """
        clawbench_path = self.config.clawbench_path
        if not (clawbench_path / ".git").exists():
            logger.debug(
                "ClawBench .git not found at %s — skipping auto-sync",
                clawbench_path,
            )
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "pull", "--ff-only", "origin", "main",
                cwd=str(clawbench_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30
            )
            if proc.returncode == 0:
                result = stdout.decode().strip()
                if "Already up to date" not in result:
                    logger.info("ClawBench updated: %s", result)
                else:
                    logger.debug("ClawBench already up to date")
            else:
                logger.warning(
                    "ClawBench sync failed (rc=%d): %s",
                    proc.returncode,
                    stderr.decode().strip(),
                )
        except asyncio.TimeoutError:
            logger.warning("ClawBench sync timed out (30s) — using current version")
        except Exception as e:
            logger.warning("ClawBench sync error: %s", e)

    # ------------------------------------------------------------------
    # LLM key check
    # ------------------------------------------------------------------

    def _check_llm_keys(self) -> bool:
        """Return True if a ClawBench LLM API key is configured."""
        return bool(self.config.clawbench_api_key)

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

        Falls back to owner UID weights when LLM keys are missing or
        all evaluations fail (likely LLM API errors).
        """
        if not self._check_llm_keys():
            logger.warning(
                "CLAWBENCH_LLM_API_KEY not set. "
                "Skipping evaluation, setting fallback weights to owner UID.",
            )
            await self._set_fallback_weights(
                reason="No LLM API key configured"
            )
            return

        # Set weights from the previous eval's results before starting this
        # eval cycle. This caches the computed weights for mid-eval replays,
        # so the validator stays active on-chain even if eval takes longer
        # than one tempo window.
        logger.info("Setting weights from previous eval before starting eval cycle")
        await self._compute_and_set_weights(current_block)
        self.last_weight_block = current_block

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

        # 1. Clear per-cycle pack caches to prevent stale entries from
        # deregistered miners affecting NCD comparisons in the weight phase.
        self._hotkey_packs.clear()
        self._pack_by_hash.clear()

        # 2. Sync metagraph
        logger.info("Syncing metagraph...")
        self.metagraph.sync(subtensor=self.subtensor)

        # 3. Read on-chain commitments
        logger.info("Reading on-chain commitments...")
        commitments = fetch_all_commitments(
            self.subtensor, self.config.netuid, self.metagraph
        )
        logger.info(f"Found {len(commitments)} valid commitments")

        # 4. Filter to non-validator miners
        active_commitments = self._filter_active_commitments(commitments)
        logger.info(f"Active miners: {len(active_commitments)}")

        if not active_commitments:
            logger.warning("No active miners with valid commitments!")
            await self._set_fallback_weights()
            return

        # 5. Pack-hash pre-dedup: for miners with identical pack_hash,
        # only evaluate the first mover (lowest block_number).
        # Saves LLM API calls. Full NCD dedup happens in weight phase.
        # Skipped miners will have no entries in scores/costs/qualified
        # and receive weight 0 — this is intentional since their pack is
        # identical to the evaluated first mover.
        hash_earliest: Dict[str, Tuple[int, int]] = {}
        for uid, commitment in active_commitments.items():
            ph = commitment.pack_hash
            if ph not in hash_earliest or commitment.block_number < hash_earliest[ph][1]:
                hash_earliest[ph] = (uid, commitment.block_number)

        skip_uids: set = set()
        for uid, commitment in active_commitments.items():
            earliest_uid, _ = hash_earliest[commitment.pack_hash]
            if uid != earliest_uid:
                skip_uids.add(uid)
                logger.info(
                    f"Miner {uid} ({commitment.hotkey[:8]}): skipping eval "
                    f"(duplicate pack_hash={commitment.pack_hash[:12]})"
                )

        # 6. Evaluate miners
        eval_scenarios = sorted(self.scenarios.keys())
        evaluated_count = 0
        attempted_count = 0

        for uid, commitment in active_commitments.items():
            hotkey = commitment.hotkey

            if uid in skip_uids:
                continue

            needs_eval = self._needs_evaluation(
                hotkey, commitment.pack_hash, current_block
            )
            if not needs_eval:
                logger.debug(
                    f"Miner {uid} ({hotkey[:8]}): skipping, "
                    f"within eval interval"
                )
                continue

            attempted_count += 1
            eval_result = await self._evaluate_miner(
                uid, commitment, eval_scenarios, epoch_seed,
                context_preamble, user_context,
                block_height=current_block,
            )

            if eval_result is not None:
                ema_reset = self._ema_pack_hash.get(hotkey) != commitment.pack_hash
                if ema_reset:
                    self._eval_counts[hotkey] = 0
                self._eval_counts[hotkey] = self._eval_counts.get(hotkey, 0) + 1
                eval_count = self._eval_counts[hotkey]

                self._update_ema(
                    hotkey, commitment.pack_hash,
                    scenario_costs=eval_result.get("costs"),
                    scenario_qualified=eval_result.get("qualified"),
                )
                self.last_eval_block[hotkey] = current_block
                evaluated_count += 1

                # Store latest token & model usage for metadata reporting
                if eval_result.get("token_usage"):
                    self.latest_token_usage[hotkey] = eval_result["token_usage"]
                if eval_result.get("model_usage"):
                    self.latest_model_usage[hotkey] = eval_result["model_usage"]

                # Submit eval result to dashboard (fire-and-forget)
                asyncio.ensure_future(
                    self._fire_submit_eval(
                        uid, commitment, eval_result, eval_count, ema_reset, current_block
                    )
                )

                # First-mover tracks cost (lower = better)
                total_cost = self.compute_total_cost_from_ema(hotkey)
                if total_cost is not None:
                    self._update_first_mover(
                        uid, hotkey, total_cost, float(commitment.block_number)
                    )

            # Mid-eval tempo refresh: replay the last computed weights so
            # the validator stays active on-chain without exposing partial
            # current-cycle results.
            mid_block = self.subtensor.get_current_block()
            if mid_block - self.last_weight_block >= self.config.weight_interval_blocks:
                logger.info(
                    f"Mid-eval tempo refresh at block {mid_block} "
                    f"({mid_block - self.last_weight_block} blocks since last set_weights)"
                )
                await self._replay_last_weights()
                self.last_weight_block = mid_block

        logger.info(f"Evaluated {evaluated_count}/{attempted_count} miners this cycle")

        # Log EMA cost summary for all active miners
        if evaluated_count > 0:
            logger.info("-" * 40)
            logger.info("COST EMA SUMMARY (all active miners)")
            logger.info("-" * 40)
            for uid, commitment in active_commitments.items():
                hk = commitment.hotkey
                ema_cost = self.compute_total_cost_from_ema(hk)
                if ema_cost is not None:
                    per_scenario = self.ema_costs.get(hk, {})
                    scenario_str = ", ".join(
                        f"{s}=${c:.4f}" for s, c in sorted(per_scenario.items())
                    )
                    logger.info(
                        f"  Miner {uid} ({hk[:8]}): "
                        f"ema_total=${ema_cost:.4f} ({scenario_str})"
                    )
            logger.info("-" * 40)

        # All attempted evaluations failed — likely an LLM API issue
        if attempted_count > 0 and evaluated_count == 0:
            logger.error(
                f"All {attempted_count} miner evaluations failed. "
                "Possible LLM API key issue or service outage. "
                "Setting fallback weights to owner UID."
            )
            await self._set_fallback_weights(
                reason="All evaluations failed (LLM error)"
            )
            return

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
        """Filter commitments to non-validator miners, excluding blacklisted coldkeys."""
        blacklist = set(self.config.coldkey_blacklist)
        active: Dict[int, MinerCommitment] = {}
        for uid, commitment in commitments.items():
            if uid < len(self.metagraph.validator_permit) and self.metagraph.validator_permit[uid]:
                continue
            if blacklist:
                coldkey = self.metagraph.coldkeys[uid] if uid < len(self.metagraph.coldkeys) else None
                if coldkey in blacklist:
                    logger.info(
                        f"Miner {uid} ({commitment.hotkey[:8]}): skipping eval "
                        f"(coldkey {coldkey} is blacklisted)"
                    )
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
        block_height: int = 0,
    ) -> Optional[Dict]:
        """Evaluate a single miner on all scenarios.

        v4.0 flow:
        1. Fetch + verify pack
        2. Schema validation
        3. Phase 1: LLM integrity check (cached by pack_hash)
        4. Run episodes (single per scenario, no consensus voting)
        5. Phase 2: LLM trajectory judge per scenario
        6. Return costs + judge-based qualification

        Returns:
            Dict with keys "costs", "qualified" mapping
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

        # Step 3: Phase 1 — LLM pack integrity analysis (cached by pack_hash)
        integrity = self.integrity_judge.check_integrity(
            pack, pack_hash=commitment.pack_hash
        )
        if not integrity.passed:
            logger.warning(
                f"Miner {miner_uid}: Pack integrity FAILED: {integrity.summary}"
            )
            for flag in integrity.flags:
                logger.warning(
                    f"  Flag: {flag.type} ({flag.severity}): {flag.explanation}"
                )
            asyncio.ensure_future(
                submit_eval(
                    self.wallet,
                    miner_hotkey=commitment.hotkey,
                    miner_uid=miner_uid,
                    block_height=block_height,
                    score=0.0,
                    ema_score=0.0,
                    cost=0.0,
                    ema_cost=0.0,
                    weight=0.0,
                    qualified=False,
                    pack_url=commitment.pack_url,
                    pack_hash=commitment.pack_hash,
                    llm_base_url=self._judge_base_url,
                    llm_model=self._judge_model,
                    rejected=True,
                    rejection_stage="integrity_check",
                    rejection_detail=integrity.summary,
                )
            )
            return None

        if integrity.flags:
            logger.info(
                f"Miner {miner_uid}: Integrity passed with "
                f"{len(integrity.flags)} non-critical flags"
            )

        self._hotkey_packs[commitment.hotkey] = pack
        self._pack_by_hash[commitment.pack_hash] = pack

        # Step 4+5: Run episodes and judge trajectories
        scenario_costs: Dict[str, float] = {}
        scenario_qualified: Dict[str, bool] = {}
        scenario_token_usage: Dict[str, Dict[str, int]] = {}
        scenario_model_usage: Dict[str, List[Dict[str, Any]]] = {}
        scenario_judge_details: Dict[str, Dict[str, Any]] = {}

        for scenario_name in eval_scenarios:
            try:
                # Single episode per scenario (no consensus voting in v4.0)
                result = await self.harness.evaluate_pack(
                    pack=pack,
                    scenario_name=scenario_name,
                    seed=epoch_seed,
                    context_preamble=context_preamble,
                    user_context=user_context,
                )

                if result.error:
                    logger.warning(
                        f"Miner {miner_uid}: {scenario_name} episode error: "
                        f"{result.error}"
                    )
                    scenario_qualified[scenario_name] = False
                    continue

                if result.cost_usd is not None:
                    scenario_costs[scenario_name] = result.cost_usd
                if result.token_usage:
                    scenario_token_usage[scenario_name] = result.token_usage
                if result.model_usage:
                    scenario_model_usage[scenario_name] = result.model_usage

                # Phase 2: LLM trajectory judge
                scenario_config = self.scenarios.get(scenario_name, {})
                trajectory = result.trajectory or []
                judge_result = self.trajectory_judge.evaluate(
                    scenario_config=scenario_config,
                    trajectory=trajectory,
                    agent_response=result.response,
                )

                qualified = judge_result.qualification_gate
                scenario_qualified[scenario_name] = qualified

                # Store full judge details for dashboard reporting
                _criteria = judge_result.criteria_results
                _n = len(_criteria)
                _passed = sum(1 for cr in _criteria if cr.verdict == "PASS")
                _grounded = sum(1 for cr in _criteria if cr.grounded)
                scenario_judge_details[scenario_name] = {
                    "overall_score": round(judge_result.overall_score, 4),
                    "safety_passed": judge_result.safety_passed,
                    "correctness_passed": judge_result.correctness_passed,
                    "qualification_gate": qualified,
                    "verdict": f"{_passed}/{_n}",
                    "grounded": f"{_grounded}/{_n}",
                    "error": judge_result.error,
                }

                cost_str = (
                    f", cost=${result.cost_usd:.4f}"
                    if result.cost_usd is not None
                    else ""
                )
                gate_str = "PASS" if qualified else "FAIL"
                logger.info(
                    f"Miner {miner_uid}: {scenario_name} -> "
                    f"judge={judge_result.overall_score:.3f}{cost_str}, "
                    f"gate={gate_str}, tool_calls={result.tool_calls}"
                )
                if result.token_usage:
                    tu = result.token_usage
                    logger.info(
                        f"Miner {miner_uid}: {scenario_name}   "
                        f"tokens: input={tu.get('input_tokens', 0)}, "
                        f"output={tu.get('output_tokens', 0)}, "
                        f"cache_read={tu.get('cache_read_tokens', 0)}, "
                        f"cache_write={tu.get('cache_write_tokens', 0)}"
                    )

                if judge_result.error:
                    logger.warning(
                        f"Miner {miner_uid}: {scenario_name} judge error: "
                        f"{judge_result.error}"
                    )

                # Log per-criterion details
                for cr in judge_result.criteria_results:
                    if cr.verdict != "PASS":
                        logger.info(
                            f"  FAIL {cr.id}: {cr.justification}"
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
                scenario_qualified[scenario_name] = False

        if not scenario_qualified:
            logger.warning(f"Miner {miner_uid}: No scenario results!")
            return None

        # Log per-miner cost summary
        if scenario_costs:
            total_cost = sum(scenario_costs.values())
            cost_details = ", ".join(
                f"{s}=${c:.4f}" for s, c in scenario_costs.items()
            )
            logger.info(
                f"Miner {miner_uid}: Cost summary: "
                f"total=${total_cost:.4f} ({cost_details})"
            )

        return {
            "costs": scenario_costs,
            "qualified": scenario_qualified,
            "token_usage": scenario_token_usage,
            "model_usage": scenario_model_usage,
            "judge_details": scenario_judge_details,
        }

    # ------------------------------------------------------------------
    # Eval submission
    # ------------------------------------------------------------------

    async def _fire_submit_eval(
        self,
        uid: int,
        commitment: "MinerCommitment",
        eval_result: Dict,
        eval_count: int,
        ema_reset: bool,
        block_height: int,
    ) -> None:
        """Build and fire the /api/scores/submit payload for one miner eval.

        Fire-and-forget: any error is logged and discarded.
        """
        hotkey = commitment.hotkey
        scenario_weights = {
            name: cfg.get("weight", 1.0)
            for name, cfg in self.scenarios.items()
        }

        # v4.0: qualification is binary (LLM judge verdict), no score EMA.
        # Derive scores from qualified dict: 1.0 if passed, 0.0 if failed.
        raw_qualified = eval_result.get("qualified") or {}
        raw_costs = eval_result.get("costs") or {}

        # Aggregate raw score (weighted mean of binary qualification)
        total_w = sum(scenario_weights.get(s, 1.0) for s in raw_qualified)
        raw_score = (
            sum(scenario_weights.get(s, 1.0) * (1.0 if q else 0.0)
                for s, q in raw_qualified.items()) / total_w
            if total_w > 0 else 0.0
        )

        # Aggregate raw cost (weighted mean across scenarios)
        cost_total_w = sum(scenario_weights.get(s, 1.0) for s in raw_costs)
        raw_cost = (
            sum(scenario_weights.get(s, 1.0) * v for s, v in raw_costs.items()) / cost_total_w
            if cost_total_w > 0 else 0.0
        )

        # Per-scenario results
        scenario_results: Dict[str, Any] = {}
        for sname, q in raw_qualified.items():
            entry: Dict[str, Any] = {
                "score": 1.0 if q else 0.0,
                "weight": round(scenario_weights.get(sname, 1.0), 4),
                "qualified": q,
            }
            if sname in raw_costs:
                entry["cost"] = round(raw_costs[sname], 4)
                entry["ema_cost"] = round(self.ema_costs.get(hotkey, {}).get(sname, 0.0), 4)
            tu = (eval_result.get("token_usage") or {}).get(sname)
            if tu:
                entry["token_usage"] = tu
            mu = (eval_result.get("model_usage") or {}).get(sname)
            if mu:
                entry["model_usage"] = mu
            jd = (eval_result.get("judge_details") or {}).get(sname)
            if jd:
                entry["judge"] = jd
            scenario_results[sname] = entry

        await submit_eval(
            self.wallet,
            miner_hotkey=hotkey,
            miner_uid=uid,
            block_height=block_height,
            score=round(raw_score, 4),
            ema_score=round(raw_score, 4),
            cost=round(raw_cost, 4),
            ema_cost=round(self.compute_total_cost_from_ema(hotkey) or 0.0, 4),
            weight=0.0,
            qualified=self.is_fully_qualified(hotkey),
            pack_url=commitment.pack_url,
            pack_hash=commitment.pack_hash,
            eval_count=eval_count,
            ema_reset=ema_reset,
            scenario_results=scenario_results,
            llm_base_url=self._judge_base_url,
            llm_model=self._judge_model,
        )

    # ------------------------------------------------------------------
    # Weight setting
    # ------------------------------------------------------------------

    async def _compute_and_set_weights(self, current_block: int):
        """Compute weights from EMA scores, set on-chain, and cache the result.

        Maps hotkey -> UID via metagraph, applies winner selection,
        and calls set_weights. The resulting uids/weights are cached in
        self._last_weights_uids / self._last_weights for mid-eval replays.
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

        # Build costs and qualification from EMA state
        scores: Dict[int, float] = {}  # kept for report metadata compatibility
        costs: Dict[int, float] = {}
        qualified: Dict[int, bool] = {}
        uid_to_hotkey: Dict[int, str] = {}

        for uid, commitment in active.items():
            hotkey = commitment.hotkey
            total_cost = self.compute_total_cost_from_ema(hotkey)
            is_qualified = self.is_fully_qualified(hotkey)

            # Only include miners that have been evaluated (have cost data)
            if total_cost is not None:
                costs[uid] = total_cost
                qualified[uid] = is_qualified
                uid_to_hotkey[uid] = hotkey
                # Score = 1.0 if qualified, 0.0 otherwise (for report compat)
                scores[uid] = 1.0 if is_qualified else 0.0

                total_cost = self.compute_total_cost_from_ema(hotkey)
                if total_cost is not None:
                    costs[uid] = total_cost
                qualified[uid] = self.is_fully_qualified(hotkey)

        if not scores:
            logger.warning("All miners have zero EMA score")
            await self._set_fallback_weights()
            return

        # Pairwise NCD dedup: exclude copy-cat miners before winner selection.
        # Layer 1 (pack_hash grouping) catches exact copies.
        # Layer 2 (NCD compression) catches paraphrased copies.
        # Priority: lower on-chain block_number = original.
        pack_info: Dict[str, Tuple[dict, int, str]] = {}
        for uid in list(costs.keys()):
            hotkey = uid_to_hotkey[uid]
            pack = self._hotkey_packs.get(hotkey)
            commitment = active.get(uid)
            if pack is not None and commitment is not None:
                pack_info[hotkey] = (
                    pack, commitment.block_number, commitment.pack_hash
                )

        ncd_excluded = deduplicate_packs(
            pack_info, self.config.similarity_threshold
        )

        if ncd_excluded:
            hotkey_to_uid = {v: k for k, v in uid_to_hotkey.items()}
            for copier_hk, original_hk in ncd_excluded.items():
                copier_uid = hotkey_to_uid.get(copier_hk)
                if copier_uid is not None:
                    logger.warning(
                        f"Miner {copier_uid} ({copier_hk[:8]}): "
                        f"weight zeroed (NCD copy of {original_hk[:8]})"
                    )
                    scores.pop(copier_uid, None)
                    costs.pop(copier_uid, None)
                    qualified.pop(copier_uid, None)
                    uid_to_hotkey.pop(copier_uid, None)

        if not scores:
            logger.warning("All scored miners excluded by NCD dedup")
            await self._set_fallback_weights()
            return

        num_active = len(scores)

        if not costs:
            logger.warning("No cost data available, setting fallback weights")
            await self._set_fallback_weights()
            return

        weights_dict = self.scorer.select_winner_by_cost(
            costs=costs,
            qualified=qualified,
            first_mover_data=self.first_mover_data,
            cost_delta=self.config.cost_delta,
            num_active_miners=num_active,
            uid_to_hotkey=uid_to_hotkey,
        )

        # Apply burn: scale miner weights by (1 - BURN_FRACTION),
        # give BURN_FRACTION to owner UID (burned by the chain).
        for uid in weights_dict:
            weights_dict[uid] *= (1.0 - BURN_FRACTION)
        weights_dict[OWNER_UID] = (
            weights_dict.get(OWNER_UID, 0.0) + BURN_FRACTION
        )

        # Log results
        logger.info("=" * 60)
        logger.info("WEIGHT RESULTS")
        logger.info(f"Burn fraction: {BURN_FRACTION:.0%} to owner UID {OWNER_UID}")
        logger.info("=" * 60)
        for uid, weight in sorted(
            weights_dict.items(),
            key=lambda x: costs.get(x[0], scores.get(x[0], 0)),
        ):
            if uid == OWNER_UID and uid not in uid_to_hotkey:
                logger.info(
                    f"  Owner UID {uid}: weight={weight:.4f} (burn)"
                )
                continue
            marker = ""
            hk = uid_to_hotkey.get(uid, "?")
            if weight > 0:
                marker = " <- WINNER" if weight == 0.5 else f" <- TOP-{sum(1 for u, w in weights_dict.items() if w >= weight and u != OWNER_UID)}"
            gate = "PASS" if qualified.get(uid, False) else "FAIL"
            cost_str = f"${costs[uid]:.4f}" if uid in costs else "n/a"
            logger.info(
                f"  Miner {uid} ({hk[:8]}): weight={weight:.4f}, "
                f"cost={cost_str}, gate={gate}, "
                f"score={scores.get(uid, 0):.3f}{marker}"
            )

        # Set weights on chain
        if SHADOW_MODE:
            await self._set_fallback_weights(reason="SHADOW MODE: eval complete")
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
                self._last_set_weights_at = int(time.time())
                self._last_weights_uids = uids
                self._last_weights = weights
            except Exception as e:
                logger.error(f"Error setting weights: {e}", exc_info=True)

    async def _replay_last_weights(self):
        """Re-set the last computed weights on-chain without recomputing.

        Used for mid-eval tempo refreshes to keep the validator active
        while eval is still running. Falls back to fallback weights if
        no previous weights are cached.
        """
        if self._last_weights_uids is None or self._last_weights is None:
            logger.info("No cached weights to replay, setting fallback weights")
            await self._set_fallback_weights(reason="No cached weights for replay")
            return
        try:
            self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=self._last_weights_uids,
                weights=self._last_weights,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )
            logger.info("Replayed last weights successfully")
            self._last_set_weights_at = int(time.time())
        except Exception as e:
            logger.error(f"Error replaying weights: {e}", exc_info=True)

    async def _set_fallback_weights(self, reason: str = "No eligible miners"):
        """Set weights to subnet owner UID when no miners qualify.

        Miner incentive directed to the owner hotkey is burned by the
        chain (not paid to the owner), so this effectively burns miner
        emissions until a qualifying miner submits.  The validator must
        always call set_weights every tempo to avoid deregistration.
        """
        try:
            # Verify wallet is accessible before attempting on-chain call
            _ = self.wallet.hotkey
        except Exception:
            logger.debug("Skipping fallback weights: wallet hotkey not available")
            return

        try:
            uids = [OWNER_UID]
            weights = [1.0]

            logger.info(
                f"{reason} — setting fallback weight to "
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
