"""TrajectoryRL Validator — Main validator implementation.

Architecture (v4.0 — LLM-as-Judge):
    1. Continuous evaluation loop with dual cadence:
       - eval_interval (~24h): re-evaluate all active packs
       - tempo (~72 min): compute weights from qualification + cost, set_weights
    2. Read on-chain commitments (subtensor.get_all_commitments)
    3. Fetch packs from miners' public HTTP URLs
    4. NCD pairwise dedup (before ClawBench); schema validation in _evaluate_miner
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
import datetime
import hashlib
import json
import logging
import os
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
from ..utils.eval_window import (
    WindowConfig, WindowPhase, EvaluationWindow,
    compute_window, is_new_window, can_evaluate,
    should_submit, should_aggregate,
)
from ..utils.consensus import (
    ConsensusPayload, ConsensusPointer,
    CONSENSUS_PROTOCOL_VERSION,
)
from ..utils.consensus_store import (
    ConsensusStore, IPFSBackend, TrajRLAPIBackend,
)
from ..utils.consensus_filter import (
    run_filter_pipeline, ValidatedSubmission,
)
from ..scoring import compute_consensus_costs
from ..utils.commitments import MinerCommitment, fetch_all_commitments
from ..utils.ncd import deduplicate_packs
from ..utils.status_reporter import (
    heartbeat, pre_eval, submit_eval, upload_eval_logs, upload_cycle_logs,
    fetch_weight_override,
)
from ..utils.llm_judge import PackIntegrityJudge, TrajectoryJudge
from .. import __version__

logger = logging.getLogger(__name__)

OWNER_UID = 74
BURN_FRACTION = 0.50  # 50% of miner emissions burned via owner UID
EVAL_START_BLOCK = 0
# Shadow mode runs real evals and logs results, but always sets weights to owner UID 74.
SHADOW_MODE = False

_EVAL_CACHE_MAX_RETRIES = int(os.getenv("TRAJECTORYRL_CACHE_MAX_RETRIES", "3"))
_EVAL_CACHE_TTL_DAYS = int(os.getenv("TRAJECTORYRL_CACHE_TTL_DAYS", "14"))
_EVAL_CACHE_ENABLED = os.getenv("TRAJECTORYRL_EVAL_CACHE_ENABLED", "1") != "0"


class TrajectoryValidator:
    """TrajectoryRL validator that evaluates policy packs using ClawBench.

    The validator (v4.0):
    1. Reads on-chain commitments from miners
    2. Fetches and verifies packs from miners' public HTTP URLs
    3. NCD pairwise dedup before ClawBench (copiers rejected like integrity fail)
    4. Phase 1: LLM pack integrity analysis (rejects gaming packs)
    5. Runs ALL ClawBench scenarios
    6. Phase 2: LLM trajectory judge per scenario (replaces regex scoring)
    7. Updates per-scenario cost EMA (keyed by miner hotkey)
    8. Sets on-chain weights (winner-take-all or bootstrap by cost)
    9. Re-sets weights every tempo (~72 min) for convergence

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

        logger.debug("Initializing Bittensor components...")
        self.wallet = bt.Wallet(
            name=config.wallet_name,
            hotkey=config.wallet_hotkey,
        )
        self.subtensor = bt.Subtensor(network=config.network)
        self.metagraph = self.subtensor.metagraph(config.netuid)

        logger.info(f"Wallet hotkey: {self.wallet.hotkey.ss58_address[:16]}...")
        logger.info(f"Network: {config.network}")
        logger.info(f"Netuid: {config.netuid}")

        logger.debug("Initializing ClawBench harness...")
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
        logger.debug("Initializing LLM judges (model=%s)...", judge_model)

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

        logger.debug("Initializing pack fetcher...")
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

        # First-mover tracking: {hotkey: (best_cost, block_number)}
        self.first_mover_data: Dict[str, Tuple[float, float]] = {}

        # Previous cycle's winner hotkey (for δ champion protection)
        self.champion_hotkey: Optional[str] = None

        # Tracks which UID each hotkey was last evaluated at for re-registration detection.
        self._hotkey_uid_map: Dict[str, int] = {}

        # Weight cadence tracking
        self.last_weight_block: int = 0

        # Eval count per hotkey for the current pack (resets on pack change)
        self._eval_counts: Dict[str, int] = {}

        # Timestamp of the most recent successful set_weights call
        self._last_set_weights_at: Optional[int] = None

        # Timestamp of the most recent completed full evaluation cycle
        self._last_eval_at: Optional[int] = None

        # UTC date of the most recent completed eval cycle (legacy daily scheduling)
        self._last_eval_date: Optional[datetime.date] = None

        # Block-based window tracking (replaces _last_eval_date for consensus protocol)
        self._last_eval_window: int = -1
        self._window_config = WindowConfig(
            window_length=config.eval_interval_blocks,
            global_anchor=config.global_anchor_block,
            publish_pct=config.window_publish_pct,
            aggregate_pct=config.window_aggregate_pct,
        )
        # Whether this window's evaluation results have been submitted to CAS
        self._window_submitted: bool = False
        # Whether this window's consensus aggregation has been performed
        self._window_aggregated: bool = False

        # Consensus CAS store (IPFS + API)
        self._consensus_store = ConsensusStore(
            ipfs=IPFSBackend(api_url=config.ipfs_api_url),
            api=TrajRLAPIBackend(
                base_url=config.consensus_api_url,
                validator_hotkey=self.wallet.hotkey.ss58_address,
            ),
        )

        # Latest consensus results from aggregation (used by weight setting)
        self._consensus_costs: Dict[str, float] = {}
        self._consensus_qualified: Dict[str, bool] = {}

        # Set to True on startup when eval_on_startup=True; cleared after first eval
        self._startup_eval_pending: bool = config.eval_on_startup

        # When True, tempo weight refreshes use fallback weights instead of
        # computing from (potentially stale) EMA data.  Set when the eval
        # cycle exits via a fallback path (no LLM key, all evals failed,
        # no active miners) and cleared after a successful eval cycle.
        self._eval_fallback_active: bool = False

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

        # Eval result cache: pack_hash -> {status, result, failure_count, ...}
        # Keyed by miner-submitted pack_hash; avoids re-running ClawBench + LLM
        # judge for packs that have already been evaluated this cycle.
        self._eval_cache: Dict[str, dict] = {}
        self._load_eval_cache()

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
            logger.debug("No persisted EMA state found, starting fresh")
            return
        try:
            data = json.loads(path.read_text())
            if data.get("scenario_config_hash") != self._scenario_config_hash:
                logger.debug(
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
            self.champion_hotkey = data.get("champion_hotkey")
            self._last_eval_window = data.get("last_eval_window", -1)

            # Load integrity judge cache if present
            integrity_cache = data.get("integrity_cache")
            if integrity_cache:
                self.integrity_judge.load_cache(integrity_cache)

            logger.debug(
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
            "champion_hotkey": self.champion_hotkey,
            "last_eval_window": self._last_eval_window,
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
    # Eval result cache
    # ------------------------------------------------------------------

    @property
    def _eval_cache_path(self) -> Path:
        return self.config.ema_state_path.parent / "eval_cache.json"

    def _load_eval_cache(self):
        """Load eval result cache from disk, pruning expired entries."""
        if not _EVAL_CACHE_ENABLED:
            return
        path = self._eval_cache_path
        if not path.exists():
            logger.debug("No eval cache found, starting fresh")
            return
        try:
            data = json.loads(path.read_text())
            cutoff = time.time() - _EVAL_CACHE_TTL_DAYS * 86400
            self._eval_cache = {
                k: v for k, v in data.items()
                if v.get("last_eval_at", 0) > cutoff
            }
            pruned = len(data) - len(self._eval_cache)
            logger.debug(
                f"Loaded eval cache: {len(self._eval_cache)} entries"
                + (f" (pruned {pruned} expired)" if pruned else "")
            )
        except Exception as e:
            logger.warning(f"Failed to load eval cache: {e}")

    def _save_eval_cache(self):
        """Persist eval result cache to disk."""
        if not _EVAL_CACHE_ENABLED:
            return
        try:
            self._eval_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._eval_cache_path.write_text(
                json.dumps(self._eval_cache, indent=2, sort_keys=True)
            )
        except Exception as e:
            logger.warning(f"Failed to save eval cache: {e}")

    def _check_eval_cache(self, pack_hash: str) -> Tuple[bool, Optional[Dict]]:
        """Check whether a cached eval result can be reused for pack_hash.

        Returns:
            (should_use_cache, cached_result)
            - (True, result_dict)  success hit — use result directly
            - (True, None)         failed hit, max retries reached — treat as None
            - (False, None)        no usable cache — run full evaluation
        """
        if not _EVAL_CACHE_ENABLED:
            return False, None

        entry = self._eval_cache.get(pack_hash)
        if entry is None:
            logger.debug(f"[EVAL_CACHE] MISS pack_hash={pack_hash[:12]}")
            return False, None

        if entry["status"] == "success":
            logger.info(f"[EVAL_CACHE] HIT  pack_hash={pack_hash[:12]} status=success")
            return True, entry["result"]

        # status == "failed"
        failure_count = entry.get("failure_count", 1)
        if failure_count >= _EVAL_CACHE_MAX_RETRIES:
            logger.info(
                f"[EVAL_CACHE] SKIP pack_hash={pack_hash[:12]} status=failed "
                f"failure_count={failure_count}/{_EVAL_CACHE_MAX_RETRIES} "
                f"(max retries reached)"
            )
            return True, None

        logger.info(
            f"[EVAL_CACHE] HIT  pack_hash={pack_hash[:12]} status=failed "
            f"failure_count={failure_count}/{_EVAL_CACHE_MAX_RETRIES} (retrying)"
        )
        return False, None

    def _update_eval_cache(
        self,
        pack_hash: str,
        eval_result: Optional[Dict],
        failure_reason: Optional[str] = None,
    ):
        """Write or update the eval cache entry for pack_hash.

        On success, stores the full result dict and resets failure_count.
        On failure, increments failure_count (capped behaviour handled by
        _check_eval_cache on the next call).
        """
        if not _EVAL_CACHE_ENABLED:
            return
        now = time.time()
        existing = self._eval_cache.get(pack_hash)
        first_eval_at = existing["first_eval_at"] if existing else now

        if eval_result is not None:
            self._eval_cache[pack_hash] = {
                "status": "success",
                "result": eval_result,
                "failure_count": 0,
                "first_eval_at": first_eval_at,
                "last_eval_at": now,
                "failure_reason": None,
            }
        else:
            prev_count = existing.get("failure_count", 0) if existing else 0
            self._eval_cache[pack_hash] = {
                "status": "failed",
                "result": None,
                "failure_count": prev_count + 1,
                "first_eval_at": first_eval_at,
                "last_eval_at": now,
                "failure_reason": failure_reason,
            }

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
            self._get_miner_logger(hotkey).info(
                f"Pack changed "
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
        self._validator_log_path = self.config.log_dir / f"validator_{int(time.time())}.log"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self._validator_log_path),
            ],
        )

        # Per-miner log directory
        self._miner_log_dir = self.config.log_dir / "miners"
        self._miner_log_dir.mkdir(parents=True, exist_ok=True)
        self._miner_loggers: Dict[str, logging.Logger] = {}

    def _get_miner_logger(self, hotkey: str) -> logging.Logger:
        """Get or create a per-miner file logger (INFO level, no console output)."""
        if hotkey in self._miner_loggers:
            return self._miner_loggers[hotkey]

        mlog = logging.getLogger(f"trajectoryrl.miner.{hotkey[:16]}")
        mlog.setLevel(logging.INFO)
        mlog.propagate = False  # don't bubble up to root / console

        log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
        fh = logging.FileHandler(
            self._miner_log_dir / f"{hotkey[:16]}.log"
        )
        fh.setFormatter(logging.Formatter(log_format))
        mlog.addHandler(fh)

        self._miner_loggers[hotkey] = mlog
        return mlog

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
                    last_eval_at=self._last_eval_at,
                )
            except Exception as e:
                logger.warning("Heartbeat error: %s", e)
            await asyncio.sleep(600)

    async def _submit_consensus_payload(self, window: EvaluationWindow):
        """Build and upload consensus payload to CAS, then register pointer."""
        from trajectoryrl import __version__

        costs_by_hotkey: Dict[str, float] = {}
        qualified_by_hotkey: Dict[str, bool] = {}

        for hotkey, scenario_costs in self.ema_costs.items():
            if scenario_costs:
                total_cost = sum(scenario_costs.values()) / len(scenario_costs)
                costs_by_hotkey[hotkey] = total_cost
            scenario_q = self.scenario_qualified.get(hotkey, {})
            qualified_by_hotkey[hotkey] = (
                bool(scenario_q) and all(scenario_q.values())
            )

        if not costs_by_hotkey:
            logger.warning(
                "Window %d: no evaluation data to submit", window.window_number
            )
            return

        payload = ConsensusPayload(
            protocol_version=self.config.consensus_protocol_version,
            window_number=window.window_number,
            validator_hotkey=self.wallet.hotkey.ss58_address,
            software_version=__version__,
            costs=costs_by_hotkey,
            qualified=qualified_by_hotkey,
            timestamp=int(time.time()),
        )

        content_address = await self._consensus_store.upload_payload(payload)
        if content_address is None:
            logger.error(
                "Window %d: failed to upload consensus payload",
                window.window_number,
            )
            return

        pointer = ConsensusPointer(
            protocol_version=self.config.consensus_protocol_version,
            window_number=window.window_number,
            content_address=content_address,
            validator_hotkey=self.wallet.hotkey.ss58_address,
        )

        await self._consensus_store.write_pointer(pointer)
        logger.info(
            "Window %d: consensus payload submitted (address=%s, %d miners)",
            window.window_number, content_address[:24], len(costs_by_hotkey),
        )

    async def _run_consensus_aggregation(self, window: EvaluationWindow):
        """Read submissions, filter, compute stake-weighted consensus costs."""
        pointers = await self._consensus_store.read_all_pointers(
            window.window_number
        )
        if not pointers:
            logger.warning(
                "Window %d: no pointers found, using local results",
                window.window_number,
            )
            return

        submissions = []
        for ptr in pointers:
            payload = await self._consensus_store.download_payload(
                ptr.content_address
            )
            if payload is not None:
                submissions.append((ptr, payload))

        if not submissions:
            logger.warning(
                "Window %d: all payload downloads failed, using local results",
                window.window_number,
            )
            return

        metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        validator_stakes: Dict[str, float] = {}
        for neuron in metagraph.neurons:
            if neuron.is_null:
                continue
            validator_stakes[neuron.hotkey] = float(neuron.stake)

        from trajectoryrl import __version__
        min_stake = getattr(self.config, "min_validator_stake", 0.0)
        validated, stats = run_filter_pipeline(
            submissions=submissions,
            expected_window=window.window_number,
            validator_stakes=validator_stakes,
            min_stake=min_stake,
            local_version=__version__,
            expected_protocol=self.config.consensus_protocol_version,
        )

        if not validated:
            logger.warning(
                "Window %d: all submissions filtered out (%s), using local results",
                window.window_number, stats.summary(),
            )
            return

        consensus_costs, consensus_qualified = compute_consensus_costs(validated)

        self._consensus_costs = consensus_costs
        self._consensus_qualified = consensus_qualified

        logger.info(
            "Window %d: consensus aggregation complete — %d miners, "
            "%d validators contributed (%s)",
            window.window_number, len(consensus_costs),
            stats.passed, stats.summary(),
        )

    def _should_start_evaluation(self, current_block: int) -> bool:
        """Return True if a new evaluation cycle should start.

        Block-based: fires when we enter a new evaluation window and are
        in the evaluation phase.  Also fires on startup if eval_on_startup
        is set and no eval has run yet.
        """
        if self._startup_eval_pending:
            return True

        window = compute_window(current_block, self._window_config)
        if window.phase != WindowPhase.EVALUATION:
            return False
        return window.window_number > self._last_eval_window

    async def run(self):
        """Main validator loop with block-based window phases.

        Window phases (per eval_interval_blocks window):
          - evaluation (0% - 80%):   run ClawBench, compute local cost EMA
          - propagation (80% - 90%): submit results to CAS, wait for others
          - aggregation (90% - 100%): read submissions, consensus, select winner

        Independent cadence:
          - tempo (~72 min / 360 blocks): set_weights using latest consensus
        """
        self._start_time = time.time()

        asyncio.create_task(self._heartbeat_loop())

        logger.info("Starting validator main loop...")
        logger.info(
            f"Eval window: {self._window_config.window_length} blocks "
            f"(~{self._window_config.window_length * 12 // 3600:.0f}h), "
            f"anchor={self._window_config.global_anchor}, "
            f"publish={self._window_config.publish_pct:.0%}, "
            f"aggregate={self._window_config.aggregate_pct:.0%}"
        )
        logger.info(
            f"Weight interval: {self.config.weight_interval_blocks} blocks "
            f"(~{self.config.weight_interval_blocks * 12 // 60}min)"
        )

        while True:
            try:
                current_block = self.subtensor.get_current_block()
                window = compute_window(current_block, self._window_config)

                # --- Pre-launch phase: fallback weights only ---
                if current_block < EVAL_START_BLOCK:
                    blocks_since_weights = current_block - self.last_weight_block
                    if blocks_since_weights >= self.config.weight_interval_blocks:
                        logger.info(
                            f"Pre-launch phase (block {current_block} < "
                            f"{EVAL_START_BLOCK}), setting fallback weights"
                        )
                        if not await self._apply_weight_override():
                            await self._set_fallback_weights()
                        self.last_weight_block = current_block
                    await asyncio.sleep(60)
                    continue

                # --- Window phase: evaluation ---
                if self._should_start_evaluation(current_block):
                    logger.info("=" * 60)
                    logger.info(
                        f"Evaluation window {window.window_number} at block "
                        f"{current_block} (phase={window.phase.value}, "
                        f"offset={window.block_offset}/{self._window_config.window_length})"
                    )
                    logger.info("=" * 60)

                    await self._sync_clawbench()

                    await self._run_evaluation_cycle(current_block)
                    self._last_eval_at = int(time.time())
                    self._last_eval_window = window.window_number
                    self._last_eval_date = datetime.datetime.utcnow().date()
                    self._startup_eval_pending = False
                    self._window_submitted = False
                    self._window_aggregated = False
                    self.last_weight_block = self.subtensor.get_current_block()

                    self.pack_fetcher.cleanup_cache(
                        self.config.pack_cache_max_size
                    )
                    self._save_ema_state()
                    self._save_eval_cache()

                # --- Window phase: submission (T_publish) ---
                if (window.phase == WindowPhase.PROPAGATION
                        and not self._window_submitted
                        and self._last_eval_window == window.window_number):
                    logger.info(
                        f"Window {window.window_number}: T_publish reached "
                        f"at block {current_block}, submitting evaluation results"
                    )
                    await self._submit_consensus_payload(window)
                    self._window_submitted = True

                # --- Window phase: aggregation (T_aggregate) ---
                if (window.phase == WindowPhase.AGGREGATION
                        and not self._window_aggregated
                        and self._last_eval_window == window.window_number):
                    logger.info(
                        f"Window {window.window_number}: T_aggregate reached "
                        f"at block {current_block}, running consensus aggregation"
                    )
                    await self._run_consensus_aggregation(window)
                    self._window_aggregated = True

                # --- Tempo cadence: re-set weights (independent of window) ---
                current_block = self.subtensor.get_current_block()
                blocks_since_weights = current_block - self.last_weight_block
                if blocks_since_weights >= self.config.weight_interval_blocks:
                    logger.info(
                        f"Tempo weight refresh at block {current_block} "
                        f"({blocks_since_weights} blocks since last, "
                        f"window={window.window_number} phase={window.phase.value})"
                    )
                    if not await self._apply_weight_override():
                        if self._eval_fallback_active:
                            await self._set_fallback_weights(
                                reason="Eval cycle in fallback mode"
                            )
                        else:
                            await self._compute_and_set_weights(current_block)
                    self.last_weight_block = current_block

                await asyncio.sleep(60)

            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                self._save_ema_state()
                self._save_eval_cache()
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

    async def _prefetch_packs_and_ncd_gate(
        self,
        active_commitments: Dict[int, MinerCommitment],
        skip_uids: set,
    ) -> Dict[str, str]:
        """Fetch packs for eligible miners, run NCD dedup, cache packs for eval.

        Miners marked as copiers (values map to originals) must not run ClawBench;
        they are rejected in the main loop like integrity failures. Non-copiers
        with a successful fetch get _hotkey_packs / _pack_by_hash populated so
        _evaluate_miner hits verify_submission cache.

        Miners whose fetch fails are omitted from this NCD round but may still
        be evaluated later if _evaluate_miner can fetch them.
        """
        pack_info: Dict[str, Tuple[dict, int, str]] = {}
        for uid, commitment in active_commitments.items():
            if uid in skip_uids:
                continue
            hk = commitment.hotkey
            result = await self.pack_fetcher.verify_submission(
                commitment.pack_url,
                commitment.pack_hash,
            )
            if not result.valid or result.pack_content is None:
                logger.warning(
                    f"NCD prefetch: uid={uid} ({hk[:8]}…): "
                    f"{result.error or 'unknown'}"
                )
                continue
            pack_info[hk] = (
                result.pack_content,
                commitment.block_number,
                commitment.pack_hash,
            )

        ncd_excluded = deduplicate_packs(
            pack_info, self.config.similarity_threshold
        )

        for hk, (pack, _bn, ph) in pack_info.items():
            if hk in ncd_excluded:
                continue
            self._hotkey_packs[hk] = pack
            self._pack_by_hash[ph] = pack

        if ncd_excluded:
            logger.info(
                f"NCD pre-eval gate: {len(ncd_excluded)} miner(s) flagged as copies"
            )
        return ncd_excluded

    # ------------------------------------------------------------------
    # Evaluation cycle
    # ------------------------------------------------------------------

    async def _run_evaluation_cycle(self, current_block: int):
        """Run one evaluation cycle with guaranteed cycle log upload.

        Wraps _execute_evaluation_cycle in try/finally to ensure the
        cycle-level validator log is always uploaded to the dashboard,
        regardless of early returns (no LLM key, no miners) or exceptions.
        """
        cycle_start = time.time()
        cycle_log_offset = self._get_validator_log_offset()
        cycle_eval_id = time.strftime("%Y%m%d_%H%M%S")
        try:
            await self._execute_evaluation_cycle(
                current_block, cycle_eval_id, cycle_start,
            )
        finally:
            asyncio.ensure_future(
                self._fire_upload_cycle_logs(
                    cycle_eval_id, cycle_log_offset, current_block,
                )
            )

    async def _execute_evaluation_cycle(
        self,
        current_block: int,
        cycle_eval_id: str,
        cycle_start: float,
    ):
        """Execute the full evaluation cycle logic.

        Falls back to owner UID weights when LLM keys are missing or
        all evaluations fail (likely LLM API errors).
        """
        if not self._check_llm_keys():
            logger.warning(
                "CLAWBENCH_LLM_API_KEY not set. "
                "Skipping evaluation, setting fallback weights to owner UID.",
            )
            self._eval_fallback_active = True
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

        logger.debug(
            f"Eval cycle: block={current_block}, seed={epoch_seed}, "
            f"context=[{epoch_ctx.user_name}, {epoch_ctx.user_role}]"
        )

        # 1. Clear per-cycle pack caches to prevent stale entries from
        # deregistered miners affecting NCD comparisons in the weight phase.
        self._hotkey_packs.clear()
        self._pack_by_hash.clear()

        # 2. Sync metagraph
        logger.debug("Syncing metagraph...")
        self.metagraph.sync(subtensor=self.subtensor)
        logger.info(
            f"Metagraph synced: {self.metagraph.n} neurons, "
            f"block {self.metagraph.block}"
        )

        # 3. Read on-chain commitments
        logger.debug("Reading on-chain commitments...")
        commitments = fetch_all_commitments(
            self.subtensor, self.config.netuid, self.metagraph
        )

        # 4. Filter to non-validator miners
        active_commitments = self._filter_active_commitments(commitments)
        logger.info(
            f"Commitments: {len(commitments)} total, "
            f"{len(active_commitments)} active miners"
        )

        if not active_commitments:
            logger.warning("No active miners with valid commitments!")
            self._eval_fallback_active = True
            await self._set_fallback_weights()
            return

        # 5. Pack-hash pre-dedup: for miners with identical pack_hash,
        # only evaluate the first mover (lowest block_number).
        # Saves LLM API calls. Paraphrase copies are caught by NCD next
        # (_prefetch_packs_and_ncd_gate); weight phase re-runs NCD as a safety net.
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
                self._get_miner_logger(commitment.hotkey).info(
                    f"Skipping eval (duplicate pack_hash={commitment.pack_hash[:12]})"
                )

        # 6. Evaluate miners
        # Order scenarios hardest-first so weak packs fail fast and we
        # skip the remaining (cheaper) scenarios, saving LLM tokens.
        # Difficulty ranking based on empirical pass-rate data.
        _SCENARIO_DIFFICULTY_ORDER = [
            "morning_brief",
            "inbox_to_action",
            "client_escalation",
            "team_standup",
            "inbox_triage",
        ]
        eval_scenarios = sorted(
            self.scenarios.keys(),
            key=lambda s: (
                _SCENARIO_DIFFICULTY_ORDER.index(s)
                if s in _SCENARIO_DIFFICULTY_ORDER
                else len(_SCENARIO_DIFFICULTY_ORDER)
            ),
        )
        evaluated_count = 0
        attempted_count = 0
        skipped_interval_count = 0
        rejected_pre_eval_count = 0
        ncd_rejected_count = 0
        cached_count = 0
        total_eligible = len(active_commitments) - len(skip_uids)
        total_scenarios = len(eval_scenarios)
        logger.info(
            f"=== Eval cycle: {total_eligible} eligible miners, "
            f"{total_scenarios} scenarios each ==="
        )

        ncd_excluded = await self._prefetch_packs_and_ncd_gate(
            active_commitments, skip_uids
        )

        miner_idx = 0
        for uid, commitment in active_commitments.items():
            hotkey = commitment.hotkey

            if uid in skip_uids:
                continue

            miner_idx += 1

            if hotkey in ncd_excluded:
                ncd_rejected_count += 1
                original_hk = ncd_excluded[hotkey]
                detail = (
                    f"NCD: policy too similar to earlier commitment "
                    f"(original hotkey {original_hk[:8]}…)"
                )
                logger.info(
                    f"[{miner_idx}/{total_eligible}] Miner {uid} ({hotkey[:8]}): "
                    f"{detail} — skipping eval"
                )
                asyncio.ensure_future(
                    submit_eval(
                        self.wallet,
                        miner_hotkey=hotkey,
                        miner_uid=uid,
                        block_height=current_block,
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
                        rejection_detail=detail,
                    )
                )
                self.ema_costs.pop(hotkey, None)
                self.scenario_qualified.pop(hotkey, None)
                self._ema_pack_hash.pop(hotkey, None)
                continue

            # Pre-eval gate: ask the server whether this miner's submission
            # is allowed before spending LLM tokens on a full evaluation.
            # Controlled by TRAJECTORYRL_PRE_EVAL_ENABLED (default: 1).
            # Fail-open on network/API errors so validators are self-sufficient.
            if os.getenv("TRAJECTORYRL_PRE_EVAL_ENABLED", "1") != "0":
                pre_eval_result = await pre_eval(
                    hotkey,
                    commitment.pack_hash,
                    commitment.pack_url,
                    wallet=self.wallet,
                )
                if pre_eval_result is not None and not pre_eval_result.get("allowed", True):
                    rejected_pre_eval_count += 1
                    reason = pre_eval_result.get("reason", "unknown")
                    logger.info(
                        f"[{miner_idx}/{total_eligible}] Miner {uid} ({hotkey[:8]}): "
                        f"pre-eval rejected (reason={reason}) — skipping eval"
                    )
                    _stage = "integrity_check" if reason == "hardcoded" else "pack_fetch"
                    _detail = (
                        f"pre-eval rejected: {reason}"
                        + (
                            f", banned_until={pre_eval_result['banned_until']}"
                            if "banned_until" in pre_eval_result
                            else ""
                        )
                    )
                    asyncio.ensure_future(
                        submit_eval(
                            self.wallet,
                            miner_hotkey=hotkey,
                            miner_uid=uid,
                            block_height=current_block,
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
                            rejection_stage=_stage,
                            rejection_detail=_detail,
                        )
                    )
                    # Clear stale EMA state so rejected miners cannot
                    # retain weight from a previous successful evaluation.
                    self.ema_costs.pop(hotkey, None)
                    self.scenario_qualified.pop(hotkey, None)
                    self._ema_pack_hash.pop(hotkey, None)
                    continue
            
            needs_eval = self._needs_evaluation(
                hotkey, commitment.pack_hash, current_block
            )
            if not needs_eval:
                skipped_interval_count += 1
                self._get_miner_logger(hotkey).info(
                    f"[{miner_idx}/{total_eligible}] Skipping, within eval interval"
                )
                continue

            # Check eval cache before spending LLM tokens.
            # Pre-eval always runs; cache is keyed by pack_hash.
            eval_elapsed = 0.0
            cache_hit, cache_result = self._check_eval_cache(commitment.pack_hash)
            if cache_hit:
                cached_count += 1
                eval_result = cache_result
                if cache_result is not None:
                    logger.info(
                        f"[{miner_idx}/{total_eligible}] Miner {uid} ({hotkey[:8]}): "
                        f"using cached eval result (pack_hash={commitment.pack_hash[:12]})"
                    )
                else:
                    logger.info(
                        f"[{miner_idx}/{total_eligible}] Miner {uid} ({hotkey[:8]}): "
                        f"cached failure — max retries reached, skipping eval"
                    )
            else:
                attempted_count += 1
                logger.info(
                    f"[{miner_idx}/{total_eligible}] Evaluating miner {uid} "
                    f"({hotkey[:8]}) ..."
                )
                eval_dir, vlog_offset, mlog_offset = self._prepare_eval_log_capture(cycle_eval_id, hotkey)
                eval_start = time.time()
                eval_result = await self._evaluate_miner(
                    uid, commitment, eval_scenarios, epoch_seed,
                    context_preamble, user_context,
                    block_height=current_block,
                )
                eval_elapsed = time.time() - eval_start
                self._update_eval_cache(commitment.pack_hash, eval_result)

                # Upload eval logs to dashboard (fire-and-forget)
                asyncio.ensure_future(
                    self._fire_upload_eval_logs(
                        cycle_eval_id, uid, commitment, eval_scenarios, eval_result,
                        eval_dir, vlog_offset, mlog_offset, current_block,
                    )
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
                q = eval_result.get("qualified", {})
                passed = sum(1 for v in q.values() if v)
                elapsed_str = f" ({eval_elapsed:.1f}s)" if eval_elapsed else ""
                logger.info(
                    f"[{miner_idx}/{total_eligible}] Miner {uid} done: "
                    f"{passed}/{len(q)} scenarios passed"
                    f"{elapsed_str} "
                    f"({evaluated_count} evaluated so far)"
                )

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
                        uid, hotkey, total_cost, float(commitment.block_number),
                        pack_hash=commitment.pack_hash,
                    )

            else:
                elapsed_str = f" ({eval_elapsed:.1f}s)" if eval_elapsed else ""
                logger.info(
                    f"[{miner_idx}/{total_eligible}] Miner {uid} eval returned None"
                    f"{elapsed_str} "
                    f"(pack fetch/integrity failed)"
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
                if not await self._apply_weight_override():
                    await self._replay_last_weights()
                self.last_weight_block = mid_block

        parts = [f"{evaluated_count} evaluated"]
        if cached_count:
            parts.append(f"{cached_count} cached")
        if rejected_pre_eval_count:
            parts.append(f"{rejected_pre_eval_count} pre-eval rejected")
        if ncd_rejected_count:
            parts.append(f"{ncd_rejected_count} NCD rejected")
        if skipped_interval_count:
            parts.append(f"{skipped_interval_count} skipped (interval)")
        failed_count = attempted_count - evaluated_count
        if failed_count > 0:
            parts.append(f"{failed_count} failed")
        cycle_elapsed = time.time() - cycle_start
        cycle_min = cycle_elapsed / 60
        logger.info(
            f"Eval cycle complete ({cycle_min:.1f}min): {total_eligible} eligible, "
            + ", ".join(parts)
        )

        # Log EMA cost summary to per-miner files
        if evaluated_count > 0:
            for uid, commitment in active_commitments.items():
                hk = commitment.hotkey
                ema_cost = self.compute_total_cost_from_ema(hk)
                if ema_cost is not None:
                    per_scenario = self.ema_costs.get(hk, {})
                    scenario_str = ", ".join(
                        f"{s}=${c:.4f}" for s, c in sorted(per_scenario.items())
                    )
                    self._get_miner_logger(hk).info(
                        f"EMA cost: total=${ema_cost:.4f} ({scenario_str})"
                    )

        # All attempted evaluations failed — likely an LLM API issue
        if attempted_count > 0 and evaluated_count == 0:
            logger.error(
                f"All {attempted_count} miner evaluations failed. "
                "Possible LLM API key issue or service outage. "
                "Setting fallback weights to owner UID."
            )
            self._eval_fallback_active = True
            await self._set_fallback_weights(
                reason="All evaluations failed (LLM error)"
            )
            return

        # Eval completed successfully — clear fallback flag so tempo
        # refreshes use real computed weights.
        self._eval_fallback_active = False

        # 5. Set weights from EMA scores
        await self._compute_and_set_weights(current_block)

    def _should_run_eval_today(self) -> bool:
        """Return True if an eval should be triggered now.

        Fires when:
        - eval_on_startup=True and no eval has run yet this process, OR
        - UTC hour >= eval_utc_hour and today's eval has not yet completed.
        """
        if self._startup_eval_pending:
            return True
        now = datetime.datetime.utcnow()
        if now.hour < self.config.eval_utc_hour:
            return False
        today = now.date()
        if self._last_eval_date is not None and self._last_eval_date >= today:
            return False
        return True

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
            self._get_miner_logger(hotkey).info(
                f"pack_hash changed, marking for eval"
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
                    self._get_miner_logger(commitment.hotkey).info(
                        f"Skipping eval (coldkey {coldkey} is blacklisted)"
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
                    self._get_miner_logger(hotkey).info(
                        f"Inactive for {blocks_since} blocks > "
                        f"{self.config.inactivity_blocks}, "
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
        pack_hash: Optional[str] = None,
    ) -> None:
        """Detect re-registration and update first-mover data for a miner.

        Tracks best (lowest) cost. Lower cost = better.

        When a miner submits a new pack (pack_hash changed), the block
        number is reset to the new commitment's block.  This ensures
        first-mover protection reflects the current pack's submission
        time, not a historical one.
        """
        prev_uid = self._hotkey_uid_map.get(hotkey)
        if prev_uid is not None and prev_uid != miner_uid:
            self._get_miner_logger(hotkey).info(
                f"Previously at UID {prev_uid}; "
                f"re-registration detected, resetting first-mover data"
            )
            self.first_mover_data.pop(hotkey, None)
        self._hotkey_uid_map[hotkey] = miner_uid

        # Detect pack change: if pack_hash differs from EMA tracking,
        # the miner submitted a new pack → reset block to new commitment.
        current_ema_hash = self._ema_pack_hash.get(hotkey)
        pack_changed = (
            pack_hash is not None
            and current_ema_hash is not None
            and pack_hash != current_ema_hash
        )

        if hotkey not in self.first_mover_data:
            self.first_mover_data[hotkey] = (cost, block_number)
            self._get_miner_logger(hotkey).info(
                f"First submission (cost=${cost:.4f}, block={block_number:.0f})"
            )
        elif pack_changed:
            # New pack → reset block number to new commitment block
            self.first_mover_data[hotkey] = (cost, block_number)
            self._get_miner_logger(hotkey).info(
                f"Pack changed, reset first-mover block to {block_number:.0f} "
                f"(cost=${cost:.4f})"
            )
        elif cost < self.first_mover_data[hotkey][0]:
            # Same pack, cost improved (EMA drift) → keep original block
            original_block = self.first_mover_data[hotkey][1]
            self.first_mover_data[hotkey] = (cost, original_block)
            self._get_miner_logger(hotkey).info(
                f"Cost improved to ${cost:.4f}"
            )

    # ------------------------------------------------------------------
    # Episode detail logging (file-only, no console output)
    # ------------------------------------------------------------------

    def _log_episode_details(
        self,
        mlog: logging.Logger,
        scenario_name: str,
        result: "EvaluationResult",
    ) -> None:
        """Log detailed OpenClaw and mock_tool request/response to the
        per-miner file logger.  The miner logger has ``propagate=False``
        and only a FileHandler, so this does NOT appear in docker logs."""
        mlog.info(
            f"--- {scenario_name} episode detail ---"
        )

        # -- OpenClaw request --
        if result.input_message:
            mlog.info(
                f"[openclaw-request] message={result.input_message}"
            )

        # -- OpenClaw response --
        if result.raw_llm_response:
            resp = result.raw_llm_response
            if "error" in resp:
                mlog.info(f"[openclaw-response] error={resp['error']}")
            else:
                model = resp.get("model", "?")
                finish = "?"
                choices = resp.get("choices") or []
                if choices:
                    finish = choices[0].get("finish_reason", "?")
                mlog.info(
                    f"[openclaw-response] model={model} finish_reason={finish}"
                )
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content", "")
                    mlog.info(
                        f"[openclaw-response] content={content}"
                    )

        # -- mock_tool calls (from trajectory) --
        trajectory = result.trajectory or []
        if trajectory:
            mlog.info(
                f"[mock-tools] {len(trajectory)} tool call(s):"
            )
            for idx, tc in enumerate(trajectory, 1):
                tool = tc.get("tool", "?")
                args = tc.get("args", {})
                resp = tc.get("response", {})
                mlog.info(
                    f"  [{idx}] tool={tool} "
                    f"args={json.dumps(args, ensure_ascii=False, default=str)}"
                )
                mlog.info(
                    f"  [{idx}] response="
                    f"{json.dumps(resp, ensure_ascii=False, default=str)}"
                )

        # -- failed requests (from all_requests) --
        all_reqs = result.all_requests or []
        failed = [r for r in all_reqs if not r.get("success")]
        if failed:
            mlog.info(f"[mock-tools] {len(failed)} failed request(s):")
            for idx, fr in enumerate(failed, 1):
                mlog.info(
                    f"  [FAIL-{idx}] tool={fr.get('tool', '?')} "
                    f"status={fr.get('status_code', '?')} "
                    f"body={json.dumps(fr.get('body'), ensure_ascii=False, default=str)}"
                )

        mlog.info(f"--- end {scenario_name} episode detail ---")

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
        1. Fetch + verify pack (usually cache hit after cycle NCD prefetch)
        2. Schema validation
        3. Phase 1: LLM integrity check (cached by pack_hash)
        4. Run episodes (single per scenario, no consensus voting)
        5. Phase 2: LLM trajectory judge per scenario
        6. Return costs + judge-based qualification

        Returns:
            Dict with keys "costs", "qualified" mapping
            scenario_name to values, or None if pre-evaluation checks fail.
        """
        mlog = self._get_miner_logger(commitment.hotkey)
        mlog.info(
            f"Evaluating miner {miner_uid} "
            f"(hotkey={commitment.hotkey[:8]}, hash={commitment.pack_hash[:12]}...)"
        )

        # Step 1: Fetch and verify pack from HTTP URL
        verification = await self.pack_fetcher.verify_submission(
            pack_url=commitment.pack_url,
            pack_hash=commitment.pack_hash,
        )

        if not verification.valid:
            mlog.warning(
                f"Pack verification failed: {verification.error}"
            )
            return None

        pack = verification.pack_content

        # Step 2: Schema validation
        lint_result = validate_opp_schema(pack)
        if not lint_result.passed:
            mlog.warning(f"Schema failed: {lint_result.issues}")
            return None

        # Step 3: Phase 1 — LLM pack integrity analysis (cached by pack_hash)
        integrity = self.integrity_judge.check_integrity(
            pack, pack_hash=commitment.pack_hash
        )
        if not integrity.passed:
            mlog.warning(f"Pack integrity FAILED: {integrity.summary}")
            for flag in integrity.flags:
                mlog.info(
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
            # Clear EMA so weight phase doesn't score this miner using
            # stale data accumulated before the hardcoding was detected.
            self.ema_costs.pop(commitment.hotkey, None)
            self.scenario_qualified.pop(commitment.hotkey, None)
            self._ema_pack_hash.pop(commitment.hotkey, None)
            return None

        if integrity.flags:
            mlog.info(
                f"Integrity passed with {len(integrity.flags)} non-critical flags"
            )

        self._hotkey_packs[commitment.hotkey] = pack
        self._pack_by_hash[commitment.pack_hash] = pack

        # Step 4+5: Run episodes and judge trajectories
        scenario_costs: Dict[str, float] = {}
        scenario_qualified: Dict[str, bool] = {}
        scenario_token_usage: Dict[str, Dict[str, int]] = {}
        scenario_model_usage: Dict[str, List[Dict[str, Any]]] = {}
        scenario_judge_details: Dict[str, Dict[str, Any]] = {}
        scenario_session_keys: Dict[str, str] = {}

        total_scenarios = len(eval_scenarios)
        for scenario_idx, scenario_name in enumerate(eval_scenarios, 1):
            mlog.info(
                f"scenario [{scenario_idx}/{total_scenarios}] {scenario_name} ..."
            )
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
                    mlog.warning(
                        f"{scenario_name} episode error: {result.error}"
                    )
                    scenario_qualified[scenario_name] = False
                    remaining = [s for s in eval_scenarios if s not in scenario_qualified]
                    if remaining:
                        mlog.info(
                            f"fail-fast, skipping {len(remaining)} remaining scenarios"
                        )
                        for s in remaining:
                            scenario_qualified[s] = False
                    break

                self._log_episode_details(mlog, scenario_name, result)

                if result.cost_usd is not None:
                    scenario_costs[scenario_name] = result.cost_usd
                if result.token_usage:
                    scenario_token_usage[scenario_name] = result.token_usage
                if result.model_usage:
                    scenario_model_usage[scenario_name] = result.model_usage
                if result.session_key:
                    scenario_session_keys[scenario_name] = result.session_key

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
                mlog.info(
                    f"{scenario_name} -> "
                    f"judge={judge_result.overall_score:.3f}{cost_str}, "
                    f"gate={gate_str}, tool_calls={result.tool_calls}"
                )
                if result.token_usage:
                    tu = result.token_usage
                    mlog.info(
                        f"{scenario_name}   "
                        f"tokens: input={tu.get('input_tokens', 0)}, "
                        f"output={tu.get('output_tokens', 0)}, "
                        f"cache_read={tu.get('cache_read_tokens', 0)}, "
                        f"cache_write={tu.get('cache_write_tokens', 0)}"
                    )

                if judge_result.error:
                    mlog.warning(
                        f"{scenario_name} judge error: {judge_result.error}"
                    )

                # Log per-criterion details
                for cr in judge_result.criteria_results:
                    if cr.verdict != "PASS":
                        mlog.info(f"  FAIL {cr.id}: {cr.justification}")
                if result.model_usage:
                    for m in result.model_usage:
                        mlog.info(
                            f"{scenario_name}   "
                            f"{m.get('model', '?')}: "
                            f"${m.get('cost_usd', 0):.4f} "
                            f"({m.get('count', 0)} calls)"
                        )

                # Fail fast: any scenario failure → skip remaining
                if not qualified:
                    remaining = [s for s in eval_scenarios if s not in scenario_qualified]
                    if remaining:
                        mlog.info(
                            f"fail-fast on {scenario_name}, "
                            f"skipping {len(remaining)} remaining scenarios"
                        )
                        for s in remaining:
                            scenario_qualified[s] = False
                    break

            except Exception as e:
                mlog.error(
                    f"{scenario_name} failed: {e}", exc_info=True,
                )
                scenario_qualified[scenario_name] = False
                remaining = [s for s in eval_scenarios if s not in scenario_qualified]
                if remaining:
                    mlog.info(
                        f"fail-fast on {scenario_name} exception, "
                        f"skipping {len(remaining)} remaining scenarios"
                    )
                    for s in remaining:
                        scenario_qualified[s] = False
                break

        if not scenario_qualified:
            mlog.warning("No scenario results!")
            return None

        # Log per-miner cost summary
        if scenario_costs:
            total_cost = sum(scenario_costs.values())
            cost_details = ", ".join(
                f"{s}=${c:.4f}" for s, c in scenario_costs.items()
            )
            mlog.info(f"Cost summary: total=${total_cost:.4f} ({cost_details})")

        return {
            "costs": scenario_costs,
            "qualified": scenario_qualified,
            "token_usage": scenario_token_usage,
            "model_usage": scenario_model_usage,
            "judge_details": scenario_judge_details,
            "session_keys": scenario_session_keys,
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

        # Log the full eval summary to the per-miner file for debugging.
        mlog = self._get_miner_logger(hotkey)
        fully_qualified = self.is_fully_qualified(hotkey)
        total_ema_cost = self.compute_total_cost_from_ema(hotkey) or 0.0
        mlog.info(
            f"=== eval summary (uid={uid}, eval#{eval_count}) ==="
        )
        mlog.info(
            f"  score={raw_score:.4f} cost=${raw_cost:.4f} "
            f"ema_cost=${total_ema_cost:.4f} qualified={fully_qualified} "
            f"ema_reset={ema_reset}"
        )
        mlog.info(
            f"  pack_url={commitment.pack_url} "
            f"pack_hash={commitment.pack_hash}"
        )
        for sname, sr in scenario_results.items():
            jd = sr.get("judge", {})
            mlog.info(
                f"  {sname}: qualified={sr.get('qualified')} "
                f"cost=${sr.get('cost', 0):.4f} "
                f"ema_cost=${sr.get('ema_cost', 0):.4f} "
                f"judge_score={jd.get('overall_score', '?')} "
                f"verdict={jd.get('verdict', '?')} "
                f"grounded={jd.get('grounded', '?')} "
                f"safety={jd.get('safety_passed', '?')} "
                f"correctness={jd.get('correctness_passed', '?')}"
            )
            tu = sr.get("token_usage")
            if tu:
                mlog.info(
                    f"    tokens: in={tu.get('input_tokens', 0)} "
                    f"out={tu.get('output_tokens', 0)} "
                    f"cache_r={tu.get('cache_read_tokens', 0)} "
                    f"cache_w={tu.get('cache_write_tokens', 0)}"
                )
            mu = sr.get("model_usage")
            if mu:
                for m in mu:
                    mlog.info(
                        f"    model={m.get('model', '?')} "
                        f"cost=${m.get('cost_usd', 0):.4f} "
                        f"calls={m.get('count', 0)}"
                    )
        mlog.info("=== end eval summary ===")

        await submit_eval(
            self.wallet,
            miner_hotkey=hotkey,
            miner_uid=uid,
            block_height=block_height,
            score=round(raw_score, 4),
            ema_score=round(raw_score, 4),
            cost=round(raw_cost, 4),
            ema_cost=round(total_ema_cost, 4),
            weight=0.0,
            qualified=fully_qualified,
            pack_url=commitment.pack_url,
            pack_hash=commitment.pack_hash,
            eval_count=eval_count,
            ema_reset=ema_reset,
            scenario_results=scenario_results,
            llm_base_url=self._judge_base_url,
            llm_model=self._judge_model,
        )

    # ------------------------------------------------------------------
    # Eval log upload
    # ------------------------------------------------------------------

    _MAX_LOG_ARCHIVE_BYTES = 10 * 1024 * 1024  # 10 MB

    def _get_validator_log_offset(self) -> int:
        """Return the current byte offset of the main validator log file."""
        try:
            return (
                self._validator_log_path.stat().st_size
                if self._validator_log_path.exists() else 0
            )
        except OSError:
            return 0

    def _prepare_eval_log_capture(self, eval_id: str, hotkey: str) -> Tuple[Path, int, int]:
        """Prepare for log capture before an eval run.

        Creates a unique eval directory, truncates mock-tools JSONL log
        files, and records file offsets for the main validator log and the
        per-miner log so only new content is captured.

        Args:
            eval_id: Cycle-level eval identifier (YYYYMMDD_HHMMSS).
            hotkey: Miner hotkey.

        Returns:
            (eval_dir, validator_log_offset, miner_log_offset) tuple.
        """
        eval_dir = self.config.log_dir / "evals" / eval_id / hotkey[:16]
        eval_dir.mkdir(parents=True, exist_ok=True)

        for pattern in ("*_calls.jsonl", "*_all_requests.jsonl"):
            for path in self.config.log_dir.glob(pattern):
                try:
                    path.write_text("")
                except OSError:
                    pass

        try:
            vlog_offset = (
                self._validator_log_path.stat().st_size
                if self._validator_log_path.exists() else 0
            )
        except OSError:
            vlog_offset = 0

        miner_log = self._miner_log_dir / f"{hotkey[:16]}.log"
        try:
            mlog_offset = miner_log.stat().st_size if miner_log.exists() else 0
        except OSError:
            mlog_offset = 0

        return eval_dir, vlog_offset, mlog_offset

    def _collect_eval_logs(
        self,
        hotkey: str,
        eval_scenarios: List[str],
        eval_dir: Path,
        validator_log_offset: int,
        miner_log_offset: int,
        session_keys: Dict[str, str],
    ) -> Optional[bytes]:
        """Snapshot log files into *eval_dir* and package as tar.gz.

        Copies the main validator log segment, per-miner log segment, and
        per-scenario JSONL files into the eval directory so they persist
        locally for debugging, then creates an in-memory tar.gz archive
        for upload.

        Args:
            hotkey: Miner hotkey (used to locate per-miner log).
            eval_scenarios: Scenario names evaluated in this run.
            eval_dir: Unique eval directory created by _prepare_eval_log_capture.
            validator_log_offset: Main validator log byte offset before eval.
            miner_log_offset: Per-miner log byte offset before eval.
            session_keys: Mapping of scenario name to OpenClaw session key.

        Returns:
            Gzipped tar archive bytes, or None on failure / over-size.
        """
        import io
        import shutil
        import tarfile

        try:
            for src_path, dst_name, offset in (
                (self._validator_log_path, "validator.log", validator_log_offset),
                (self._miner_log_dir / f"{hotkey[:16]}.log", "miner.log", miner_log_offset),
            ):
                if src_path.exists():
                    with open(src_path, "rb") as f:
                        f.seek(offset)
                        segment = f.read()
                    if segment:
                        (eval_dir / dst_name).write_bytes(segment)

            # Copy OpenClaw session transcript files (full LLM conversation logs)
            openclaw_sessions_dir = Path("/root/.openclaw/agents/main/sessions")
            for scenario, session_key in session_keys.items():
                if session_key:
                    src = openclaw_sessions_dir / f"{session_key}.jsonl"
                    if src.exists() and src.stat().st_size > 0:
                        dst = eval_dir / f"{scenario}_conversation.jsonl"
                        shutil.copy2(str(src), str(dst))
                        logger.info(f"Copied session transcript for {scenario}: {session_key}")

            for scenario in eval_scenarios:
                for suffix in ("_calls.jsonl", "_all_requests.jsonl"):
                    src = self.config.log_dir / f"{scenario}{suffix}"
                    if src.exists() and src.stat().st_size > 0:
                        shutil.copy2(str(src), str(eval_dir / f"{scenario}{suffix}"))

            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for child in sorted(eval_dir.iterdir()):
                    tar.add(str(child), arcname=child.name)

            archive = buf.getvalue()
            if len(archive) > self._MAX_LOG_ARCHIVE_BYTES:
                logger.warning(
                    "Log archive too large (%d bytes), skipping upload",
                    len(archive),
                )
                return None
            return archive
        except Exception as e:
            logger.warning("Failed to collect eval logs: %s", e)
            return None

    async def _fire_upload_eval_logs(
        self,
        eval_id: str,
        uid: int,
        commitment: "MinerCommitment",
        eval_scenarios: List[str],
        eval_result: Optional[Dict],
        eval_dir: Path,
        validator_log_offset: int,
        miner_log_offset: int,
        block_height: int,
    ) -> None:
        """Collect and upload eval logs. Fire-and-forget."""
        session_keys = eval_result.get("session_keys", {}) if eval_result else {}
        log_archive = self._collect_eval_logs(
            commitment.hotkey, eval_scenarios, eval_dir,
            validator_log_offset, miner_log_offset, session_keys,
        )
        if log_archive:
            await upload_eval_logs(
                self.wallet,
                eval_id=eval_id,
                miner_hotkey=commitment.hotkey,
                miner_uid=uid,
                block_height=block_height,
                pack_hash=commitment.pack_hash,
                log_archive=log_archive,
            )

    async def _fire_upload_cycle_logs(
        self,
        eval_id: str,
        cycle_log_offset: int,
        block_height: int,
    ) -> None:
        """Upload the full eval cycle validator log. Fire-and-forget."""
        import io
        import tarfile

        try:
            if not self._validator_log_path.exists():
                return
            with open(self._validator_log_path, "rb") as f:
                f.seek(cycle_log_offset)
                segment = f.read()
            if not segment:
                return

            cycle_dir = self.config.log_dir / "evals" / eval_id
            cycle_dir.mkdir(parents=True, exist_ok=True)
            (cycle_dir / "validator.log").write_bytes(segment)

            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(
                    str(cycle_dir / "validator.log"),
                    arcname="validator.log",
                )
            archive = buf.getvalue()
            if len(archive) > self._MAX_LOG_ARCHIVE_BYTES:
                logger.warning(
                    "Cycle log archive too large (%d bytes), skipping",
                    len(archive),
                )
                return

            await upload_cycle_logs(
                self.wallet,
                eval_id=eval_id,
                block_height=block_height,
                log_archive=archive,
            )
        except Exception as e:
            logger.warning("Cycle log upload error: %s", e)

    # ------------------------------------------------------------------
    # Weight setting
    # ------------------------------------------------------------------

    async def _apply_weight_override(self) -> bool:
        """Check for a forced weight override and apply it if active.

        Only activated in exceptional circumstances (e.g. consensus
        divergence) to save validator vtrust by forcing a known-good
        winner UID from the server side.

        Returns True if an override was applied (caller should skip normal
        weight setting), False otherwise.
        """
        from datetime import datetime, timezone

        override = await fetch_weight_override(self.wallet)
        if override is None:
            return False

        expires_str = override.get("expiresAt")
        winner_uid = override.get("winnerUid")
        if expires_str is None or winner_uid is None:
            return False

        try:
            expires_at = datetime.fromisoformat(
                expires_str.replace("+00", "+00:00")
            )
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            if now >= expires_at:
                logger.info(
                    "Weight override expired at %s, ignoring", expires_str
                )
                return False
        except Exception as e:
            logger.warning("Failed to parse override expiresAt: %s", e)
            return False

        if winner_uid == OWNER_UID:
            uids = [OWNER_UID]
            weights = [1.0]
            logger.info(
                "Weight override: winnerUid=%d is owner UID, "
                "setting weight 1.0 to owner UID %d",
                winner_uid, OWNER_UID,
            )
        else:
            uids = [int(winner_uid), OWNER_UID]
            weights = [0.5, 0.5]
            logger.info(
                "Weight override: setting weight 0.5 to winnerUid %d "
                "and 0.5 to owner UID %d",
                winner_uid, OWNER_UID,
            )

        try:
            self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )
            logger.info("Weight override applied successfully")
            self._last_set_weights_at = int(time.time())
            self._last_weights_uids = uids
            self._last_weights = weights
        except Exception as e:
            logger.error("Error applying weight override: %s", e, exc_info=True)
        return True

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

        if not scores:
            logger.warning("All miners have zero EMA score")
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
            champion_hotkey=self.champion_hotkey,
        )

        # Update champion_hotkey to this cycle's winner (only if there IS a winner)
        winner_uid = max(weights_dict, key=weights_dict.get)
        if weights_dict.get(winner_uid, 0) > 0 and winner_uid in uid_to_hotkey:
            self.champion_hotkey = uid_to_hotkey[winner_uid]

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

    def _fallback_owner_weights(self) -> Optional[Tuple[list, list]]:
        """Read on-chain weights set by OWNER_UID and return (uids, weights).

        Returns None if the owner has no weights or the read fails.
        """
        try:
            self.metagraph.sync(subtensor=self.subtensor)
            W = self.metagraph.W  # (n, n) weight matrix
            if OWNER_UID >= W.shape[0]:
                logger.warning(
                    f"OWNER_UID {OWNER_UID} out of range (metagraph size {W.shape[0]})"
                )
                return None

            owner_weights = W[OWNER_UID]
            # Extract non-zero entries
            uids = []
            weights = []
            for uid, w in enumerate(owner_weights.tolist()):
                if w > 0:
                    uids.append(uid)
                    weights.append(float(w))

            if not uids:
                logger.warning(
                    f"Owner UID {OWNER_UID} has no non-zero weights on chain"
                )
                return None

            logger.info(
                f"Copied {len(uids)} weight entries from owner UID {OWNER_UID}"
            )
            return uids, weights
        except Exception as e:
            logger.warning(f"Failed to read owner weights from chain: {e}")
            return None

    def _fallback_to_owner(self) -> Tuple[list, list]:
        """Return weight=1.0 on OWNER_UID only (burns emissions)."""
        return [OWNER_UID], [1.0]

    async def _set_fallback_weights(self, reason: str = "No eligible miners"):
        """Set weights when no miners qualify.

        Fallback order:
        1. Copy on-chain weights from OWNER_UID (UID 74).
        2. If that fails, set weight=1.0 to OWNER_UID only (burns emissions).

        The validator must always call set_weights every tempo to avoid
        deregistration.
        """
        try:
            _ = self.wallet.hotkey
        except Exception:
            logger.debug("Skipping fallback weights: wallet hotkey not available")
            return

        try:
            copied = self._fallback_owner_weights()
            if copied is not None:
                uids, weights = copied
                logger.info(
                    f"{reason} — copying weights from owner UID {OWNER_UID} "
                    f"({len(uids)} entries)"
                )
            else:
                uids, weights = self._fallback_to_owner()
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
            self._last_set_weights_at = int(time.time())
        except Exception as e:
            logger.error(f"Error setting fallback weights: {e}", exc_info=True)


async def main():
    """Entry point for validator."""
    config = ValidatorConfig.from_env()
    validator = TrajectoryValidator(config)
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
