"""TrajectoryRL Validator — Main validator implementation.

Architecture (Season 1 — trajrl-bench):
    1. Continuous evaluation loop with dual cadence:
       - eval_interval (~24h): re-evaluate all active packs
       - tempo (~72 min): compute weights from qualification + cost, set_weights
    2. Read on-chain commitments (subtensor.get_all_commitments)
    3. Fetch packs from miners' public HTTP URLs
    4. NCD pairwise dedup; schema validation in _evaluate_miner
    5. Run trajrl-bench sandbox evaluation (SKILL.md packs, SSH sandbox, LLM judge)
    6. Compute split-half delta scoring (quality-based)
    7. Set on-chain weights (winner-take-all / bootstrap)

Each validator operates independently — YC3 aggregates on-chain.
"""

import asyncio
import datetime
import hashlib
import json
import logging
import os
import time

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
import docker.errors

from ..utils.config import ValidatorConfig
from ..utils.sandbox_harness import TrajectorySandboxHarness, SandboxEvaluationResult
from ..scoring import TrajectoryScorer
from ..utils.github import PackFetcher
from ..utils.eval_window import (
    WindowConfig, WindowPhase, EvaluationWindow,
    compute_window, is_new_window, can_evaluate,
    should_submit, should_aggregate,
)
from ..utils import consensus as _consensus_mod
from ..utils.consensus import (
    ConsensusPayload, ConsensusPointer,
    CONSENSUS_PROTOCOL_VERSION,
)

def _scoring_version() -> int:
    """Current scoring version — major version of trajrl-bench (e.g. v3.0.1 → 3).
    Set dynamically after pulling the sandbox image at each eval cycle."""
    return _consensus_mod.SCORING_VERSION
from ..utils.consensus_store import (
    ConsensusStore, IPFSBackend, TrajRLAPIBackend,
)
from ..utils.consensus_filter import (
    run_filter_pipeline, ValidatedSubmission,
)
from ..scoring import compute_consensus_scores
from ..utils.winner_state import (
    WinnerState, select_winner_with_protection,
    save_winner_state, load_winner_state,
)
from ..utils.commitments import (
    MinerCommitment, fetch_all_commitments,
    ValidatorConsensusCommitment, fetch_validator_consensus_commitments,
    format_consensus_commitment, decode_dual_address,
    is_consensus_commitment, parse_consensus_commitment,
)
from ..utils.ncd import deduplicate_packs
from ..utils.status_reporter import (
    heartbeat, pre_eval, submit_eval, upload_eval_logs, upload_cycle_logs,
)
from ..utils.llm_judge import PackIntegrityJudge, TrajectoryJudge
from .. import __version__

logger = logging.getLogger(__name__)

OWNER_UID = 74
BURN_FRACTION = 0.50  # 50% of miner emissions burned via owner UID
EVAL_START_BLOCK = 7986780  # 2026-04-17 08:00 UTC (ref: block 7986030 @ 05:30:09 UTC, ~12s/block)

_METAGRAPH_SYNC_RETRIES = 3
_METAGRAPH_SYNC_DELAY = 10  # seconds between retries
_METAGRAPH_MIN_NEURONS = 1  # minimum expected neurons for a healthy metagraph

_SET_WEIGHTS_MAX_RETRIES = 3
_SET_WEIGHTS_RETRY_DELAY = 12  # seconds; roughly 1 block interval


class TrajectoryValidator:
    """TrajectoryRL validator that evaluates SKILL.md packs via trajrl-bench.

    Season 1 flow:
    1. Reads on-chain commitments from miners
    2. Fetches and verifies packs from miners' public HTTP URLs
    3. NCD pairwise dedup (copiers rejected like integrity fail)
    4. Runs trajrl-bench sandbox evaluation (SSH sandbox, LLM judge)
    5. Computes split-half delta scoring (quality-based)
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
        logger.info(f"TrajectoryRL Validator v{__version__}")
        logger.info("=" * 60)

        logger.debug("Initializing Bittensor components...")
        self.wallet = bt.Wallet(
            name=config.wallet_name,
            hotkey=config.wallet_hotkey,
        )
        self.subtensor = bt.Subtensor(network=config.network)
        self.metagraph = self.subtensor.metagraph(config.netuid)

        mg_n = getattr(self.metagraph, "n", 0)
        logger.info(f"Wallet hotkey: {self.wallet.hotkey.ss58_address[:16]}...")
        logger.info(f"Network: {config.network}")
        logger.info(f"Netuid: {config.netuid}")
        logger.info(f"Metagraph: n={mg_n}, block={getattr(self.metagraph, 'block', '?')}")
        if not mg_n:
            logger.error(
                "METAGRAPH EMPTY at startup (n=0). "
                "This validator will not be able to set weights until "
                "metagraph sync is healthy. Check subtensor endpoint: %s",
                config.network,
            )

        logger.info("Initializing trajrl-bench sandbox harness...")
        self._sandbox_harness = TrajectorySandboxHarness(config)

        self.scorer = TrajectoryScorer(
            consensus_epsilon=config.consensus_epsilon,
            bootstrap_threshold=config.bootstrap_threshold,
        )

        judge_model = config.judge_model or config.llm_model
        judge_api_key = config.judge_api_key or config.llm_api_key
        judge_base_url = config.judge_base_url or config.llm_base_url
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

        # Per-scenario continuous quality scores: {hotkey: {scenario: float in [0.0, 1.0]}}
        # Source: judge_details[scenario].overall_score (S1 sandbox eval).
        self.scenario_scores: Dict[str, Dict[str, float]] = {}

        # Latest token usage per hotkey/scenario: {hotkey: {scenario: {input_tokens, ...}}}
        self.latest_token_usage: Dict[str, Dict[str, Dict[str, int]]] = {}

        # Latest model usage per hotkey/scenario: {hotkey: {scenario: [model_entry, ...]}}
        self.latest_model_usage: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        # Track the pack_hash that each hotkey was last evaluated with.
        self._eval_pack_hash: Dict[str, str] = {}

        # Last eval block for each hotkey (for rate-limiting and inactivity)
        self.last_eval_block: Dict[str, int] = {}

        # Pack content cache: pack_hash -> pack dict
        self._pack_by_hash: Dict[str, dict] = {}

        # Packs by hotkey (populated during evaluation for NCD dedup)
        self._hotkey_packs: Dict[str, dict] = {}

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
            window_shift=config.window_shift,
        )
        # Which window's aggregation has completed (persisted in eval state).
        # Guards one-shot aggregation per window — survives restarts.
        self._consensus_window: int = -1

        # Cycle log state — uploaded after each window phase completes
        self._cycle_eval_id: Optional[str] = None
        self._cycle_log_offset: int = 0
        self._cycle_log_block: int = 0

        # Consensus CAS store (IPFS + API)
        def _sign_consensus_msg(msg: str) -> str:
            sig = self.wallet.hotkey.sign(msg.encode())
            return "0x" + (sig if isinstance(sig, bytes) else bytes(sig)).hex()

        self._consensus_store = ConsensusStore(
            ipfs=IPFSBackend(
                api_url=config.ipfs_api_url,
                gateway_urls=config.ipfs_gateway_urls,
            ),
            api=TrajRLAPIBackend(
                base_url=config.consensus_api_url,
                sign_fn=_sign_consensus_msg,
                validator_hotkey=self.wallet.hotkey.ss58_address,
            ),
        )


        # Miners disqualified during current window's evaluation (pre-eval, integrity)
        # Reset at each new eval cycle. Included in ConsensusPayload.
        self._disqualified_miners: Dict[str, str] = {}

        # Winner Protection state
        self._winner_state_path = str(config.winner_state_path)
        self._winner_state = load_winner_state(self._winner_state_path)


        # Scenario config hash for state invalidation — driven by bench version
        self._scenario_config_hash = "trajrl-bench"

        # Load persisted evaluation state
        self._load_eval_state()

        logger.info("Validator initialization complete!")

    # ------------------------------------------------------------------
    # Metagraph sync helpers
    # ------------------------------------------------------------------

    def _sync_metagraph(
        self,
        *,
        retries: int = _METAGRAPH_SYNC_RETRIES,
        caller: str = "",
    ) -> bool:
        """Sync metagraph with retries.  Returns True if healthy (n > 0).

        On each failed attempt the subtensor connection is rebuilt so that
        transient network / websocket issues are recovered from automatically.
        """
        label = f"[{caller}] " if caller else ""

        for attempt in range(1, retries + 1):
            try:
                self.metagraph.sync(subtensor=self.subtensor)
            except Exception as e:
                logger.warning(
                    "%sMetagraph sync attempt %d/%d failed: %s",
                    label, attempt, retries, e,
                )
                if attempt < retries:
                    time.sleep(_METAGRAPH_SYNC_DELAY)
                    self._reconnect_subtensor(label)
                continue

            n = getattr(self.metagraph, "n", 0)
            if n and n >= _METAGRAPH_MIN_NEURONS:
                if attempt > 1:
                    logger.info(
                        "%sMetagraph sync recovered on attempt %d/%d "
                        "(n=%d, block=%s)",
                        label, attempt, retries, n,
                        getattr(self.metagraph, "block", "?"),
                    )
                return True

            logger.warning(
                "%sMetagraph sync returned n=%d (expected >= %d), "
                "attempt %d/%d",
                label, n, _METAGRAPH_MIN_NEURONS, attempt, retries,
            )
            if attempt < retries:
                time.sleep(_METAGRAPH_SYNC_DELAY)
                self._reconnect_subtensor(label)

        n = getattr(self.metagraph, "n", 0)
        logger.error(
            "%sMETAGRAPH UNHEALTHY after %d attempts — n=%d, "
            "block=%s. On-chain operations (set_weights, commitments) "
            "will likely fail silently. Check subtensor endpoint: %s",
            label, retries, n,
            getattr(self.metagraph, "block", "?"),
            self.config.network,
        )
        return False

    def _reconnect_subtensor(self, label: str = "") -> None:
        """Rebuild the subtensor connection to recover from stale websockets."""
        try:
            logger.info("%sReconnecting subtensor (network=%s)...",
                        label, self.config.network)
            self.subtensor = bt.Subtensor(network=self.config.network)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            logger.info("%sSubtensor reconnected", label)
        except Exception as e:
            logger.error(
                "%sFailed to reconnect subtensor: %s", label, e,
                exc_info=True,
            )

    def _is_metagraph_healthy(self) -> bool:
        """Quick check whether the cached metagraph has any neurons."""
        n = getattr(self.metagraph, "n", 0)
        return bool(n and n >= _METAGRAPH_MIN_NEURONS)

    # ------------------------------------------------------------------
    # Evaluation state persistence
    # ------------------------------------------------------------------

    def _compute_scenario_config_hash(self) -> str:
        """Hash the bench version for detecting configuration changes."""
        config_str = f"trajrl-bench:{self._sandbox_harness.sandbox_version}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _load_eval_state(self):
        """Load persisted evaluation state from disk.

        Invalidates all state when either ``scoring_version`` or
        ``scenario_config_hash`` no longer matches the running code.
        """
        path = self.config.eval_state_path
        if not path.exists():
            legacy = path.parent / "ema_state.json"
            if legacy.exists():
                logger.info("Migrating legacy ema_state.json → %s", path.name)
                path = legacy
            else:
                logger.debug("No persisted eval state found, starting fresh")
                return
        try:
            data = json.loads(path.read_text())

            file_sv = data.get("scoring_version", 1)
            if file_sv != _scoring_version():
                logger.info(
                    "Eval state scoring_version mismatch (%d != %d), "
                    "invalidating all eval state",
                    file_sv, _scoring_version(),
                )
                return

            if data.get("scenario_config_hash") != self._scenario_config_hash:
                logger.debug(
                    "Scenario pool changed, invalidating all eval state"
                )
                return

            self.scenario_scores = data.get("scenario_scores", {})
            self._eval_pack_hash = data.get("eval_pack_hash", data.get("ema_pack_hash", {}))
            self.last_eval_block = {
                k: int(v) for k, v in data.get("last_eval_block", {}).items()
            }
            self._last_eval_window = data.get("last_eval_window", -1)

            self._consensus_window = data.get("consensus_window", -1)

            integrity_cache = data.get("integrity_cache")
            if integrity_cache:
                self.integrity_judge.load_cache(integrity_cache)

            logger.debug(
                f"Loaded eval state: {len(self.scenario_scores)} hotkeys"
            )
        except Exception as e:
            logger.warning(f"Failed to load eval state: {e}, starting fresh")

    def _save_eval_state(self):
        """Persist evaluation state to disk for restart recovery."""
        data = {
            "scoring_version": _scoring_version(),
            "scenario_config_hash": self._scenario_config_hash,
            "scenario_scores": self.scenario_scores,
            "eval_pack_hash": self._eval_pack_hash,
            "last_eval_block": self.last_eval_block,
            "last_eval_window": self._last_eval_window,
            "consensus_window": self._consensus_window,
            "integrity_cache": self.integrity_judge.dump_cache(),
        }
        try:
            self.config.eval_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.eval_state_path.write_text(
                json.dumps(data, indent=2, sort_keys=True)
            )
        except Exception as e:
            logger.warning(f"Failed to save eval state: {e}")

    # ------------------------------------------------------------------
    # Eval state update
    # ------------------------------------------------------------------

    def _update_eval_results(
        self,
        hotkey: str,
        pack_hash: str,
        scenario_scores: Optional[Dict[str, float]] = None,
    ):
        """Record per-scenario quality scores.

        Resets data when pack_hash changes (new pack = new observations).
        """
        if self._eval_pack_hash.get(hotkey) != pack_hash:
            self._get_miner_logger(hotkey).info(
                f"Pack changed "
                f"({self._eval_pack_hash.get(hotkey, 'none')[:8]} -> {pack_hash[:8]}), "
                f"resetting eval data"
            )
            self.scenario_scores[hotkey] = {}
            self.latest_token_usage.pop(hotkey, None)
            self.latest_model_usage.pop(hotkey, None)
            self._eval_pack_hash[hotkey] = pack_hash

        if scenario_scores:
            if hotkey not in self.scenario_scores:
                self.scenario_scores[hotkey] = {}
            self.scenario_scores[hotkey].update(scenario_scores)

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
                    bench_image_hash=self._sandbox_harness.bench_image_hash,
                    harness_image_hash=self._sandbox_harness.harness_image_hash,
                    bench_version=self._sandbox_harness.sandbox_version,
                )
            except Exception as e:
                logger.warning("Heartbeat error: %s", e)
            await asyncio.sleep(600)

    def _check_own_commitment_on_chain(self, window_number: int) -> bool:
        """Check if this validator's consensus commitment exists on-chain for the given window."""
        my_hotkey = self.wallet.hotkey.ss58_address
        try:
            raw_commitments = self.subtensor.get_all_commitments(
                netuid=self.config.netuid,
            )
        except Exception as e:
            logger.warning(
                "Window %d: failed to read on-chain commitments for self-check: %s",
                window_number, e,
            )
            return False

        if not raw_commitments:
            return False

        raw = raw_commitments.get(my_hotkey)
        if not raw or not is_consensus_commitment(raw):
            return False

        parsed = parse_consensus_commitment(raw)
        if parsed is None:
            return False

        _, on_chain_window, _, _ = parsed
        return on_chain_window == window_number

    async def _submit_consensus_payload(self, window: EvaluationWindow) -> bool:
        """Build and upload consensus payload to CAS, then write pointer on-chain.

        Returns True if the on-chain commitment was written successfully,
        False otherwise (caller may retry).
        """
        payload = self._build_local_consensus_payload(window)
        if payload is None:
            logger.warning(
                "Window %d: no evaluation data to submit", window.window_number
            )
            return False

        content_address = await self._consensus_store.upload_payload(payload)
        if content_address is None:
            logger.error(
                "Window %d: failed to upload consensus payload",
                window.window_number,
            )
            return False

        MAX_COMMITMENT_BYTES = 128

        commitment_str = format_consensus_commitment(
            protocol_version=self.config.consensus_protocol_version,
            window_number=window.window_number,
            content_address=content_address,
            scoring_version=_scoring_version(),
        )

        if len(commitment_str.encode("utf-8")) > MAX_COMMITMENT_BYTES:
            ipfs_cid, _ = decode_dual_address(content_address)
            fallback_address = ipfs_cid or content_address
            commitment_str = format_consensus_commitment(
                protocol_version=self.config.consensus_protocol_version,
                window_number=window.window_number,
                content_address=fallback_address,
                scoring_version=_scoring_version(),
            )
            logger.warning(
                "Window %d: commitment too long with dual address, "
                "using IPFS-only (%d bytes)",
                window.window_number, len(commitment_str.encode("utf-8")),
            )

        try:
            self.subtensor.set_commitment(
                wallet=self.wallet,
                netuid=self.config.netuid,
                data=commitment_str,
            )
            logger.info(
                "Window %d: consensus pointer written on-chain "
                "(address=%s, %d miners, commitment=%s, %d bytes)",
                window.window_number, content_address[:24],
                len(payload.scores), commitment_str[:60],
                len(commitment_str.encode("utf-8")),
            )
            return True
        except Exception as e:
            logger.error(
                "Window %d: failed to write consensus pointer on-chain: %s",
                window.window_number, e,
            )
            return False

    def _build_local_consensus_payload(
        self, window: EvaluationWindow,
    ) -> Optional[ConsensusPayload]:
        """Build a ConsensusPayload from in-memory evaluation data.

        Used as a fallback when this validator's submission is absent from
        CAS/chain during aggregation.  Returns None if no eval data exists.
        """
        scores_by_hotkey: Dict[str, float] = {}
        disqualified_by_hotkey: Dict[str, str] = {}

        # Aggregate continuous quality scores from judge overall_score.
        # Winner-take-all downstream, so the mean is just used to rank miners.
        for hotkey, scenario_s in self.scenario_scores.items():
            if scenario_s:
                score = sum(scenario_s.values()) / len(scenario_s)
                scores_by_hotkey[hotkey] = round(score, 4)

        for hotkey, reason in getattr(self, "_disqualified_miners", {}).items():
            disqualified_by_hotkey[hotkey] = reason

        if not scores_by_hotkey and not disqualified_by_hotkey:
            return None

        bench_ver = self._sandbox_harness.sandbox_version

        return ConsensusPayload(
            protocol_version=self.config.consensus_protocol_version,
            window_number=window.window_number,
            validator_hotkey=self.wallet.hotkey.ss58_address,
            bench_version=bench_ver,
            scores=scores_by_hotkey,
            timestamp=int(time.time()),
            scoring_version=_scoring_version(),
            disqualified=disqualified_by_hotkey,
        )

    async def _run_consensus_aggregation(self, window: EvaluationWindow):
        """Read on-chain consensus pointers, download payloads, filter, aggregate."""
        if not self._is_metagraph_healthy():
            logger.warning(
                "Window %d: metagraph unhealthy (n=%d) before consensus "
                "aggregation — re-syncing...",
                window.window_number, getattr(self.metagraph, "n", 0),
            )
            self._sync_metagraph(caller="consensus_aggregation")

        chain_commitments = fetch_validator_consensus_commitments(
            self.subtensor, self.config.netuid, self.metagraph,
        )
        if not chain_commitments:
            logger.warning(
                "Window %d: no consensus commitments on chain, using local results",
                window.window_number,
            )
            return

        submissions = []
        skipped_sv = 0
        download_failed = 0
        for vc in chain_commitments:
            if vc.scoring_version != _scoring_version():
                skipped_sv += 1
                continue
            pointer = ConsensusPointer(
                protocol_version=vc.protocol_version,
                window_number=vc.window_number,
                content_address=vc.content_address,
                validator_hotkey=vc.validator_hotkey,
            )
            payload = await self._consensus_store.download_payload(
                vc.content_address
            )
            if payload is not None:
                submissions.append((pointer, payload))
            else:
                download_failed += 1
        if skipped_sv:
            logger.info(
                "Window %d: skipped %d commitments with mismatched "
                "scoring_version (expected %d)",
                window.window_number, skipped_sv, _scoring_version(),
            )

        my_hotkey = self.wallet.hotkey.ss58_address
        own_found = any(
            p.validator_hotkey == my_hotkey for p, _ in submissions
        )
        if not own_found:
            logger.info(
                "Window %d: own submission not found in CAS downloads; "
                "aggregating from other validators only",
                window.window_number,
            )

        if not submissions:
            total = len(chain_commitments)
            if skipped_sv == total:
                logger.warning(
                    "Window %d: no usable commitments — all %d filtered out "
                    "due to scoring_version mismatch (local=%d), "
                    "using local results",
                    window.window_number, total, _scoring_version(),
                )
            else:
                logger.warning(
                    "Window %d: no usable submissions from %d commitments "
                    "(%d version-filtered, %d download-failed), "
                    "using local results",
                    window.window_number, total, skipped_sv, download_failed,
                )
            return

        validator_stakes: Dict[str, float] = {}
        for uid in range(len(self.metagraph.hotkeys)):
            hotkey = self.metagraph.hotkeys[uid]
            stake = float(self.metagraph.stake[uid])
            if stake > 0:
                validator_stakes[hotkey] = stake

        bench_ver = self._sandbox_harness.sandbox_version
        min_stake = getattr(self.config, "min_validator_stake", 0.0)
        validated, stats = run_filter_pipeline(
            submissions=submissions,
            expected_window=window.window_number,
            validator_stakes=validator_stakes,
            min_stake=min_stake,
            local_version=bench_ver,
            expected_protocol=self.config.consensus_protocol_version,
            expected_scoring_version=_scoring_version(),
        )

        if not validated:
            logger.warning(
                "Window %d: all submissions filtered out (%s), using local results",
                window.window_number, stats.summary(),
            )
            return

        consensus_scores, disqualified = compute_consensus_scores(validated)

        # Build hotkey → UID mapping (used by pre-eval reporting, winner
        # selection, and logging)
        hk_to_uid: Dict[str, int] = {}
        for uid in range(len(self.metagraph.hotkeys)):
            hk_to_uid[self.metagraph.hotkeys[uid]] = uid

        # Pre-eval gate during aggregation: disqualify miners rejected by the
        # platform before selecting a winner.  Mirrors the per-miner pre-eval
        # check in the evaluation loop.
        if os.getenv("TRAJECTORYRL_PRE_EVAL_ENABLED", "1") != "0":
            miner_commitments = fetch_all_commitments(
                self.subtensor, self.config.netuid, self.metagraph,
            )
            hk_to_commitment: Dict[str, MinerCommitment] = {
                c.hotkey: c for c in miner_commitments.values()
            }
            current_block = self.subtensor.get_current_block()

            miners_to_check = [
                (miner_hk, hk_to_commitment[miner_hk])
                for miner_hk in consensus_scores
                if miner_hk in hk_to_commitment
            ]

            if miners_to_check:
                sem = asyncio.Semaphore(8)

                async def _limited_pre_eval(hk: str, c: MinerCommitment):
                    async with sem:
                        return await pre_eval(
                            hk, c.pack_hash, c.pack_url, wallet=self.wallet,
                        )

                results = await asyncio.gather(*(
                    _limited_pre_eval(miner_hk, commitment)
                    for miner_hk, commitment in miners_to_check
                ))

                pre_eval_disqualified = 0
                for (miner_hk, commitment), pre_eval_result in zip(miners_to_check, results):
                    if pre_eval_result is not None and not pre_eval_result.get("allowed", True):
                        reason = pre_eval_result.get("reason", "unknown")
                        disqualified[miner_hk] = f"pre_eval:{reason}"
                        pre_eval_disqualified += 1
                        _stage = "integrity_check" if reason == "hardcoded" else "pack_fetch"
                        _detail = (
                            f"pre-eval rejected: {reason}"
                            + (
                                f", banned_until={pre_eval_result['banned_until']}"
                                if "banned_until" in pre_eval_result
                                else ""
                            )
                        )
                        logger.info(
                            "Window %d: miner %s pre-eval rejected during "
                            "aggregation (reason=%s) — marked disqualified",
                            window.window_number, miner_hk[:8], reason,
                        )
                        asyncio.ensure_future(
                            submit_eval(
                                self.wallet,
                                miner_hotkey=miner_hk,
                                miner_uid=hk_to_uid.get(miner_hk, -1),
                                block_height=current_block,
                                score=0.0,
                                weight=0.0,
                                qualified=False,
                                pack_url=commitment.pack_url,
                                pack_hash=commitment.pack_hash,
                                llm_base_url=self._judge_base_url,
                                llm_model=self._judge_model,
                                rejected=True,
                                rejection_stage=_stage,
                                rejection_detail=_detail,
                                scoring_version=_scoring_version(),
                                **self._harness_metadata(),
                            )
                        )
                if pre_eval_disqualified:
                    logger.info(
                        "Window %d: %d miner(s) disqualified by pre-eval "
                        "during aggregation",
                        window.window_number, pre_eval_disqualified,
                    )

        self._consensus_window = window.window_number

        # Filter disqualified miners from consensus scores before winner selection
        eligible_scores = {
            hk: s for hk, s in consensus_scores.items() if hk not in disqualified
        }

        # Apply Winner Protection (stores winner_uid so set_weights
        # can skip the metagraph lookup later)
        winner_hk, updated_state = select_winner_with_protection(
            consensus_scores=eligible_scores,
            state=self._winner_state,
            score_delta=self.config.score_delta,
            hk_to_uid=hk_to_uid,
            disable_winner_protection=self.config.disable_winner_protection,
        )
        self._winner_state = updated_state
        save_winner_state(updated_state, self._winner_state_path)

        logger.info("=" * 60)
        logger.info(
            "Window %d: CONSENSUS RESULTS — %d miners (%d eligible, %d disqualified), "
            "%d validators (%s)",
            window.window_number, len(consensus_scores),
            len(eligible_scores), len(disqualified),
            stats.passed, stats.summary(),
        )
        logger.info(f"Burn fraction: {BURN_FRACTION:.0%} to owner UID {OWNER_UID}")
        if winner_hk:
            winner_uid = hk_to_uid.get(winner_hk, -1)
            logger.info(
                "Winner: %s (UID %d, score=%.4f, weight=%.4f)",
                winner_hk[:8], winner_uid,
                consensus_scores.get(winner_hk, 0),
                1.0 - BURN_FRACTION,
            )
        else:
            logger.info("No winner — all miners disqualified or no scores")
        logger.info("=" * 60)
        for hk, score in sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True):
            uid = hk_to_uid.get(hk, -1)
            status = "DISQ" if hk in disqualified else "OK"
            marker = " <- WINNER" if hk == winner_hk else ""
            logger.info(
                "  Miner %d (%s): score=%.4f, status=%s%s",
                uid, hk[:8], score, status, marker,
            )

        self._save_eval_state()

    async def _aggregate_on_startup(self):
        """Run consensus aggregation once at startup before entering the main loop.

        Reads on-chain validator commitments to find the latest window with
        submissions, then reuses ``_run_consensus_aggregation`` and
        ``_set_winner_weights`` to compute the winner and set weights
        immediately — without waiting for the normal window lifecycle.

        This is a side-effect-free operation with respect to the main loop:
        ``_consensus_window`` is saved and restored so the normal per-window
        aggregation guard is not consumed.
        """
        logger.info("=" * 60)
        logger.info("aggregate_when_start enabled — running startup aggregation")
        logger.info("=" * 60)

        self._sync_metagraph(caller="aggregate_on_startup")

        chain_commitments = fetch_validator_consensus_commitments(
            self.subtensor, self.config.netuid, self.metagraph,
        )
        if not chain_commitments:
            logger.warning(
                "Startup aggregation: no consensus commitments on chain, skipping"
            )
            return

        window_counts = Counter(vc.window_number for vc in chain_commitments)
        target_window = max(window_counts.keys())
        logger.info(
            "Startup aggregation: found %d commitments across windows %s, "
            "targeting latest window %d (%d submissions)",
            len(chain_commitments), dict(window_counts),
            target_window, window_counts[target_window],
        )

        synthetic_window = EvaluationWindow(
            window_number=target_window,
            window_start=self._window_config.effective_anchor
            + target_window * self._window_config.window_length,
            block_offset=self._window_config.window_length - 1,
            phase=WindowPhase.AGGREGATION,
            blocks_into_phase=0,
            blocks_remaining_in_phase=1,
            publish_offset=self._window_config.publish_block,
            aggregate_offset=self._window_config.aggregate_block,
        )

        saved_consensus_window = self._consensus_window
        await self._run_consensus_aggregation(synthetic_window)
        aggregation_succeeded = (self._consensus_window == target_window)

        self._consensus_window = saved_consensus_window
        self._save_eval_state()

        if aggregation_succeeded:
            await self._set_winner_weights()
            current_block = self.subtensor.get_current_block()
            self.last_weight_block = current_block
            logger.info(
                "Startup aggregation complete: winner=%s, weights set at block %d",
                self._winner_state.winner_hotkey
                and self._winner_state.winner_hotkey[:8],
                current_block,
            )
        else:
            logger.warning(
                "Startup aggregation: consensus aggregation did not produce "
                "a result for window %d, skipping weight setting",
                target_window,
            )

    async def _full_cycle_on_startup(self):
        """Run a complete eval → propagation → aggregation cycle at startup.

        Executes the three window phases sequentially regardless of the
        current on-chain window phase, then sets winner weights.  The main
        loop guards (``_last_eval_window``, ``_consensus_window``) are
        updated so the loop will not re-run these phases for the same window.
        """
        logger.info("=" * 60)
        logger.info("full_cycle_on_startup enabled — running full cycle")
        logger.info("=" * 60)

        current_block = self.subtensor.get_current_block()
        window = compute_window(current_block, self._window_config)

        # --- Phase 1: Evaluation ---
        logger.info(
            "Startup full cycle [1/3]: evaluation "
            "(window=%d, block=%d)",
            window.window_number, current_block,
        )
        await self._run_evaluation_cycle(current_block, window.window_number)
        self._last_eval_at = int(time.time())
        self._last_eval_window = window.window_number
        self._last_eval_date = datetime.datetime.utcnow().date()
        self.last_weight_block = self.subtensor.get_current_block()
        self._save_eval_state()

        # --- Phase 2: Propagation ---
        logger.info(
            "Startup full cycle [2/3]: propagation (window=%d)",
            window.window_number,
        )
        ok = await self._submit_consensus_payload(window)
        if ok:
            logger.info(
                "Startup full cycle: consensus payload submitted successfully"
            )
        else:
            logger.warning(
                "Startup full cycle: consensus payload submission failed, "
                "aggregation will proceed with available data"
            )

        # --- Phase 3: Aggregation ---
        logger.info(
            "Startup full cycle [3/3]: aggregation (window=%d)",
            window.window_number,
        )
        await self._run_consensus_aggregation(window)

        if self._consensus_window == window.window_number:
            await self._set_winner_weights()
            current_block = self.subtensor.get_current_block()
            self.last_weight_block = current_block
            logger.info(
                "Startup full cycle complete: winner=%s, weights set at block %d",
                self._winner_state.winner_hotkey
                and self._winner_state.winner_hotkey[:8],
                current_block,
            )
        else:
            logger.warning(
                "Startup full cycle: aggregation did not produce a result "
                "for window %d",
                window.window_number,
            )

    def _should_start_evaluation(self, current_block: int) -> bool:
        """Return True if a new evaluation cycle should start.

        Block-based: fires once per window during the evaluation phase.
        """
        window = compute_window(current_block, self._window_config)
        if window.phase != WindowPhase.EVALUATION:
            return False
        return window.window_number > self._last_eval_window

    async def run(self):
        """Main validator loop with block-based window phases.

        Window phases (per eval_interval_blocks window):
          - evaluation (0% - 80%):   run trajrl-bench, compute scores
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
            f"shift={self._window_config.window_shift}, "
            f"publish={self._window_config.publish_pct:.0%}, "
            f"aggregate={self._window_config.aggregate_pct:.0%}"
        )
        logger.info(
            f"Weight interval: {self.config.weight_interval_blocks} blocks "
            f"(~{self.config.weight_interval_blocks * 12 // 60}min)"
        )

        if self.config.full_cycle_on_startup:
            try:
                await self._full_cycle_on_startup()
            except Exception as e:
                logger.error(
                    "Startup full cycle failed: %s — continuing to main loop",
                    e, exc_info=True,
                )
        elif self.config.aggregate_when_start:
            try:
                await self._aggregate_on_startup()
            except Exception as e:
                logger.error(
                    "Startup aggregation failed: %s — continuing to main loop",
                    e, exc_info=True,
                )

        # --- Replay pending uploads from recent days ---
        try:
            await self._replay_pending_uploads()
        except Exception as e:
            logger.warning("Startup pending upload replay failed: %s", e, exc_info=True)

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
                        await self._set_fallback_weights()
                        self.last_weight_block = current_block
                    await asyncio.sleep(300)
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

                    await self._run_evaluation_cycle(current_block, window.window_number)
                    self._last_eval_at = int(time.time())
                    self._last_eval_window = window.window_number
                    self._last_eval_date = datetime.datetime.utcnow().date()
                    self.last_weight_block = self.subtensor.get_current_block()

                    self.pack_fetcher.cleanup_cache(
                        self.config.pack_cache_max_size
                    )
                    self._save_eval_state()

                    if self._cycle_eval_id is not None:
                        asyncio.ensure_future(
                            self._fire_upload_cycle_logs(
                                self._cycle_eval_id,
                                self._cycle_log_offset,
                                self._cycle_log_block,
                            )
                        )

                # --- Propagation window: submit results (idempotent — checks on-chain) ---
                if (window.phase == WindowPhase.PROPAGATION
                        and not self._check_own_commitment_on_chain(window.window_number)):
                    logger.info(
                        "Window %d: submitting evaluation results at block %d",
                        window.window_number, current_block,
                    )
                    ok = await self._submit_consensus_payload(window)
                    if not ok:
                        logger.warning(
                            "Window %d: submission attempt failed, "
                            "will retry next loop iteration",
                            window.window_number,
                        )

                    if self._cycle_eval_id is not None:
                        asyncio.ensure_future(
                            self._fire_upload_cycle_logs(
                                self._cycle_eval_id,
                                self._cycle_log_offset,
                                self._cycle_log_block,
                            )
                        )

                # --- Window phase: aggregation (idempotent — checks _consensus_window) ---
                if (window.phase == WindowPhase.AGGREGATION
                        and self._consensus_window != window.window_number):
                    logger.info(
                        f"Window {window.window_number}: T_aggregate reached "
                        f"at block {current_block}, running consensus aggregation"
                    )
                    await self._run_consensus_aggregation(window)

                    if self._consensus_window == window.window_number:
                        await self._set_winner_weights()
                        self.last_weight_block = current_block

                    if self._cycle_eval_id is not None:
                        asyncio.ensure_future(
                            self._fire_upload_cycle_logs(
                                self._cycle_eval_id,
                                self._cycle_log_offset,
                                self._cycle_log_block,
                            )
                        )
                        self._cycle_eval_id = None

                # --- Tempo cadence: re-assert winner weights ---
                current_block = self.subtensor.get_current_block()
                blocks_since_weights = current_block - self.last_weight_block
                if blocks_since_weights >= self.config.weight_interval_blocks:
                    logger.info(
                        f"Tempo weight refresh at block {current_block} "
                        f"({blocks_since_weights} blocks since last, "
                        f"window={window.window_number} phase={window.phase.value})"
                    )
                    await self._set_winner_weights()
                    self.last_weight_block = current_block

                await asyncio.sleep(300)

            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                self._save_eval_state()
                if self._cycle_eval_id is not None:
                    await self._fire_upload_cycle_logs(
                        self._cycle_eval_id,
                        self._cycle_log_offset,
                        self._cycle_log_block,
                    )
                    self._cycle_eval_id = None
                break
            except Exception as e:
                logger.error(f"Error in main loop: %s", e, exc_info=True)
                await asyncio.sleep(60)

    
    # ------------------------------------------------------------------
    # LLM key check
    # ------------------------------------------------------------------

    def _check_llm_keys(self) -> bool:
        """Return True if an LLM API key is configured."""
        return bool(self.config.llm_api_key)

    async def _prefetch_packs_and_ncd_gate(
        self,
        active_commitments: Dict[int, MinerCommitment],
        skip_uids: set,
    ) -> Dict[str, str]:
        """Fetch packs for eligible miners, run NCD dedup, cache packs for eval.

        Miners marked as copiers (values map to originals) are rejected in the
        main loop like integrity failures. Non-copiers with a successful fetch
        get _hotkey_packs / _pack_by_hash populated so _evaluate_miner hits
        verify_submission cache.

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

    async def _run_evaluation_cycle(
        self, current_block: int, window_number: int,
    ):
        """Run one evaluation cycle.

        The log offset is saved to instance state so the main loop can
        upload after each window phase (evaluation, submission,
        aggregation), progressively capturing more context.  The server
        is expected to handle duplicate/overlapping uploads.
        """
        cycle_start = time.time()
        cycle_eval_id = time.strftime("%Y%m%d_%H%M") + f"_w{window_number}"

        self._cycle_eval_id = cycle_eval_id
        self._cycle_log_offset = self._get_validator_log_offset()
        self._cycle_log_block = current_block

        await self._execute_evaluation_cycle(
            current_block, cycle_eval_id, cycle_start,
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
                "LLM_API_KEY not set. "
                "Skipping evaluation, setting fallback weights to owner UID.",
            )
            await self._set_fallback_weights(
                reason="No LLM API key configured"
            )
            return

        # Re-assert the current winner's weights before starting this eval
        # cycle, so the validator is active on-chain from the start.
        logger.info("Setting winner weights before starting eval cycle")
        await self._set_winner_weights()
        self.last_weight_block = current_block

        self._disqualified_miners = {}

        # Pull latest sandbox image before eval (gets new scenarios + version)
        try:
            await self._sandbox_harness.pull_latest()
        except docker.errors.DockerException as e:
            logger.error(
                "Docker is not available — cannot run sandbox evaluations. "
                "If running inside Docker, ensure docker.sock is mounted. "
                "Run:\n"
                "  docker compose -f docker/docker-compose.validator.yml "
                "--env-file .env.validator up -d\n"
                "Skipping this eval cycle. Error: %s", e,
            )
            return
        # scoring_version = trajrl-bench major version (v3.0.1 → 3)
        import trajectoryrl.utils.consensus as _consensus
        _consensus_mod.SCORING_VERSION = self._sandbox_harness.scoring_version
        logger.info("scoring_version=%d (bench_version=%s)",
                    self._sandbox_harness.scoring_version,
                    self._sandbox_harness.sandbox_version)

        # Epoch seed for context variation
        epoch = current_block // self.config.eval_interval_blocks
        epoch_seed = self.compute_epoch_seed(epoch, self.config.netuid)
        logger.debug(
            f"Eval cycle: block={current_block}, seed={epoch_seed}"
        )

        # 1. Clear per-cycle pack caches to prevent stale entries from
        # deregistered miners affecting NCD comparisons in the weight phase.
        self._hotkey_packs.clear()
        self._pack_by_hash.clear()

        # 2. Sync metagraph (with retries + reconnect)
        logger.debug("Syncing metagraph...")
        mg_healthy = self._sync_metagraph(caller="eval_cycle")
        logger.info(
            "Metagraph synced: n=%d neurons, block=%s, healthy=%s",
            getattr(self.metagraph, "n", 0),
            getattr(self.metagraph, "block", "?"),
            mg_healthy,
        )
        if not mg_healthy:
            logger.error(
                "Metagraph has 0 neurons — cannot evaluate or set weights. "
                "Skipping eval cycle. This usually means the subtensor "
                "endpoint is unreachable or returning stale data."
            )
            return

        # 3. Read on-chain commitments
        logger.debug("Reading on-chain commitments...")
        commitments = fetch_all_commitments(
            self.subtensor, self.config.netuid, self.metagraph
        )

        # 4. Filter to active non-validator miners (drops stale submissions)
        active_commitments = self._filter_active_commitments(
            commitments, current_block,
        )
        logger.info(
            f"Commitments: {len(commitments)} total, "
            f"{len(active_commitments)} active miners"
        )

        if not active_commitments:
            logger.warning("No active miners with valid commitments!")
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

        # 6. Evaluate miners (scenarios selected by sandbox image)
        eval_scenarios: List[str] = []  # not used in S1 — sandbox picks scenario
        evaluated_count = 0
        attempted_count = 0
        skipped_interval_count = 0
        rejected_pre_eval_count = 0
        ncd_rejected_count = 0
        total_eligible = len(active_commitments) - len(skip_uids)
        logger.info(
            f"=== Eval cycle: {total_eligible} eligible miners ==="
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
                        weight=0.0,
                        qualified=False,
                        pack_url=commitment.pack_url,
                        pack_hash=commitment.pack_hash,
                        llm_base_url=self._judge_base_url,
                        llm_model=self._judge_model,
                        rejected=True,
                        rejection_stage="integrity_check",
                        rejection_detail=detail,
                        scoring_version=_scoring_version(),
                        **self._harness_metadata(),
                    )
                )
                self.scenario_scores.pop(hotkey, None)
                self._eval_pack_hash.pop(hotkey, None)
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
                            weight=0.0,
                            qualified=False,
                            pack_url=commitment.pack_url,
                            pack_hash=commitment.pack_hash,
                            llm_base_url=self._judge_base_url,
                            llm_model=self._judge_model,
                            rejected=True,
                            rejection_stage=_stage,
                            rejection_detail=_detail,
                            scoring_version=_scoring_version(),
                            **self._harness_metadata(),
                        )
                    )
                    self._disqualified_miners[hotkey] = f"pre_eval_rejected:{reason}"
                    self.scenario_scores.pop(hotkey, None)
                    self._eval_pack_hash.pop(hotkey, None)
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

            attempted_count += 1
            logger.info(
                f"[{miner_idx}/{total_eligible}] Evaluating miner {uid} "
                f"({hotkey[:8]}) ..."
            )
            eval_dir, vlog_offset, mlog_offset = self._prepare_eval_log_capture(cycle_eval_id, hotkey)
            eval_start = time.time()
            eval_result = await self._evaluate_miner(
                uid, commitment, epoch_seed,
                block_height=current_block,
            )
            eval_elapsed = time.time() - eval_start

            # Upload eval logs to dashboard (fire-and-forget)
            asyncio.ensure_future(
                self._fire_upload_eval_logs(
                    cycle_eval_id, uid, commitment, eval_scenarios, eval_result,
                    eval_dir, vlog_offset, mlog_offset, current_block,
                )
            )

            if eval_result is not None:
                pack_changed = self._eval_pack_hash.get(hotkey) != commitment.pack_hash
                if pack_changed:
                    self._eval_counts[hotkey] = 0
                self._eval_counts[hotkey] = self._eval_counts.get(hotkey, 0) + 1
                eval_count = self._eval_counts[hotkey]

                raw_q = eval_result.get("qualified") or {}
                raw_jd = eval_result.get("judge_details") or {}
                scenario_scores_map = {}
                for sname, qv in raw_q.items():
                    jd = raw_jd.get(sname) or {}
                    overall = jd.get("overall_score")
                    if overall is not None:
                        scenario_scores_map[sname] = float(overall)
                    else:
                        logger.warning(
                            "Miner %d scenario %s: missing overall_score "
                            "in judge_details, defaulting to 0.0",
                            uid, sname,
                        )
                        scenario_scores_map[sname] = 0.0

                self._update_eval_results(
                    hotkey, commitment.pack_hash,
                    scenario_scores=scenario_scores_map,
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
                        uid, commitment, eval_result, eval_count, pack_changed, current_block
                    )
                )

                self._track_uid_change(uid, hotkey)

            else:
                elapsed_str = f" ({eval_elapsed:.1f}s)" if eval_elapsed else ""
                logger.info(
                    f"[{miner_idx}/{total_eligible}] Miner {uid} eval returned None"
                    f"{elapsed_str} "
                    f"(pack fetch/integrity failed)"
                )

            # Mid-eval tempo refresh: re-assert the current winner's weights
            # to keep the validator active on-chain without recomputing.
            mid_block = self.subtensor.get_current_block()
            if mid_block - self.last_weight_block >= self.config.weight_interval_blocks:
                logger.info(
                    f"Mid-eval tempo refresh at block {mid_block} "
                    f"({mid_block - self.last_weight_block} blocks since last set_weights)"
                )
                await self._set_winner_weights()
                self.last_weight_block = mid_block

        parts = [f"{evaluated_count} evaluated"]
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

        # All attempted evaluations failed — likely a validator-side
        # issue.  Mark unhealthy so the next cycle ignores cached results.
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

        # 5. Re-assert winner weights after eval completes
        await self._set_winner_weights()

    def _needs_evaluation(
        self, hotkey: str, pack_hash: str, current_block: int
    ) -> bool:
        """Check if a miner needs re-evaluation.

        Returns True if:
        - pack_hash changed since last eval
        - Time since last eval >= eval_interval
        - Never evaluated before
        """
        if self._eval_pack_hash.get(hotkey) != pack_hash:
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
        current_block: int,
    ) -> Dict[int, MinerCommitment]:
        """Filter commitments to active, non-validator miners.

        Excludes:
        - UIDs with validator_permit
        - Blacklisted coldkeys
        - Stale submissions (commitment older than ``inactivity_blocks``)
        """
        blacklist = set(self.config.coldkey_blacklist)
        max_age = self.config.inactivity_blocks
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
            age = current_block - commitment.block_number
            if age > max_age:
                self._get_miner_logger(commitment.hotkey).info(
                    f"Stale submission: {age} blocks old > "
                    f"{max_age} block limit (~48h), skipping"
                )
                continue
            active[uid] = commitment
        return active

    # ------------------------------------------------------------------
    # UID re-registration tracking
    # ------------------------------------------------------------------

    def _track_uid_change(
        self,
        miner_uid: int,
        hotkey: str,
    ) -> None:
        """Detect re-registration for a miner hotkey."""
        prev_uid = self._hotkey_uid_map.get(hotkey)
        if prev_uid is not None and prev_uid != miner_uid:
            self._get_miner_logger(hotkey).info(
                f"Previously at UID {prev_uid}; "
                f"re-registration detected"
            )
        self._hotkey_uid_map[hotkey] = miner_uid

    # ------------------------------------------------------------------
    # Episode detail logging (file-only, no console output)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Miner evaluation
    # ------------------------------------------------------------------

    async def _evaluate_miner(
        self,
        miner_uid: int,
        commitment: MinerCommitment,
        epoch_seed: int,
        block_height: int = 0,
    ) -> Optional[Dict]:
        """Evaluate a single miner via trajrl-bench sandbox.

        Flow:
        1. Fetch + verify submission (usually cache hit after cycle NCD prefetch)
        2. Run trajrl-bench sandbox evaluation (SKILL.md + SSH + LLM judge)
        3. Return quality-based scores

        Returns:
            Dict with keys "qualified", "judge_details" mapping
            scenario_name to values, or None if pre-evaluation checks fail.
        """
        mlog = self._get_miner_logger(commitment.hotkey)
        mlog.info(
            f"Evaluating miner {miner_uid} "
            f"(hotkey={commitment.hotkey[:8]}, hash={commitment.pack_hash[:12]}...)"
        )

        # Step 1: Fetch and verify submission from HTTP URL
        verification = await self.pack_fetcher.verify_submission(
            pack_url=commitment.pack_url,
            pack_hash=commitment.pack_hash,
        )

        if not verification.valid:
            mlog.warning(
                f"Pack verification failed: {verification.error}"
            )
            return None

        return await self._evaluate_miner_s1(
            miner_uid=miner_uid,
            commitment=commitment,
            pack=verification.pack_content,
            epoch_seed=epoch_seed,
            block_height=block_height,
        )

    async def _evaluate_miner_s1(
        self,
        miner_uid: int,
        commitment: "MinerCommitment",
        pack: dict,
        epoch_seed: int,
        block_height: int = 0,
    ) -> Optional[Dict]:
        """Season 1 evaluation path: trajrl-bench with LLM judge.

        Runs N episodes of the same scenario in an SSH sandbox with
        Hermes Agent. Uses split-half delta scoring (quality-based,
        not cost-based).

        Returns same shape as _evaluate_miner for pipeline compatibility:
            {"qualified": {...}, "judge_details": {...}, ...}
        """
        mlog = self._get_miner_logger(commitment.hotkey)

        files = pack.get("files", {})
        if "SKILL.md" not in files:
            mlog.warning("Pack missing SKILL.md in files")
            self._disqualified_miners[commitment.hotkey] = "missing_skill_md"
            return None

        extra_files = [f for f in files if f != "SKILL.md"]
        if extra_files:
            mlog.warning("S1 pack contains unexpected files: %s", extra_files)

        skill_md = files["SKILL.md"]
        if not skill_md or not skill_md.strip():
            mlog.warning("Empty SKILL.md submission")
            self._disqualified_miners[commitment.hotkey] = "empty_skill_md"
            return None

        mlog.info(
            "S1 evaluation: %d episodes, skill_md=%d chars",
            self.config.sandbox_num_episodes, len(skill_md),
        )

        try:
            result = await self._sandbox_harness.evaluate_miner(
                skill_md=skill_md,
                epoch_seed=epoch_seed,
                pack_hash=commitment.pack_hash,
                validator_salt=self._default_validator_salt(),
            )
        except Exception as e:
            mlog.error("S1 evaluation failed: %s", e, exc_info=True)
            self._disqualified_miners[commitment.hotkey] = "s1_eval_error"
            return None

        if result.error:
            mlog.warning("S1 evaluation error: %s", result.error)
            self._disqualified_miners[commitment.hotkey] = "s1_eval_error"
            return None

        # Map S1 result to validator pipeline format.
        # S1 uses a single "scenario" (incident_response) with quality scoring.
        scenario_name = result.scenario_name
        qualified = result.success

        mlog.info(
            "S1 result: final_score=%.3f, mean_quality=%.3f, delta=%.3f, "
            "episodes=%s, qualified=%s",
            result.score, result.mean_quality, result.delta,
            result.episode_qualities, qualified,
        )

        scenario_qualified = {scenario_name: qualified}
        scenario_judge_details = {
            scenario_name: {
                "overall_score": round(result.score, 4),
                "mean_quality": round(result.mean_quality, 4),
                "delta": round(result.delta, 4),
                "early_mean": round(result.early_mean, 4),
                "late_mean": round(result.late_mean, 4),
                "episode_qualities": [round(q, 4) for q in result.episode_qualities],
                "qualification_gate": qualified,
                "harness": "trajrl-bench",
                "sandbox_version": self._sandbox_harness.sandbox_version,
            },
        }

        return {
            "qualified": scenario_qualified,
            "token_usage": {},
            "model_usage": {},
            "judge_details": scenario_judge_details,
            "session_keys": {},
            "session_files": {},
            # S1-specific: full SandboxEvaluationResult for eval log upload.
            # Not serialized — consumed by _fire_upload_eval_logs and dropped.
            "_s1_sandbox_result": result,
        }

    def _default_validator_salt(self) -> str:
        """Derive a stable validator salt from the wallet hotkey."""
        import hashlib
        data = f"{self.wallet.hotkey.ss58_address}:{self.config.netuid}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Eval submission
    # ------------------------------------------------------------------

    def _harness_metadata(self) -> Dict[str, str]:
        """Return bench/harness image hashes and bench version for submit payloads."""
        h = self._sandbox_harness
        meta: Dict[str, str] = {}
        if h.bench_image_hash != "unknown":
            meta["bench_image_hash"] = h.bench_image_hash
        if h.harness_image_hash != "unknown":
            meta["harness_image_hash"] = h.harness_image_hash
        if h.sandbox_version != "unknown":
            meta["bench_version"] = h.sandbox_version
        return meta

    async def _fire_submit_eval(
        self,
        uid: int,
        commitment: "MinerCommitment",
        eval_result: Dict,
        eval_count: int,
        pack_changed: bool,
        block_height: int,
    ) -> None:
        """Build and fire the /api/scores/submit payload for one miner eval.

        Fire-and-forget: any error is logged and discarded.
        """
        hotkey = commitment.hotkey

        raw_qualified = eval_result.get("qualified") or {}
        raw_judge_details = eval_result.get("judge_details") or {}

        def _scenario_score(sname: str, qualified_val: bool) -> float:
            jd = raw_judge_details.get(sname) or {}
            overall = jd.get("overall_score")
            if overall is not None:
                return float(overall)
            logger.warning(
                "Scenario %s: missing overall_score in judge_details, "
                "defaulting to 0.0", sname,
            )
            return 0.0

        # Aggregate raw score (mean across scenarios)
        raw_score = (
            sum(_scenario_score(s, q) for s, q in raw_qualified.items())
            / len(raw_qualified)
            if raw_qualified else 0.0
        )

        # Per-scenario results
        scenario_results: Dict[str, Any] = {}
        for sname, q in raw_qualified.items():
            entry: Dict[str, Any] = {
                "score": round(_scenario_score(sname, q), 4),
                "weight": 1.0,
                "qualified": q,
            }
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
        mlog.info(
            f"=== eval summary (uid={uid}, eval#{eval_count}) ==="
        )
        mlog.info(
            f"  score={raw_score:.4f} pack_changed={pack_changed}"
        )
        mlog.info(
            f"  pack_url={commitment.pack_url} "
            f"pack_hash={commitment.pack_hash}"
        )
        for sname, sr in scenario_results.items():
            jd = sr.get("judge", {})
            mlog.info(
                f"  {sname}: qualified={sr.get('qualified')} "
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
            weight=0.0,
            qualified=bool(raw_qualified and all(raw_qualified.values())),
            pack_url=commitment.pack_url,
            pack_hash=commitment.pack_hash,
            eval_count=eval_count,
            scenario_results=scenario_results,
            llm_base_url=self._judge_base_url,
            llm_model=self._judge_model,
            scoring_version=_scoring_version(),
            **self._harness_metadata(),
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
            eval_id: Cycle-level eval identifier (e.g. "20260329_1430_w42").
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
        session_files: Optional[Dict[str, str]] = None,
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
            session_files: Mapping of scenario name to actual session filename
                (UUID.jsonl) in the OpenClaw sessions directory.

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
            # OpenClaw creates files with UUID names, not the logical session key.
            # Use session_files (actual filenames) when available, fall back to
            # session_keys for backwards compatibility.
            openclaw_sessions_dir = Path("/root/.openclaw/agents/main/sessions")
            _session_files = session_files or {}
            for scenario in eval_scenarios:
                src = None
                # Prefer the actual session file detected by run_episode.py
                fname = _session_files.get(scenario)
                if fname:
                    candidate = openclaw_sessions_dir / fname
                    if candidate.exists() and candidate.stat().st_size > 0:
                        src = candidate
                # Fall back to session_key lookup (legacy, usually won't match)
                if src is None:
                    skey = session_keys.get(scenario)
                    if skey:
                        candidate = openclaw_sessions_dir / f"{skey}.jsonl"
                        if candidate.exists() and candidate.stat().st_size > 0:
                            src = candidate
                if src:
                    dst = eval_dir / f"{scenario}_conversation.jsonl"
                    shutil.copy2(str(src), str(dst))
                    logger.info(f"Copied session transcript for {scenario}: {src.name}")

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
        session_files = eval_result.get("session_files", {}) if eval_result else {}

        # If this was an S1 eval, write the sandbox artifacts (transcripts,
        # evaluation.json, JUDGE.md, fixtures) into eval_dir so they're
        # included in the tar.gz uploaded to the dashboard.
        s1_result = eval_result.get("_s1_sandbox_result") if eval_result else None
        if s1_result is not None:
            try:
                s1_result.write_artifacts(eval_dir)
            except Exception as e:
                logger.warning("Failed to write S1 eval artifacts: %s", e)

        log_archive = self._collect_eval_logs(
            commitment.hotkey, eval_scenarios, eval_dir,
            validator_log_offset, miner_log_offset, session_keys,
            session_files,
        )
        if log_archive:
            meta = {
                "eval_id": eval_id,
                "window_number": int(eval_id.rsplit("_w", 1)[-1]) if "_w" in eval_id else None,
                "miner_hotkey": commitment.hotkey,
                "miner_uid": uid,
                "block_height": block_height,
                "pack_hash": commitment.pack_hash,
                "type": "eval",
            }
            try:
                (eval_dir / "upload_meta.json").write_text(
                    json.dumps(meta), encoding="utf-8",
                )
                (eval_dir / "logs.tar.gz").write_bytes(log_archive)
            except OSError as e:
                logger.warning("Failed to persist eval upload metadata: %s", e)

            ok = await upload_eval_logs(
                self.wallet,
                eval_id=eval_id,
                miner_hotkey=commitment.hotkey,
                miner_uid=uid,
                block_height=block_height,
                pack_hash=commitment.pack_hash,
                log_archive=log_archive,
            )
            if ok:
                try:
                    (eval_dir / ".uploaded").touch()
                except OSError:
                    pass

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

            meta = {
                "eval_id": eval_id,
                "window_number": int(eval_id.rsplit("_w", 1)[-1]) if "_w" in eval_id else None,
                "block_height": block_height,
                "type": "cycle",
            }
            try:
                (cycle_dir / "cycle_upload_meta.json").write_text(
                    json.dumps(meta), encoding="utf-8",
                )
                (cycle_dir / "cycle.tar.gz").write_bytes(archive)
            except OSError as e:
                logger.warning("Failed to persist cycle upload metadata: %s", e)

            ok = await upload_cycle_logs(
                self.wallet,
                eval_id=eval_id,
                block_height=block_height,
                log_archive=archive,
            )
            if ok:
                try:
                    (cycle_dir / ".cycle_uploaded").touch()
                except OSError:
                    pass
        except Exception as e:
            logger.warning("Cycle log upload error: %s", e)

    # ------------------------------------------------------------------
    # Log replay (startup)
    # ------------------------------------------------------------------

    async def _replay_pending_uploads(self):
        """Re-upload logs from the last 2 days that have metadata but no .uploaded marker."""
        evals_root = self.config.log_dir / "evals"
        if not evals_root.exists():
            return

        today = datetime.datetime.utcnow().date()
        recent_prefixes = {
            (today - datetime.timedelta(days=d)).strftime("%Y%m%d")
            for d in range(2)
        }

        eval_uploaded = 0
        cycle_uploaded = 0

        for eval_id_dir in sorted(evals_root.iterdir()):
            if not eval_id_dir.is_dir():
                continue
            if not any(eval_id_dir.name.startswith(p) for p in recent_prefixes):
                continue

            eval_id = eval_id_dir.name

            # --- Per-miner eval logs with persisted metadata ---
            for miner_dir in sorted(eval_id_dir.iterdir()):
                if not miner_dir.is_dir():
                    continue
                if (miner_dir / ".uploaded").exists():
                    continue
                meta_path = miner_dir / "upload_meta.json"
                archive_path = miner_dir / "logs.tar.gz"
                if not meta_path.exists() or not archive_path.exists():
                    continue

                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue

                ok = await upload_eval_logs(
                    self.wallet,
                    eval_id=meta.get("eval_id", eval_id),
                    miner_hotkey=meta["miner_hotkey"],
                    miner_uid=meta["miner_uid"],
                    block_height=meta.get("block_height", 0),
                    pack_hash=meta.get("pack_hash", "unknown"),
                    log_archive=archive_path.read_bytes(),
                )
                if ok:
                    eval_uploaded += 1
                    try:
                        (miner_dir / ".uploaded").touch()
                    except OSError:
                        pass

            # --- Cycle logs with persisted metadata ---
            if (eval_id_dir / ".cycle_uploaded").exists():
                continue
            cycle_meta_path = eval_id_dir / "cycle_upload_meta.json"
            cycle_archive_path = eval_id_dir / "cycle.tar.gz"
            if not cycle_meta_path.exists() or not cycle_archive_path.exists():
                continue

            try:
                meta = json.loads(cycle_meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            ok = await upload_cycle_logs(
                self.wallet,
                eval_id=meta.get("eval_id", eval_id),
                block_height=meta.get("block_height", 0),
                log_archive=cycle_archive_path.read_bytes(),
            )
            if ok:
                cycle_uploaded += 1
                try:
                    (eval_id_dir / ".cycle_uploaded").touch()
                except OSError:
                    pass

        if eval_uploaded or cycle_uploaded:
            logger.info(
                "Startup log replay: re-uploaded %d eval + %d cycle logs",
                eval_uploaded, cycle_uploaded,
            )

    # ------------------------------------------------------------------
    # Weight setting
    # ------------------------------------------------------------------

    async def _do_set_weights(
        self,
        uids: list,
        weights: list,
        *,
        label: str = "",
    ) -> bool:
        """Call subtensor.set_weights with retry on failure.

        Returns True if weights were set successfully.
        """
        for attempt in range(1, _SET_WEIGHTS_MAX_RETRIES + 1):
            try:
                result = self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=uids,
                    weights=weights,
                    wait_for_inclusion=True,
                    wait_for_finalization=False,
                )
                success = getattr(result, "success", None)
                message = getattr(result, "message", str(result))
                if success is False:
                    logger.warning(
                        "%sset_weights attempt %d/%d FAILED — "
                        "success=%s, message=%s, uids=%s",
                        label, attempt, _SET_WEIGHTS_MAX_RETRIES,
                        success, message, uids,
                    )
                else:
                    logger.info(
                        "%sset_weights OK — uids=%s, success=%s, message=%s",
                        label, uids, success, message,
                    )
                    self._last_set_weights_at = int(time.time())
                    return True
            except Exception as e:
                logger.warning(
                    "%sset_weights attempt %d/%d exception: %s",
                    label, attempt, _SET_WEIGHTS_MAX_RETRIES, e,
                )

            if attempt < _SET_WEIGHTS_MAX_RETRIES:
                delay = _SET_WEIGHTS_RETRY_DELAY * attempt
                logger.info(
                    "%sRetrying set_weights in %ds...", label, delay,
                )
                await asyncio.sleep(delay)

        logger.error(
            "%sset_weights FAILED after %d attempts (uids=%s)",
            label, _SET_WEIGHTS_MAX_RETRIES, uids,
        )
        return False

    async def _set_winner_weights(self):
        """Set on-chain weights from persisted WinnerState.

        Reads winner_uid directly from the persisted state — no metagraph
        lookup required.  The UID is captured at consensus aggregation time
        when the metagraph is known to be healthy.
        """
        winner_hk = self._winner_state.winner_hotkey
        winner_uid = self._winner_state.winner_uid

        if not winner_hk or winner_uid is None:
            reason = (
                "No winner in WinnerState"
                if not winner_hk
                else f"Winner {winner_hk[:8]} has no UID in state"
            )
            await self._set_fallback_weights(reason=reason)
            return

        if winner_uid == OWNER_UID:
            uids = [OWNER_UID]
            weights = [1.0]
        else:
            uids = [winner_uid, OWNER_UID]
            weights = [1.0 - BURN_FRACTION, BURN_FRACTION]

        await self._do_set_weights(
            uids, weights,
            label=f"[winner={winner_hk[:8]} UID {winner_uid}] ",
        )

    def _fallback_owner_weights(self) -> Optional[Tuple[list, list]]:
        """Read on-chain weights set by OWNER_UID and return (uids, weights).

        Returns None if the owner has no weights or the read fails.
        """
        if not self._is_metagraph_healthy():
            logger.warning(
                "Cannot read owner weights: metagraph unhealthy (n=%d). "
                "Attempting re-sync...",
                getattr(self.metagraph, "n", 0),
            )
            if not self._sync_metagraph(caller="fallback_owner_weights"):
                logger.error(
                    "Metagraph still unhealthy after retry — cannot read "
                    "owner weight row. Falling back to owner-only weight."
                )
                return None

        try:
            W = self.metagraph.W  # (n, n) weight matrix
            if OWNER_UID >= W.shape[0]:
                logger.warning(
                    "OWNER_UID %d out of range (metagraph size %d)",
                    OWNER_UID, W.shape[0],
                )
                return None

            owner_weights = W[OWNER_UID]
            uids = []
            weights = []
            for uid, w in enumerate(owner_weights.tolist()):
                if w > 0:
                    uids.append(uid)
                    weights.append(float(w))

            if not uids:
                logger.warning(
                    "Owner UID %d has no non-zero weights on chain",
                    OWNER_UID,
                )
                return None

            logger.info(
                "Copied %d weight entries from owner UID %d: uids=%s",
                len(uids), OWNER_UID, uids,
            )
            return uids, weights
        except Exception as e:
            logger.warning("Failed to read owner weights from chain: %s", e)
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

        if not self._is_metagraph_healthy():
            logger.error(
                "METAGRAPH UNHEALTHY (n=%d) during fallback weight setting. "
                "Attempting reconnect before set_weights. Reason: %s",
                getattr(self.metagraph, "n", 0), reason,
            )
            self._sync_metagraph(caller="fallback_weights")

        copied = self._fallback_owner_weights()
        if copied is not None:
            uids, weights = copied
            logger.info(
                "%s — copying weights from owner UID %d "
                "(%d entries, uids=%s)",
                reason, OWNER_UID, len(uids), uids,
            )
        else:
            uids, weights = self._fallback_to_owner()
            logger.info(
                "%s — setting fallback weight to "
                "owner UID %d (uids=%s, weights=%s)",
                reason, OWNER_UID, uids, weights,
            )

        await self._do_set_weights(
            uids, weights, label=f"[fallback: {reason}] ",
        )


async def main():
    """Entry point for validator."""
    config = ValidatorConfig.from_env()
    validator = TrajectoryValidator(config)
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
