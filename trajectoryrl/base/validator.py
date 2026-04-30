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

Each validator operates independently — Yuma Consensus aggregates on-chain.
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
from ..utils import config as _config_mod
from ..utils.consensus import (
    ConsensusPayload, ConsensusPointer,
    CONSENSUS_PROTOCOL_VERSION,
)


def _spec_number() -> int:
    """Current scoring spec identifier (validator-side constant).

    See ``trajectoryrl/utils/config.py::SPEC_NUMBER`` for bump policy.
    Used as the value written into outgoing commitments / payloads and as
    the fallback target when no on-chain spec_number group reaches
    stake-weighted majority.
    """
    return _config_mod.SPEC_NUMBER


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
from ..utils.pack_ownership import (
    claim_owner, evict_orphans,
    save_pack_first_seen, load_pack_first_seen,
    EVICTION_GRACE_WINDOWS,
    _decode_table as _decode_pack_first_seen_table,
)
from ..utils.miner_eval import (
    evaluate_miner_s1,
    SKIP_PACK_VERIFY,
)
from ..utils.commitments import (
    MinerCommitment, fetch_all_commitments,
    ValidatorConsensusCommitment, fetch_validator_consensus_commitments,
    format_consensus_commitment, decode_dual_address,
    is_consensus_commitment, parse_consensus_commitment,
)
from ..utils.eval_snapshot import (
    EvalSnapshot, take_snapshot, load_snapshot, save_snapshot,
)
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

        # Last eval block for each hotkey (telemetry only — eval gating is
        # by pack_hash, see _needs_evaluation).
        self.last_eval_block: Dict[str, int] = {}

        # Last eval window for each hotkey. Used by _needs_evaluation logging
        # and surfaces "which window did we last score this miner in" for
        # debugging restart / cross-epoch reuse behavior.
        self.last_eval_window: Dict[str, int] = {}

        # Per-validator ownership lock for exact pack_hash dedup.
        # pack_hash -> (first_observed_hotkey, first_observed_block).
        # Set once via setdefault, never updated. No succession: if the
        # original owner goes inactive, no other miner inherits — copies
        # always receive weight 0 until eviction (see end-of-cycle sweep
        # in _execute_evaluation_cycle).
        self.pack_first_seen: Dict[str, Tuple[str, int]] = {}

        # pack_hash -> last window number we observed it in any active
        # commitment. Updated by `evict_orphans` at the end of each
        # evaluation cycle. Drives the grace-windowed eviction policy:
        # a pack_first_seen entry is dropped only after its hash has
        # been absent for `EVICTION_GRACE_WINDOWS` consecutive
        # wall-clock windows (any re-activation resets the clock).
        self.pack_last_seen_window: Dict[str, int] = {}

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
        # Which window's aggregation has completed (persisted in eval state).
        # Guards one-shot aggregation per window — survives restarts.
        self._consensus_window: int = -1
        # Logical consensus anchor (Issue 1): the only window allowed to
        # progress eval/submit/aggregation state. Decoupled from physical
        # window derived from current block.
        self._target_window: int = -1
        # Submit idempotency marker for the target window.
        self._target_submit_done: bool = False
        # Sticky marker: quorum not yet met for current target window.
        self._waiting_for_quorum: bool = False

        # Cycle log state — uploaded after each window phase completes
        self._cycle_eval_id: Optional[str] = None
        self._cycle_log_offset: int = 0
        self._cycle_log_block: int = 0
        self._cycle_window_number: int = 0

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

        # Load persisted evaluation state. Eval state is invalidated when the
        # persisted spec_number disagrees with the current SPEC_NUMBER (any
        # bump in scenario set or scoring methodology renders cached scores
        # incomparable).
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

    def _load_eval_state(self):
        """Load persisted evaluation state from disk.

        Invalidates all state when the persisted ``spec_number`` no longer
        matches the running ``SPEC_NUMBER`` — any spec bump implies cached
        scores are incomparable with new ones. Accepts either ``spec_number``
        (current) or ``scoring_version`` (legacy) JSON keys.

        ``pack_first_seen`` is loaded from its own file
        (``self.config.pack_first_seen_path``); for backward compatibility
        with earlier internal layouts where the table lived inside
        ``eval_state.json``, the legacy key is migrated on first load and
        immediately persisted to the new path so subsequent restarts read
        from the dedicated file.
        """
        # pack_first_seen lives in its own file. Load it first so we can
        # detect whether legacy migration is needed regardless of the
        # eval_state.json outcome (spec_number mismatch must NOT wipe
        # ownership locks — those are spec-agnostic). The companion
        # last-seen-window dict drives the grace-period eviction logic;
        # v1 files yield an empty side dict and the clock starts on the
        # next eviction sweep.
        self.pack_first_seen, self.pack_last_seen_window = load_pack_first_seen(
            self.config.pack_first_seen_path
        )

        path = self.config.eval_state_path
        if not path.exists():
            logger.debug("No persisted eval state found, starting fresh")
            return
        try:
            data = json.loads(path.read_text())

            file_spec = data.get("spec_number", data.get("scoring_version", 1))
            if file_spec != _spec_number():
                logger.info(
                    "Eval state spec_number mismatch (%d != %d), "
                    "invalidating all eval state",
                    file_spec, _spec_number(),
                )
                self._migrate_legacy_pack_first_seen(data)
                return

            self.scenario_scores = data.get("scenario_scores", {})
            self._eval_pack_hash = data.get("eval_pack_hash", {})
            self.last_eval_block = {
                k: int(v) for k, v in data.get("last_eval_block", {}).items()
            }
            self.last_eval_window = {
                k: int(v) for k, v in data.get("last_eval_window_per_hotkey", {}).items()
            }
            self._last_eval_window = data.get("last_eval_window", -1)

            self._migrate_legacy_pack_first_seen(data)

            self._consensus_window = data.get("consensus_window", -1)
            self._target_window = int(
                data.get("target_window", self._consensus_window + 1)
            )
            self._target_submit_done = bool(data.get("target_submit_done", False))
            self._waiting_for_quorum = bool(data.get("waiting_for_quorum", False))

            # Heal legacy false-positive from pre-PR197 validator binaries:
            # the old aggregation-phase gate could latch _waiting_for_quorum
            # on a freshly-bumped target where the validator had not yet
            # submitted. That combination is unreachable after the fix, so
            # treat it as a corrupt carryover and clear it — otherwise the
            # main-loop tempo refresh would keep burning until the next
            # quorum success flips the flag back.
            if self._waiting_for_quorum and not self._target_submit_done:
                logger.warning(
                    "Eval state heal: clearing waiting_for_quorum on "
                    "target_window=%d with target_submit_done=False "
                    "(legacy pre-PR197 false-positive)",
                    self._target_window,
                )
                self._waiting_for_quorum = False
                self._save_eval_state()

            integrity_cache = data.get("integrity_cache")
            if integrity_cache:
                self.integrity_judge.load_cache(integrity_cache)

            logger.debug(
                f"Loaded eval state: {len(self.scenario_scores)} hotkeys"
            )
        except Exception as e:
            logger.warning(f"Failed to load eval state: {e}, starting fresh")

    def _migrate_legacy_pack_first_seen(self, eval_state_data: dict) -> None:
        """One-time migration of legacy ``pack_first_seen`` from eval_state.json.

        Earlier internal builds wrote the ownership table into the same
        file as eval state. New code reads from a dedicated path; if
        that file is empty but the legacy key carries entries, decode
        them and immediately rewrite to the new path. After this, the
        legacy key stops being written by ``_save_eval_state`` so
        future restarts see only the new file.
        """
        if self.pack_first_seen:
            return
        legacy = eval_state_data.get("pack_first_seen")
        if not legacy:
            return
        migrated = _decode_pack_first_seen_table(legacy)
        if not migrated:
            return
        self.pack_first_seen = migrated
        # Legacy files predate the grace-window tracker; start every
        # entry's clock on the next eviction sweep so we don't surprise
        # operators with mass evictions immediately after the upgrade.
        self.pack_last_seen_window = {}
        try:
            save_pack_first_seen(
                self.pack_first_seen,
                self.pack_last_seen_window,
                self.config.pack_first_seen_path,
            )
            logger.info(
                "Migrated %d pack_first_seen entries from eval_state.json "
                "to %s",
                len(self.pack_first_seen),
                self.config.pack_first_seen_path,
            )
        except Exception as e:
            logger.warning(
                "pack_first_seen legacy migration: failed to persist new "
                "file (%s); will retry on next save",
                e,
            )

    def _save_eval_state(self):
        """Persist evaluation state to disk for restart recovery.

        ``pack_first_seen`` is persisted to its own file (see
        ``self.config.pack_first_seen_path``); failures there are
        independent of the main eval-state write so a single bad disk
        does not lose both.
        """
        # Emit both keys for backward-compat with older validator binaries.
        spec = _spec_number()
        data = {
            "scoring_version": spec,
            "spec_number": spec,
            "scenario_scores": self.scenario_scores,
            "eval_pack_hash": self._eval_pack_hash,
            "last_eval_block": self.last_eval_block,
            "last_eval_window": self._last_eval_window,
            "last_eval_window_per_hotkey": self.last_eval_window,
            "consensus_window": self._consensus_window,
            "target_window": self._target_window,
            "target_submit_done": self._target_submit_done,
            "waiting_for_quorum": self._waiting_for_quorum,
            "integrity_cache": self.integrity_judge.dump_cache(),
        }
        try:
            self.config.eval_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.eval_state_path.write_text(
                json.dumps(data, indent=2, sort_keys=True)
            )
        except Exception as e:
            logger.warning(f"Failed to save eval state: {e}")

        try:
            save_pack_first_seen(
                self.pack_first_seen,
                self.pack_last_seen_window,
                self.config.pack_first_seen_path,
            )
        except Exception as e:
            logger.warning(f"Failed to save pack_first_seen: {e}")

    def _drop_miner_eval_state(self, hotkey: str):
        """Remove all per-miner eval state and persist.

        Invariant: eval_state only contains miners that successfully
        completed an evaluation. Any pre-evaluation gate failure
        (duplicate pack_hash, NCD copy, pre-eval rejection, pack
        fetch / integrity failure, ...) must wipe the hotkey here so
        stale data from a previous cycle cannot leak into scoring or
        survive across restarts.
        """
        changed = False
        for store in (
            self.scenario_scores,
            self._eval_pack_hash,
            self.last_eval_block,
            self.last_eval_window,
            self._eval_counts,
            self.latest_token_usage,
            self.latest_model_usage,
        ):
            if hotkey in store:
                store.pop(hotkey, None)
                changed = True
        if changed:
            self._save_eval_state()

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
            spec_number=_spec_number(),
        )

        if len(commitment_str.encode("utf-8")) > MAX_COMMITMENT_BYTES:
            ipfs_cid, _ = decode_dual_address(content_address)
            fallback_address = ipfs_cid or content_address
            commitment_str = format_consensus_commitment(
                protocol_version=self.config.consensus_protocol_version,
                window_number=window.window_number,
                content_address=fallback_address,
                spec_number=_spec_number(),
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
            spec_number=_spec_number(),
            disqualified=disqualified_by_hotkey,
        )

    async def _run_consensus_aggregation(self, window: EvaluationWindow):
        """Read on-chain consensus pointers, download payloads, filter, aggregate.

        All on-chain commitments are downloaded regardless of their
        ``spec_number`` value; the target spec_number for filtering is
        derived from the on-chain stake distribution by the filter pipeline
        (see ``consensus_filter.select_target_spec_number``).
        """
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
        download_failed = 0
        for vc in chain_commitments:
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
            logger.warning(
                "Window %d: no usable submissions from %d commitments "
                "(%d download-failed), using local results",
                window.window_number, total, download_failed,
            )
            return

        validator_stakes: Dict[str, float] = {}
        for uid in range(len(self.metagraph.hotkeys)):
            hotkey = self.metagraph.hotkeys[uid]
            stake = float(self.metagraph.stake[uid])
            if stake > 0:
                validator_stakes[hotkey] = stake

        min_stake = getattr(self.config, "min_validator_stake", 0.0)
        zero_threshold = getattr(self.config, "zero_signal_threshold", 1.0)
        validated, stats = run_filter_pipeline(
            submissions=submissions,
            expected_window=window.window_number,
            validator_stakes=validator_stakes,
            min_stake=min_stake,
            local_spec_number=_spec_number(),
            expected_protocol=self.config.consensus_protocol_version,
            zero_signal_threshold=zero_threshold,
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
        # platform before selecting a winner.  Uses epoch_number so the server
        # resolves the pack that was active during this window (prevents
        # pack-switch escapes).
        if os.getenv("TRAJECTORYRL_PRE_EVAL_ENABLED", "1") != "0":
            current_block = self.subtensor.get_current_block()
            miners_to_check = list(consensus_scores.keys())

            if miners_to_check:
                sem = asyncio.Semaphore(8)

                async def _limited_pre_eval(hk: str):
                    async with sem:
                        return await pre_eval(
                            hk, epoch_number=window.window_number,
                            wallet=self.wallet,
                        )

                results = await asyncio.gather(*(
                    _limited_pre_eval(miner_hk)
                    for miner_hk in miners_to_check
                ))

                pre_eval_disqualified = 0
                for miner_hk, pre_eval_result in zip(miners_to_check, results):
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
                                pack_url=pre_eval_result.get("pack_url", ""),
                                pack_hash=pre_eval_result.get("pack_hash", ""),
                                llm_base_url=self._judge_base_url,
                                llm_model=self._judge_model,
                                rejected=True,
                                rejection_stage=_stage,
                                rejection_detail=_detail,
                                spec_number=_spec_number(),
                                epoch_number=window.window_number,
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
        # can skip the metagraph lookup later). target_spec_number gates
        # the cross-spec bypass: when the chain-derived target spec
        # differs from the persisted winner's spec, the δ threshold is
        # skipped and the new spec's best miner takes over immediately.
        winner_hk, updated_state = select_winner_with_protection(
            consensus_scores=eligible_scores,
            state=self._winner_state,
            score_delta=self.config.score_delta,
            hk_to_uid=hk_to_uid,
            disable_winner_protection=self.config.disable_winner_protection,
            target_spec_number=stats.target_spec_number,
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

        On success we **persist** ``_consensus_window = target_window`` so
        the main loop's "advance target_window past the just-aggregated
        window" check ([1404]) actually fires after restart. On failure we
        restore the saved value so the main loop can retry aggregation
        normally without poisoning its per-window guard.
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
            window_start=self._window_config.global_anchor
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

        # Persist _consensus_window = target_window when aggregation succeeded
        # so the main loop's "target_window <= _consensus_window → advance"
        # check fires next tick. Restoring the saved value here would leave
        # _consensus_window stuck at -1 (or the pre-restart value), which
        # deadlocks the main loop: target_window stays at the just-aggregated
        # window, _should_start_target_evaluation refuses to start the next
        # eval cycle (target != physical), and _submit_consensus_payload
        # retries forever with no local data to submit.
        if not aggregation_succeeded:
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

    def _ensure_target_window(self, physical_window: EvaluationWindow) -> int:
        """Initialize target window from persisted state or physical window."""
        if self._target_window < 0:
            self._target_window = max(self._consensus_window + 1, 0)
            if self._target_window < physical_window.window_number:
                self._target_window = physical_window.window_number
            self._target_submit_done = False
            self._save_eval_state()
        return self._target_window

    def _should_start_target_evaluation(
        self,
        physical_window: EvaluationWindow,
        target_window: int,
    ) -> bool:
        """Return True if target window evaluation should start now."""
        if target_window != physical_window.window_number:
            return False
        return target_window > self._last_eval_window

    def _make_window_view(
        self,
        physical_window: EvaluationWindow,
        window_number: int,
    ) -> EvaluationWindow:
        """Build an EvaluationWindow view for a logical target window."""
        window_start = (
            self._window_config.global_anchor
            + window_number * self._window_config.window_length
        )
        return EvaluationWindow(
            window_number=window_number,
            window_start=window_start,
            block_offset=physical_window.block_offset,
            phase=physical_window.phase,
            blocks_into_phase=physical_window.blocks_into_phase,
            blocks_remaining_in_phase=physical_window.blocks_remaining_in_phase,
            publish_offset=self._window_config.publish_block,
            aggregate_offset=self._window_config.aggregate_block,
        )

    def _compute_quorum_status(
        self, target_window: int
    ) -> Tuple[bool, float, float, float]:
        """Return quorum verdict and stake metrics for target window."""
        chain_commitments = fetch_validator_consensus_commitments(
            self.subtensor, self.config.netuid, self.metagraph,
        )
        # Total validator stake in subnet (permit + positive stake).
        total_validator_stake = 0.0
        for uid in range(len(self.metagraph.hotkeys)):
            permitted = (
                uid < len(self.metagraph.validator_permit)
                and self.metagraph.validator_permit[uid]
            )
            if not permitted:
                continue
            stake = float(self.metagraph.stake[uid])
            if stake > 0:
                total_validator_stake += stake

        if total_validator_stake <= 0:
            return False, 0.0, 0.0, 0.0

        hk_to_stake: Dict[str, float] = {}
        for uid in range(len(self.metagraph.hotkeys)):
            hotkey = self.metagraph.hotkeys[uid]
            stake = float(self.metagraph.stake[uid])
            permitted = (
                uid < len(self.metagraph.validator_permit)
                and self.metagraph.validator_permit[uid]
            )
            if permitted and stake > 0:
                hk_to_stake[hotkey] = stake

        target_commitments: List[ValidatorConsensusCommitment] = []
        for vc in chain_commitments:
            if vc.window_number == target_window:
                target_commitments.append(vc)

        stake_by_spec: Dict[int, float] = {}
        seen_for_spec: set = set()
        for vc in target_commitments:
            if vc.validator_hotkey in seen_for_spec:
                continue
            stake = hk_to_stake.get(vc.validator_hotkey, 0.0)
            if stake <= 0:
                continue
            seen_for_spec.add(vc.validator_hotkey)
            stake_by_spec[vc.spec_number] = stake_by_spec.get(vc.spec_number, 0.0) + stake

        effective_spec = _spec_number()
        spec_total_stake = sum(stake_by_spec.values())
        if spec_total_stake > 0 and stake_by_spec:
            dominant_spec = max(stake_by_spec, key=lambda s: stake_by_spec[s])
            dominant_share = stake_by_spec[dominant_spec] / spec_total_stake
            if dominant_share > 0.5:
                effective_spec = dominant_spec

        submitted_stake = 0.0
        seen_hotkeys = set()
        for vc in target_commitments:
            if vc.spec_number != effective_spec:
                continue
            if vc.validator_hotkey in seen_hotkeys:
                continue
            stake = hk_to_stake.get(vc.validator_hotkey, 0.0)
            if stake <= 0:
                continue
            seen_hotkeys.add(vc.validator_hotkey)
            submitted_stake += stake

        ratio = submitted_stake / total_validator_stake
        meets = ratio > float(getattr(self.config, "quorum_threshold", 0.5))
        return meets, ratio, submitted_stake, total_validator_stake

    async def _set_burn_weights(self, reason: str) -> None:
        """Always set burn weights (no copy-owner fallback)."""
        uids, weights = self._fallback_to_owner()
        await self._do_set_weights(
            uids, weights, label=f"[burn: {reason}] ",
        )

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
            f"publish={self._window_config.publish_pct:.0%}, "
            f"aggregate={self._window_config.aggregate_pct:.0%}"
        )
        logger.info(
            f"Weight interval: {self.config.weight_interval_blocks} blocks "
            f"(~{self.config.weight_interval_blocks * 12 // 60}min)"
        )

        # Pull sandbox image early so audit logs (bench_version) are accurate
        # for startup aggregation. SPEC_NUMBER is a code-level constant and
        # no longer derived from the bench image.
        if self.config.full_cycle_on_startup or self.config.aggregate_when_start:
            try:
                await self._sandbox_harness.pull_latest()
                logger.info(
                    "spec_number=%d (bench_version=%s) — pulled before "
                    "startup aggregation",
                    _spec_number(),
                    self._sandbox_harness.sandbox_version,
                )
            except Exception as e:
                logger.warning(
                    "Failed to pull sandbox image before startup aggregation: "
                    "%s — bench_version audit field may be stale", e,
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

                target_window = self._ensure_target_window(window)
                if target_window > window.window_number:
                    logger.warning(
                        "Target window %d is ahead of physical window %d; "
                        "clamping to physical window",
                        target_window, window.window_number,
                    )
                    self._target_window = window.window_number
                    self._target_submit_done = self._check_own_commitment_on_chain(
                        self._target_window
                    )
                    self._save_eval_state()
                    target_window = self._target_window

                if target_window <= self._consensus_window:
                    self._target_window = self._consensus_window + 1
                    self._target_submit_done = self._check_own_commitment_on_chain(
                        self._target_window
                    )
                    self._save_eval_state()
                    target_window = self._target_window

                target_window_view = self._make_window_view(window, target_window)

                # --- Target window: evaluation ---
                if self._should_start_target_evaluation(window, target_window):
                    logger.info("=" * 60)
                    logger.info(
                        f"Evaluation target_window={target_window} at block "
                        f"{current_block} (phase={window.phase.value}, "
                        f"offset={window.block_offset}/{self._window_config.window_length})"
                    )
                    logger.info("=" * 60)

                    await self._run_evaluation_cycle(current_block, target_window)
                    self._last_eval_at = int(time.time())
                    self._last_eval_window = target_window
                    self._last_eval_date = datetime.datetime.utcnow().date()
                    self.last_weight_block = self.subtensor.get_current_block()
                    self._target_submit_done = False

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
                                self._cycle_window_number,
                            )
                        )

                # --- Submit when target eval is fully done (phase-decoupled) ---
                if (
                    self._last_eval_window >= target_window
                    and not self._target_submit_done
                ):
                    if self._check_own_commitment_on_chain(target_window):
                        self._target_submit_done = True
                        self._save_eval_state()
                    else:
                        logger.info(
                            "Target window %d: eval complete, submitting payload at block %d",
                            target_window, current_block,
                        )
                        ok = await self._submit_consensus_payload(target_window_view)
                        if ok:
                            self._target_submit_done = True
                            self._save_eval_state()
                        else:
                            logger.warning(
                                "Target window %d: submission attempt failed; "
                                "will retry next loop iteration",
                                target_window,
                            )

                        if self._cycle_eval_id is not None:
                            asyncio.ensure_future(
                                self._fire_upload_cycle_logs(
                                    self._cycle_eval_id,
                                    self._cycle_log_offset,
                                    self._cycle_log_block,
                                    self._cycle_window_number,
                                )
                            )

                # --- Aggregation phase: quorum gate on target window ---
                # Only assess quorum on a target this validator has actually
                # submitted for. A freshly-bumped target (e.g. consensus+1
                # right after a successful aggregation) has no own commitment
                # yet; checking quorum on it would trivially miss and latch
                # _waiting_for_quorum=True, poisoning the upcoming eval phase
                # via the main-loop tempo refresh's burn branch.
                if (
                    window.phase == WindowPhase.AGGREGATION
                    and self._consensus_window < target_window
                    and self._target_submit_done
                ):
                    logger.info(
                        "Physical window %d in aggregation; checking quorum for target window %d",
                        window.window_number, target_window,
                    )
                    meets, ratio, submitted_stake, total_stake = self._compute_quorum_status(
                        target_window
                    )

                    if not meets:
                        self._waiting_for_quorum = True
                        logger.warning(
                            "Target window %d quorum miss: submitted_stake=%.4f total_validator_stake=%.4f "
                            "ratio=%.4f threshold=%.4f — setting burn weights",
                            target_window,
                            submitted_stake,
                            total_stake,
                            ratio,
                            float(getattr(self.config, "quorum_threshold", 0.5)),
                        )
                        await self._set_burn_weights(
                            reason=(
                                f"quorum-miss target={target_window} "
                                f"ratio={ratio:.4f}"
                            )
                        )
                        self.last_weight_block = current_block
                        self._save_eval_state()
                    else:
                        logger.info(
                            "Target window %d quorum met: submitted_stake=%.4f total_validator_stake=%.4f "
                            "ratio=%.4f — running consensus aggregation",
                            target_window, submitted_stake, total_stake, ratio,
                        )
                        await self._run_consensus_aggregation(target_window_view)

                        if self._consensus_window == target_window:
                            await self._set_winner_weights()
                            self.last_weight_block = current_block
                            self._waiting_for_quorum = False
                            self._target_window = window.window_number
                            self._target_submit_done = self._check_own_commitment_on_chain(
                                self._target_window
                            )
                            self._save_eval_state()
                        else:
                            logger.warning(
                                "Target window %d aggregation did not produce consensus result",
                                target_window,
                            )

                        if self._cycle_eval_id is not None:
                            asyncio.ensure_future(
                                self._fire_upload_cycle_logs(
                                    self._cycle_eval_id,
                                    self._cycle_log_offset,
                                    self._cycle_log_block,
                                    self._cycle_window_number,
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
                    if self._waiting_for_quorum and self._consensus_window < self._target_window:
                        await self._set_burn_weights(
                            reason=(
                                f"tempo-refresh waiting target={self._target_window}"
                            )
                        )
                    else:
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
                        self._cycle_window_number,
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

    async def _prefetch_packs(
        self,
        active_commitments: Dict[int, MinerCommitment],
    ) -> None:
        """Warm PackFetcher's disk cache for every unique pack_hash.

        Each unique `pack_hash` triggers a single `verify_submission` call;
        the fetched content is dropped on the floor — it lives in
        PackFetcher's per-hash disk cache so the later `_evaluate_miner`
        path hits a cached entry instead of re-downloading. Failed fetches
        are logged and skipped; they will surface again during
        `_evaluate_miner`.
        """
        seen: set = set()
        for uid, commitment in active_commitments.items():
            ph = commitment.pack_hash
            if ph in seen:
                continue
            seen.add(ph)
            result = await self.pack_fetcher.verify_submission(
                commitment.pack_url, ph,
            )
            if not result.valid or result.pack_content is None:
                logger.warning(
                    f"Pack prefetch failed: uid={uid} "
                    f"({commitment.hotkey[:8]}…): {result.error or 'unknown'}"
                )

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
        self._cycle_window_number = window_number

        await self._execute_evaluation_cycle(
            current_block, window_number, cycle_eval_id, cycle_start,
        )

    async def _execute_evaluation_cycle(
        self,
        current_block: int,
        window_number: int,
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
        # spec_number is a validator-side constant (see config.SPEC_NUMBER);
        # bench_version is logged purely for audit.
        logger.info("spec_number=%d (bench_version=%s)",
                    _spec_number(),
                    self._sandbox_harness.sandbox_version)

        # Epoch seed for context variation
        epoch = current_block // self.config.eval_interval_blocks
        epoch_seed = self.compute_epoch_seed(epoch, self.config.netuid)
        logger.debug(
            f"Eval cycle: block={current_block}, seed={epoch_seed}"
        )

        # 1. Sync metagraph (with retries + reconnect)
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

        # 2. Acquire the window-N active-set snapshot.
        #
        # Each window has exactly one snapshot
        # (``active_set_window_{N}.json``) freezing the deterministic
        # commitment subset for that window (commit_block < window_start
        # and not stale). The snapshot is the source of truth for the
        # remainder of the cycle: cross-validator consistency comes
        # from chain commit_block, restart consistency from the file.
        #
        # On the first cycle inside a window the file is absent, so we
        # build it from a fresh chain query and persist immediately.
        # On subsequent cycles (same window) the file hits and we
        # avoid the chain round-trip entirely.
        window = compute_window(current_block, self._window_config)
        snapshot = self._acquire_window_snapshot(window, current_block)
        if snapshot is None:
            return

        # Per-validator runtime filters (validator_permit, blacklist)
        # are dynamic and applied on top of the frozen snapshot every
        # cycle so that permit / blacklist changes take effect without
        # invalidating the snapshot itself.
        active_commitments = self._filter_active_commitments_runtime(
            snapshot.commitments,
        )
        logger.info(
            f"Window {window.window_number} snapshot: "
            f"{len(snapshot.commitments)} eligible, "
            f"{len(active_commitments)} after runtime filter"
        )

        if not active_commitments:
            logger.warning("No active miners with valid commitments!")
            await self._set_fallback_weights()
            return

        # 4. Evaluate miners (scenarios selected by sandbox image).
        # Pack-hash dedup is handled inside the loop via `pack_first_seen`
        # ownership lock: the first hotkey to register a given pack_hash
        # owns it permanently (for this validator instance). Subsequent
        # submitters of the same pack_hash are treated as copies, get
        # weight 0, and skip the LLM eval entirely.
        eval_scenarios: List[str] = []  # not used in S1 — sandbox picks scenario
        evaluated_count = 0
        attempted_count = 0
        skipped_interval_count = 0
        rejected_pre_eval_count = 0
        copy_rejected_count = 0
        total_eligible = len(active_commitments)
        logger.info(
            f"=== Eval cycle: {total_eligible} eligible miners ==="
        )

        await self._prefetch_packs(active_commitments)

        # Iterate by chain commit_block ascending so the earliest
        # committer of a duplicated pack_hash is processed first and
        # claim_owner records them as the canonical owner. Even if a
        # later iteration order would observe the same final state
        # (claim_owner is order-independent under the new semantics),
        # ascending order also avoids wasting an eval round-trip on a
        # copy that is about to be demoted in the same cycle.
        eval_order = sorted(
            active_commitments.items(),
            key=lambda item: (item[1].block_number, item[1].hotkey),
        )

        miner_idx = 0
        for uid, commitment in eval_order:
            hotkey = commitment.hotkey
            miner_idx += 1

            # Ownership lock: earliest chain commit_block wins (Issue 4
            # anti-snipe). Passing the on-chain block keeps the recorded
            # ordering deterministic across validators rather than
            # tied to local observation time.
            ph = commitment.pack_hash
            owner_hk, _owner_blk = claim_owner(
                self.pack_first_seen, ph, hotkey, commitment.block_number,
            )
            if owner_hk != hotkey:
                copy_rejected_count += 1
                detail = (
                    f"pack_hash {ph[:12]} owned by {owner_hk[:8]}; "
                    f"treating as copy"
                )
                self._get_miner_logger(hotkey).info(
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
                        spec_number=_spec_number(),
                        epoch_number=window_number,
                        **self._harness_metadata(),
                    )
                )
                self._disqualified_miners[hotkey] = (
                    f"copy_of:{owner_hk}"
                )
                self._drop_miner_eval_state(hotkey)
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
                            spec_number=_spec_number(),
                            epoch_number=window_number,
                            **self._harness_metadata(),
                        )
                    )
                    self._disqualified_miners[hotkey] = f"pre_eval_rejected:{reason}"
                    self._drop_miner_eval_state(hotkey)
                    continue

            needs_eval = self._needs_evaluation(
                hotkey, commitment.pack_hash, window_number
            )
            if not needs_eval:
                skipped_interval_count += 1
                last_w = self.last_eval_window.get(hotkey)
                self._get_miner_logger(hotkey).info(
                    f"[{miner_idx}/{total_eligible}] Skipping, "
                    f"pack_hash unchanged "
                    f"(last evaluated in window {last_w})"
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
                    window_number,
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
                self.last_eval_window[hotkey] = window_number
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
                        uid, commitment, eval_result, eval_count, pack_changed,
                        current_block, window_number,
                    )
                )

                self._track_uid_change(uid, hotkey)

                # Persist per-miner so a crash mid-cycle doesn't lose
                # work already paid for in LLM tokens.
                self._save_eval_state()

            else:
                elapsed_str = f" ({eval_elapsed:.1f}s)" if eval_elapsed else ""
                logger.info(
                    f"[{miner_idx}/{total_eligible}] Miner {uid} eval returned None"
                    f"{elapsed_str} "
                    f"(pack fetch/integrity failed)"
                )
                self._drop_miner_eval_state(hotkey)

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

        # End-of-cycle eviction: drop pack_first_seen entries whose
        # pack_hash has been absent from every active commitment for
        # `EVICTION_GRACE_WINDOWS` consecutive wall-clock windows.
        # This bounds the table's growth and enables a "resurrection"
        # path for long-orphaned packs while shielding original authors
        # from a single short outage costing them ownership (see
        # trajectoryrl.utils.pack_ownership).
        active_hashes = {c.pack_hash for c in active_commitments.values()}
        evicted = evict_orphans(
            self.pack_first_seen,
            self.pack_last_seen_window,
            active_hashes,
            current_window=window_number,
        )
        if evicted:
            logger.info(
                f"pack_first_seen eviction: {len(evicted)} orphaned entries removed "
                f"after {EVICTION_GRACE_WINDOWS}-window grace at window {window_number}"
            )

        parts = [f"{evaluated_count} evaluated"]
        if rejected_pre_eval_count:
            parts.append(f"{rejected_pre_eval_count} pre-eval rejected")
        if copy_rejected_count:
            parts.append(f"{copy_rejected_count} copy rejected")
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
        self, hotkey: str, pack_hash: str, current_window: int
    ) -> bool:
        """Return True iff the miner needs (re-)evaluation in this window.

        Dedup is keyed purely on ``pack_hash``. Since
        ``_filter_active_commitments`` drops commitments older than
        ``inactivity_blocks`` (~2 windows), an unchanged ``pack_hash``
        transitively means the miner is still actively re-submitting the
        same pack — safe to reuse cached scenario scores.

        ``current_window`` is accepted for symmetry / future use; it is
        not part of the gating decision.
        """
        del current_window  # signature only; kept for caller clarity

        if self._eval_pack_hash.get(hotkey) != pack_hash:
            self._get_miner_logger(hotkey).info(
                f"pack_hash changed, marking for eval"
            )
            return True

        if not self.scenario_scores.get(hotkey):
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

    def _filter_active_commitments_runtime(
        self,
        commitments: Dict[int, MinerCommitment],
    ) -> Dict[int, MinerCommitment]:
        """Apply per-validator runtime filters on top of a snapshot.

        The snapshot already enforces the deterministic filters
        (``commit_block < window_start`` and stale-age vs window_start).
        This method layers on the per-validator dynamic filters that
        intentionally do NOT participate in cross-validator consistency:

        * ``validator_permit`` — current metagraph state (a permit can
          appear or disappear mid-window).
        * coldkey blacklist — operator-local policy.
        """
        blacklist = set(self.config.coldkey_blacklist)
        active: Dict[int, MinerCommitment] = {}
        for uid, commitment in commitments.items():
            if uid < len(self.metagraph.validator_permit) and self.metagraph.validator_permit[uid]:
                continue
            if blacklist:
                coldkey = (
                    self.metagraph.coldkeys[uid]
                    if uid < len(self.metagraph.coldkeys) else None
                )
                if coldkey in blacklist:
                    self._get_miner_logger(commitment.hotkey).info(
                        f"Skipping eval (coldkey {coldkey} is blacklisted)"
                    )
                    continue
            active[uid] = commitment
        return active

    def _acquire_window_snapshot(
        self,
        window: EvaluationWindow,
        current_block: int,
    ) -> Optional[EvalSnapshot]:
        """Load or build the active-set snapshot for ``window``.

        Order of operations:

        1. Try to load ``active_set_window_{N}.json``. A successful
           load is the fast path for any non-first cycle in window N
           (and for restart recovery within the same window).
        2. On miss, query the chain once and apply the deterministic
           filters (``commit_block < window_start`` plus stale-age vs
           window_start). Persist the result so subsequent cycles in
           the same window hit the cache.

        Returns the snapshot, or None if there are no eligible
        commitments at all (caller should fall back to owner weights).
        """
        snapshot = load_snapshot(self.config.active_set_dir, window.window_number)
        if snapshot is not None:
            return snapshot

        logger.info(
            "Window %d: building active-set snapshot from chain "
            "(window_start=%d, current_block=%d)",
            window.window_number, window.window_start, current_block,
        )
        raw = fetch_all_commitments(
            self.subtensor, self.config.netuid, self.metagraph,
        )
        snapshot = take_snapshot(
            raw,
            window_number=window.window_number,
            window_start=window.window_start,
            snapshot_block=current_block,
            inactivity_blocks=self.config.inactivity_blocks,
        )
        save_snapshot(snapshot, self.config.active_set_dir)
        return snapshot

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

        Delegates the real work to ``evaluate_miner_s1`` (shared with
        ``scripts/eval_miners.py``). This method only:
        - Resolves the per-miner file logger and the wallet-derived salt.
        - Maps the helper's outcome back into the validator pipeline dict
          shape and updates ``_disqualified_miners`` for non-pack-verify
          skips (matching pre-refactor behavior).

        Returns:
            Dict with keys "qualified", "judge_details", ... or ``None``
            if pre-evaluation checks fail.
        """
        mlog = self._get_miner_logger(commitment.hotkey)
        mlog.info(
            f"Evaluating miner {miner_uid} "
            f"(hotkey={commitment.hotkey[:8]}, hash={commitment.pack_hash[:12]}...)"
        )

        outcome = await evaluate_miner_s1(
            harness=self._sandbox_harness,
            pack_fetcher=self.pack_fetcher,
            commitment=commitment,
            epoch_seed=epoch_seed,
            validator_salt=self._default_validator_salt(),
            mlog=mlog,
        )

        if not outcome.success:
            # Match pre-refactor semantics: pack-verify failures return
            # None silently; SKILL.md and harness errors mark the miner
            # as disqualified for this evaluation cycle.
            if outcome.skip_reason and outcome.skip_reason != SKIP_PACK_VERIFY:
                self._disqualified_miners[commitment.hotkey] = outcome.skip_reason
            return None

        # Defensive: judge_details is built in-place by evaluate_miner_s1 today
        # so this normally cannot raise, but a future contract drift (renamed
        # field, missing scenario, None outcome) must not crash the whole cycle
        # — that would silently kill all subsequent miners until restart.
        scenario_qualified = {
            sn: bool((d or {}).get("qualification_gate", False))
            for sn, d in (outcome.judge_details or {}).items()
            if isinstance(d, dict)
        }

        return {
            "qualified": scenario_qualified,
            "token_usage": {},
            "model_usage": {},
            "judge_details": outcome.judge_details,
            "session_keys": {},
            "session_files": {},
            # S1-specific: full SandboxEvaluationResult for eval log upload.
            # Not serialized — consumed by _fire_upload_eval_logs and dropped.
            "_s1_sandbox_result": outcome.sandbox_result,
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
        window_number: int,
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
            spec_number=_spec_number(),
            epoch_number=window_number,
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
        window_number: int,
    ) -> None:
        """Collect and upload eval logs. Fire-and-forget."""

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
            validator_log_offset, miner_log_offset,
        )
        if log_archive:
            meta = {
                "eval_id": eval_id,
                "window_number": window_number,
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
                epoch_number=window_number,
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
        window_number: int,
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
                "window_number": window_number,
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
                epoch_number=window_number,
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
                    epoch_number=meta.get("window_number"),
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
                epoch_number=meta.get("window_number"),
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
