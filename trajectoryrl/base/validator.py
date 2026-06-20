"""TrajectoryRL Validator — v6.0 winner-challenger daemon.

Architecture (Season 1 — trajrl-bench, v6.0 IM):
    1. Polls ``GET /api/v2/epoch/current`` (~10 s) for the active
       challenger and the seated winner.
    2. Refreshes the local winner cache from the server response on every
       successful poll.
    3. Fetches and verifies the challenger pack (SHA256), runs trajrl-bench
       sandbox evaluation, and posts the signed score to
       ``POST /api/v2/epoch/{challenge_epoch_id}/score``.
    4. Sets on-chain weights from the cached winner; tempo-gated so the
       chain accepts at most one weight write per tempo regardless of
       how often the daemon ticks.
    5. Heartbeats every ~10 min with version + image digests.

The validator carries no per-miner evaluation cache, no client-side
integrity LLM judge, no ``pack_first_seen`` ownership state, and no
``epoch_snapshot`` polling. Pre-eval, integrity gating, and ownership
locks are all handled server-side (see INCENTIVE_MECHANISM.md §4 and §3).

Each validator runs independently — Yuma Consensus aggregates on-chain.
"""

import asyncio
import datetime
import hashlib
import json
import logging
import time

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

from ..utils.config import ValidatorConfig
from ..utils.sandbox_harness import (
    TrajectorySandboxHarness,
    scenarios_for_spec,
    LATEST_SPEC,
)
from ..utils.github import PackFetcher
from ..utils import config as _config_mod
from ..utils.commitments import MinerCommitment
from ..utils.miner_eval import (
    evaluate_miner_s1,
    SKIP_PACK_VERIFY,
    SKIP_EPOCH_CHANGED,
    SKIP_PROVIDER_FAILURE,
)
from ..utils.trajrl_api import (
    heartbeat,
    fetch_current_epoch,
    fetch_current_winner,
    submit_challenge_score,
    submit_scenario_progress,
    upload_eval_logs,
    upload_cycle_logs,
)
from ..utils.winner_state import (
    WinnerState,
    derive_winner_state,
    save_winner_state,
    load_winner_state,
    WINNER_FALLBACK_TTL_SECONDS,
)
from .. import __version__


def _spec_number() -> int:
    """Current scoring spec identifier (validator-side constant).

    See ``trajectoryrl/utils/config.py::SPEC_NUMBER`` for bump policy.
    """
    return _config_mod.SPEC_NUMBER


logger = logging.getLogger(__name__)

OWNER_UID = 74
BURN_FRACTION = 0.75  # 75% of miner emissions burned via owner UID

_METAGRAPH_SYNC_RETRIES = 3
_METAGRAPH_SYNC_DELAY = 10  # seconds between retries
_METAGRAPH_MIN_NEURONS = 1

# Total set_weights attempts per tick: 1 initial + 2 retries on failure.
_SET_WEIGHTS_MAX_RETRIES = 3
_SET_WEIGHTS_RETRY_DELAY = 12  # seconds; roughly 1 block interval

# Daemon cadence (v6.0).
#
# Eval-tick polling is fixed at _EPOCH_POLL_INTERVAL — every tick we
# poll /epoch/current + /winner/current and exit early if there's
# nothing to do. We deliberately do NOT derive a long sleep from
# `remaining_blocks * 12s`: under v6 dynamic-epoch the active epoch
# can finalise early once every whitelisted validator has submitted,
# so a long sleep would overshoot the early-finalise boundary and the
# next epoch wouldn't be observed until the (now-stale) sleep elapses.
_EPOCH_POLL_INTERVAL = 10
_WEIGHT_CHECK_INTERVAL = 60
_HEARTBEAT_INTERVAL = 600

# Mid-epoch budget gate. We don't pin an absolute eval-duration budget
# (operator-tunable env vars would just shift mis-configuration risk
# from one knob to another). Instead use a progress ratio against the
# server-reported `elapsed_blocks` / `epoch_length_blocks`: if more than
# `_EVAL_LATEST_START_RATIO` of the epoch is already gone when we look,
# skip and wait for the next one. This auto-adapts to whatever
# `EPOCH_LENGTH_BLOCKS` the server is configured with — e.g. on a
# 150-block (~30 min) epoch with ratio 0.5 the daemon must start eval
# within the first ~15 min of the window.
_EVAL_LATEST_START_RATIO = 0.5


class TrajectoryValidator:
    """v6.0 thin validator daemon."""

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
                "METAGRAPH EMPTY at startup (n=0). On-chain operations "
                "will fail until sync recovers. Endpoint: %s",
                config.network,
            )

        logger.info("Initializing trajrl-bench sandbox harness...")
        self._sandbox_harness = TrajectorySandboxHarness(config)

        logger.debug("Initializing pack fetcher...")
        self.pack_fetcher = PackFetcher(cache_dir=config.pack_cache_dir)

        # Winner cache (server-canonical, mirrored locally for fallback)
        self._winner_state_path = str(config.winner_state_path)
        self._winner_state: WinnerState = load_winner_state(self._winner_state_path)

        # Telemetry timestamps (set after each successful action)
        self._last_set_weights_at: Optional[int] = None
        self._last_eval_at: Optional[int] = None

        # Tempo gate: which block we last attempted set_weights at
        self._last_set_weights_block: int = 0

        # Idempotency: don't re-evaluate the same challenge epoch on retry
        self._last_scored_challenge_epoch_id: Optional[int] = None

        # Cycle log state — rolled per-eval to bundle the validator log
        # captured during a single challenger evaluation
        self._cycle_eval_id: Optional[str] = None
        self._cycle_log_offset: int = 0
        self._cycle_log_block: int = 0

        self._load_eval_state()

        logger.info("Validator initialization complete!")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logging(self):
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        self._validator_log_path = (
            self.config.log_dir / f"validator_{int(time.time())}.log"
        )
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self._validator_log_path),
            ],
        )

        # Per-miner log directory (kept for per-challenger eval traces)
        self._miner_log_dir = self.config.log_dir / "miners"
        self._miner_log_dir.mkdir(parents=True, exist_ok=True)
        self._miner_loggers: Dict[str, logging.Logger] = {}

    def _get_miner_logger(self, hotkey: str) -> logging.Logger:
        """Get or create a per-miner file logger (INFO, no console)."""
        if hotkey in self._miner_loggers:
            return self._miner_loggers[hotkey]

        mlog = logging.getLogger(f"trajectoryrl.miner.{hotkey[:16]}")
        mlog.setLevel(logging.INFO)
        mlog.propagate = False

        log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
        fh = logging.FileHandler(
            self._miner_log_dir / f"{hotkey[:16]}.log"
        )
        fh.setFormatter(logging.Formatter(log_format))
        mlog.addHandler(fh)

        self._miner_loggers[hotkey] = mlog
        return mlog

    # ------------------------------------------------------------------
    # Metagraph sync helpers
    # ------------------------------------------------------------------

    def _sync_metagraph(
        self,
        *,
        retries: int = _METAGRAPH_SYNC_RETRIES,
        caller: str = "",
    ) -> bool:
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
                        "%sMetagraph sync recovered on attempt %d/%d (n=%d)",
                        label, attempt, retries, n,
                    )
                return True

            logger.warning(
                "%sMetagraph sync returned n=%d, attempt %d/%d",
                label, n, attempt, retries,
            )
            if attempt < retries:
                time.sleep(_METAGRAPH_SYNC_DELAY)
                self._reconnect_subtensor(label)

        logger.error(
            "%sMETAGRAPH UNHEALTHY after %d attempts — n=%d. "
            "On-chain set_weights will likely fail. Endpoint: %s",
            label, retries, getattr(self.metagraph, "n", 0),
            self.config.network,
        )
        return False

    def _reconnect_subtensor(self, label: str = ""):
        try:
            logger.info("%sReconnecting subtensor (network=%s)...",
                        label, self.config.network)
            self.subtensor = bt.Subtensor(network=self.config.network)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            logger.info("%sSubtensor reconnected", label)
            return self.subtensor
        except Exception as e:
            logger.error(
                "%sFailed to reconnect subtensor: %s", label, e,
                exc_info=True,
            )
            return None

    def _is_metagraph_healthy(self) -> bool:
        n = getattr(self.metagraph, "n", 0)
        return bool(n and n >= _METAGRAPH_MIN_NEURONS)

    # ------------------------------------------------------------------
    # Persistent eval state (v6 minimal)
    # ------------------------------------------------------------------

    def _load_eval_state(self):
        """Load persisted state from disk.

        v6.0 keeps almost no per-miner state on the validator. The only
        fields that matter across restarts are:
          - ``last_scored_challenge_epoch_id`` so a restart mid-epoch
            doesn't double-submit the same score.
          - ``last_set_weights_at`` / ``last_eval_at`` for telemetry.

        Everything else (winner cache) lives in the dedicated
        ``winner_state.json``.
        """
        path = self.config.eval_state_path
        if not path.exists():
            logger.debug("No persisted eval state found, starting fresh")
            return
        try:
            data = json.loads(path.read_text())
            file_spec = data.get("spec_number", data.get("scoring_version", _spec_number()))
            if file_spec != _spec_number():
                logger.info(
                    "Eval state spec_number mismatch (%s != %d), "
                    "resetting per-spec idempotency markers",
                    file_spec, _spec_number(),
                )
                return

            self._last_scored_challenge_epoch_id = data.get(
                "last_scored_challenge_epoch_id"
            )
            self._last_set_weights_at = data.get("last_set_weights_at")
            self._last_eval_at = data.get("last_eval_at")
            self._last_set_weights_block = int(
                data.get("last_set_weights_block", 0) or 0
            )
        except Exception as e:
            logger.warning("Failed to load eval state from %s: %s", path, e)

    def _save_eval_state(self):
        path = self.config.eval_state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "spec_number": _spec_number(),
                "scoring_version": _spec_number(),  # legacy mirror
                "last_scored_challenge_epoch_id": (
                    self._last_scored_challenge_epoch_id
                ),
                "last_set_weights_at": self._last_set_weights_at,
                "last_eval_at": self._last_eval_at,
                "last_set_weights_block": self._last_set_weights_block,
            }
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save eval state to %s: %s", path, e)

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self):
        """Send heartbeat every ~10 min."""
        while True:
            try:
                await heartbeat(
                    self.wallet,
                    last_set_weights_at=self._last_set_weights_at,
                    last_eval_at=self._last_eval_at,
                    bench_image_hash=self._sandbox_harness.bench_image_hash,
                    harness_image_hash=self._sandbox_harness.scenario_image_hash,
                    bench_version=self._sandbox_harness.sandbox_version,
                    llm_model=self.config.llm_model,
                    llm_base_url=self.config.llm_base_url,
                )
            except Exception as e:
                logger.warning("Heartbeat error: %s", e)
            await asyncio.sleep(_HEARTBEAT_INTERVAL)

    # ------------------------------------------------------------------
    # Validator-side metadata for outgoing scores
    # ------------------------------------------------------------------

    def _default_validator_salt(self) -> str:
        data = f"{self.wallet.hotkey.ss58_address}:{self.config.netuid}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _harness_metadata(self) -> Dict[str, str]:
        """bench/harness image hashes, bench version, and agent-harness
        name+version for outgoing payloads."""
        h = self._sandbox_harness
        meta: Dict[str, str] = {}
        if h.bench_image_hash != "unknown":
            meta["bench_image_hash"] = h.bench_image_hash
        if h.scenario_image_hash != "unknown":
            meta["harness_image_hash"] = h.scenario_image_hash
        if h.sandbox_version != "unknown":
            meta["bench_version"] = h.sandbox_version
        if h.harness_name:
            meta["harness_name"] = h.harness_name
        if h.harness_version:
            meta["harness_version"] = h.harness_version
        return meta

    # ------------------------------------------------------------------
    # Eval log capture / upload (kept verbatim from v5.x — same contract)
    # ------------------------------------------------------------------

    _MAX_LOG_ARCHIVE_BYTES = 10 * 1024 * 1024  # 10 MB

    def _get_validator_log_offset(self) -> int:
        try:
            return (
                self._validator_log_path.stat().st_size
                if self._validator_log_path.exists() else 0
            )
        except OSError:
            return 0

    def _prepare_eval_log_capture(
        self, eval_id: str, hotkey: str,
    ) -> Tuple[Path, int, int]:
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
        commitment: MinerCommitment,
        eval_scenarios: List[str],
        eval_result: Optional[Dict],
        eval_dir: Path,
        validator_log_offset: int,
        miner_log_offset: int,
        block_height: int,
        challenge_epoch_id: int,
    ) -> None:
        """Collect and upload eval logs. Fire-and-forget."""
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
                "challenge_epoch_id": challenge_epoch_id,
                "miner_hotkey": commitment.hotkey,
                "miner_uid": uid,
                "block_height": block_height,
                "pack_hash": commitment.pack_hash,
                "spec_number": _spec_number(),
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
                epoch_number=challenge_epoch_id,
                spec_number=_spec_number(),
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
        challenge_epoch_id: int,
    ) -> None:
        """Upload the validator log segment for one challenger cycle."""
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
                "challenge_epoch_id": challenge_epoch_id,
                "block_height": block_height,
                "spec_number": _spec_number(),
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
                epoch_number=challenge_epoch_id,
                spec_number=_spec_number(),
            )
            if ok:
                try:
                    (cycle_dir / ".cycle_uploaded").touch()
                except OSError:
                    pass
        except Exception as e:
            logger.warning("Cycle log upload error: %s", e)

    async def _replay_pending_uploads(self):
        """Re-upload eval/cycle logs from the last 2 days that lack a marker."""
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
                    epoch_number=meta.get(
                        "challenge_epoch_id", meta.get("window_number")
                    ),
                    spec_number=meta.get("spec_number"),
                )
                if ok:
                    eval_uploaded += 1
                    try:
                        (miner_dir / ".uploaded").touch()
                    except OSError:
                        pass

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
                epoch_number=meta.get(
                    "challenge_epoch_id", meta.get("window_number")
                ),
                spec_number=meta.get("spec_number"),
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
    # Challenger evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _challenge_seed(challenge_epoch_id: int, netuid: int = 11) -> int:
        raw = f"trajectoryrl-{netuid}-challenge-{challenge_epoch_id}".encode()
        return int(hashlib.sha256(raw).hexdigest()[:8], 16)

    @staticmethod
    def _epoch_has_budget(
        resp: Dict[str, Any],
        epoch: Dict[str, Any],
        challenge_epoch_id: int,
    ) -> bool:
        """Mid-epoch start gate based on server-reported elapsed progress.

        Returns False (skip) when more than ``_EVAL_LATEST_START_RATIO``
        of the epoch has already passed; True otherwise. When the server
        cannot report ``elapsed_blocks`` (chain RPC down), or epoch
        length is missing/zero, the gate is bypassed and the caller
        proceeds — the daemon has no better signal locally.
        """
        elapsed_blocks = resp.get("elapsed_blocks")
        if elapsed_blocks is None:
            return True

        # Prefer the explicit epoch_length_blocks field; fall back to
        # end_block - start_block. The two should agree but the latter
        # is the safer ground truth.
        start_block = epoch.get("start_block")
        end_block = epoch.get("end_block")
        try:
            length = int(epoch.get("epoch_length_blocks") or 0)
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            try:
                length = int((end_block or 0) - (start_block or 0))
            except (TypeError, ValueError):
                length = 0
        if length <= 0:
            return True

        progress = elapsed_blocks / length
        if progress > _EVAL_LATEST_START_RATIO:
            logger.info(
                "Skipping epoch %s: elapsed %d/%d blocks (%.0f%%) "
                "past latest-start cutoff %.0f%%; waiting for next epoch.",
                challenge_epoch_id, elapsed_blocks, length,
                progress * 100, _EVAL_LATEST_START_RATIO * 100,
            )
            return False
        return True

    @staticmethod
    def _commitment_from_epoch(epoch_block: Dict[str, Any]) -> Optional[MinerCommitment]:
        """Build a ``MinerCommitment`` from the API ``epoch`` block.

        Returns None when the block is missing required fields. UID is
        unknown at this layer (not in the v6 epoch block); ``-1`` is a
        sentinel that surfaces in eval logs but does not affect scoring.
        """
        if not isinstance(epoch_block, dict):
            return None

        hotkey = epoch_block.get("challenger_hotkey")
        pack_hash = epoch_block.get("challenger_pack_hash")
        pack_url = epoch_block.get("challenger_pack_url")
        if not hotkey or not pack_hash or not pack_url:
            return None

        try:
            uid = int(epoch_block.get("challenger_uid", -1))
        except (TypeError, ValueError):
            uid = -1

        try:
            block_number = int(
                epoch_block.get("start_block")
                or epoch_block.get("created_block")
                or 0
            )
        except (TypeError, ValueError):
            block_number = 0

        return MinerCommitment(
            uid=uid,
            hotkey=hotkey,
            pack_hash=pack_hash,
            pack_url=pack_url,
            block_number=block_number,
            raw=f"{pack_hash}|{pack_url}",
        )

    async def _evaluate_challenger(
        self,
        commitment: MinerCommitment,
        challenge_epoch_id: int,
        spec: int,
        scenarios: tuple,
    ) -> Optional[Dict[str, Any]]:
        """Run a single trajrl-bench evaluation on the active challenger."""
        mlog = self._get_miner_logger(commitment.hotkey)
        mlog.info(
            f"Evaluating challenger uid={commitment.uid} "
            f"hotkey={commitment.hotkey[:8]} "
            f"hash={commitment.pack_hash[:12]}... "
            f"epoch_id={challenge_epoch_id}"
        )

        # Per-scenario streaming hooks for the live /challenge page. The
        # eval runs inside run_in_executor so callbacks fire on a worker
        # thread; we schedule the async POST back on the main loop via
        # run_coroutine_threadsafe (fire-and-forget). Submits are
        # bounded by a 10s timeout, 404s log at debug, other errors at
        # warning — none affect eval correctness.
        #
        # Three lifecycle events per scenario:
        #   start  → state="running"   (sandbox up, hermes chat about to run)
        #   chat-end → state="verifying" (chat done, verifier about to run)
        #   done   → state="complete"  (verifier scored — final quality)
        loop = asyncio.get_running_loop()

        def _post_progress(scenario_name, scenario_index, total_scenarios,
                           state, *, quality=0.0, cost_usd=None,
                           duration_s=None, timed_out=None):
            coro = submit_scenario_progress(
                self.wallet,
                challenge_epoch_id=challenge_epoch_id,
                miner_hotkey=commitment.hotkey,
                miner_uid=commitment.uid,
                scenario_name=scenario_name,
                scenario_index=scenario_index,
                total_scenarios=total_scenarios,
                quality=quality,
                state=state,
                cost_usd=cost_usd,
                duration_s=duration_s,
                timed_out=timed_out,
                spec_number=spec,
                harness_name=self._sandbox_harness.harness_name,
                harness_version=self._sandbox_harness.harness_version,
                llm_model=self.config.llm_model,
            )
            asyncio.run_coroutine_threadsafe(coro, loop)

        def on_episode_start(scenario_name, scenario_index, total_scenarios):
            _post_progress(scenario_name, scenario_index, total_scenarios,
                           "running")

        def on_episode_verifying(scenario_name, scenario_index, total_scenarios):
            _post_progress(scenario_name, scenario_index, total_scenarios,
                           "verifying")

        def on_episode_done(episode, scenario_index, total_scenarios):
            _post_progress(
                episode.scenario, scenario_index, total_scenarios,
                "complete",
                quality=episode.quality,
                cost_usd=episode.cost_usd,
                duration_s=episode.duration_s,
                timed_out=episode.timed_out,
            )

        # Per-scenario abort gate: if the epoch has moved on while we
        # were grinding through earlier scenarios, bail out and let the
        # validator pick up the new challenger. Saves ~20-30 min of
        # compute on a session whose submission would 409 anyway.
        # Synchronous fetch of the current epoch via the existing
        # trajrl_api helper, executed on the harness's worker thread.
        def is_epoch_still_current(pack_hash):
            from ..utils.trajrl_api import fetch_current_epoch  # local import — avoid cycle
            fut = asyncio.run_coroutine_threadsafe(fetch_current_epoch(), loop)
            cur = fut.result(timeout=8.0)
            if cur is None:
                return True  # transient — assume still current to avoid false-aborts
            epoch = cur.get("epoch") or {}
            if epoch.get("status") != "in_progress":
                return False
            cur_pack = epoch.get("challenger_pack_hash") or ""
            # pack_hash here is the full hex sha256 from the chain commitment;
            # the API also returns the full hex. Compare on first 12 chars to
            # tolerate any historical truncation.
            return cur_pack.startswith(pack_hash[:12])

        outcome = await evaluate_miner_s1(
            harness=self._sandbox_harness,
            pack_fetcher=self.pack_fetcher,
            commitment=commitment,
            epoch_seed=self._challenge_seed(challenge_epoch_id, self.config.netuid),
            validator_salt=self._default_validator_salt(),
            scenarios=scenarios,
            mlog=mlog,
            on_episode_start=on_episode_start,
            on_episode_verifying=on_episode_verifying,
            on_episode_done=on_episode_done,
            is_epoch_still_current=is_epoch_still_current,
        )

        if not outcome.success:
            return {
                "success": False,
                "skip_reason": outcome.skip_reason,
            }

        scenario_qualified = {
            sn: bool((d or {}).get("qualification_gate", False))
            for sn, d in (outcome.judge_details or {}).items()
            if isinstance(d, dict)
        }

        return {
            "success": True,
            "qualified": scenario_qualified,
            "judge_details": outcome.judge_details,
            "_s1_sandbox_result": outcome.sandbox_result,
        }

    @staticmethod
    def _summarize_eval(
        eval_result: Dict[str, Any],
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Collapse the per-scenario eval into (score, qualified, scenario_results).

        score = Σ per-scenario quality (passed/total ∈ [0, 1] each)
        qualified = at least one scenario passed at least one test (any-pass).
        """
        raw_qualified = eval_result.get("qualified") or {}
        raw_judge = eval_result.get("judge_details") or {}

        def _scenario_score(sname: str) -> float:
            jd = raw_judge.get(sname) or {}
            overall = jd.get("overall_score")
            if overall is not None:
                try:
                    return float(overall)
                except (TypeError, ValueError):
                    pass
            return 0.0

        raw_score = sum(_scenario_score(s) for s in raw_qualified)

        scenario_results: Dict[str, Any] = {}
        for sname, q in raw_qualified.items():
            entry: Dict[str, Any] = {
                "score": round(_scenario_score(sname), 4),
                "weight": 1.0,
                "qualified": bool(q),
            }
            jd = raw_judge.get(sname)
            if jd:
                entry["judge"] = jd
            scenario_results[sname] = entry

        qualified = bool(raw_qualified) and any(raw_qualified.values())
        return round(raw_score, 4), qualified, scenario_results

    # ------------------------------------------------------------------
    # Eval loop
    # ------------------------------------------------------------------

    async def _eval_loop(self):
        """Two-poll tick on the v6 daemon hot path:

        - ``GET /api/v2/epoch/current`` for the eval task (challenger pack).
        - ``GET /api/v2/winner/current`` for the local-derivation inputs
          that drive the winner cache (and therefore ``set_weights``).

        Both endpoints are independent and share one tick. The loop
        polls at a fixed cadence of ``_EPOCH_POLL_INTERVAL`` (10 s)
        regardless of how much time is left in the current epoch — see
        the comment on ``_EPOCH_POLL_INTERVAL`` for why we no longer
        derive a long sleep from ``remaining_blocks``.
        """
        logger.info("Eval loop started (poll %ds)", _EPOCH_POLL_INTERVAL)
        while True:
            try:
                await self._eval_loop_tick()
            except Exception as e:
                logger.exception("Eval loop tick failed: %s", e)
            await asyncio.sleep(_EPOCH_POLL_INTERVAL)

    async def _eval_loop_tick(self) -> None:
        """Single eval-tick: refresh the winner cache, poll the active
        challenge epoch, and (if not yet scored and budget permits) run
        the eval and submit the score. Returns nothing — the caller
        always sleeps for ``_EPOCH_POLL_INTERVAL``.
        """
        # 1) Refresh the winner cache from /api/v2/winner/current. The
        #    server's `winner` field is advisory; derive_winner_state
        #    runs the same stake-weighted aggregation locally and warns
        #    on divergence.
        try:
            await self._refresh_winner_cache()
        except Exception as e:
            logger.warning("Winner cache refresh failed: %s", e)

        # 2) Poll the active challenge epoch. Sign the request with the
        #    validator wallet so the response includes
        #    `epoch.challenger_pack_url` (gated to validators-or-24h per
        #    docs/API.md). The `winner` block on this response is
        #    intentionally ignored — see fetch_current_winner /
        #    derive_winner_state.
        resp = await fetch_current_epoch(self.wallet)
        if resp is None:
            return  # transient → next tick

        epoch = resp.get("epoch")
        if not epoch:
            return  # 404 / between epochs → next tick

        try:
            challenge_epoch_id = int(epoch.get("challenge_epoch_id"))
        except (TypeError, ValueError):
            logger.warning("epoch missing or invalid challenge_epoch_id: %r", epoch)
            return

        # Already scored this epoch — nothing to do until the server
        # rolls to the next one (which it may do early under
        # dynamic-epoch). Next tick will see the new challenge_epoch_id.
        if self._last_scored_challenge_epoch_id == challenge_epoch_id:
            return

        # Mid-epoch budget gate (docs/API.md "Mid-epoch start"). When
        # there isn't enough chain time left, skip this epoch.
        if not self._epoch_has_budget(resp, epoch, challenge_epoch_id):
            return

        commitment = self._commitment_from_epoch(epoch)
        if commitment is None:
            logger.warning(
                "epoch %s missing challenger fields; skipping",
                challenge_epoch_id,
            )
            return

        # Resolve the active scoring spec for this epoch. The server dictates
        # it (epoch.spec_number, resolved from the web spec_schedule), so the
        # validator auto-switches scenario sets at the scheduled cutover epoch
        # with no redeploy. Fall back to this build's max-known spec when the
        # field is absent (older web). If the server's spec is newer than this
        # build knows, abstain — operators must upgrade before the schedule's
        # effective_epoch; a stale-spec submission is dropped by the web spec
        # floor anyway.
        try:
            spec = int(epoch.get("spec_number", LATEST_SPEC))
        except (TypeError, ValueError):
            spec = LATEST_SPEC
        scenarios = scenarios_for_spec(spec)
        if scenarios is None:
            logger.error(
                "epoch %d requires spec %d but this validator build only "
                "knows up to spec %d — upgrade the validator. Abstaining "
                "(no submission) until then.",
                challenge_epoch_id, spec, LATEST_SPEC,
            )
            return

        await self._score_challenger(
            challenge_epoch_id, commitment, spec, scenarios,
        )

    async def _refresh_winner_cache(self):
        """Pull /api/v2/winner/current, run local derivation, persist cache."""
        winner_resp = await fetch_current_winner()
        if winner_resp is None:
            return  # transport error; keep the previous cache + TTL countdown

        new_winner = derive_winner_state(winner_resp)
        self._winner_state = new_winner
        try:
            save_winner_state(self._winner_state, self._winner_state_path)
        except Exception as e:
            logger.warning("Failed to persist winner cache: %s", e)

    async def _score_challenger(
        self, challenge_epoch_id: int, commitment: MinerCommitment,
        spec: int, scenarios: tuple,
    ):
        eval_id = (
            datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            + f"_e{challenge_epoch_id}"
        )

        # Begin cycle log capture
        self._cycle_eval_id = eval_id
        self._cycle_log_offset = self._get_validator_log_offset()
        block_height = 0
        try:
            block_height = int(self.subtensor.get_current_block() or 0)
        except Exception as e:
            logger.debug("Could not read current block for eval: %s", e)
        self._cycle_log_block = block_height

        # Per-miner log capture
        eval_dir, vlog_offset, mlog_offset = self._prepare_eval_log_capture(
            eval_id, commitment.hotkey,
        )

        eval_result = await self._evaluate_challenger(
            commitment, challenge_epoch_id, spec, scenarios,
        )

        # Harness aborted mid-session because the epoch had moved on.
        # Drop the eval entirely — POSTing the partial sum would punish
        # the miner for scenarios we never ran (server-side 409s most of
        # these anyway, but the intent here is "no submission", not "low
        # submission that hopefully gets rejected").
        if eval_result and eval_result.get("skip_reason") == SKIP_EPOCH_CHANGED:
            logger.info(
                "Discarding eval for epoch %d: harness aborted mid-session "
                "(epoch moved on); no score submitted",
                challenge_epoch_id,
            )
            return

        # Infra-failure discard: when hermes itself failed on every
        # scenario this session (non-zero exits / deadline kills, no
        # billed LLM call anywhere), the per-scenario qualities are
        # baseline credit on empty agent outputs — not a miner signal.
        # POSTing that score would mark this miner as broken on our
        # behalf and pollute consensus. Skip the submission entirely;
        # the validator picks up the next challenger on the next eval
        # tick, and once the operator restores the LLM the evaluation
        # resumes normally.
        if (
            eval_result
            and eval_result.get("skip_reason") == SKIP_PROVIDER_FAILURE
        ):
            logger.error(
                "Discarding eval for epoch %d: hermes failed on every "
                "scenario (LLM round-trip never produced output, no "
                "billing). Validator infrastructure is broken — check "
                "LLM key/credits/network. No score submitted.",
                challenge_epoch_id,
            )
            return

        rejected = False
        rejection_detail: Optional[str] = None
        score = 0.0
        qualified = False
        scenario_results: Dict[str, Any] = {}

        if eval_result is None or not eval_result.get("success"):
            rejected = True
            rejection_detail = (
                eval_result.get("skip_reason") if eval_result else "evaluation_skipped"
            )
            # Pack-verify failures from the helper are silent skips in
            # v5.x; in v6 we still surface them as rejected so the server
            # has an explicit verdict for this validator-epoch pair.
            if rejection_detail == SKIP_PACK_VERIFY:
                rejection_detail = "pack_verify_failed"
        else:
            score, qualified, scenario_results = self._summarize_eval(eval_result)

        ok = await submit_challenge_score(
            self.wallet,
            challenge_epoch_id=challenge_epoch_id,
            score=score,
            qualified=qualified,
            rejected=rejected if rejected else None,
            rejection_detail=rejection_detail if rejected else None,
            scenario_results=scenario_results or None,
            spec_number=spec,
            llm_base_url=self.config.llm_base_url,
            llm_model=self.config.llm_model,
            judge_model=self.config.judge_model or None,
            **self._harness_metadata(),
        )

        if ok:
            self._last_scored_challenge_epoch_id = challenge_epoch_id
            self._last_eval_at = int(time.time())
            self._save_eval_state()

        # Fire-and-forget log uploads
        eval_scenarios = list(scenario_results.keys()) if scenario_results else []
        try:
            await self._fire_upload_eval_logs(
                eval_id, commitment.uid, commitment, eval_scenarios,
                eval_result, eval_dir, vlog_offset, mlog_offset,
                block_height, challenge_epoch_id,
            )
        except Exception as e:
            logger.warning("Eval log upload failed: %s", e)

        try:
            await self._fire_upload_cycle_logs(
                eval_id, self._cycle_log_offset, self._cycle_log_block,
                challenge_epoch_id,
            )
        except Exception as e:
            logger.warning("Cycle log upload failed: %s", e)

    # ------------------------------------------------------------------
    # Weight loop
    # ------------------------------------------------------------------

    def _should_set_weights(self, current_block: int) -> bool:
        """Tempo-gated check: enough blocks elapsed since last set_weights."""
        if self._last_set_weights_block <= 0:
            return True
        elapsed = current_block - self._last_set_weights_block
        return elapsed >= self.config.weight_interval_blocks

    async def _weight_loop(self):
        """Periodically drive on-chain set_weights from the winner cache."""
        logger.info(
            "Weight loop started (check interval %ds, tempo gate %d blocks)",
            _WEIGHT_CHECK_INTERVAL, self.config.weight_interval_blocks,
        )
        while True:
            try:
                await self._weight_loop_tick()
            except Exception as e:
                logger.exception("Weight loop tick failed: %s", e)
            await asyncio.sleep(_WEIGHT_CHECK_INTERVAL)

    async def _weight_loop_tick(self):
        try:
            current_block = int(self.subtensor.get_current_block() or 0)
        except Exception as e:
            logger.warning(
                "Could not read current block for set_weights gate: %s", e,
            )
            return

        if not self._should_set_weights(current_block):
            return

        if not self._winner_state.is_seated:
            await self._set_fallback_weights(reason="No winner seated yet")
            self._last_set_weights_block = current_block
            self._save_eval_state()
            return

        if not self._winner_state.is_fresh(WINNER_FALLBACK_TTL_SECONDS):
            logger.error(
                "Winner cache stale (>%ds since last server refresh); "
                "refusing set_weights and burning to owner UID",
                WINNER_FALLBACK_TTL_SECONDS,
            )
            await self._set_burn_weights(reason="winner_cache_stale")
            self._last_set_weights_block = current_block
            self._save_eval_state()
            return

        await self._set_winner_weights()
        self._last_set_weights_block = current_block
        self._save_eval_state()

    # ------------------------------------------------------------------
    # set_weights
    # ------------------------------------------------------------------

    async def _do_set_weights(
        self,
        uids: list,
        weights: list,
        *,
        label: str = "",
    ) -> bool:
        """Call subtensor.set_weights with retry on failure."""
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
                logger.info("%sRetrying set_weights in %ds...", label, delay)
                await asyncio.sleep(delay)

        logger.error(
            "%sset_weights FAILED after %d attempts (uids=%s)",
            label, _SET_WEIGHTS_MAX_RETRIES, uids,
        )
        return False

    async def _set_winner_weights(self):
        """Drive set_weights from the cached winner."""
        winner_hk = self._winner_state.winner_hotkey
        winner_uid = self._winner_state.winner_uid

        if not winner_hk or winner_uid is None:
            await self._set_fallback_weights(reason="No winner in cache")
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
        """Read on-chain weights set by OWNER_UID and return (uids, weights)."""
        if not self._is_metagraph_healthy():
            logger.warning(
                "Cannot read owner weights: metagraph unhealthy (n=%d). "
                "Attempting re-sync...",
                getattr(self.metagraph, "n", 0),
            )
            if not self._sync_metagraph(caller="fallback_owner_weights"):
                return None

        try:
            W = self.metagraph.W
            if OWNER_UID >= W.shape[0]:
                logger.warning(
                    "OWNER_UID %d out of range (metagraph size %d)",
                    OWNER_UID, W.shape[0],
                )
                return None

            owner_weights = W[OWNER_UID]
            uids: list = []
            weights: list = []
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
        """Set weights when no winner exists. Always call set_weights."""
        try:
            _ = self.wallet.hotkey
        except Exception:
            logger.debug("Skipping fallback weights: wallet hotkey not available")
            return

        if not self._is_metagraph_healthy():
            logger.error(
                "METAGRAPH UNHEALTHY (n=%d) during fallback weight setting. "
                "Reason: %s",
                getattr(self.metagraph, "n", 0), reason,
            )
            self._sync_metagraph(caller="fallback_weights")

        copied = self._fallback_owner_weights()
        if copied is not None:
            uids, weights = copied
            logger.info(
                "%s — copying weights from owner UID %d (%d entries)",
                reason, OWNER_UID, len(uids),
            )
        else:
            uids, weights = self._fallback_to_owner()
            logger.info(
                "%s — burning to owner UID %d (uids=%s, weights=%s)",
                reason, OWNER_UID, uids, weights,
            )

        await self._do_set_weights(
            uids, weights, label=f"[fallback: {reason}] ",
        )

    async def _set_burn_weights(self, reason: str) -> None:
        uids, weights = self._fallback_to_owner()
        await self._do_set_weights(
            uids, weights, label=f"[burn: {reason}] ",
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self):
        """Run eval loop, weight loop, and heartbeat loop concurrently."""
        try:
            await self._replay_pending_uploads()
        except Exception as e:
            logger.warning("Startup log replay failed: %s", e)

        await asyncio.gather(
            self._eval_loop(),
            self._weight_loop(),
            self._heartbeat_loop(),
        )


async def main():
    """Entry point for validator."""
    config = ValidatorConfig.from_env()
    validator = TrajectoryValidator(config)
    await validator.run()


if __name__ == "__main__":
    asyncio.run(main())
