"""TrajectoryRL Validator — v6.0 winner-challenger daemon.

Architecture (Season 1 — trajrl-bench, v6.0 IM):
    1. Polls ``GET /api/v2/epoch/current`` (~30 s) for the active
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
import os
import time

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt

from ..utils.config import ValidatorConfig
from ..utils.sandbox_harness import TrajectorySandboxHarness
from ..utils.github import PackFetcher
from ..utils import config as _config_mod
from ..utils.commitments import MinerCommitment
from ..utils.miner_eval import evaluate_miner_s1, SKIP_PACK_VERIFY
from ..utils.status_reporter import (
    heartbeat,
    fetch_current_epoch,
    submit_challenge_score,
    upload_eval_logs,
    upload_cycle_logs,
)
from ..utils.winner_state import (
    WinnerState,
    winner_from_server_block,
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
BURN_FRACTION = 0.95  # 95% of miner emissions burned via owner UID

_METAGRAPH_SYNC_RETRIES = 3
_METAGRAPH_SYNC_DELAY = 10  # seconds between retries
_METAGRAPH_MIN_NEURONS = 1

_SET_WEIGHTS_MAX_RETRIES = 3
_SET_WEIGHTS_RETRY_DELAY = 12  # seconds; roughly 1 block interval

# Daemon cadence (v6.0)
_EPOCH_POLL_INTERVAL = 30        # seconds between GET /api/v2/epoch/current
_WEIGHT_CHECK_INTERVAL = 300     # seconds between weight-set attempts
_HEARTBEAT_INTERVAL = 600        # seconds between heartbeats


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
        """bench/harness image hashes + bench version for outgoing payloads."""
        h = self._sandbox_harness
        meta: Dict[str, str] = {}
        if h.bench_image_hash != "unknown":
            meta["bench_image_hash"] = h.bench_image_hash
        if h.scenario_image_hash != "unknown":
            meta["harness_image_hash"] = h.scenario_image_hash
        if h.sandbox_version != "unknown":
            meta["bench_version"] = h.sandbox_version
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
        pack_url = epoch_block.get("challenger_pack_url") or epoch_block.get("pack_url")
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
    ) -> Optional[Dict[str, Any]]:
        """Run a single trajrl-bench evaluation on the active challenger."""
        mlog = self._get_miner_logger(commitment.hotkey)
        mlog.info(
            f"Evaluating challenger uid={commitment.uid} "
            f"hotkey={commitment.hotkey[:8]} "
            f"hash={commitment.pack_hash[:12]}... "
            f"epoch_id={challenge_epoch_id}"
        )

        outcome = await evaluate_miner_s1(
            harness=self._sandbox_harness,
            pack_fetcher=self.pack_fetcher,
            commitment=commitment,
            epoch_seed=self._challenge_seed(challenge_epoch_id, self.config.netuid),
            validator_salt=self._default_validator_salt(),
            mlog=mlog,
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
        """Poll /api/v2/epoch/current; eval the challenger; submit score."""
        logger.info("Eval loop started (poll interval %ds)", _EPOCH_POLL_INTERVAL)
        while True:
            try:
                await self._eval_loop_tick()
            except Exception as e:
                logger.exception("Eval loop tick failed: %s", e)
            await asyncio.sleep(_EPOCH_POLL_INTERVAL)

    async def _eval_loop_tick(self):
        resp = await fetch_current_epoch()
        if resp is None:
            return  # transient error, retry next tick

        # Refresh the winner cache on every successful poll. Server is
        # canonical; we overwrite local state regardless of staleness.
        winner_block = resp.get("winner")
        new_winner = winner_from_server_block(winner_block)
        self._winner_state = new_winner
        try:
            save_winner_state(self._winner_state, self._winner_state_path)
        except Exception as e:
            logger.warning("Failed to persist winner cache: %s", e)

        epoch = resp.get("epoch")
        if not epoch:
            return  # no in-progress epoch; just keep polling

        try:
            challenge_epoch_id = int(epoch.get("challenge_epoch_id"))
        except (TypeError, ValueError):
            logger.warning("epoch missing or invalid challenge_epoch_id: %r", epoch)
            return

        if self._last_scored_challenge_epoch_id == challenge_epoch_id:
            return  # idempotent: already evaluated this epoch

        commitment = self._commitment_from_epoch(epoch)
        if commitment is None:
            logger.warning(
                "epoch %s missing challenger fields; skipping",
                challenge_epoch_id,
            )
            return

        await self._score_challenger(challenge_epoch_id, commitment)

    async def _score_challenger(
        self, challenge_epoch_id: int, commitment: MinerCommitment,
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

        eval_result = await self._evaluate_challenger(commitment, challenge_epoch_id)

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
            spec_number=_spec_number(),
            llm_base_url=self.config.llm_base_url,
            llm_model=self.config.llm_model,
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
