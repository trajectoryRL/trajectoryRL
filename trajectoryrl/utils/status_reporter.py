"""Validator status reporting to the TrajectoryRL web dashboard."""

import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

from trajectoryrl import __version__

logger = logging.getLogger(__name__)

_BASE_URL = os.getenv("TRAJECTORYRL_API_BASE_URL", "https://trajrl.com")
DEFAULT_HEARTBEAT_URL = f"{_BASE_URL}/api/v2/validators/heartbeat"
DEFAULT_SUBMIT_URL = f"{_BASE_URL}/api/v2/scores/submit"
DEFAULT_EPOCH_SNAPSHOT_URL = f"{_BASE_URL}/api/v2/validators/epoch_snapshot"
DEFAULT_LOGS_UPLOAD_URL = f"{_BASE_URL}/api/validators/logs/upload"
DEFAULT_CYCLE_LOGS_URL = f"{_BASE_URL}/api/validators/logs/cycle"


async def heartbeat(
    wallet,
    *,
    heartbeat_url: str = DEFAULT_HEARTBEAT_URL,
    last_set_weights_at: Optional[int] = None,
    last_eval_at: Optional[int] = None,
    bench_image_hash: Optional[str] = None,
    harness_image_hash: Optional[str] = None,
    bench_version: Optional[str] = None,
) -> bool:
    """Send a validator heartbeat to the dashboard API (v2).

    Args:
        wallet: bt.Wallet with accessible hotkey for signing.
        heartbeat_url: Dashboard heartbeat endpoint.
        last_set_weights_at: Unix timestamp of most recent set_weights call.
        last_eval_at: Unix timestamp of most recent completed full eval cycle.
        bench_image_hash: Docker image digest of the trajrl-bench sandbox.
        harness_image_hash: Docker image digest of the hermes-agent harness.
        bench_version: Version string reported by the trajrl-bench CLI.

    Returns:
        True on success (HTTP 200), False otherwise.
    """
    try:
        hotkey_kp = wallet.hotkey
    except Exception:
        logger.debug("Skipping heartbeat: wallet hotkey not available")
        return False
    hotkey_addr = hotkey_kp.ss58_address
    timestamp = int(time.time())

    message = f"trajectoryrl-heartbeat:{hotkey_addr}:{timestamp}"
    sig = hotkey_kp.sign(message.encode())
    signature = "0x" + (sig if isinstance(sig, bytes) else bytes(sig)).hex()

    payload: Dict[str, Any] = {
        "hotkey": hotkey_addr,
        "version": __version__,
        "timestamp": timestamp,
        "signature": signature,
    }
    if last_set_weights_at is not None:
        payload["last_set_weights_at"] = last_set_weights_at
    if last_eval_at is not None:
        payload["last_eval_at"] = last_eval_at
    if bench_image_hash is not None:
        payload["bench_image_hash"] = bench_image_hash
    if harness_image_hash is not None:
        payload["harness_image_hash"] = harness_image_hash
    if bench_version is not None:
        payload["bench_version"] = bench_version

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(heartbeat_url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.debug("Heartbeat sent (hotkey=%s...)", hotkey_addr[:8])
            return True
        logger.warning(
            "Heartbeat failed: %d %s", resp.status_code, resp.text[:200]
        )
        return False
    except Exception as e:
        logger.warning("Heartbeat error: %s", e)
        return False


class SnapshotNotReady(Exception):
    """Raised when /api/v2/validators/epoch_snapshot returns 404 for the
    requested epoch (sync worker hasn't built it yet). Caller should
    retry on the next loop iteration without treating this as a hard
    failure."""


async def fetch_epoch_snapshot(
    epoch_number: int,
    wallet,
    *,
    snapshot_url: str = DEFAULT_EPOCH_SNAPSHOT_URL,
    timeout: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Fetch the eval target set for ``epoch_number`` from trajrl.com.

    The endpoint is the validator's only source for the eval set: it
    absorbs both the on-chain commitment query and per-miner pre-eval
    (each entry comes back with ``pre_eval_status`` baked in). The
    snapshot is precomputed by the sync worker and frozen — every
    validator gets byte-identical bytes for the same epoch.

    Args:
        epoch_number: Epoch (window) number for which to fetch the
            snapshot. Non-negative integer.
        wallet: bt.Wallet with accessible hotkey for signing. Signing
            is required by the endpoint because the response includes
            sensitive ``pack_url`` values for web-source submissions.
        snapshot_url: Endpoint URL (override for tests).
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed response dict on HTTP 200 (with ``entries`` list, etc.).
        ``None`` for any other outcome — 404 (snapshot not ready),
        network error, auth/signature error, or unparseable JSON. The
        caller is expected to retry on its next loop iteration.

    Raises:
        SnapshotNotReady: When the server replies 404. Distinct from
            ``None`` so callers that want to log "still building" vs.
            "transient error" can differentiate. Currently the
            validator path treats both as "skip this cycle, retry."
    """
    try:
        hotkey_kp = wallet.hotkey
    except Exception:
        logger.error("fetch_epoch_snapshot: wallet hotkey not available")
        return None
    hotkey_addr = hotkey_kp.ss58_address
    timestamp = int(time.time())

    # The endpoint expects its own dedicated signing prefix
    # (``trajectoryrl-snapshot``); other v2 endpoints use different
    # prefixes (e.g. /api/v2/scores/submit uses ``trajectoryrl-submit``,
    # /api/v2/validators/heartbeat uses ``trajectoryrl-heartbeat``).
    message = f"trajectoryrl-snapshot:{hotkey_addr}:{timestamp}"
    sig = hotkey_kp.sign(message.encode())
    signature = "0x" + (sig if isinstance(sig, bytes) else bytes(sig)).hex()

    payload: Dict[str, Any] = {
        "epoch_number": epoch_number,
        "validator_hotkey": hotkey_addr,
        "timestamp": timestamp,
        "signature": signature,
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(snapshot_url, json=payload, timeout=timeout)
    except Exception as e:
        logger.warning("fetch_epoch_snapshot error for epoch %d: %s", epoch_number, e)
        return None

    if resp.status_code == 200:
        try:
            data = resp.json()
        except Exception as e:
            logger.warning(
                "fetch_epoch_snapshot epoch %d: response not JSON-decodable: %s",
                epoch_number, e,
            )
            return None
        if not isinstance(data, dict) or "entries" not in data:
            logger.warning(
                "fetch_epoch_snapshot epoch %d: unexpected payload shape (keys=%s)",
                epoch_number,
                list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
            return None
        return data

    if resp.status_code == 404:
        logger.info(
            "fetch_epoch_snapshot epoch %d: snapshot not yet built (HTTP 404)",
            epoch_number,
        )
        raise SnapshotNotReady(epoch_number)

    logger.warning(
        "fetch_epoch_snapshot epoch %d: HTTP %d %s",
        epoch_number, resp.status_code, resp.text[:200],
    )
    return None


async def upload_eval_logs(
    wallet,
    *,
    eval_id: str,
    miner_hotkey: str,
    miner_uid: int,
    block_height: int,
    pack_hash: str,
    log_archive: bytes,
    epoch_number: Optional[int] = None,
    upload_url: str = DEFAULT_LOGS_UPLOAD_URL,
) -> bool:
    """Upload eval log archive to the dashboard API.

    Fire-and-forget: failures are logged and silently discarded.
    Must never block or affect the validator's eval loop.

    Args:
        wallet: bt.Wallet with accessible hotkey for signing.
        eval_id: Eval run identifier (e.g. "20260329_1430_w42").
        miner_hotkey: Hotkey of the evaluated miner.
        miner_uid: UID of the evaluated miner.
        block_height: Block height of this eval.
        pack_hash: SHA-256 hash of the pack being evaluated.
        log_archive: Gzipped tar archive bytes containing eval log files.
        epoch_number: Optional eval window number. Sent as a string in the
            multipart form so the server can store it on the eval_logs row
            without depending on its own block_height -> epoch derivation.
        upload_url: Dashboard log upload endpoint.

    Returns:
        True on success (HTTP 200), False otherwise.
    """
    try:
        hotkey_kp = wallet.hotkey
    except Exception:
        logger.debug("Skipping log upload: wallet hotkey not available")
        return False
    hotkey_addr = hotkey_kp.ss58_address
    timestamp = int(time.time())

    message = f"trajectoryrl-logs:{hotkey_addr}:{timestamp}"
    sig = hotkey_kp.sign(message.encode())
    signature = "0x" + (sig if isinstance(sig, bytes) else bytes(sig)).hex()

    form_data: Dict[str, str] = {
        "validator_hotkey": hotkey_addr,
        "eval_id": eval_id,
        "miner_hotkey": miner_hotkey,
        "miner_uid": str(miner_uid),
        "block_height": str(block_height),
        "pack_hash": pack_hash,
        "timestamp": str(timestamp),
        "signature": signature,
    }
    if epoch_number is not None:
        form_data["epoch_number"] = str(epoch_number)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                upload_url,
                data=form_data,
                files={
                    "log_archive": (
                        "logs.tar.gz",
                        log_archive,
                        "application/gzip",
                    ),
                },
                timeout=30,
            )
        if resp.status_code == 200:
            logger.debug(
                "Eval logs uploaded (miner=%s...)", miner_hotkey[:8]
            )
            return True
        logger.warning(
            "Eval log upload failed: %d %s",
            resp.status_code,
            resp.text[:200],
        )
        return False
    except Exception as e:
        logger.warning("Eval log upload error: %s", e)
        return False


async def upload_cycle_logs(
    wallet,
    *,
    eval_id: str,
    block_height: int,
    log_archive: bytes,
    epoch_number: Optional[int] = None,
    upload_url: str = DEFAULT_CYCLE_LOGS_URL,
) -> bool:
    """Upload eval cycle log archive to the dashboard API.

    Fire-and-forget: failures are logged and silently discarded.
    Must never block or affect the validator's eval loop.

    Args:
        wallet: bt.Wallet with accessible hotkey for signing.
        eval_id: Eval cycle identifier (e.g. "20260329_1430_w42").
        block_height: Block height at cycle start.
        log_archive: Gzipped tar archive bytes containing the cycle log.
        epoch_number: Optional eval window number. Sent as a string in the
            multipart form so the server can store it on the eval_logs row
            without depending on its own block_height -> epoch derivation.
        upload_url: Dashboard cycle log upload endpoint.

    Returns:
        True on success (HTTP 200), False otherwise.
    """
    try:
        hotkey_kp = wallet.hotkey
    except Exception:
        logger.debug("Skipping cycle log upload: wallet hotkey not available")
        return False
    hotkey_addr = hotkey_kp.ss58_address
    timestamp = int(time.time())

    message = f"trajectoryrl-logs:{hotkey_addr}:{timestamp}"
    sig = hotkey_kp.sign(message.encode())
    signature = "0x" + (sig if isinstance(sig, bytes) else bytes(sig)).hex()

    form_data: Dict[str, str] = {
        "validator_hotkey": hotkey_addr,
        "eval_id": eval_id,
        "block_height": str(block_height),
        "timestamp": str(timestamp),
        "signature": signature,
    }
    if epoch_number is not None:
        form_data["epoch_number"] = str(epoch_number)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                upload_url,
                data=form_data,
                files={
                    "log_archive": (
                        "cycle.tar.gz",
                        log_archive,
                        "application/gzip",
                    ),
                },
                timeout=30,
            )
        if resp.status_code == 200:
            logger.debug("Cycle logs uploaded")
            return True
        logger.warning(
            "Cycle log upload failed: %d %s",
            resp.status_code,
            resp.text[:200],
        )
        return False
    except Exception as e:
        logger.warning("Cycle log upload error: %s", e)
        return False


async def submit_eval(
    wallet,
    *,
    miner_hotkey: str,
    miner_uid: int,
    block_height: int,
    score: float,
    weight: float,
    qualified: bool,
    pack_url: Optional[str] = None,
    pack_hash: Optional[str] = None,
    eval_count: Optional[int] = None,
    scenario_results: Optional[Dict[str, Any]] = None,
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    rejected: Optional[bool] = None,
    rejection_stage: Optional[str] = None,
    rejection_detail: Optional[str] = None,
    spec_number: Optional[int] = None,
    bench_image_hash: Optional[str] = None,
    harness_image_hash: Optional[str] = None,
    bench_version: Optional[str] = None,
    epoch_number: Optional[int] = None,
    submit_url: str = DEFAULT_SUBMIT_URL,
) -> bool:
    """Submit a single miner eval result to the dashboard API.

    Fire-and-forget: failures are logged and silently discarded.
    Must never block or affect the validator's eval loop.

    Returns:
        True on success (HTTP 200), False otherwise.
    """
    try:
        hotkey_kp = wallet.hotkey
    except Exception:
        logger.debug("Skipping eval submit: wallet hotkey not available")
        return False
    hotkey_addr = hotkey_kp.ss58_address
    timestamp = int(time.time())

    message = f"trajectoryrl-submit:{hotkey_addr}:{timestamp}"
    sig = hotkey_kp.sign(message.encode())
    signature = "0x" + (sig if isinstance(sig, bytes) else bytes(sig)).hex()

    payload: Dict[str, Any] = {
        "validator_hotkey": hotkey_addr,
        "miner_hotkey": miner_hotkey,
        "miner_uid": miner_uid,
        "block_height": block_height,
        "timestamp": timestamp,
        "signature": signature,
        "version": __version__,
        "score": score,
        "weight": weight,
        "qualified": qualified,
    }
    if pack_url is not None:
        payload["pack_url"] = pack_url
    if pack_hash is not None:
        payload["pack_hash"] = pack_hash
    if eval_count is not None:
        payload["eval_count"] = eval_count
    if scenario_results is not None:
        payload["scenario_results"] = scenario_results
    if llm_base_url is not None:
        payload["llm_base_url"] = llm_base_url
    if llm_model is not None:
        payload["llm_model"] = llm_model
    if rejected is not None:
        payload["rejected"] = rejected
    if rejection_stage is not None:
        payload["rejection_stage"] = rejection_stage
    if rejection_detail is not None:
        payload["rejection_detail"] = rejection_detail
    if spec_number is not None:
        # Emit both keys for backward compatibility with the central status
        # service while it migrates to the new column name.
        payload["scoring_version"] = spec_number
        payload["spec_number"] = spec_number
    if bench_image_hash is not None:
        payload["bench_image_hash"] = bench_image_hash
    if harness_image_hash is not None:
        payload["harness_image_hash"] = harness_image_hash
    if bench_version is not None:
        payload["bench_version"] = bench_version
    if epoch_number is not None:
        payload["epoch_number"] = epoch_number

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(submit_url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.debug(
                "Eval submitted (miner=%s...)", miner_hotkey[:8]
            )
            return True
        logger.warning(
            "Eval submit failed: %d %s", resp.status_code, resp.text[:200]
        )
        return False
    except Exception as e:
        logger.warning("Eval submit error: %s", e)
        return False
