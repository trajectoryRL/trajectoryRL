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
DEFAULT_PRE_EVAL_URL = f"{_BASE_URL}/api/v2/miners/pre-eval"
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


# Cache keyed by (miner_hotkey, identifier) → last valid pre-eval response.
_pre_eval_cache: Dict[tuple, Dict[str, Any]] = {}


def _is_valid_pre_eval_response(data: Any) -> bool:
    """Check that the response has the expected pre-eval format."""
    return isinstance(data, dict) and "allowed" in data


async def pre_eval(
    miner_hotkey: str,
    pack_hash: Optional[str] = None,
    pack_url: Optional[str] = None,
    *,
    epoch_number: Optional[int] = None,
    wallet=None,
    pre_eval_url: str = DEFAULT_PRE_EVAL_URL,
) -> Optional[Dict[str, Any]]:
    """Call the pre-eval API to check whether a miner's submission is allowed.

    Supports two query modes:
      - By pack_hash: checks a specific pack (used during per-miner evaluation)
      - By epoch_number: server resolves the most recently submitted pack in
        that epoch (used during aggregation to avoid pack-switch escapes)

    At least one of pack_hash or epoch_number must be provided.

    Signing is optional — the v2 endpoint is read-only. When wallet is
    provided, the request is signed for auditability.

    Uses a local cache so that when the API is unreachable or returns an
    unexpected format, the last known-good response is used instead of
    blindly failing open.

    Args:
        miner_hotkey: Miner hotkey to check.
        pack_hash: Pack hash submitted by the miner. Required if
            epoch_number is not provided.
        pack_url: Pack download URL (optional context for the server).
        epoch_number: Epoch/window number. When provided, the server
            returns the eval result for the miner's most recently
            submitted pack in that epoch.
        wallet: bt.Wallet with accessible hotkey for signing (optional).
        pre_eval_url: Pre-eval API endpoint.

    Returns:
        Parsed response dict on success or from cache, or None only if the
        request failed AND no cached result exists (fail-open).
    """
    if pack_hash is None and epoch_number is None:
        logger.warning("pre_eval: neither pack_hash nor epoch_number provided — failing open")
        return None

    cache_key = (miner_hotkey, pack_hash or f"epoch:{epoch_number}")

    payload: Dict[str, Any] = {
        "miner_hotkey": miner_hotkey,
    }
    if pack_hash is not None:
        payload["pack_hash"] = pack_hash
    if epoch_number is not None:
        payload["epoch_number"] = epoch_number
    if pack_url is not None:
        payload["pack_url"] = pack_url

    if wallet is not None:
        try:
            hotkey_kp = wallet.hotkey
            validator_hotkey = hotkey_kp.ss58_address
            timestamp = int(time.time())
            message = f"trajectoryrl-report:{validator_hotkey}:{timestamp}"
            sig = hotkey_kp.sign(message.encode())
            signature = "0x" + (sig if isinstance(sig, bytes) else bytes(sig)).hex()
            payload["validator_hotkey"] = validator_hotkey
            payload["timestamp"] = timestamp
            payload["signature"] = signature
        except Exception:
            logger.debug("pre_eval: wallet hotkey not available, sending unsigned")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(pre_eval_url, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if _is_valid_pre_eval_response(data):
                _pre_eval_cache[cache_key] = data
                logger.info(
                    "Pre-eval check passed: allowed=%s reason=%s",
                    data.get("allowed"),
                    data.get("reason", ""),
                )
                return data
            logger.warning(
                "Pre-eval returned unexpected format (keys=%s)",
                list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
        else:
            logger.warning(
                "Pre-eval check failed: %d %s",
                resp.status_code,
                resp.text[:200],
            )
    except Exception as e:
        logger.warning("Pre-eval check error: %s", e)

    # API failed or returned bad data — fall back to cache.
    cached = _pre_eval_cache.get(cache_key)
    if cached is not None:
        identifier = pack_hash[:12] if pack_hash else f"epoch:{epoch_number}"
        logger.info(
            "Using cached pre-eval result for %s/%s: allowed=%s",
            miner_hotkey[:8],
            identifier,
            cached.get("allowed"),
        )
        return cached

    # No cache available — fail-open.
    identifier = pack_hash[:12] if pack_hash else f"epoch:{epoch_number}"
    logger.warning(
        "No cached pre-eval for %s/%s — failing open",
        miner_hotkey[:8],
        identifier,
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

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                upload_url,
                data={
                    "validator_hotkey": hotkey_addr,
                    "eval_id": eval_id,
                    "miner_hotkey": miner_hotkey,
                    "miner_uid": str(miner_uid),
                    "block_height": str(block_height),
                    "pack_hash": pack_hash,
                    "timestamp": str(timestamp),
                    "signature": signature,
                },
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

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                upload_url,
                data={
                    "validator_hotkey": hotkey_addr,
                    "eval_id": eval_id,
                    "block_height": str(block_height),
                    "timestamp": str(timestamp),
                    "signature": signature,
                },
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
