"""HTTP client for the TrajectoryRL platform server (`trajrl.com`).

Implements the validator-side surface of the v6 winner-challenger
protocol described in ``docs/API.md``:

* ``fetch_current_epoch``  → ``GET /api/v2/epoch/current``
* ``fetch_current_winner`` → ``GET /api/v2/winner/current``
* ``submit_challenge_score`` → ``POST /api/v2/epoch/{id}/score``
* ``heartbeat``            → ``POST /api/v2/validators/heartbeat``
* ``upload_eval_logs`` / ``upload_cycle_logs`` → ``POST /api/validators/logs/{upload,cycle}``

Read endpoints are public; write endpoints sign their payload with the
hotkey via the ``_sign`` helper. Each endpoint uses a distinct signing
prefix so signatures are not interchangeable across endpoints.
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from trajectoryrl import __version__

logger = logging.getLogger(__name__)

_BASE_URL = os.getenv("TRAJECTORYRL_API_BASE_URL", "https://trajrl.com")
DEFAULT_HEARTBEAT_URL = f"{_BASE_URL}/api/v2/validators/heartbeat"
DEFAULT_EPOCH_CURRENT_URL = f"{_BASE_URL}/api/v2/epoch/current"
DEFAULT_EPOCH_SCORE_URL_TEMPLATE = f"{_BASE_URL}/api/v2/epoch/{{challenge_epoch_id}}/score"
DEFAULT_WINNER_CURRENT_URL = f"{_BASE_URL}/api/v2/winner/current"
DEFAULT_LOGS_UPLOAD_URL = f"{_BASE_URL}/api/validators/logs/upload"
DEFAULT_CYCLE_LOGS_URL = f"{_BASE_URL}/api/validators/logs/cycle"


def _sign(prefix: str, hotkey_kp, timestamp: int, *extras: str) -> Tuple[str, str, str]:
    """Build a signed message for a v2 endpoint.

    Returns (hotkey_address, message, hex_signature).
    Trailing extras are joined with ":" after the hotkey + timestamp,
    matching the format documented per-endpoint in API.md.
    """
    hotkey_addr = hotkey_kp.ss58_address
    parts = [prefix, hotkey_addr, str(timestamp), *extras]
    message = ":".join(parts)
    sig = hotkey_kp.sign(message.encode())
    signature = "0x" + (sig if isinstance(sig, bytes) else bytes(sig)).hex()
    return hotkey_addr, message, signature


async def heartbeat(
    wallet,
    *,
    heartbeat_url: str = DEFAULT_HEARTBEAT_URL,
    last_set_weights_at: Optional[int] = None,
    last_eval_at: Optional[int] = None,
    bench_image_hash: Optional[str] = None,
    harness_image_hash: Optional[str] = None,
    bench_version: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
) -> bool:
    """Send a validator heartbeat to the dashboard API (v2).

    Wire contract: ``docs/API.md`` POST /api/v2/validators/heartbeat.
    """
    try:
        hotkey_kp = wallet.hotkey
    except Exception:
        logger.debug("Skipping heartbeat: wallet hotkey not available")
        return False

    timestamp = int(time.time())
    hotkey_addr, _msg, signature = _sign(
        "trajectoryrl-heartbeat", hotkey_kp, timestamp
    )

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
    if llm_model is not None:
        payload["llm_model"] = llm_model
    if llm_base_url is not None:
        payload["llm_base_url"] = llm_base_url

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


async def fetch_current_epoch(
    wallet=None,
    *,
    epoch_url: str = DEFAULT_EPOCH_CURRENT_URL,
    timeout: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Fetch the active challenge epoch from trajrl.com.

    Public read with **optional signed auth** to unlock pack URLs.

    Per ``docs/API.md`` § ``GET /api/v2/epoch/current``, ``pack_url``
    fields (``epoch.challenger_pack_url`` and ``winner.pack_url``) are
    gated: they are returned only when the request is signed by an
    on-chain validator, OR when the epoch was opened more than 24 h
    ago. The endpoint soft-falls-through on any signature problem
    (malformed param, bad signature, non-validator hotkey) — the URLs
    just stay hidden, no 401/403 is returned.

    Pass a ``wallet`` to sign the request and unlock the URLs while
    the epoch is fresh; without it the daemon will only see the public
    fields and ``_commitment_from_epoch`` will skip evals it can't
    fetch a pack for.

    Returns the parsed response dict::

        {
          "epoch": { "challenge_epoch_id", "challenger_hotkey",
                     "challenger_pack_hash", "challenger_pack_url",
                     "start_block", "end_block",
                     "epoch_length_blocks", "status", ... } | None,
          "current_block":    int | None,   # server-stamped chain block
          "elapsed_blocks":   int | None,   # current_block - start_block
          "remaining_blocks": int | None,   # end_block - current_block
        }

    ``current_block`` / ``remaining_blocks`` are ``None`` when the
    server's chain RPC is unreachable; callers that gate on remaining
    eval budget should fall back to their own block reading in that
    case (see ``docs/API.md`` "Mid-epoch start").

    The response also carries a ``winner`` block, but v6 daemons must
    ignore it: the authoritative source of seated-winner state is
    ``GET /api/v2/winner/current`` (see ``fetch_current_winner``). The
    ``winner`` field on this endpoint is retained as a non-authoritative
    convenience for the website. See ``docs/API.md`` "Migration note".

    ``epoch`` is ``None`` when no epoch is in progress (404 from server
    or response without an epoch block). Returns ``None`` on transport /
    parse errors so callers can simply retry on the next poll.
    """
    params: Optional[Dict[str, str]] = None
    if wallet is not None:
        try:
            hotkey_kp = wallet.hotkey
        except Exception:
            hotkey_kp = None
        if hotkey_kp is not None:
            timestamp = int(time.time())
            hotkey_addr, _msg, signature = _sign(
                "trajectoryrl-epoch-current", hotkey_kp, timestamp
            )
            params = {
                "hotkey": hotkey_addr,
                "timestamp": str(timestamp),
                "signature": signature,
            }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(epoch_url, params=params, timeout=timeout)
    except Exception as e:
        logger.warning("fetch_current_epoch error: %s", e)
        return None

    if resp.status_code == 404:
        return {
            "epoch": None,
            "current_block": None,
            "elapsed_blocks": None,
            "remaining_blocks": None,
        }

    if resp.status_code != 200:
        logger.warning(
            "fetch_current_epoch: HTTP %d %s",
            resp.status_code, resp.text[:200],
        )
        return None

    try:
        data = resp.json()
    except Exception as e:
        logger.warning("fetch_current_epoch: response not JSON-decodable: %s", e)
        return None

    if not isinstance(data, dict):
        logger.warning(
            "fetch_current_epoch: unexpected payload type %s", type(data).__name__
        )
        return None

    def _opt_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return {
        "epoch": data.get("epoch"),
        "current_block": _opt_int(data.get("current_block")),
        "elapsed_blocks": _opt_int(data.get("elapsed_blocks")),
        "remaining_blocks": _opt_int(data.get("remaining_blocks")),
    }


async def fetch_current_winner(
    *,
    winner_url: str = DEFAULT_WINNER_CURRENT_URL,
    timeout: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Fetch the seated-winner inputs from trajrl.com.

    Public read endpoint — no signing required. The response carries the
    raw inputs needed for **local** winner derivation: per-validator
    submissions with ``validator_stake`` snapshotted at score-POST time,
    plus the server's advisory ``winner`` claim for cross-checking.

    Shape::

        {
          "winner": { "hotkey", "uid", "pack_hash", "pack_url",
                      "score", "since_epoch_id" } | None,
          "finalized_epoch": { "challenge_epoch_id", "challenger_hotkey",
                               "challenger_pack_hash",
                               "outcome", "winner_replaced",
                               "finalized_at" } | None,
          "submissions": [
            { "validator_hotkey", "validator_stake",
              "challenger_pack_hash", "challenger_score",
              "challenger_qualified", "challenger_rejected",
              "winner_pack_hash", "winner_score",
              "winner_qualified", "winner_rejected" }, ...
          ],
        }

    Cold start (no finalized epoch yet) has all three fields null /
    empty. The caller (typically ``derive_winner_state``) is responsible
    for running the local aggregation and comparing to ``response.winner``.

    Returns ``None`` on transport / parse errors so callers can simply
    retry on the next poll.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(winner_url, timeout=timeout)
    except Exception as e:
        logger.warning("fetch_current_winner error: %s", e)
        return None

    if resp.status_code == 404:
        return {"winner": None, "finalized_epoch": None, "submissions": []}

    if resp.status_code != 200:
        logger.warning(
            "fetch_current_winner: HTTP %d %s",
            resp.status_code, resp.text[:200],
        )
        return None

    try:
        data = resp.json()
    except Exception as e:
        logger.warning("fetch_current_winner: response not JSON-decodable: %s", e)
        return None

    if not isinstance(data, dict):
        logger.warning(
            "fetch_current_winner: unexpected payload type %s", type(data).__name__
        )
        return None

    return {
        "winner": data.get("winner"),
        "finalized_epoch": data.get("finalized_epoch"),
        "submissions": data.get("submissions") or [],
    }


async def submit_challenge_score(
    wallet,
    *,
    challenge_epoch_id: int,
    score: float,
    qualified: bool,
    rejected: Optional[bool] = None,
    rejection_detail: Optional[str] = None,
    scenario_results: Optional[Dict[str, Any]] = None,
    spec_number: Optional[int] = None,
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    bench_image_hash: Optional[str] = None,
    harness_image_hash: Optional[str] = None,
    bench_version: Optional[str] = None,
    submit_url_template: str = DEFAULT_EPOCH_SCORE_URL_TEMPLATE,
    timeout: float = 30.0,
) -> bool:
    """Post the validator's signed score for one challenge epoch.

    Implements ``POST /api/v2/epoch/{challenge_epoch_id}/score``. The
    signing prefix is ``trajectoryrl-challenge-score`` and includes the
    epoch id, making the signature replay-safe across epochs.

    The body carries a ``challenger`` block (always populated). Server
    stamps ``challenger_hotkey`` / ``challenger_pack_hash`` from the
    epoch row, so the validator only sends the eval-derived fields.

    Returns True on HTTP 200, False on any failure. Callers should not
    treat failures as fatal — the score loop retries on the next epoch
    finalize cycle if needed.
    """
    try:
        hotkey_kp = wallet.hotkey
    except Exception:
        logger.debug("Skipping challenge score: wallet hotkey not available")
        return False

    timestamp = int(time.time())
    hotkey_addr, _msg, signature = _sign(
        "trajectoryrl-challenge-score",
        hotkey_kp,
        timestamp,
        str(challenge_epoch_id),
    )

    challenger: Dict[str, Any] = {
        "score": score,
        "qualified": qualified,
    }
    if rejected is not None:
        challenger["rejected"] = rejected
    if rejection_detail is not None:
        challenger["rejection_detail"] = rejection_detail
    if scenario_results is not None:
        challenger["scenario_results"] = scenario_results

    payload: Dict[str, Any] = {
        "validator_hotkey": hotkey_addr,
        "timestamp": timestamp,
        "signature": signature,
        "version": __version__,
        "challenger": challenger,
    }
    if spec_number is not None:
        payload["spec_number"] = spec_number
    if llm_base_url is not None:
        payload["llm_base_url"] = llm_base_url
    if llm_model is not None:
        payload["llm_model"] = llm_model
    if bench_image_hash is not None:
        payload["bench_image_hash"] = bench_image_hash
    if harness_image_hash is not None:
        payload["harness_image_hash"] = harness_image_hash
    if bench_version is not None:
        payload["bench_version"] = bench_version

    submit_url = submit_url_template.format(challenge_epoch_id=challenge_epoch_id)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(submit_url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            logger.debug(
                "Challenge score submitted (epoch=%d, hotkey=%s...)",
                challenge_epoch_id, hotkey_addr[:8],
            )
            return True
        logger.warning(
            "Challenge score submit failed: %d %s",
            resp.status_code, resp.text[:200],
        )
        return False
    except Exception as e:
        logger.warning("Challenge score submit error: %s", e)
        return False


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
    """
    try:
        hotkey_kp = wallet.hotkey
    except Exception:
        logger.debug("Skipping log upload: wallet hotkey not available")
        return False

    timestamp = int(time.time())
    hotkey_addr, _msg, signature = _sign(
        "trajectoryrl-logs", hotkey_kp, timestamp
    )

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
    """
    try:
        hotkey_kp = wallet.hotkey
    except Exception:
        logger.debug("Skipping cycle log upload: wallet hotkey not available")
        return False

    timestamp = int(time.time())
    hotkey_addr, _msg, signature = _sign(
        "trajectoryrl-logs", hotkey_kp, timestamp
    )

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
