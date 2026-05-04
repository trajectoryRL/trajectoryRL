"""TrajectoryRL Miner — Pack building and on-chain submission.

Season 1 workflow:
    1. Write SKILL.md
    2. Build pack: build_s1_pack(skill_content) -> {"schema_version": 1, "files": {"SKILL.md": "..."}}
    3. Host pack.json — either:
        a. Self-hosted (S3-compatible bucket, R2, etc.), or
        b. Submit to the managed web service via ``submit_pack_via_api``
           (POST /api/v2/miners/submit) — server stores the pack at a
           random GCS key and returns the URL
    4. Submit on-chain commitment via submit_commitment(pack_hash, pack_url)

The on-chain commitment is block-timestamped, establishing first-mover
precedence. Validators read commitments and fetch packs via HTTP.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import bittensor as bt
import httpx

from ..utils.commitments import parse_commitment

logger = logging.getLogger(__name__)

MAX_COMMITMENT_BYTES = 128
MAX_PACK_SIZE = 32768

# /api/v2/miners/submit lives under the same TrajectoryRL web service
# the validator talks to (heartbeat / scores submit / epoch_snapshot).
DEFAULT_API_BASE_URL = os.environ.get(
    "TRAJECTORYRL_API_BASE_URL", "https://trajrl.com"
)
DEFAULT_MINER_SUBMIT_URL = f"{DEFAULT_API_BASE_URL}/api/v2/miners/submit"


class TrajectoryMiner:
    """TrajectoryRL miner for building and submitting policy packs.

    Example::

        miner = TrajectoryMiner(wallet_name="miner", wallet_hotkey="default")
        pack = TrajectoryMiner.build_s1_pack(open("SKILL.md").read())
        pack_hash = TrajectoryMiner.save_pack(pack, "pack.json")
        # Upload pack.json to a public URL, then:
        miner.submit_commitment(pack_hash, "https://example.com/pack.json")
    """

    def __init__(
        self,
        wallet_name: Optional[str] = None,
        wallet_hotkey: Optional[str] = None,
        netuid: Optional[int] = None,
        network: Optional[str] = None,
    ):
        self.wallet_name = wallet_name or os.environ.get("WALLET_NAME", "miner")
        self.wallet_hotkey = wallet_hotkey or os.environ.get("WALLET_HOTKEY", "default")
        self.netuid = netuid if netuid is not None else int(os.environ.get("NETUID", "11"))
        self.network = network or os.environ.get("NETWORK", "finney")

        self._wallet: Optional[bt.Wallet] = None
        self._subtensor: Optional[bt.Subtensor] = None

    @property
    def wallet(self) -> bt.Wallet:
        if self._wallet is None:
            self._wallet = bt.Wallet(
                name=self.wallet_name, hotkey=self.wallet_hotkey
            )
        return self._wallet

    @property
    def subtensor(self) -> bt.Subtensor:
        if self._subtensor is None:
            self._subtensor = bt.Subtensor(network=self.network)
        return self._subtensor

    def close(self):
        """Close the Subtensor websocket so the process can exit cleanly."""
        if self._subtensor is not None:
            try:
                self._subtensor.substrate.close()
            except Exception:
                pass
            self._subtensor = None

    # ------------------------------------------------------------------
    # Pack building (Season 1)
    # ------------------------------------------------------------------

    @staticmethod
    def build_s1_pack(skill_md: str) -> dict:
        """Build a Season 1 pack from SKILL.md content.

        Args:
            skill_md: SKILL.md content string.

        Returns:
            Pack dict: {"schema_version": 1, "files": {"SKILL.md": content}}
        """
        return {
            "schema_version": 1,
            "files": {
                "SKILL.md": skill_md,
            },
        }

    @staticmethod
    def validate_s1(pack: dict) -> List[str]:
        """Validate a pack against Season 1 requirements.

        Returns:
            List of issues (empty list = valid).
        """
        issues = []

        if pack.get("schema_version") != 1:
            issues.append(f"schema_version must be 1, got {pack.get('schema_version')}")

        files = pack.get("files")
        if not isinstance(files, dict):
            issues.append("missing or invalid 'files' dict")
            return issues

        if "SKILL.md" not in files:
            issues.append("files must contain 'SKILL.md'")
            return issues

        skill = files["SKILL.md"]
        if not isinstance(skill, str):
            issues.append("files['SKILL.md'] must be a string")
        elif not skill.strip():
            issues.append("SKILL.md must not be empty")

        size = len(json.dumps(pack, sort_keys=True))
        if size > MAX_PACK_SIZE:
            issues.append(f"pack size {size} bytes exceeds limit ({MAX_PACK_SIZE})")

        return issues

    # ------------------------------------------------------------------
    # Pack I/O
    # ------------------------------------------------------------------

    @staticmethod
    def load_pack(path: str) -> dict:
        """Load a pack.json from disk."""
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def save_pack(pack: dict, path: str) -> str:
        """Save pack to disk and return its SHA256 hash."""
        canonical = json.dumps(pack, sort_keys=True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(canonical)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def compute_pack_hash(pack: dict) -> str:
        """Compute content-addressed SHA256 hash of a pack.

        Uses ``json.dumps(pack, sort_keys=True)`` for deterministic
        serialization, matching the validator's hash computation.
        """
        canonical = json.dumps(pack, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ------------------------------------------------------------------
    # On-chain commitment
    # ------------------------------------------------------------------

    @staticmethod
    def format_commitment(
        pack_hash: str,
        pack_url: str,
    ) -> str:
        """Format a commitment string for on-chain submission.

        Returns:
            Pipe-delimited commitment string (≤128 bytes).

        Raises:
            ValueError: If inputs are invalid.
        """
        if len(pack_hash) != 64:
            raise ValueError(f"pack_hash must be 64 hex chars, got {len(pack_hash)}")
        if not pack_url.startswith(("https://", "http://")):
            raise ValueError(f"pack_url must be an HTTP(S) URL, got: {pack_url}")

        commitment = f"{pack_hash}|{pack_url}"
        if len(commitment.encode()) > MAX_COMMITMENT_BYTES:
            raise ValueError(
                f"Commitment too long: {len(commitment.encode())} bytes "
                f"(max {MAX_COMMITMENT_BYTES}). Use a shorter URL."
            )

        parsed = parse_commitment(commitment)
        if parsed is None:
            raise ValueError(f"Commitment failed round-trip validation: {commitment}")

        return commitment

    def submit_commitment(
        self,
        pack_hash: str,
        pack_url: str,
    ) -> bool:
        """Submit on-chain commitment via set_commitment.

        Returns:
            True if commitment was submitted successfully.
        """
        commitment = self.format_commitment(pack_hash, pack_url)
        logger.info(f"Submitting commitment: {commitment}")

        try:
            result = self.subtensor.set_commitment(
                wallet=self.wallet,
                netuid=self.netuid,
                data=commitment,
            )
            if result is False or (result is not None and result is not True and not result):
                logger.error("set_commitment returned failure: %s", result)
                return False

            block = self.subtensor.get_current_block()
            logger.info(
                "Commitment submitted at block %d (first-mover precedence)",
                block,
            )

            stored = self.get_current_commitment()
            if stored is None:
                logger.warning(
                    "Commitment not found on-chain after submission. "
                    "It may take a few blocks to finalize."
                )
                return False

            parsed = parse_commitment(stored)
            if parsed and parsed[0] == pack_hash:
                logger.info("On-chain verification passed: hash matches")
            else:
                logger.warning(
                    "On-chain commitment mismatch: expected hash %s..., got: %s",
                    pack_hash[:16], stored[:40] if stored else "(empty)",
                )
                return False

            return True
        except bt.NotRegisteredError:
            logger.error(
                "Miner hotkey is not registered on subnet %d. "
                "Run: btcli subnet register --netuid %d --wallet.name %s --wallet.hotkey %s",
                self.netuid, self.netuid, self.wallet_name, self.wallet_hotkey,
            )
            return False
        except bt.ChainConnectionError as e:
            logger.error(
                "Cannot connect to %s network: %s", self.network, e,
            )
            return False
        except bt.KeyFileError as e:
            logger.error("Wallet key file error: %s", e)
            return False
        except bt.ChainTransactionError as e:
            logger.error("Chain transaction failed: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to submit commitment (%s): %s", type(e).__name__, e)
            return False

    # ------------------------------------------------------------------
    # Managed web submission (POST /api/v2/miners/submit)
    # ------------------------------------------------------------------

    def submit_pack_via_api(
        self,
        pack: dict,
        *,
        submit_url: str = DEFAULT_MINER_SUBMIT_URL,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Submit a pack to the managed web service (POST /api/v2/miners/submit).

        The web service stores the pack in GCS under a random key and
        returns a public ``pack_url`` that the miner subsequently
        commits on-chain via ``submit_commitment``. This is an
        alternative to self-hosting (R2, S3, etc.) — the chain
        commitment step is identical in both cases.

        Args:
            pack: The pack dict (will be canonicalized and hashed).
            submit_url: Endpoint URL (override for tests / staging).
            timeout: HTTP timeout in seconds.

        Returns:
            The parsed server response on HTTP 200 — keys include
            ``pack_url``, ``pack_hash``, ``submission_id``,
            ``cooldown_seconds``, ``next_upload_allowed_at``,
            ``pre_eval_status``. ``None`` on any non-200 outcome
            (logged as warning/error). 429 (cooldown) is returned as
            ``None`` — inspect logs for ``next_upload_allowed_at``.

        Notes:
            * Per-miner cooldown is enforced server-side (default 1h).
              Resubmitting the same ``(hotkey, pack_hash)`` after the
              cooldown lifts is idempotent — the existing pack_url is
              reused without a fresh GCS upload.
            * Pre-eval runs asynchronously after the response. The
              miner / validators see the final verdict in subsequent
              ``/api/v2/validators/epoch_snapshot`` reads.
        """
        canonical = json.dumps(pack, sort_keys=True)
        if len(canonical.encode("utf-8")) > MAX_PACK_SIZE:
            logger.error(
                "Pack canonical size exceeds %d bytes — refusing to submit",
                MAX_PACK_SIZE,
            )
            return None

        pack_hash = hashlib.sha256(canonical.encode()).hexdigest()

        try:
            hotkey_kp = self.wallet.hotkey
        except Exception as e:
            logger.error("Cannot access wallet hotkey for signing: %s", e)
            return None
        hotkey_addr = hotkey_kp.ss58_address
        timestamp = int(time.time())

        # Endpoint-specific signing prefix per API.md § /api/v2/miners/submit.
        message = f"trajectoryrl-miner-submit:{hotkey_addr}:{timestamp}"
        sig = hotkey_kp.sign(message.encode())
        signature = "0x" + (sig if isinstance(sig, bytes) else bytes(sig)).hex()

        payload = {
            "miner_hotkey": hotkey_addr,
            "timestamp": timestamp,
            "signature": signature,
            "pack_hash": pack_hash,
            "pack_content": canonical,
        }

        logger.info(
            "Submitting pack to %s (hash=%s..., %d bytes)",
            submit_url, pack_hash[:12], len(canonical.encode("utf-8")),
        )

        try:
            with httpx.Client() as client:
                resp = client.post(submit_url, json=payload, timeout=timeout)
        except Exception as e:
            logger.error("Network error submitting pack: %s", e)
            return None

        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception as e:
                logger.error("Submit response not JSON-decodable: %s", e)
                return None
            if not isinstance(data, dict) or "pack_url" not in data:
                logger.error(
                    "Submit response shape unexpected (keys=%s)",
                    list(data.keys()) if isinstance(data, dict) else type(data).__name__,
                )
                return None
            logger.info(
                "Pack submitted successfully: submission_id=%s, "
                "cooldown_seconds=%s, pre_eval_status=%s",
                data.get("submission_id"),
                data.get("cooldown_seconds"),
                data.get("pre_eval_status"),
            )
            return data

        if resp.status_code == 429:
            try:
                data = resp.json()
                logger.error(
                    "Submission rejected (cooldown): next_upload_allowed_at=%s, "
                    "cooldown_seconds=%s",
                    data.get("next_upload_allowed_at"),
                    data.get("cooldown_seconds"),
                )
            except Exception:
                logger.error("Submission rejected (cooldown): %s", resp.text[:200])
            return None

        if resp.status_code in (400, 403):
            logger.error(
                "Submission rejected (HTTP %d): %s",
                resp.status_code, resp.text[:300],
            )
            return None

        logger.error(
            "Submission failed (HTTP %d): %s",
            resp.status_code, resp.text[:200],
        )
        return None

    def get_current_commitment(self) -> Optional[str]:
        """Read this miner's current on-chain commitment.

        Returns:
            Raw commitment string, or None if not set.
        """
        try:
            hotkey = self.wallet.hotkey.ss58_address
            commitments = self.subtensor.get_all_commitments(netuid=self.netuid)
            return commitments.get(hotkey)
        except bt.ChainConnectionError as e:
            logger.error("Cannot connect to %s network: %s", self.network, e)
            return None
        except bt.KeyFileError as e:
            logger.error("Wallet key file error: %s", e)
            return None
        except Exception as e:
            logger.error("Failed to read commitment (%s): %s", type(e).__name__, e)
            return None
