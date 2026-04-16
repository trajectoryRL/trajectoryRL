"""TrajectoryRL Miner — Pack building and on-chain submission.

Miners don't run a server. The workflow is:

Season 1 (SKILL.md — plain text):
    1. Write SKILL.md
    2. Upload SKILL.md to a public HTTP endpoint
    3. Submit on-chain commitment via set_commitment

v4.0 (OPP JSON — legacy):
    1. Write AGENTS.md (policy document)
    2. Build a pack.json (OPP v1 format)
    3. Upload pack.json to a public HTTP endpoint
    4. Submit on-chain commitment via set_commitment

The on-chain commitment is block-timestamped, establishing first-mover
precedence. Validators read commitments and fetch packs via HTTP.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import bittensor as bt

from ..utils.opp_schema import validate_opp_schema
from ..utils.commitments import parse_commitment

logger = logging.getLogger(__name__)

MAX_COMMITMENT_BYTES = 128


class TrajectoryMiner:
    """TrajectoryRL miner for building and submitting policy packs.

    Example (S1 — submit SKILL.md)::

        miner = TrajectoryMiner(wallet_name="miner", wallet_hotkey="default")
        skill_hash = TrajectoryMiner.compute_text_hash("path/to/SKILL.md")
        miner.submit_commitment(skill_hash, "https://example.com/SKILL.md")

    Example (v4.0 — submit JSON pack)::

        miner = TrajectoryMiner(wallet_name="miner", wallet_hotkey="default")
        pack = miner.build_pack(agents_md="path/to/AGENTS.md")
        miner.submit(pack, pack_url="https://trajrl.com/samples/pack.json")
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

        # Lazy-init Bittensor (only needed for on-chain operations)
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
    # Pack building
    # ------------------------------------------------------------------

    @staticmethod
    def build_pack(
        agents_md: str,
        pack_name: str = "my-pack",
        pack_version: str = "1.0.0",
        soul_md: Optional[str] = None,
        extra_files: Optional[Dict[str, str]] = None,
        tool_allow: Optional[list] = None,
        tool_deny: Optional[list] = None,
        stop_rules: Optional[list] = None,
    ) -> dict:
        """Build an OPP v1 pack from an AGENTS.md string or file path.

        Args:
            agents_md: AGENTS.md content string, or path to a file.
            pack_name: Pack name for metadata.
            pack_version: Semver string (e.g., "1.0.0").
            soul_md: Optional SOUL.md content or file path.
            extra_files: Optional dict of filename -> content to include.
            tool_allow: Tools to allow (default: exec, slack, memory_search,
                memory_get, read).
            tool_deny: Tools to deny (default: admin_*, shell).
            stop_rules: Optional stop rules list.

        Returns:
            OPP v1 pack dict, ready for JSON serialization.
        """
        agents_content = _read_or_use(agents_md)
        soul_content = _read_or_use(soul_md) if soul_md else None

        files = {"AGENTS.md": agents_content}
        if soul_content:
            files["SOUL.md"] = soul_content
        if extra_files:
            files.update(extra_files)

        if tool_allow is None:
            tool_allow = ["exec", "slack", "memory_search", "memory_get", "read"]
        if tool_deny is None:
            tool_deny = ["admin_*", "shell"]

        pack = {
            "schema_version": 1,
            "files": files,
            "tool_policy": {
                "allow": tool_allow,
                "deny": tool_deny,
            },
            "metadata": {
                "pack_name": pack_name,
                "pack_version": pack_version,
                "target_suite": "clawbench_v1",
            },
        }

        if stop_rules:
            pack["stop_rules"] = stop_rules

        return pack

    @staticmethod
    def load_pack(path: str) -> dict:
        """Load a pack.json from disk.

        Args:
            path: Path to pack.json file.

        Returns:
            Parsed pack dict.

        Raises:
            FileNotFoundError: If path doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def save_pack(pack: dict, path: str) -> str:
        """Save pack to disk and return its SHA256 hash.

        Args:
            pack: OPP v1 pack dict.
            path: Output file path.

        Returns:
            SHA256 hex digest of the canonical JSON.
        """
        canonical = json.dumps(pack, sort_keys=True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(canonical)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def compute_pack_hash(pack: dict) -> str:
        """Compute content-addressed SHA256 hash of a JSON pack (v4.0).

        Uses ``json.dumps(pack, sort_keys=True)`` for deterministic
        serialization, matching the validator's hash computation.

        Args:
            pack: OPP v1 pack dict.

        Returns:
            64-char lowercase hex SHA256 digest.
        """
        canonical = json.dumps(pack, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ------------------------------------------------------------------
    # S1: plain-text SKILL.md helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_text_hash(text_or_path: str) -> str:
        """Compute SHA256 hash of raw text content (S1 SKILL.md).

        If ``text_or_path`` is a path to an existing file, reads the file
        first.  Otherwise treats the string as content directly.

        Args:
            text_or_path: SKILL.md content string, or path to a file.

        Returns:
            64-char lowercase hex SHA256 digest.
        """
        content = _read_or_use(text_or_path)
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def save_text(text_or_path: str, output: str) -> str:
        """Save SKILL.md text to disk and return its SHA256 hash.

        Args:
            text_or_path: SKILL.md content string, or path to a source file.
            output: Output file path.

        Returns:
            64-char lowercase hex SHA256 digest.
        """
        content = _read_or_use(text_or_path)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(content)
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def validate(pack: dict) -> "ValidationResult":
        """Validate pack against OPP v1 schema locally.

        Args:
            pack: Pack dict to validate.

        Returns:
            ValidationResult with passed/issues.
        """
        return validate_opp_schema(pack)

    # ------------------------------------------------------------------
    # On-chain commitment
    # ------------------------------------------------------------------

    @staticmethod
    def format_commitment(
        pack_hash: str,
        pack_url: str,
    ) -> str:
        """Format a commitment string for on-chain submission.

        Args:
            pack_hash: SHA256 hex of submission content (64 chars).
            pack_url: HTTP(S) URL where the submission is hosted.

        Returns:
            Pipe-delimited commitment string (≤256 bytes).

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

        Args:
            pack_hash: SHA256 hex of submission content.
            pack_url: HTTP(S) URL where the submission is publicly accessible.

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

            # Verify the commitment actually landed on-chain
            stored = self.get_current_commitment()
            if stored is None:
                logger.warning(
                    "Commitment not found on-chain after submission. "
                    "It may take a few blocks to finalize, or the extrinsic may have failed silently."
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
                "Cannot connect to %s network. Check network connectivity "
                "and that the chain endpoint is reachable: %s", self.network, e,
            )
            return False
        except bt.KeyFileError as e:
            logger.error(
                "Wallet key file error: %s. "
                "Check wallet exists with: btcli wallet list", e,
            )
            return False
        except bt.ChainTransactionError as e:
            logger.error("Chain transaction failed: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to submit commitment (%s): %s", type(e).__name__, e)
            return False

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
            logger.error(
                "Cannot connect to %s network. Check network connectivity "
                "and that the chain endpoint is reachable: %s", self.network, e,
            )
            return None
        except bt.KeyFileError as e:
            logger.error(
                "Wallet key file error: %s. "
                "Check wallet exists with: btcli wallet list", e,
            )
            return None
        except Exception as e:
            logger.error("Failed to read commitment (%s): %s", type(e).__name__, e)
            return None

    # ------------------------------------------------------------------
    # Full submit workflow
    # ------------------------------------------------------------------

    def submit(
        self,
        pack: dict,
        pack_url: str,
    ) -> bool:
        """Full submission workflow: validate + commit on-chain.

        Args:
            pack: OPP v1 pack dict.
            pack_url: Public URL where pack.json is hosted.

        Returns:
            True if everything succeeded.
        """
        result = self.validate(pack)
        if not result.passed:
            logger.error(f"Pack validation failed: {result.issues}")
            return False

        pack_hash = self.compute_pack_hash(pack)
        logger.info(f"Pack hash: {pack_hash}")
        logger.info(f"Pack size: {len(json.dumps(pack))} bytes")

        return self.submit_commitment(pack_hash, pack_url)


def _read_or_use(value: str) -> str:
    """If value is a file path that exists, read it. Otherwise use as-is."""
    if len(value) > 4096:
        return value
    try:
        p = Path(value)
        if p.exists() and p.is_file():
            return p.read_text()
    except OSError:
        pass
    return value
