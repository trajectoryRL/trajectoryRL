"""On-chain commitment reading and parsing for TrajectoryRL validators.

Miners submit pack metadata via ``subtensor.set_commitment(netuid, data)``.
Validators read all commitments with ``subtensor.get_all_commitments(netuid)``
and parse them into structured ``MinerCommitment`` objects.

Commitment wire format (pipe-delimited, ≤256 bytes):
    {pack_hash_hex}|{pack_url}

Where:
    pack_hash_hex: SHA256 hex digest of the canonical pack JSON (64 chars)
    pack_url: HTTP(S) URL where validators can GET the pack.json
              (any public HTTP(S) endpoint)

Reference: INCENTIVE_MECHANISM.md § Submission Protocol
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HTTP_URL = re.compile(r"^https?://\S+$")


@dataclass
class MinerCommitment:
    """Parsed on-chain commitment from a miner."""

    uid: int
    hotkey: str
    pack_hash: str  # SHA256 hex, 64 chars
    pack_url: str  # HTTP(S) URL to pack.json
    block_number: int  # on-chain block when commitment was set
    raw: str  # original commitment string


def parse_commitment(raw: str) -> Optional[Tuple[str, str]]:
    """Parse a compact commitment string into (pack_hash, pack_url).

    Supports pipe-delimited format:
        ``{pack_hash}|{pack_url}``

    Returns:
        Tuple of (pack_hash, pack_url) or None if unparseable.
    """
    if not raw or not isinstance(raw, str):
        return None

    raw = raw.strip()
    parts = raw.split("|", maxsplit=1)
    if len(parts) != 2:
        return None

    pack_hash, pack_url = parts[0].strip(), parts[1].strip()

    if not _HEX64.match(pack_hash):
        return None

    if not _HTTP_URL.match(pack_url):
        return None

    return pack_hash, pack_url


def fetch_all_commitments(
    subtensor,
    netuid: int,
    metagraph,
) -> Dict[int, MinerCommitment]:
    """Read all miner commitments from the chain.

    Calls ``subtensor.get_all_commitments(netuid)`` and parses each entry.
    Skips UIDs with no commitment or unparseable data.

    Args:
        subtensor: Bittensor subtensor instance.
        netuid: Subnet UID (11 for TrajectoryRL).
        metagraph: Bittensor metagraph for hotkey/UID mapping.

    Returns:
        Dict mapping UID -> MinerCommitment for all valid entries.
    """
    commitments: Dict[int, MinerCommitment] = {}

    try:
        raw_commitments = subtensor.get_all_commitments(netuid=netuid)
    except Exception as e:
        logger.error(f"Failed to read on-chain commitments: {e}")
        return commitments

    if not raw_commitments:
        return commitments

    # Build hotkey -> uid mapping
    hotkey_to_uid: Dict[str, int] = {}
    for uid in range(len(metagraph.hotkeys)):
        hotkey_to_uid[metagraph.hotkeys[uid]] = uid

    # raw_commitments: {hotkey_ss58: commitment_string}
    for hotkey, raw in raw_commitments.items():
        uid = hotkey_to_uid.get(hotkey)
        if uid is None:
            continue

        parsed = parse_commitment(raw)
        if parsed is None:
            logger.debug(f"UID {uid}: unparseable commitment, skipping")
            continue

        pack_hash, pack_url = parsed

        # Get commitment block number for first-mover ordering
        block_number = _get_commitment_block(subtensor, netuid, hotkey)

        commitments[uid] = MinerCommitment(
            uid=uid,
            hotkey=hotkey,
            pack_hash=pack_hash,
            pack_url=pack_url,
            block_number=block_number,
            raw=raw,
        )
        logger.info(
            f"UID {uid}: commitment found — hash={pack_hash[:12]}… "
            f"url={pack_url} block={block_number}"
        )

    return commitments


def _get_commitment_block(subtensor, netuid: int, hotkey: str) -> int:
    """Get the block number at which a UID's commitment was set.

    Tries ``get_commitment_metadata`` first; falls back to current block.
    """
    try:
        meta = subtensor.get_commitment_metadata(netuid=netuid, hotkey_ss58=hotkey)
        if meta and isinstance(meta, dict) and "block" in meta:
            return int(meta["block"])
    except Exception as e:
        logger.debug(f"get_commitment_metadata failed for {hotkey[:8]}… ({e}), falling back to current block")

    # Fallback: use current block (less precise but functional).
    # This inflates the apparent first-mover block number so the miner
    # looks like a late submitter — safer than defaulting to 0 (which
    # would give them an artificially early timestamp).
    try:
        return subtensor.get_current_block()
    except Exception:
        # Both API calls failed.  Return a large sentinel so the miner
        # is treated as a late submitter rather than getting an
        # artificially early timestamp (the previous default of 0).
        return 2**63
