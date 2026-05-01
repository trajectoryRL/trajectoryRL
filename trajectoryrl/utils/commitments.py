"""On-chain commitment reading and parsing for TrajectoryRL validators.

Miners submit pack metadata via ``subtensor.set_commitment(netuid, data)``.
Validators read all commitments with ``subtensor.get_all_commitments(netuid)``
and parse them into structured ``MinerCommitment`` objects.

Two commitment formats coexist on the same ``set_commitment`` / ``get_all_commitments`` channel:

Miner commitment (pipe-delimited, ≤256 bytes):
    {pack_hash_hex}|{pack_url}

Validator consensus commitment (pipe-delimited, ≤256 bytes):
    consensus:{protocol_version}|{window_number}|{content_address}

``content_address`` may be a single CAS address (IPFS CID or GCS URL) or
a dual-address string ``{ipfs_cid};{gcs_url}`` when both backends
succeeded.  Use ``decode_dual_address`` / ``encode_dual_address`` to
convert between the wire format and separate (ipfs, gcs) components.

The ``consensus:`` prefix distinguishes validator consensus pointers from
miner pack commitments.

Reference: INCENTIVE_MECHANISM.md § Submission Protocol
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HTTP_URL = re.compile(r"^https?://\S+$")


_CONSENSUS_PREFIX = "consensus:"
DUAL_ADDRESS_SEPARATOR = ";"


@dataclass
class MinerCommitment:
    """Parsed on-chain commitment from a miner."""

    uid: int
    hotkey: str
    pack_hash: str  # SHA256 hex, 64 chars
    pack_url: str  # HTTP(S) URL to pack.json
    block_number: int  # on-chain block when commitment was set
    raw: str  # original commitment string


@dataclass
class ValidatorConsensusCommitment:
    """Parsed on-chain consensus pointer from a validator.

    Written by validators after uploading evaluation payloads to CAS.
    Other validators discover these by reading ``get_all_commitments``
    and filtering for the ``consensus:`` prefix.
    """

    protocol_version: int
    window_number: int
    content_address: str  # IPFS CID or GCS URL
    validator_hotkey: str
    block_number: int
    raw: str
    spec_number: int = 1


def encode_dual_address(
    ipfs_cid: Optional[str] = None,
    gcs_url: Optional[str] = None,
) -> Optional[str]:
    """Encode IPFS CID and/or GCS URL into a single content-address string.

    When both are present, joins them with ``;`` so both can be recovered
    during download.  When only one is available the raw address is returned
    unchanged, which keeps the format identical to the legacy single-address
    encoding.

    Returns None if neither address is provided.
    """
    if ipfs_cid and gcs_url:
        return f"{ipfs_cid}{DUAL_ADDRESS_SEPARATOR}{gcs_url}"
    return ipfs_cid or gcs_url or None


def decode_dual_address(
    content_address: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Decode a content-address string into (ipfs_cid, gcs_url).

    Handles three formats:
    * Dual:   ``{ipfs_cid};{gcs_url}``  → both fields populated
    * Legacy URL:  ``https://…``          → (None, url)
    * Legacy CID:  ``Qm…`` / ``bafy…``   → (cid, None)
    """
    if DUAL_ADDRESS_SEPARATOR in content_address:
        parts = content_address.split(DUAL_ADDRESS_SEPARATOR, maxsplit=1)
        ipfs_cid = parts[0].strip() or None
        gcs_url = parts[1].strip() or None
        return ipfs_cid, gcs_url

    addr = content_address.strip()
    if addr.startswith("http://") or addr.startswith("https://"):
        return None, addr
    return addr, None


def is_consensus_commitment(raw: str) -> bool:
    """Check if a raw commitment string is a validator consensus pointer."""
    return isinstance(raw, str) and raw.strip().startswith(_CONSENSUS_PREFIX)


def parse_consensus_commitment(raw: str) -> Optional[Tuple[int, int, str, int]]:
    """Parse a validator consensus commitment string.

    Supports two formats:
        Old (3-field): ``consensus:{pv}|{window}|{content_address}``
        New (4-field): ``consensus:{pv}|{window}|{spec_number}|{content_address}``

    The integer at field 3 is the scoring spec identifier. Older commitments
    written under the legacy ``scoring_version`` name parse to the same int
    (the on-chain wire format is positional, so the rename does not break
    backward compatibility). Old 3-field commitments default to
    ``spec_number=1``.

    Returns:
        Tuple of (protocol_version, window_number, content_address,
        spec_number) or None if unparseable.
    """
    if not raw or not isinstance(raw, str):
        return None

    raw = raw.strip()
    if not raw.startswith(_CONSENSUS_PREFIX):
        return None

    body = raw[len(_CONSENSUS_PREFIX):]
    parts = body.split("|", maxsplit=3)

    if len(parts) == 3:
        try:
            protocol_version = int(parts[0].strip())
            window_number = int(parts[1].strip())
        except (ValueError, TypeError):
            return None
        content_address = parts[2].strip()
        if not content_address:
            return None
        return protocol_version, window_number, content_address, 1

    if len(parts) == 4:
        try:
            protocol_version = int(parts[0].strip())
            window_number = int(parts[1].strip())
            spec_number = int(parts[2].strip())
        except (ValueError, TypeError):
            return None
        content_address = parts[3].strip()
        if not content_address:
            return None
        return protocol_version, window_number, content_address, spec_number

    return None


def format_consensus_commitment(
    protocol_version: int,
    window_number: int,
    content_address: str,
    spec_number: int = 1,
) -> str:
    """Build a consensus commitment string for ``set_commitment``.

    Returns:
        Formatted string:
        ``consensus:{protocol_version}|{window_number}|{spec_number}|{content_address}``
    """
    return (
        f"{_CONSENSUS_PREFIX}{protocol_version}|{window_number}"
        f"|{spec_number}|{content_address}"
    )


def parse_commitment(raw: str) -> Optional[Tuple[str, str]]:
    """Parse a miner commitment string into (pack_hash, pack_url).

    Supports pipe-delimited format:
        ``{pack_hash}|{pack_url}``

    Consensus commitments (``consensus:`` prefix) are not miner
    commitments and return None.

    Returns:
        Tuple of (pack_hash, pack_url) or None if unparseable.
    """
    if not raw or not isinstance(raw, str):
        return None

    raw = raw.strip()
    if raw.startswith(_CONSENSUS_PREFIX):
        return None

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
    *,
    reconnect: Optional[Callable[[], object]] = None,
) -> Optional[Dict[int, MinerCommitment]]:
    """Read all miner commitments from the chain.

    Calls ``subtensor.get_all_commitments(netuid)`` and parses each entry.
    Skips UIDs with no commitment or unparseable data.

    Args:
        subtensor: Bittensor subtensor instance.
        netuid: Subnet UID (11 for TrajectoryRL).
        metagraph: Bittensor metagraph for hotkey/UID mapping.
        reconnect: Optional callback that rebuilds the subtensor connection
            and returns the new instance. Invoked once if the chain query
            raises (typically due to a stale websocket). The retry uses the
            returned subtensor.

    Returns:
        ``None`` if the chain query failed (caller MUST NOT persist this
        as an empty active set — see ``_acquire_window_snapshot``).
        ``{}`` if the chain query succeeded but returned no commitments.
        A populated dict (UID → MinerCommitment) otherwise.
    """
    try:
        raw_commitments = subtensor.get_all_commitments(netuid=netuid)
    except Exception as e:
        if reconnect is None:
            logger.error(
                f"Failed to read on-chain commitments (no reconnect cb): {e}"
            )
            return None
        logger.warning(
            f"Failed to read on-chain commitments: {e} — reconnecting and retrying once"
        )
        try:
            subtensor = reconnect()
            raw_commitments = subtensor.get_all_commitments(netuid=netuid)
        except Exception as e2:
            logger.error(
                f"Failed to read on-chain commitments after reconnect: {e2}"
            )
            return None

    commitments: Dict[int, MinerCommitment] = {}

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
        logger.debug(
            f"UID {uid}: commitment found — hash={pack_hash[:12]}… "
            f"url={pack_url} block={block_number}"
        )

    if commitments:
        uids = sorted(commitments.keys())
        logger.info(
            f"Fetched {len(commitments)} commitments "
            f"(UIDs: {uids[0]}–{uids[-1]})"
        )
    else:
        logger.info("Fetched 0 commitments")

    return commitments


def fetch_validator_consensus_commitments(
    subtensor,
    netuid: int,
    metagraph,
) -> List[ValidatorConsensusCommitment]:
    """Read all validator consensus commitments from the chain.

    Calls ``get_all_commitments`` and filters for entries with the
    ``consensus:`` prefix.  Only entries from hotkeys that hold a
    ``validator_permit`` in the metagraph are returned; this prevents
    miners from injecting fake consensus data by submitting a
    ``consensus:``-prefixed commitment.

    Returns:
        List of ValidatorConsensusCommitment from permitted validators.
    """
    results: List[ValidatorConsensusCommitment] = []

    try:
        raw_commitments = subtensor.get_all_commitments(netuid=netuid)
    except Exception as e:
        logger.error(f"Failed to read on-chain commitments: {e}")
        return results

    if not raw_commitments:
        return results

    permitted_validators: set = set()
    permit_check_available = False
    try:
        hotkeys = metagraph.hotkeys
        permits = metagraph.validator_permit
        for uid in range(len(hotkeys)):
            if uid < len(permits) and permits[uid]:
                permitted_validators.add(hotkeys[uid])
        permit_check_available = True
    except Exception as e:
        logger.warning(
            "Failed to read validator_permit from metagraph: %s. "
            "Falling back to prefix-only filtering.", e,
        )

    skipped_non_validator = 0
    for hotkey, raw in raw_commitments.items():
        if not is_consensus_commitment(raw):
            continue

        if permit_check_available and hotkey not in permitted_validators:
            skipped_non_validator += 1
            logger.debug(
                "Hotkey %s: consensus commitment ignored — "
                "no validator_permit in metagraph",
                hotkey[:8],
            )
            continue

        parsed = parse_consensus_commitment(raw)
        if parsed is None:
            logger.debug(
                "Validator %s: unparseable consensus commitment, skipping",
                hotkey[:8],
            )
            continue

        protocol_version, window_number, content_address, spec_number = parsed
        block_number = _get_commitment_block(subtensor, netuid, hotkey)

        results.append(ValidatorConsensusCommitment(
            protocol_version=protocol_version,
            window_number=window_number,
            content_address=content_address,
            validator_hotkey=hotkey,
            block_number=block_number,
            raw=raw,
            spec_number=spec_number,
        ))
        logger.debug(
            "Validator %s: consensus commitment — v%d window=%d spec=%d addr=%s",
            hotkey[:8], protocol_version, window_number,
            spec_number, content_address[:24],
        )

    if skipped_non_validator:
        logger.info(
            "Skipped %d consensus commitments from non-validator hotkeys",
            skipped_non_validator,
        )
    logger.info(
        "Fetched %d validator consensus commitments", len(results),
    )
    return results


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
