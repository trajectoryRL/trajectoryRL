"""Consensus payload and pointer data models.

These types define the wire format for cross-validator evaluation sharing
in the two-phase consensus protocol.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Optional


CONSENSUS_PROTOCOL_VERSION = 1


@dataclass
class ConsensusPayload:
    """Evaluation results published by one validator for one window.

    Content-addressed: the canonical JSON serialization determines the
    content hash used for CAS storage and integrity verification.
    """
    protocol_version: int
    window_number: int
    validator_hotkey: str
    software_version: str
    costs: Dict[str, float]         # miner hotkey -> EMA cost (USD)
    qualified: Dict[str, bool]      # miner hotkey -> qualification gate
    timestamp: int                  # unix seconds when payload was built

    def to_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "window_number": self.window_number,
            "validator_hotkey": self.validator_hotkey,
            "software_version": self.software_version,
            "costs": self.costs,
            "qualified": self.qualified,
            "timestamp": self.timestamp,
        }

    def serialize(self) -> bytes:
        """Canonical JSON serialization (sorted keys, no extra whitespace)."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def content_hash(self) -> str:
        """SHA-256 of canonical serialization, prefixed with 'sha256:'."""
        return "sha256:" + hashlib.sha256(self.serialize()).hexdigest()

    @classmethod
    def deserialize(cls, data: bytes) -> "ConsensusPayload":
        d = json.loads(data.decode("utf-8"))
        return cls(
            protocol_version=d["protocol_version"],
            window_number=d["window_number"],
            validator_hotkey=d["validator_hotkey"],
            software_version=d["software_version"],
            costs=d["costs"],
            qualified=d["qualified"],
            timestamp=d["timestamp"],
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ConsensusPayload":
        return cls(
            protocol_version=d["protocol_version"],
            window_number=d["window_number"],
            validator_hotkey=d["validator_hotkey"],
            software_version=d["software_version"],
            costs=d["costs"],
            qualified=d["qualified"],
            timestamp=d["timestamp"],
        )


def verify_payload_integrity(data: bytes, expected_hash: str) -> bool:
    """Verify that raw payload bytes match expected content hash.

    Works for both sha256: prefixed hashes and bare hex strings.
    """
    hex_digest = hashlib.sha256(data).hexdigest()
    expected_hex = expected_hash.removeprefix("sha256:")
    return hex_digest == expected_hex


@dataclass
class ConsensusPointer:
    """Lightweight pointer registered in the pointer registry.

    Points from (window_number, validator_hotkey) to a CAS content address
    where the full payload can be retrieved.
    """
    protocol_version: int
    window_number: int
    content_address: str            # "sha256:{hex}" or IPFS CID
    validator_hotkey: str

    def to_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "window_number": self.window_number,
            "content_address": self.content_address,
            "validator_hotkey": self.validator_hotkey,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConsensusPointer":
        return cls(
            protocol_version=d["protocol_version"],
            window_number=d["window_number"],
            content_address=d["content_address"],
            validator_hotkey=d["validator_hotkey"],
        )
