"""Consensus payload and pointer data models.

These types define the wire format for cross-validator evaluation sharing
in the two-phase consensus protocol.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict


CONSENSUS_PROTOCOL_VERSION = 2

# Scoring version = major version of trajrl-bench (e.g. v3.0.1 → 3).
# Overwritten at runtime by TrajectorySandboxHarness.scoring_version after
# pulling the sandbox image.  This default is only used before the first pull.
# Results from different scoring versions are never mixed during aggregation,
# cached-result lookup, or winner selection.
SCORING_VERSION = 1


@dataclass
class ConsensusPayload:
    """Evaluation results published by one validator for one window.

    Content-addressed: the canonical JSON serialization determines the
    content hash used for CAS storage and integrity verification.

    Protocol v2 fields:
      - scores: miner hotkey -> quality score (0.0–1.0)
      - bench_version: trajrl-bench version string
      - disqualified: miner hotkey -> reason
    """
    protocol_version: int
    window_number: int
    validator_hotkey: str
    bench_version: str
    scores: Dict[str, float]        # miner hotkey -> quality score (0.0–1.0)
    timestamp: int                   # unix seconds when payload was built
    scoring_version: int = 1         # major version of trajrl-bench
    disqualified: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "bench_version": self.bench_version,
            "disqualified": self.disqualified,
            "protocol_version": self.protocol_version,
            "scores": self.scores,
            "scoring_version": self.scoring_version,
            "timestamp": self.timestamp,
            "validator_hotkey": self.validator_hotkey,
            "window_number": self.window_number,
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
            bench_version=d.get("bench_version", ""),
            scores=d.get("scores", {}),
            timestamp=d["timestamp"],
            scoring_version=d.get("scoring_version", 1),
            disqualified=d.get("disqualified", {}),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ConsensusPayload":
        return cls(
            protocol_version=d["protocol_version"],
            window_number=d["window_number"],
            validator_hotkey=d["validator_hotkey"],
            bench_version=d.get("bench_version", ""),
            scores=d.get("scores", {}),
            timestamp=d["timestamp"],
            scoring_version=d.get("scoring_version", 1),
            disqualified=d.get("disqualified", {}),
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

    ``content_address`` stores the raw on-chain string.  It may be a single
    CAS address (IPFS CID or GCS URL) or a dual-address string
    ``{ipfs_cid};{gcs_url}`` when both backends succeeded.
    Use :func:`commitments.decode_dual_address` to split into components.
    """
    protocol_version: int
    window_number: int
    content_address: str            # raw on-chain value; may contain ";" for dual-address
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
