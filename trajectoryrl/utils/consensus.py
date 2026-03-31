"""Consensus payload and pointer data models.

These types define the wire format for cross-validator evaluation sharing
in the two-phase consensus protocol.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict


CONSENSUS_PROTOCOL_VERSION = 1

# Scoring version tracks evaluation criteria (scenario set, rubric, judge logic).
# Bump this whenever scenarios are added/removed or scoring semantics change
# so that results from different versions are never mixed during aggregation,
# cached-result lookup, or winner selection.
SCORING_VERSION = 1


@dataclass
class ConsensusPayload:
    """Evaluation results published by one validator for one window.

    Content-addressed: the canonical JSON serialization determines the
    content hash used for CAS storage and integrity verification.
    """
    protocol_version: int
    window_number: int
    validator_hotkey: str
    clawbench_version: str
    costs: Dict[str, float]         # miner hotkey -> EMA cost (USD)
    qualified: Dict[str, bool]      # miner hotkey -> qualification gate (all known miners)
    timestamp: int                  # unix seconds when payload was built
    scoring_version: int = 1        # evaluation criteria version (scenario set + rubric)
    disqualified: Dict[str, str] = field(default_factory=dict)
    # miner hotkey -> reason (e.g. "pre_eval_rejected", "integrity_failed")
    # miners in disqualified also appear in qualified with value=False

    def to_dict(self) -> dict:
        d = {
            "clawbench_version": self.clawbench_version,
            "costs": self.costs,
            "disqualified": self.disqualified,
            "protocol_version": self.protocol_version,
            "qualified": self.qualified,
            "scoring_version": self.scoring_version,
            "timestamp": self.timestamp,
            "validator_hotkey": self.validator_hotkey,
            "window_number": self.window_number,
        }
        return d

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
            clawbench_version=d.get("clawbench_version", d.get("software_version", "")),
            costs=d["costs"],
            qualified=d["qualified"],
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
            clawbench_version=d.get("clawbench_version", d.get("software_version", "")),
            costs=d["costs"],
            qualified=d["qualified"],
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
