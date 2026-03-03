"""Synapse definitions for TrajectoryRL subnet communication.

Based on Bittensor synapse patterns:
- https://docs.learnbittensor.org/learn/neurons
- https://docs.learnbittensor.org/tutorials/ocr-subnet-tutorial
"""

import hashlib
import json
from typing import Any, Dict, Optional

import bittensor as bt
from pydantic import Field, field_validator


class PackRequest(bt.Synapse):
    """Validator requests the miner's current policy pack.

    Attributes:
        suite_id: Target task suite (e.g., "clawbench_v1")
        schema_version: OPP schema version (currently 1)
        max_bytes: Maximum inline payload size
        want_pointer_ok: If True, miner can return URL instead of inline
    """

    suite_id: str = Field(
        default="clawbench_v1",
        description="Target task suite identifier"
    )
    schema_version: int = Field(
        default=1,
        description="OpenClaw Policy Pack schema version"
    )
    max_bytes: int = Field(
        default=65536,  # 64KB
        description="Maximum size for inline pack_b64 payload"
    )
    want_pointer_ok: bool = Field(
        default=True,
        description="Whether validator accepts pack_url instead of inline"
    )


class PackResponse(bt.Synapse):
    """Miner returns their policy pack submission info.

    Attributes:
        pack_hash: SHA256 hash of the pack JSON (hex digest)
        pack_url: Public HTTP(S) URL where pack.json is hosted
        metadata: Declared pack metadata (not verified by validator)
    """

    pack_hash: str = Field(
        default="",
        description="SHA256 hash of pack content (hex digest)"
    )
    pack_url: str = Field(
        default="",
        description="Public HTTP(S) URL where pack.json is hosted"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pack metadata (author, version, target_suite, etc.)"
    )

    @field_validator("pack_hash")
    @classmethod
    def validate_pack_hash(cls, v: str) -> str:
        """Ensure pack_hash is a valid SHA256 hex digest."""
        if v and len(v) != 64:
            raise ValueError(f"pack_hash must be 64 hex chars, got {len(v)}")
        if v and not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("pack_hash must be lowercase hex")
        return v.lower()

    @field_validator("pack_url")
    @classmethod
    def validate_pack_url(cls, v: str) -> str:
        """Ensure pack_url is a valid HTTP(S) URL."""
        if v and not v.startswith(("https://", "http://")):
            raise ValueError("pack_url must be an HTTP(S) URL")
        return v

    @staticmethod
    def compute_pack_hash(pack: dict) -> str:
        """Compute SHA256 hash from pack dict.

        Args:
            pack: Policy pack dictionary

        Returns:
            Hex digest of SHA256 hash
        """
        content = json.dumps(pack, sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()
