"""Dual-backend CAS for consensus payloads: IPFS primary, trajrl.com fallback.

Upload: try IPFS first, then trajrl.com API if IPFS fails.
Download: try IPFS first, then trajrl.com API if IPFS fails.
Both backends are written on upload (best-effort) for redundancy.

Pointer registry: trajrl.com API serves as the pointer registry.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import aiohttp

from .consensus import (
    ConsensusPayload, ConsensusPointer,
    verify_payload_integrity,
)

logger = logging.getLogger(__name__)


class CASBackend(ABC):
    """Abstract content-addressed storage backend."""

    @abstractmethod
    async def upload(self, data: bytes) -> Optional[str]:
        """Upload data, return content address or None on failure."""
        ...

    @abstractmethod
    async def download(self, address: str) -> Optional[bytes]:
        """Download data by content address, return bytes or None."""
        ...


class IPFSBackend(CASBackend):
    """IPFS via local kubo node HTTP API.

    Upload:   POST /api/v0/add (auto-pins)
    Download: POST /api/v0/cat?arg={cid}
    """

    def __init__(self, api_url: str = "http://localhost:5001"):
        self.api_url = api_url.rstrip("/")

    async def upload(self, data: bytes) -> Optional[str]:
        try:
            url = f"{self.api_url}/api/v0/add"
            form = aiohttp.FormData()
            form.add_field("file", data, content_type="application/octet-stream")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.warning("IPFS upload failed: HTTP %d", resp.status)
                        return None
                    result = await resp.json()
                    cid = result.get("Hash")
                    if cid:
                        logger.info("IPFS upload OK: CID=%s", cid)
                    return cid
        except Exception as e:
            logger.warning("IPFS upload error: %s", e)
            return None

    async def download(self, address: str) -> Optional[bytes]:
        try:
            url = f"{self.api_url}/api/v0/cat"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    params={"arg": address},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("IPFS download failed: HTTP %d (CID=%s)", resp.status, address)
                        return None
                    data = await resp.read()
                    logger.debug("IPFS download OK: CID=%s, %d bytes", address, len(data))
                    return data
        except Exception as e:
            logger.warning("IPFS download error (CID=%s): %s", address, e)
            return None


class TrajRLAPIBackend(CASBackend):
    """Centralized fallback via trajrl.com API.

    Upload:   POST /api/v1/consensus/payload
    Download: GET  /api/v1/consensus/payload/{content_hash}
    """

    def __init__(
        self,
        base_url: str = "https://api.trajrl.com",
        sign_fn=None,
        validator_hotkey: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self._sign_fn = sign_fn
        self._validator_hotkey = validator_hotkey

    def _auth_headers(self, timestamp: int) -> dict:
        """Build authentication headers for trajrl.com API."""
        headers = {"Content-Type": "application/json"}
        return headers

    async def upload(self, data: bytes) -> Optional[str]:
        """Upload payload via API. Returns sha256: content hash."""
        content_hash = "sha256:" + hashlib.sha256(data).hexdigest()
        try:
            import time
            ts = int(time.time())
            payload_dict = json.loads(data.decode("utf-8"))

            body = {
                "validator_hotkey": self._validator_hotkey,
                "timestamp": ts,
                "signature": "",
                "payload": payload_dict,
            }
            if self._sign_fn:
                msg = f"trajectoryrl-consensus:{self._validator_hotkey}:{ts}"
                body["signature"] = self._sign_fn(msg)

            url = f"{self.base_url}/api/v1/consensus/payload"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=body,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status in (200, 409):
                        logger.info("API upload OK: hash=%s", content_hash)
                        return content_hash
                    logger.warning("API upload failed: HTTP %d", resp.status)
                    return None
        except Exception as e:
            logger.warning("API upload error: %s", e)
            return None

    async def download(self, address: str) -> Optional[bytes]:
        """Download payload by content hash from API."""
        try:
            normalized = address.removeprefix("sha256:")
            url = f"{self.base_url}/api/v1/consensus/payload/sha256:{normalized}"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("API download failed: HTTP %d (hash=%s)", resp.status, address)
                        return None
                    result = await resp.json()
                    payload_dict = result.get("payload")
                    if payload_dict is None:
                        return None
                    data = json.dumps(payload_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    logger.debug("API download OK: hash=%s, %d bytes", address, len(data))
                    return data
        except Exception as e:
            logger.warning("API download error (hash=%s): %s", address, e)
            return None


class ConsensusStore:
    """Dual-backend CAS with IPFS primary, trajrl.com API fallback.

    Pointer registry is always via the API backend.
    """

    def __init__(self, ipfs: IPFSBackend, api: TrajRLAPIBackend):
        self.ipfs = ipfs
        self.api = api

    async def upload_payload(self, payload: ConsensusPayload) -> Optional[str]:
        """Upload payload to CAS. Returns content address or None.

        Strategy: try IPFS first, then API. Best-effort mirror to both.
        """
        data = payload.serialize()
        sha_hash = payload.content_hash()

        ipfs_cid = await self.ipfs.upload(data)

        api_hash = await self.api.upload(data)

        if ipfs_cid:
            return ipfs_cid
        if api_hash:
            return api_hash

        logger.error("Both IPFS and API upload failed for window %d", payload.window_number)
        return None

    async def download_payload(self, content_address: str) -> Optional[ConsensusPayload]:
        """Download and verify payload from CAS.

        Try IPFS first (if address looks like a CID), then API.
        Always verify integrity via sha256.
        """
        data = None

        is_ipfs_cid = not content_address.startswith("sha256:")
        if is_ipfs_cid:
            data = await self.ipfs.download(content_address)

        if data is None:
            sha_addr = content_address if content_address.startswith("sha256:") else None
            if sha_addr:
                data = await self.api.download(sha_addr)

        if data is None:
            if is_ipfs_cid:
                data = await self.api.download(content_address)
            if data is None:
                logger.warning("Failed to download payload: %s", content_address)
                return None

        try:
            payload = ConsensusPayload.deserialize(data)
        except Exception as e:
            logger.warning("Failed to deserialize payload %s: %s", content_address, e)
            return None

        expected_hash = payload.content_hash()
        if not verify_payload_integrity(data, expected_hash):
            logger.warning(
                "Payload integrity check failed: address=%s, computed=%s",
                content_address, expected_hash,
            )
            return None

        return payload

    async def write_pointer(self, pointer: ConsensusPointer, sign_fn=None) -> bool:
        """Register a pointer via the API backend."""
        try:
            import time
            ts = int(time.time())
            body = {
                "validator_hotkey": pointer.validator_hotkey,
                "timestamp": ts,
                "signature": "",
                "pointer": pointer.to_dict(),
            }
            if sign_fn:
                msg = f"trajectoryrl-consensus:{pointer.validator_hotkey}:{ts}"
                body["signature"] = sign_fn(msg)

            url = f"{self.api.base_url}/api/v1/consensus/pointer"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=body,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        logger.info(
                            "Pointer registered: window=%d, validator=%s",
                            pointer.window_number, pointer.validator_hotkey[:8],
                        )
                        return True
                    logger.warning("Pointer registration failed: HTTP %d", resp.status)
                    return False
        except Exception as e:
            logger.warning("Pointer registration error: %s", e)
            return False

    async def read_all_pointers(self, window_number: int) -> List[ConsensusPointer]:
        """Read all pointers for a window from the API backend."""
        try:
            url = f"{self.api.base_url}/api/v1/consensus/pointers/{window_number}"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "Read pointers failed: HTTP %d (window=%d)",
                            resp.status, window_number,
                        )
                        return []
                    result = await resp.json()
                    raw_pointers = result.get("pointers", [])
                    pointers = [ConsensusPointer.from_dict(p) for p in raw_pointers]
                    logger.info(
                        "Read %d pointers for window %d",
                        len(pointers), window_number,
                    )
                    return pointers
        except Exception as e:
            logger.warning("Read pointers error (window=%d): %s", window_number, e)
            return []
